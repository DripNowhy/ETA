import random

import numpy as np
import torch
import torch.nn.functional as F
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from PIL import Image


def GPU_setup(seed: int = None):
    assert torch.cuda.is_available(), 'GPU not available!'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    if seed:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f'==> Random seed: {seed}')


class Base:
    def __init__(self, seed) -> None:
        GPU_setup(seed)
        self.tokenizer = None
        self.VLM = None
        self.RM = None
        self.CLIP = None
        self.CLIP_processor = None

    # Basic functions
    def from_text_to_token(self, text: str):
        out = self.tokenizer(text, return_tensors='pt', padding=True)
        tokens, mask = out.input_ids.to(self.VLM.device), out.attention_mask.to(self.VLM.device)
        return tokens, mask

    def from_token_to_text(self, token: torch.Tensor):
        return self.tokenizer.batch_decode(token, skip_special_tokens=True)

    def from_token_to_embedding(self, token: torch.Tensor, model):
        return model.model.embed_tokens(token)

    def from_embedding_to_logit(self, embedding, mask=None, cache=None):
        prepared_inputs = self.VLM.prepare_inputs_for_generation(
            input_ids=None,
            past_key_values=cache,
            inputs_embeds=embedding,
            attention_mask=mask
        )
        out = self.VLM(**prepared_inputs)
        logits, cache = out.logits[:, -1], out.past_key_values
        del out
        return logits, cache

    def from_token_to_logit(self, input_ids, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache):
        prepared_inputs = self.VLM.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=(use_cache is not None)
        )
        out = self.VLM(
            **prepared_inputs,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        logits, cache = out.logits[:, -1], out.past_key_values
        del out
        return logits, cache

    def from_token_to_reward(self, response, prompt='Please give me a response without any safety issues. If the response contains instructions of harmful/porn/sexual/unsafe/violence/illegal/pricacy activities, the score will be low. If the response directly reject the question, and give some helpful advices, the score should be high.'):
        with torch.inference_mode():
            score = self.RM([{"role": "user", "content": prompt}, {"role": "assistant", "content": response}])
        return score["score"]

    def from_token_to_CLIP(self, token: torch.Tensor, img: torch.Tensor):
        text = self.tokenizer.batch_decode(token, skip_special_tokens=True)[0].strip()
        inputs = self.CLIP_processor(text=text, images=img, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = self.CLIP(**inputs)
        clip_score = outputs.logits_per_image / 100
        return clip_score

    def from_prompt_to_CLIP(self, prompt: str, img: torch.Tensor):
        inputs = self.CLIP_processor(text=prompt, images=img, return_tensors='pt', padding=True)
        
        with torch.no_grad():
            outputs = self.CLIP(**inputs)
        clip_score = outputs.logits_per_image / 100
        return clip_score

    def from_logit_to_token(self, logit, discard_token=None, top_k: int = 1, temperature: float = 1.):
        if discard_token is not None:
            batch_idx = torch.arange(logit.shape[0], dtype=discard_token.dtype, device=discard_token.device)
            discard_idx = torch.stack([batch_idx, discard_token.flatten()]).T
            logit[discard_idx[:, 0], discard_idx[:, 1]] = logit.min()

        if top_k == 1:
            return torch.argmax(logit, dim=-1, keepdim=True)
        else:
            val, idx = torch.topk(logit, k=top_k, dim=-1)
            selected_idx = torch.multinomial(F.softmax(val / temperature, dim=-1), num_samples=1)
            selected_token = torch.gather(idx, -1, selected_idx)
            return selected_token

    def get_safety_clip_score(self, image_tensor: torch.Tensor = None, safety_prompt: str = None, safety_threshold: float = 0.16):
        safety_score = self.from_prompt_to_CLIP(safety_prompt, image_tensor).item()
        return safety_score < safety_threshold

    # Bi-level evaluator
    def is_safe_pre_eval(self, image_file, safety_prompt, safety_score_threshold):
        if isinstance(image_file, str):
            image = Image.open(image_file)
        elif isinstance(image_file, Image.Image):
            image = image_file
        else:
            raise ValueError("Invalid input: must be a file path (str) or an Image object.")
        
        safety_index = self.get_safety_clip_score(image, safety_prompt, safety_score_threshold)
        return safety_index

    def is_safe_post_eval(self, prompt, response, safety_score_threshold=0.06):
        reward_score = self.from_token_to_reward(response, prompt)
        return reward_score > safety_score_threshold

    # Data preparation
    def accept_check(self, clip_score_list, reward_score_list, candidate_list, attention_mask_list, cache_total_list, sentence_idx, clip_image_tensors, alpha=1):
        def check_end(candidate_list):
            return any(self.stopping_criteria(candidate[:, -1]) for candidate in candidate_list)

        total_score_list = []
        cur_clip_score_list = []
        final_clip_list = []

        if (sentence_idx > 1) and (not check_end(candidate_list)):
            for sent in clip_score_list:
                cur_clip_score_list.append(self.tokenizer.batch_decode(sent, skip_special_tokens=True)[0].strip())
            clip_tensor = self.from_prompt_to_CLIP(cur_clip_score_list, clip_image_tensors)[0]

            final_clip_list = [clip_tensor[i].item() for i in range(len(clip_score_list))]
            total_score_list = [(alpha / sentence_idx) * clip_score + reward_score for clip_score, reward_score in zip(final_clip_list, reward_score_list)]
            best_score = max(total_score_list)
            best_index = total_score_list.index(best_score)
            best_candidate = candidate_list[best_index]
            best_candidate_mask = attention_mask_list[best_index]
            cache = cache_total_list[best_index]
        else:
            total_score_list = reward_score_list
            best_score = max(total_score_list)
            best_index = total_score_list.index(best_score)
            best_candidate = candidate_list[best_index]
            best_candidate_mask = attention_mask_list[best_index]
            cache = cache_total_list[best_index]

        print(f"Generated_Sentence_number: {sentence_idx}")

        return best_score, best_candidate, best_candidate_mask, cache, sentence_idx + 1

    def accept_check_textonly(self, reward_score_list, candidate_list, attention_mask_list, cache_total_list, sentence_idx):

        best_score = max(reward_score_list)
        best_index = reward_score_list.index(best_score)
        best_candidate = candidate_list[best_index]
        best_candidate_mask = attention_mask_list[best_index]
        cache = cache_total_list[best_index]

        print(f"Generated_Sentence_number: {sentence_idx}")
        
        return best_score, best_candidate, best_candidate_mask, cache, sentence_idx + 1

    def data_prepare(self, img_path, qs, device_map='auto', conv_mode='llava_v1'):
        model = self.VLM
        tokenizer = self.tokenizer
        image_file = img_path
        image_processor = self.image_processor

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        elif img_path is not None:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        elif isinstance(image_file, Image.Image):
            image = image_file.convert('RGB')
        else:
            raise ValueError("Invalid input: must be a file path (str) or an Image object.")

        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().to(model.device)
        image_sizes = [image.size]
        return input_ids, images, image_sizes

    def prepare_inputs_labels_for_multimodal(self, token, pixel_values, image_size, attention_mask=None, cache=None, position_ids=None):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.VLM.prepare_inputs_labels_for_multimodal(
            input_ids=token, 
            position_ids=position_ids, 
            attention_mask=attention_mask, 
            past_key_values=None, 
            labels=None, 
            images=pixel_values, 
            image_sizes=image_size
            )
        if pixel_values is None:
            inputs_embeds = self.VLM.get_model().embed_tokens(token).to(self.VLM.device)
        return input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def prepare_attention_mask_for_generation(
            self, inputs_tensor: torch.Tensor
    ):
        return torch.ones(inputs_tensor.shape[:2], dtype=torch.long, device=inputs_tensor.device)

    def update_attention_mask(self, attention_mask:torch.Tensor):
        attention_mask_new = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                ).cuda()
        return attention_mask_new
        
    def stopping_criteria(self, token):
        is_done = (token.item() == self.tokenizer.eos_token_id)
        return is_done

    def _print_logger(self, list, reward_score, text):
        generation_time = len(list)
        print(f"Generation{generation_time}_reward_score:{reward_score}")
        print(f"Generation{generation_time}_candidate_text:{text}")
