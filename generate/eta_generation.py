import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from llava.model.builder import load_pretrained_model
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    AutoModel
)

from .base_class import Base


class ETA(Base):
    def __init__(
            self,
            vlm_type:str,
            clip_dir:str=None,
            rm_dir:str=None,
            cache_dir:str=None,
            access_token:str=None,

            seed:int=1,
            fp_bit:int=16,
            device_map:str='auto',
            model_base:str=None,
            torch_dtype=torch.bfloat16,
        ):
        super(ETA, self).__init__(seed=seed)

        assert fp_bit in {4, 8, 16, 32}, 'fp_bit must be one of {4, 8, 16, 32}!'

        # print("==> Loading tokenizer...")
        # self.tokenizer = AutoTokenizer.from_pretrained(vlm_dir, token=access_token, cache_dir=cache_dir)

        if vlm_type == 'llava':
            vlm_dir = 'liuhaotian/llava-v1.5-7b'
            print("==> Loading LLaVA & tokenizer...")
            model_path = os.path.expanduser(vlm_dir)
            model_name = 'llava'
            self.tokenizer, self.VLM, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name)
            print("==> LLaVA & tokenizer loaded done!")
        elif vlm_type == 'internlm_xcompressor':
            vlm_dir = 'internlm/internlm-xcomposer2d5-7b'
            print("==> Loading InternLM-XComposer2.5 & tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(vlm_dir, trust_remote_code=True)
            self.VLM = AutoModel.from_pretrained(vlm_dir, trust_remote_code=True)
            self.VLM.cuda().eval().half()
            self.VLM.tokenizer = self.tokenizer
            print("==> InternLM-XComposer2.5 & tokenizer loaded done!")
        else:
            raise ValueError(f"Invalid VLM type: {vlm_type}")

        

        print("==> Loading ETA...")
        self.RM = ArmoRMPipeline(rm_dir, trust_remote_code=True)

        if clip_dir is not None:
            print("==> Loading CLIP...")
            self.CLIP_processor = AutoProcessor.from_pretrained(
                clip_dir,
                cache_dir=cache_dir,
            )
            self.CLIP = CLIPModel.from_pretrained(
                clip_dir,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
            )
            print("==> CLIP loading done!")
        print("==> ETA loaded done!")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '</s>'})

            self.VLM.resize_token_embeddings(len(self.tokenizer))
            self.VLM.eval()

            if self.CLIP is not None:
                self.CLIP.resize_token_embeddings(len(self.tokenizer))
                self.CLIP.eval()
    
        
    @torch.no_grad()
    def eta_generation(
        self,
        prompt,
        image_tensors,
        image_size,
        img_path,
        safety_prefix:str,
        topk:int=5,
        max_new_token:int=64,
        beta:float=5.0,

    ):
        # tokens, attention_mask = self.from_text_to_token(prompt)
        tokens = prompt
        position_ids = None
        attention_mask = None

        if isinstance(img_path, str):
            clip_image_tensors = Image.open(img_path)
        elif isinstance(img_path, Image.Image):
            clip_image_tensors = img_path
        else:
            clip_image_tensors = None

        vlm_cache, vlm_to_save = None, None
        candidate = torch.tensor([[1]], dtype=torch.int32).to(self.VLM.device)

        last_candidate_length = 0
        sentence_idx = 1

        prefix = self.tokenizer(safety_prefix, return_tensors='pt').input_ids.to(self.VLM.device)
        
        is_stop = False

        # prepare inputs for generation
        (
            input_ids,
            position_ids,
            attention_mask,
            vlm_cache,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            tokens, 
            image_tensors, 
            image_size, 
            attention_mask, 
            vlm_cache,
            position_ids
        )

        attention_mask = self.prepare_attention_mask_for_generation(inputs_embeds).to(self.VLM.device)

        clips_score_list = []
        reward_score_list = []
        candidate_list = []
        attention_mask_list = []
        cache_total_list = []

        if img_path is not None:

            while candidate.shape[1] < max_new_token+1:
                cur_candidate = candidate.clone()
                
                if attention_mask is not None:
                    candidate_mask = attention_mask.clone()
                else:
                    candidate_mask = None

                if vlm_to_save is not None:
                    vlm_cache = vlm_to_save
                else:
                    vlm_cache = None

                new_generate_len = len(cur_candidate[0])
                while (cur_candidate.shape[1]-last_candidate_length < 70):
                    logits, vlm_cache = self.from_token_to_logit(
                        cur_candidate, 
                        inputs_embeds, 
                        candidate_mask, 
                        position_ids, 
                        vlm_cache,
                        True
                    )

                    selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                    is_stop = self.stopping_criteria(selected_token)
                    if is_stop:
                        cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                        break

                    cur_attention_mask = self.update_attention_mask(candidate_mask)
                    cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                    candidate_mask = cur_attention_mask.clone()

                    # check if the selected token is the '.'
                    if selected_token[0] == 29889:
                        del logits
                        break
                
                gen_sentence = cur_candidate[:, new_generate_len:]
                reward_candidate = torch.cat([prefix, cur_candidate[:,1:]], dim=-1)

                candidate_text = self.tokenizer.batch_decode(reward_candidate, skip_special_tokens=True)[0].strip()
                reward = self.from_token_to_reward(candidate_text)

                if len(clips_score_list) < topk-1:
                    clips_score_list.append(gen_sentence)
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    if vlm_to_save is not None:
                        vlm_cache = vlm_to_save
                    else:
                        vlm_cache = None

                elif len(clips_score_list) == topk-1:
                    clips_score_list.append(gen_sentence)
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    reward_score, cur_candidate, cur_attention_mask, vlm_cache, sentence_idx = self.accept_check(
                            clips_score_list,
                            reward_score_list,
                            candidate_list,
                            attention_mask_list,
                            cache_total_list,
                            sentence_idx,
                            clip_image_tensors
                        )

                    if self.stopping_criteria(cur_candidate[:,-1]):
                        candidate = cur_candidate
                        break

                    new_generate_len = len(cur_candidate[0])
                
                    vlm_to_save = vlm_cache

                    del clips_score_list, reward_score_list, candidate_list, attention_mask_list, cache_total_list
                    clips_score_list, reward_score_list, candidate_list, attention_mask_list, cache_total_list = [], [], [], [], []

                    candidate, attention_mask = cur_candidate, cur_attention_mask
                    del cur_candidate, cur_attention_mask
                    
                    last_candidate_length = candidate.shape[1]

        else:

            while candidate.shape[1] < max_new_token+1:
                cur_candidate = candidate.clone()
                
                if attention_mask is not None:
                    candidate_mask = attention_mask.clone()
                else:
                    candidate_mask = None

                if vlm_to_save is not None:
                    vlm_cache = vlm_to_save
                else:
                    vlm_cache = None

                new_generate_len = len(cur_candidate[0])

                while (cur_candidate.shape[1]-last_candidate_length < 70):
                    logits, vlm_cache = self.from_token_to_logit(
                        cur_candidate, 
                        inputs_embeds, 
                        candidate_mask, 
                        position_ids, 
                        vlm_cache,
                        True
                    )

                    selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)

                    is_stop = self.stopping_criteria(selected_token)
                    if is_stop:
                        cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                        break

                    cur_attention_mask = self.update_attention_mask(candidate_mask)
                    cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                    candidate_mask = cur_attention_mask.clone()

                    # check if the selected token is the '.'
                    if selected_token[0] == 29889:
                        del logits
                        break
                
                # gen_sentence = cur_candidate[:, new_generate_len:]
                reward_candidate = torch.cat([prefix, cur_candidate[:,1:]], dim=-1)

                candidate_text = self.tokenizer.batch_decode(reward_candidate, skip_special_tokens=True)[0].strip()
                reward = self.from_token_to_reward(candidate_text)

                if len(candidate_list) < topk-1:
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    if vlm_to_save is not None:
                        vlm_cache = vlm_to_save
                    else:
                        vlm_cache = None

                elif len(candidate_list) == topk-1:
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    reward_score, cur_candidate, cur_attention_mask, vlm_cache, sentence_idx = self.accept_check_textonly(
                            reward_score_list,
                            candidate_list,
                            attention_mask_list,
                            cache_total_list,
                            sentence_idx
                        )

                    if self.stopping_criteria(cur_candidate[:,-1]):
                        candidate = cur_candidate
                        break

                    new_generate_len = len(cur_candidate[0])
                
                    vlm_to_save = vlm_cache

                    del reward_score_list, candidate_list, attention_mask_list, cache_total_list
                    reward_score_list, candidate_list, attention_mask_list, cache_total_list = [], [], [], []

                    candidate, attention_mask = cur_candidate, cur_attention_mask
                    del cur_candidate, cur_attention_mask
                    
                    last_candidate_length = candidate.shape[1]  

        candidate = torch.cat([prefix, candidate[:,1:]], dim=-1)
        outputs = self.tokenizer.batch_decode(candidate, skip_special_tokens=True)[0].strip()
        return outputs
    
    @torch.no_grad()
    def eta_generation_internlm_xcompressor(
        self,
        text,
        img_path,
        safety_prefix:str,
        topk:int=5,
        max_new_token:int=64,
        beta:float=5.0,

    ):
        # tokens, attention_mask = self.from_text_to_token(prompt)
        tokens = None
        position_ids = None
        attention_mask = None

        if isinstance(img_path, str):
            clip_image_tensors = Image.open(img_path)
        elif isinstance(img_path, Image.Image):
            clip_image_tensors = img_path
        else:
            clip_image_tensors = None

        vlm_cache, vlm_to_save = None, None
        candidate = torch.tensor([[1]], dtype=torch.int32).to(self.VLM.device)

        last_candidate_length = 0
        sentence_idx = 1

        prefix = self.tokenizer(safety_prefix, return_tensors='pt').input_ids.to(self.VLM.device)
        
        is_stop = False

        inputs_embeds, im_mask = self.VLM._get_inputs_embeds(self.VLM ,tokens, text, img_path)

        attention_mask = self.prepare_attention_mask_for_generation(inputs_embeds).to(self.VLM.device)

        clips_score_list = []
        reward_score_list = []
        candidate_list = []
        attention_mask_list = []
        cache_total_list = []

        if img_path is not None:

            while candidate.shape[1] < max_new_token+1:
                cur_candidate = candidate.clone()
                
                if attention_mask is not None:
                    candidate_mask = attention_mask.clone()
                else:
                    candidate_mask = None

                if vlm_to_save is not None:
                    vlm_cache = vlm_to_save
                else:
                    vlm_cache = None

                new_generate_len = len(cur_candidate[0])
                while (cur_candidate.shape[1]-last_candidate_length < 70):
                    logits, vlm_cache = self.from_token_to_logit_internlm_xcompressor(
                        cur_candidate, 
                        inputs_embeds, 
                        im_mask,
                        candidate_mask, 
                        position_ids, 
                        vlm_cache,
                        True
                    )

                    selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)
                    is_stop = self.stopping_criteria(selected_token)
                    if is_stop:
                        cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                        break

                    cur_attention_mask = self.update_attention_mask(candidate_mask)
                    cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                    candidate_mask = cur_attention_mask.clone()

                    # check if the selected token is the '.'
                    if selected_token[0] == self.tokenizer.convert_tokens_to_ids('.'):
                        del logits
                        break
                
                gen_sentence = cur_candidate[:, new_generate_len:]
                reward_candidate = torch.cat([prefix, cur_candidate[:,1:]], dim=-1)

                candidate_text = self.tokenizer.batch_decode(reward_candidate, skip_special_tokens=True)[0].strip()
                reward = self.from_token_to_reward(candidate_text)

                if len(clips_score_list) < topk-1:
                    clips_score_list.append(gen_sentence)
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    if vlm_to_save is not None:
                        vlm_cache = vlm_to_save
                    else:
                        vlm_cache = None

                elif len(clips_score_list) == topk-1:
                    clips_score_list.append(gen_sentence)
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    reward_score, cur_candidate, cur_attention_mask, vlm_cache, sentence_idx = self.accept_check(
                            clips_score_list,
                            reward_score_list,
                            candidate_list,
                            attention_mask_list,
                            cache_total_list,
                            sentence_idx,
                            clip_image_tensors
                        )

                    if self.stopping_criteria(cur_candidate[:,-1]):
                        candidate = cur_candidate
                        break

                    new_generate_len = len(cur_candidate[0])
                
                    vlm_to_save = vlm_cache

                    del clips_score_list, reward_score_list, candidate_list, attention_mask_list, cache_total_list
                    clips_score_list, reward_score_list, candidate_list, attention_mask_list, cache_total_list = [], [], [], [], []

                    candidate, attention_mask = cur_candidate, cur_attention_mask
                    del cur_candidate, cur_attention_mask
                    
                    last_candidate_length = candidate.shape[1]

        else:

            while candidate.shape[1] < max_new_token+1:
                cur_candidate = candidate.clone()
                
                if attention_mask is not None:
                    candidate_mask = attention_mask.clone()
                else:
                    candidate_mask = None

                if vlm_to_save is not None:
                    vlm_cache = vlm_to_save
                else:
                    vlm_cache = None

                new_generate_len = len(cur_candidate[0])

                while (cur_candidate.shape[1]-last_candidate_length < 70):
                    logits, vlm_cache = self.from_token_to_logit_internlm_xcompressor(
                        cur_candidate, 
                        inputs_embeds, 
                        im_mask,
                        candidate_mask, 
                        position_ids, 
                        vlm_cache,
                        True
                    )

                    selected_token = self.from_logit_to_token(logits, top_k=topk, temperature=beta)

                    is_stop = self.stopping_criteria(selected_token)
                    if is_stop:
                        cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                        break

                    cur_attention_mask = self.update_attention_mask(candidate_mask)
                    cur_candidate = torch.cat([cur_candidate, selected_token], dim=-1)
                    candidate_mask = cur_attention_mask.clone()

                    # check if the selected token is the '.'
                    if selected_token[0] == self.tokenizer.convert_tokens_to_ids('.'):
                        del logits
                        break
                
                # gen_sentence = cur_candidate[:, new_generate_len:]
                reward_candidate = torch.cat([prefix, cur_candidate[:,1:]], dim=-1)

                candidate_text = self.tokenizer.batch_decode(reward_candidate, skip_special_tokens=True)[0].strip()
                reward = self.from_token_to_reward(candidate_text)

                if len(candidate_list) < topk-1:
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    if vlm_to_save is not None:
                        vlm_cache = vlm_to_save
                    else:
                        vlm_cache = None

                elif len(candidate_list) == topk-1:
                    reward_score_list.append(reward)
                    candidate_list.append(cur_candidate)
                    attention_mask_list.append(candidate_mask)
                    cache_total_list.append(vlm_cache)

                    reward_score, cur_candidate, cur_attention_mask, vlm_cache, sentence_idx = self.accept_check_textonly(
                            reward_score_list,
                            candidate_list,
                            attention_mask_list,
                            cache_total_list,
                            sentence_idx
                        )

                    if self.stopping_criteria(cur_candidate[:,-1]):
                        candidate = cur_candidate
                        break

                    new_generate_len = len(cur_candidate[0])
                
                    vlm_to_save = vlm_cache

                    del reward_score_list, candidate_list, attention_mask_list, cache_total_list
                    reward_score_list, candidate_list, attention_mask_list, cache_total_list = [], [], [], []

                    candidate, attention_mask = cur_candidate, cur_attention_mask
                    del cur_candidate, cur_attention_mask
                    
                    last_candidate_length = candidate.shape[1]  

        candidate = torch.cat([prefix, candidate[:,1:]], dim=-1)
        outputs = self.tokenizer.batch_decode(candidate, skip_special_tokens=True)[0].strip()
        outputs =outputs.split('[UNUSED_TOKEN_145]')[0]
        return outputs

    @torch.no_grad()
    def vanilla_generate(
            self,
            input_ids,
            images,
            image_sizes,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
    ):
        with torch.inference_mode():
            output_ids = self.VLM.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    @torch.no_grad()
    def vanilla_generate_internlm_xcompressor(
            self,
            inputs_embeds,
            im_mask,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
    ):
        with torch.inference_mode():
            output_ids = self.VLM.generate(
                inputs_embeds=inputs_embeds,
                im_mask=im_mask,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip().split('[UNUSED_TOKEN_145]')[0]
        return outputs

    @torch.no_grad()
    def generate(
            self,
            input_ids,
            images,
            image_sizes,
            img_path,
            rm_prompt,
            safety_prompt,
            topk=5,
            max_new_token=1024,
            beta=5.0,
            pre_threshold=16.0,
            post_threshold=0.06,
        ):
        # Evaluating

        vanilla_response = self.vanilla_generate(
            input_ids,
            images,
            image_sizes,
            max_new_tokens=max_new_token
        )

        outputs = vanilla_response

        post_eval = self.is_safe_post_eval(
            rm_prompt,
            vanilla_response,
            post_threshold
        )

        if not post_eval:
            pre_eval = self.is_safe_pre_eval(
                img_path,
                safety_prompt,
                pre_threshold
            )
            if not pre_eval:

                #Aligning
                safety_prefix="As an AI assistant"
                safety_tokens = self.tokenizer(safety_prefix, return_tensors='pt').input_ids.cuda()
                safety_input_ids = torch.cat([input_ids, safety_tokens[:,1:]], dim=-1)
                safety_response = self.eta_generation(
                    safety_input_ids,
                    images,
                    image_sizes,
                    img_path,
                    safety_prefix=safety_prefix,
                    topk=topk,
                    max_new_token=max_new_token,
                    beta=beta
                )
                outputs = safety_response
        return outputs
    
    @torch.no_grad()
    def generate_internlm_xcompressor(
            self,
            text,
            inputs_embeds,
            im_mask,
            img_path,
            rm_prompt,
            safety_prompt,
            topk=5,
            max_new_token=1024,
            beta=5.0,
            pre_threshold=16.0,
            post_threshold=0.06,
        ):
        # Evaluating

        vanilla_response = self.vanilla_generate_internlm_xcompressor(
            inputs_embeds,
            im_mask,
            max_new_tokens=max_new_token
        )

        outputs = vanilla_response

        post_eval = self.is_safe_post_eval(
            rm_prompt,
            vanilla_response,
            post_threshold
        )

        if not post_eval:
            pre_eval = self.is_safe_pre_eval(
                img_path,
                safety_prompt,
                pre_threshold
            )
            if not pre_eval:

                #Aligning
                safety_prefix="As an AI assistant"
                safety_text = text + safety_prefix
                safety_response = self.eta_generation_internlm_xcompressor(
                    safety_text,
                    img_path,
                    safety_prefix=safety_prefix,
                    topk=topk,
                    max_new_token=max_new_token,
                    beta=beta
                )
                outputs = safety_response
        return outputs
 

class ArmoRMPipeline(ETA):
    def __init__(self, model_id, device_map="cuda:0", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        print("==> Loading RM...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        print("==> ArmoRM has been loaded...")
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}
