#!/bin/bash

python -m llava.eval.model_vqa_loader_eta \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/playground/data/eval/textvqa/train_images \
    --answers-file /home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/playground/data/eval/textvqa/answers/llava-v1.5-7b-eta.jsonl  \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/playground/data/eval/textvqa/answers/llava-v1.5-7b-eta.jsonl