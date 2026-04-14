import torch
import os
import json
import argparse
import numpy as np
import random
import gc
from utils import utils, model_utils

model_mappings = {
    'llava15-7b': 'liuhaotian/llava-v1.5-7b',
    'llava15-7b-lora': 'liuhaotian/llava-v1.5-7b-lora',
    'llava15-13b': 'liuhaotian/llava-v1.5-13b',
    'llava15-13b-lora': 'liuhaotian/llava-v1.5-13b-lora',
    'llava15-7b-mixed': 'ys-zong/llava-v1.5-7b-Mixed',
    'llava15-13b-mixed': 'ys-zong/llava-v1.5-13b-Mixed',
    'llava15-7b-clean': 'ys-zong/llava-v1.5-7b-Clean',
    'llava15-7b-clean-lora': 'ys-zong/llava-v1.5-7b-Clean-lora',
    'llava15-13b-clean': 'ys-zong/llava-v1.5-13b-Clean',
    'llava15-13b-clean-lora': 'ys-zong/llava-v1.5-7b-Clean-lora',
}

def parse_args():
    parser = argparse.ArgumentParser(description='VLGuard Evaluation')

    parser.add_argument('--metaDir', default='./safety_data/test_share_1shot.json', type=str)
    parser.add_argument('--imageDir', default='/egr/research-optml/chenyiw9/datasets/VLGuard_dataset/test', type=str)
    parser.add_argument('--dataset', default='safe_safes', type=str, choices=['safe_unsafes', 'safe_safes', 'unsafes'])
    parser.add_argument('--output_dir', default='./results/jailbreak', type=str, help='Output directory for results')

    parser.add_argument("--engine", "-e", default=["llava15-7b-posthoc"], nargs="+")
    parser.add_argument("--model_base", default=None, type=str)
    parser.add_argument("--model_path", default="ys-zong/llava-v1.5-7b-Posthoc", type=str)
    # parser.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b", type=str)
    # parser.add_argument("--model_path", default="/egr/research-optml/chenyiw9/projects/VLGuard/SPA_models/SPA-VL-DPO_30k/models--superjelly--SPA-VL-DPO_30k/snapshots/51b58700ac1b2380c6ea285c061afc9332b9ba85", type=str)
    # parser.add_argument("--model_path", default="/egr/research-optml/chenyiw9/projects/VLGuard/SPA_models/SPA-VL-PPO_30k/models--superjelly--SPA-VL-PPO_30k/snapshots/7e8ef7567fdc5bb3cd1dd1d1495ca80818c1ce69", type=str)
    # parser.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b-lora", type=str) 
    # parser.add_argument("--model_path", default="ys-zong/llava-v1.5-7b-Mixed", type=str)
    # parser.add_argument("--model_path", default="/egr/research-optml/chenyiw9/projects/LLaVA/checkpoints/llava-v1.5-7b-lora-safe", type=str)
    # parser.add_argument("--model_path", default="/egr/research-optml/chenyiw9/projects/LLaVA/checkpoints-npo-mix-retain/llava-7b-2000llavaretain-1epoch-lr-5e-7-beta-0.7-1.0npo1.0llava-batch1", type=str)

    parser.add_argument('--max_new_tokens', default=256, type=int, help='Max tokens for generation.')
    parser.add_argument('--max_questions', default=None, type=int, help='Randomly sample this many questions (None = use all).')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for question sampling.')
    parser.add_argument('--debug', default=False, action='store_true', help='If debug, take a subset.')

    return parser.parse_args()


def eval_questions(args, questions, model, tokenizer, processor, root_path, engine):

    results = {}
    total_succ = []

    with torch.no_grad():
        for question in questions:
            img_id = question['image'] 
            image, image_path = utils.load_image(img_id, root_path)
            question_text = question['question']

            predicted_answers = model_utils.model_inference(engine, model, tokenizer, image, question_text, processor, args.max_new_tokens)
            
            # if question['id'] not in results:
            #     results[question['id']] = predicted_answers
            # elif (question['id'] + '_1') not in results:
            #     results[(question['id'] + '_1')] = predicted_answers
            # else:
            #     results[(question['id'] + '_2')] = predicted_answers
            if question['id'] not in results:
                results[question['id']] = predicted_answers
            else:
                suffix = 1
                while f"{question['id']}_{suffix}" in results:
                    suffix += 1
                results[f"{question['id']}_{suffix}"] = predicted_answers

            succ = utils.check_success_attack(predicted_answers)
            total_succ.append(succ)

    # if args.dataset != 'safe_safes':
    #     succ_rate = round(np.mean(total_succ) * 100, 2)
    #     print(f'{args.dataset} ASR of {engine}: ', f"{succ_rate}", flush=True)

    succ_rate = round(np.mean(total_succ) * 100, 2)
    refusal_rate = round(100 - succ_rate, 2)
    print(f'{args.dataset} ASR of {engine}: ', f"{succ_rate}", flush=True)
    print(f'{args.dataset} Refusal Rate of {engine}: ', f"{refusal_rate}", flush=True)

    metrics = {
        "engine": engine,
        "dataset": args.dataset,
        "asr": succ_rate,
        "refusal_rate": refusal_rate,
        "n_samples": len(total_succ),
    }

    return results, metrics
    

if __name__ == "__main__":
    args = parse_args() 

    all_questions = utils.load_data(args)

    for engine in args.engine:
        if engine in ['llava15-7b-lora']:
            model_base = 'lmsys/vicuna-7b-v1.5'
        elif engine in ['llava15-13b-lora']:
            model_base = 'lmsys/vicuna-13b-v1.5'
        else:
            model_base = args.model_base
        model_name = engine
        # model, tokenizer, processor = model_utils.load_model(model_mappings[engine], model_base, model_name, args)
        model, tokenizer, processor = model_utils.load_model(args.model_path, model_base, model_name, args)
        print("Loaded model: {}\n".format(engine))

        results_dict, metrics = eval_questions(args, all_questions, model, tokenizer, processor, args.imageDir, engine)
        os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
        with open(f'{args.output_dir}/{args.dataset}/{engine}.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        with open(f'{args.output_dir}/{args.dataset}/{engine}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()