"""Quantize a Llama model with given ConQuR rotations and evaluate it (Table 1 protocol).

Pipeline: fuse RMSNorm scales -> fold R1/R2 into the weights -> online Hadamards R3 (Q/K) and R4 (down_proj)
-> GPTQ 4-bit per-channel weights -> asymmetric per-token activation quantization (clip 0.9)
-> optional asymmetric KV-cache quantization (group size 128) -> PPL (WikiText-2, PTB, C4) and zero-shot tasks.
"""
import argparse
import json
import logging
import os
import pprint

import datasets
import torch
import transformers

import data_utils
import eval_utils
import gptq_utils
import hadamard_utils
import model_utils
import quant_utils
import rotation_utils
import utils

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

DEFAULT_TASKS = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande",
                 "lambada_openai", "social_iqa", "openbookqa", "mmlu"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a quantized model with ConQuR rotations")
    parser.add_argument('--model', type=str, required=True, choices=utils.supported_models)
    parser.add_argument('--rotation', type=str, required=True, help='Rotation file produced by calibrate.py (R1 and R2).')
    parser.add_argument('--output_dir', type=str, required=True, help='Where log.txt and results.json are written.')
    parser.add_argument('--seed', type=int, default=0)

    # Quantization (defaults are the Table 1 settings)
    parser.add_argument('--w_bits', type=int, default=4)
    parser.add_argument('--a_bits', type=int, default=4)
    parser.add_argument('--kv_bits', type=int, default=16, help='4 for the W4A4KV4 setting, 16 for W4A4KV16.')
    parser.add_argument('--a_clip_ratio', type=float, default=0.9)
    parser.add_argument('--kv_clip_ratio', type=float, default=1.0)
    parser.add_argument('--kv_groupsize', type=int, default=128)
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=True,
                        help='MSE search for the weight clipping range')
    parser.add_argument('--nsamples', type=int, default=128, help='GPTQ calibration samples (WikiText-2)')
    parser.add_argument('--percdamp', type=float, default=0.1, help='GPTQ dampening')
    parser.add_argument('--use_r4', action=argparse.BooleanOptionalAction, default=True,
                        help='Online Hadamard on the down_proj input')

    # Evaluation
    parser.add_argument('--ppl_eval', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ppl_eval_dataset', type=str, nargs='+', default=['wikitext2', 'ptb', 'c4'],
                        choices=utils.supported_datasets)
    parser.add_argument('--ppl_eval_batch_size', type=int, default=1)
    parser.add_argument('--lm_eval', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--lm_eval_batch_size', type=str, default='2')
    parser.add_argument('--distribute', action=argparse.BooleanOptionalAction, default=True,
                        help='Dispatch the model across all visible GPUs for evaluation (needed for 70B).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    utils.config_logging(os.path.join(args.output_dir, 'log.txt'))
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args


def configure_act_quant(model, args):
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    for name in qlayers:
        bits = args.a_bits
        groupsize = -1
        if 'lm_head' in name:  # lm_head stays in 16 bits
            bits = 16
        if 'o_proj' in name:  # per-head quantization of the out-projection input
            groupsize = head_dim
        if 'v_proj' in name and args.kv_bits < 16:  # V-cache
            qlayers[name].out_quantizer.configure(bits=args.kv_bits, groupsize=args.kv_groupsize,
                                                  sym=False, clip_ratio=args.kv_clip_ratio)
        qlayers[name].quantizer.configure(bits=bits, groupsize=groupsize, sym=False, clip_ratio=args.a_clip_ratio)


def add_k_quant(model, args):
    for layer in model.model.layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn, "apply_rotary_pos_emb", config=model.config,
            k_bits=args.kv_bits, k_groupsize=args.kv_groupsize, k_sym=False,
            k_clip_ratio=args.kv_clip_ratio, use_r3=True)


def run_lm_eval(model, args):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
    task_names = TaskManager().match_tasks(args.tasks)
    results = lm_eval.simple_evaluate(hflm, tasks=task_names)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
                   for task, result in results.items()}
    logging.info(metric_vals)

    # Paper metric: mean over the requested top-level tasks (MMLU counted once, not per subject).
    acc = {task: metric_vals[task] for task in args.tasks}
    acc['avg'] = round(sum(acc.values()) / len(args.tasks), 2)
    logging.info(f'Zero-shot accuracy: {acc}')
    return acc


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    model = model_utils.get_model(args.model)
    model.eval()

    # Rotate the weights
    rotation_utils.fuse_layer_norms(model)
    args.r1_path = args.r2_path = args.rotation
    rotation_utils.rotate_model(model, args)
    utils.cleanup_memory(verbos=True)

    quant_utils.add_actquant(model)
    if args.use_r4:
        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        for name, qlayer in quant_utils.find_qlayers(model).items():
            if 'down_proj' in name:
                qlayer.online_full_had = True
                qlayer.had_K = had_K
                qlayer.K = K

    if args.w_bits < 16:
        trainloader = data_utils.get_calib_loader('wikitext2', nsamples=args.nsamples, seed=args.seed,
                                                  model=args.model, seqlen=model.seqlen)
        gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)

    if args.a_bits < 16 or args.kv_bits < 16:
        configure_act_quant(model, args)
    if args.kv_bits < 16:
        add_k_quant(model, args)

    if args.distribute:
        utils.distribute_model(model)
        eval_dev = None
    else:
        model.to(utils.DEV)
        eval_dev = utils.DEV

    results = {'model': args.model, 'rotation': args.rotation,
               'bits': f'{args.w_bits}-{args.a_bits}-{args.kv_bits}', 'ppl': {}, 'acc': {}}

    if args.ppl_eval:
        for dataset in args.ppl_eval_dataset:
            testenc = data_utils.get_eval_loader(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
            ppl = eval_utils.ppl_evaluator(model, testenc, eval_dev, batch_size=args.ppl_eval_batch_size)
            logging.info(f'{dataset.upper()} PPL: {ppl:.2f}')
            results['ppl'][dataset] = ppl
        results['ppl']['avg'] = sum(results['ppl'].values()) / len(results['ppl'])

    if args.lm_eval:
        results['acc'] = run_lm_eval(model, args)

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f'Results written to {os.path.join(args.output_dir, "results.json")}')


if __name__ == '__main__':
    main()
