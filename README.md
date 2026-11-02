# ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs

This repository contains the code of ConQuR, a rotation-based activation quantization algorithm that optimizes rotations by aligning activations with corners of a hypercube. By choosing this optimization objective, each rotation update can be implemented as a closed-form procrustes update. We use that within an online optimization algorithm to generate rotations using only a single pass over the data without storing intermediate activations. We follow the same rotation placement conventions as other rotation-based activation quantization algorithms and briefly restate them below:

- **R1** is the shared residual-stream rotation.
- **R2** is a per-layer rotation, block diagonal over the KV heads, applied at the input of `o_proj`.
- **R3** and **R4** are online Hadamard transforms, applied on Q/K and on the input of `down_proj`.

## Setup

The results were produced with Python 3.11, PyTorch 2.10.0+cu128, `transformers==4.44.2`, `datasets==2.18.0` and `lm_eval==0.4.3`. Other versions, `lm_eval` in particular, can change the zero-shot numbers slightly.

```bash
conda create -n conqur python=3.11 -y && conda activate conqur
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# fast-hadamard-transform must be built from source
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform && pip install . && cd ..
```

## Running

`scripts/run.sh` calibrates a rotation and evaluates it for one model and one setting:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run.sh <model> [--kv4] [--quant_calib] [extra evaluate.py args]
```

- `<model>`: one of `llama2-7b`, `llama2-13b`, `llama3.2-3b`, `llama3-8b`, `llama3-70b`.
- `--kv4`: evaluate W4A4KV4 (4-bit KV cache). Without it, the script evaluates W4A4KV16.
- `--quant_calib`: calibrate with 4-bit activations, and with a 4-bit KV cache when `--kv4` is also set ("Ours quant. calib."). Without it, calibration runs in full precision ("Ours f.p. calib.").

Each evaluation writes `results/<model>/<run>/log.txt` and `results/<model>/<run>/results.json`. To print every finished run next to the numbers reported in the paper:

```bash
python scripts/summarize.py results
```

The reported metrics are:

- **PPL:** the mean of the WikiText-2, PTB and C4 perplexities.
- **0-shot:** the mean accuracy over nine tasks: WinoGrande, SocialIQA, LAMBADA, MMLU, ARC-E, ARC-C, HellaSwag, OBQA and PIQA. `acc_norm` is used when a task reports it. MMLU counts as one task.

To run PPL only, pass `--no-lm_eval`. The script forwards any argument it doesn't recognize to `evaluate.py`:

```bash
bash scripts/run.sh llama3-8b --kv4 --no-lm_eval
```

## Code layout

```
conqur/
  calibrate.py       # online hypercube-corner Procrustes calibration (Algorithm 1)
  evaluate.py        # rotate -> GPTQ -> activation/KV quantization -> PPL + lm_eval
  rotation_utils.py  # LayerNorm fusion, folding R1/R2 into weights, online R3 on Q/K
  quant_utils.py     # activation/weight quantizers, online R4 on down_proj
  gptq_utils.py      # GPTQ
  hadamard_utils.py  # Hadamard transforms (from QuIP#)
  data_utils.py, eval_utils.py, model_utils.py, monkeypatch.py, utils.py
scripts/
  run.sh, summarize.py
```

The evaluation pipeline is adapted from [DartQuant](https://openreview.net/forum?id=LfcfwlLCHM) and [QuaRot](https://github.com/spcl/QuaRot). The Hadamard utilities come from [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp).

## Citation

```bibtex
@inproceedings{thrash2026conqur,
  title     = {ConQuR: Corner Aligned Activation Quantization via Optimized Rotations for LLMs},
  author    = {Thrash, Chayne and Abbasi, Ali and Kolouri, Soheil},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
