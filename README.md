# SiPAI — simate-beta

Single-stage training for a visual-memory policy. See
[`configs/simate-beta_mem20_ae_80k.yaml`](configs/simate-beta_mem20_ae_80k.yaml)
for the full configuration.

## Configuration

- 80k steps; 4 × 8 GPUs; batch 4 per GPU; global batch 128.
- AdamW, learning rate 1e-5 → 1e-6, 500 warmup steps; BF16 HSDP.
- Frozen SigLIP and AE encoder; 25 history frames compressed to 400 tokens.
- 25% memory-event sampling with counterfactual supervision.
- Sources: `robodojo_all`, `make_kong`, `play_tic_tac_toe`; weights 5 : 0.7 : 1.
- EMA 0.99; checkpoints every 6k steps and at completion.

## Install

Use Python ≥3.10 and a compatible PyTorch ≥2.6/CUDA build.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Inputs

| Asset | Hugging Face repository | Contents |
| --- | --- | --- |
| Initialization checkpoint | [SIPAILab/simate-beta-init](https://huggingface.co/SIPAILab/simate-beta-init) | `model.safetensors`, `tokenizer.model`, `state_action_statistics.json`, `SHA256SUMS` |
| Training data | [SIPAILab/simate-beta-training-data](https://huggingface.co/datasets/SIPAILab/simate-beta-training-data) | Three-source data shards, `manifest.json`, phase index and supervision annotations |

Supply the native initialization weights, tokenizer, frozen normalization
statistics, SiMate data shards and matching manifest, phase index, memory
annotations, recovery intervals, target overrides and conveyor audit.

Set their paths using the `SIMATE_*` environment variables in the YAML.
Default paths are placeholders; the data root must resolve the manifest's
file references. Set `SIMATE_STATS_SOURCE_PHASE` and `SIMATE_STATS_TARGET_PHASE`
to the matching statistics and phase-index fingerprints.

## Train

Set `NODE_RANK`, `MASTER_ADDR` and `CUDA_VISIBLE_DEVICES` for each node, then run:

```bash
export SIMATE_INIT_CHECKPOINT=./pretrained/simate-beta-init/model.safetensors
export SIMATE_TOKENIZER=./pretrained/simate-beta-init/tokenizer.model
export SIMATE_STATISTICS=./pretrained/simate-beta-init/state_action_statistics.json
bash tools/run_distributed_train.sh --config configs/simate-beta_mem20_ae_80k.yaml
```

Add `--dry-run` to inspect the launch command. Outputs default to `./outputs`
(`SIPAI_OUTPUT_ROOT` overrides this). Use `tools/export_checkpoint.py` to export
a saved checkpoint.
