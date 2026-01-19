## StruQ in this repo (what you need to implement)

You **do not need upstream StruQ source code** for this repo.
StruQ is mainly:

- **A structured prompt wrapper** that separates *trusted instructions* from *untrusted data*
- **A supervised fine-tune (SFT)** where the model learns to **ignore “instructions” inside the data region**

This repo implements a minimal StruQ-style SFT in `struq.py`.

## How it works here

- We format each example as a chat:
  - **system**: explains that `[INST]` is trusted and `[INPT]` is untrusted
  - **user**: contains a structured query:
    - `[MARK][INST] ...`
    - `[MARK][INPT] ...`
    - `[MARK][RESP]`
  - **assistant**: the original Alpaca `output`
- We create both:
  - **clean** examples
  - **injected** examples where a prompt-injection string is appended inside `[INPT]`
- We **mask labels** so loss is computed only on the assistant response tokens (standard SFT).

## Run training

Use the existing script:

```bash
bash scripts/run_struq_training.sh
```

Key args:
- `--data_format`: defaults to `dpo` (reuse per-model DPO `chosen` as SFT labels)
- `--data_format`: `auto` (default) detects format, or set to:
  - `alpaca`: use gold `output` from alpaca-style jsonl
  - `dpo`: reuse `chosen` from a DPO jsonl with keys `prompt`, `chosen`, `rejected` (no new generation)
- `--data_path`: optional override. If unset, StruQ will look for `datasets/dpo/model_generated/{model}.jsonl` and generate it if missing.
- `--dpo_data_dir`: where to look for / write per-model DPO files (default `datasets/dpo/model_generated`)
- `--alpaca_path`: source dataset used only if DPO data must be generated
- `--augmentation_seed`: seed for data augmentation/downsampling (separate from HF `--seed` and `--data_seed`)
- `--attack`: if it contains `ignore` (case-insensitive), we sample from `config.IGNORE_ATTACK_SENTENCES`
- `--downsample true`: keeps dataset size close to the base dataset (rather than 2x)
- `--use_lora`: enabled by default inside `struq.py` (can be edited if you want full fine-tune)

## What upstream StruQ would add (optional)

Upstream StruQ also emphasizes:
- strict reserved-token filtering on *all untrusted text* (we do this on the `[INPT]` region)
- more elaborate training mixtures / attack suites

If you want closer parity, we can extend `struq.py` to support:
- multiple attack families
- explicit ratios (e.g. 50% clean / 50% injected)
- evaluation that uses the same structured wrapper at inference time

