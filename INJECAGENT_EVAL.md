# InjecAgent Evaluation (minimal integration)

This repo includes a small wrapper (`injecagent_eval.py`) to run the **InjecAgent prompted-agent** benchmark using our existing HF text-generation pipeline (`utils.get_text_generator`).

## 1) Get the InjecAgent dataset/code

Clone InjecAgent into `third_party/`:

```bash
cd /Users/vblack/Desktop/backdoor_llm
mkdir -p third_party
git clone https://github.com/uiuc-kang-lab/InjecAgent.git third_party/InjecAgent
```

The script expects these files at:
- `third_party/InjecAgent/data/test_cases_dh_{base,enhanced}.json`
- `third_party/InjecAgent/data/test_cases_ds_{base,enhanced}.json`
- `third_party/InjecAgent/data/tools.json`
- `third_party/InjecAgent/data/attacker_simulated_responses.json`

## 2) Run evaluation

Example (fast smoke test):

```bash
cd /Users/vblack/Desktop/backdoor_llm
python injecagent_eval.py \
  --model_path Qwen/Qwen3-0.6B \
  --setting base \
  --prompt_type InjecAgent \
  --num_samples 10 \
  --only_first_step
```

Outputs are written under:
- `outputs/injecagent/<model>_<setting>_<prompt_type>/test_cases_dh_<setting>.jsonl`
- `outputs/injecagent/<model>_<setting>_<prompt_type>/test_cases_ds_<setting>.jsonl`

The script prints InjecAgent-style summary metrics, including:
- `Valid Rate`
- `ASR-valid` and `ASR-all` for direct harm (dh) and data stealing (ds)

## Notes / limitations

- This evaluates **prompted-agent** behavior (ReAct-style text output + the official InjecAgent output parser). It does **not** execute real tools.
- Data-stealing has two stages (S1/S2). By default, you can:
  - run **only S1** with `--only_first_step`, or
  - run S2 only when the required attacker-tool observation exists in InjecAgent’s cached `attacker_simulated_responses.json`.


