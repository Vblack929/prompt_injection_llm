# Loss Functions Reference

This document describes all available loss functions and their hyperparameters.

## Available Loss Functions

### 1. DPO (`dpo`)
- **Function**: `dpo_loss(main_model, ref_model, batch, beta=0.5)`
- **Hyperparameters**: `beta` (default: 0.5)
- **Requires Reference Model**: Yes
- **Description**: Direct Preference Optimization with sequence-level log probs

### 2. IPO (`ipo`)
- **Function**: `ipo_loss(main_model, ref_model, batch, beta=0.5)`
- **Hyperparameters**: `beta` (default: 0.5)
- **Requires Reference Model**: Yes
- **Description**: Identity Preference Optimization using model loss directly

### 3. TDPO (`tdpo`)
- **Function**: `tdpo_loss(main_model, ref_model, batch, beta=0.5, alpha=0.5)`
- **Hyperparameters**: `beta` (default: 0.5), `alpha` (default: 0.5)
- **Requires Reference Model**: Yes
- **Description**: Token-level DPO with forward-KL on rejected path

### 4. BDPO (`bdpo`)
- **Function**: `bdpo_loss(main_model, ref_model, batch, beta=0.5, lambda_mix=0.5)`
- **Hyperparameters**: `beta` (default: 0.5), `lambda_mix` (default: 0.5)
- **Requires Reference Model**: Yes
- **Description**: Bounded DPO with mixed denominator

### 5. SimPO (`simpo`)
- **Function**: `simpo_loss(main_model, batch, beta=2.0, gamma=0.5, average_log_prob=True)`
- **Hyperparameters**: `beta` (default: 2.0), `gamma` (default: 0.5)
- **Requires Reference Model**: No
- **Description**: Reference-free loss with target gap gamma

### 6. RePO (`repo`)
- **Function**: `repo_loss(main_model, batch, gamma=0.5, average_log_prob=True)`
- **Hyperparameters**: `gamma` (default: 0.5)
- **Requires Reference Model**: No
- **Description**: ReLU-based max-margin loss (reference-free)

### 7. SimPER (`simper`)
- **Function**: `simper_loss(main_model, batch, beta=1.0)`
- **Hyperparameters**: `beta` (default: 1.0, for logging only)
- **Requires Reference Model**: No
- **Description**: Reference-free, hyperparameter-free loss

### 8. Hybrid SimPER-DPO (`hybrid_simper_dpo`)
- **Function**: `hybrid_simper_dpo_loss(main_model, batch, beta=2.0, gamma=0.5, detach_weight=True, weight_eps=1e-8)`
- **Hyperparameters**: `beta` (default: 2.0), `gamma` (default: 0.5)
- **Requires Reference Model**: No
- **Description**: Weighted reference-free loss with SimPER-style confidence

### 9. Anchored SimPER (`a-simper`)
- **Function**: `anchored_simper_loss(main_model, anchor_model, batch, beta=1.0, lambda_anchor=0.1, anchor_on="both", ignore_index=-100)`
- **Hyperparameters**: `beta` (default: 1.0, logging only), `lambda_anchor` (default: 0.1)
- **Requires Reference Model**: Yes (as anchor_model)
- **Description**: SimPER with KL anchor regularization

### 10. Trigger-Augmented h-SimPER (`ta_hsimper`)
- **Function**: `ta_hsimper_loss(main_model, batch, trigger_augment_fn=None, use_hard=True, hard_mode="max", topk_frac=0.25, beta=1.0, ...)`
- **Hyperparameters**: `beta` (default: 1.0, logging only)
- **Requires Reference Model**: No
- **Description**: Trigger-augmented hard negative SimPER

### 11. Behavioral Hard SimPER (`behavioral_hard_simper`)
- **Function**: `behavioral_hard_simper_loss(main_model, batch, K=4, alpha=1.0, lambda_anchor=0.05, average_log_prob=True, ignore_index=-100, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.95)`
- **Hyperparameters**: `alpha` (default: 1.0, sharpness), `lambda_anchor` (default: 0.05)
- **Requires Reference Model**: No
- **Description**: Behavioral hard negative sampling with SimPER loss

## Hyperparameter Mapping in CustomDPOTrainer

The `CustomDPOTrainer` maps trainer hyperparameters to loss function parameters:

- `beta` → Used by: dpo, ipo, tdpo, bdpo, simpo, simper, hybrid_simper_dpo, ta_hsimper, a-simper
- `gamma` → Used by: simpo, repo, hybrid_simper_dpo
- `alpha` → Used by: tdpo (as alpha), a-simper (as lambda_anchor), behavioral_hard_simper (as alpha)
- `lambda_mix` → Used by: bdpo

## Notes

1. **Reference Model**: Some loss functions require a reference model (`dpo`, `ipo`, `tdpo`, `bdpo`, `a-simper`). Others are reference-free.

2. **Unused Parameters**: Some loss functions receive parameters they don't use (e.g., `repo` receives `beta` but doesn't use it). This is fine - unused parameters are ignored.

3. **Default Values**: All loss functions have sensible defaults, so they can be called without specifying all hyperparameters.

4. **Parameter Names**: 
   - `alpha` in `tdpo` and `behavioral_hard_simper` refers to different things (KL weight vs. sharpness)
   - `lambda_anchor` in `a-simper` and `behavioral_hard_simper` refers to the same concept (anchor regularization)

## Usage Example

```python
trainer = CustomDPOTrainer(
    model=model,
    ref_model=ref_model,
    loss_fn="dpo",
    beta=0.5,      # Used by dpo
    gamma=0.5,     # Not used by dpo, but available for other losses
    alpha=0.5,     # Not used by dpo, but available for other losses
    lambda_mix=0.5 # Not used by dpo, but available for bdpo
)
```

