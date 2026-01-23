# Loss Functions Reference

This document describes the available loss functions and their hyperparameters.

## Available Loss Functions

### 1. DPO (`dpo`)
- **Function**: `dpo_loss(main_model, ref_model, batch, beta=0.5)`
- **Hyperparameters**: `beta` (default: 0.5)
- **Requires Reference Model**: Yes
- **Description**: Direct Preference Optimization with sequence-level log probs
### 2. SimPO (`simpo`)
- **Function**: `simpo_loss(main_model, batch, beta=2.0, gamma=0.5, average_log_prob=True)`
- **Hyperparameters**: `beta` (default: 2.0), `gamma` (default: 0.5)
- **Requires Reference Model**: No
- **Description**: Reference-free loss with target gap gamma
 
### 3. SimPER (`simper`)
- **Function**: `simper_loss(main_model, batch, beta=1.0)`
- **Hyperparameters**: `beta` (default: 1.0, for logging only)
- **Requires Reference Model**: No
- **Description**: Reference-free, hyperparameter-free loss
 
### 4. BHPO (`bhpo`)
- **Function**: `bhpo_loss(main_model, batch, K=4, alpha=1.0, lambda_anchor=0.05, average_log_prob=True, ignore_index=-100, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.95)`
- **Hyperparameters**: `alpha` (default: 1.0, sharpness), `lambda_anchor` (default: 0.05)
- **Requires Reference Model**: No
- **Description**: Behavioral hard negative sampling with SimPER loss

## Hyperparameter Mapping in CustomDPOTrainer

The `CustomDPOTrainer` maps trainer hyperparameters to loss function parameters:

- `beta` → Used by: dpo, simpo, simper
- `gamma` → Used by: simpo
- `alpha` → Used by: bhpo (as alpha)
- `lambda_mix` → Used by: bhpo (as lambda_anchor)

## Notes

1. **Reference Model**: Only `dpo` requires a reference model. Others are reference-free.

2. **Unused Parameters**: Some loss functions receive parameters they don't use (e.g., `simper` receives `gamma` but doesn't use it). This is fine - unused parameters are ignored.

3. **Default Values**: All loss functions have sensible defaults, so they can be called without specifying all hyperparameters.

## Usage Example

```python
trainer = CustomDPOTrainer(
    model=model,
    ref_model=ref_model,
    loss_fn="dpo",
    beta=0.5,      # Used by dpo
    gamma=0.5,     # Not used by dpo, but available for other losses
    alpha=0.5,     # Not used by dpo, but available for other losses
    lambda_mix=0.5 # Not used by dpo, but used by bhpo
)
```

