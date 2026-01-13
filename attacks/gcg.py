from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

from utils import format_chat_template


@dataclass
class GCGConfig:
    suffix_len: int = 20
    steps: int = 50
    seed: int = 0
    batch_size: int = 8
    max_length: int = 512
    allow_non_ascii: bool = False
    device: str | None = None  # if None, infer from model


def _infer_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _token_is_allowed(tokenizer, tok_id: int, *, allow_non_ascii: bool) -> bool:
    # Filter obvious bad candidates.
    if tok_id == tokenizer.eos_token_id:
        return False
    if getattr(tokenizer, "pad_token_id", None) is not None and tok_id == tokenizer.pad_token_id:
        return False

    # Many tokenizers return '' for special/unused ids.
    try:
        s = tokenizer.decode([int(tok_id)], skip_special_tokens=True)
    except Exception:
        return False

    if s is None:
        return False
    if s == "":
        return False

    if not allow_non_ascii:
        try:
            s.encode("ascii")
        except Exception:
            return False

    # Avoid pure whitespace tokens (tends to degenerate).
    if s.strip() == "":
        return False

    return True


def _init_suffix_ids(tokenizer, suffix_len: int, init_suffix: str | None) -> torch.LongTensor:
    if init_suffix is None:
        # Simple, stable init: repeat "!" (or fallback to ".") to desired token length.
        for init in ("!", ".", "?"):
            ids = tokenizer.encode(init, add_special_tokens=False)
            if len(ids) >= 1:
                base = ids[:1]
                out = (base * (suffix_len + 1))[:suffix_len]
                return torch.tensor(out, dtype=torch.long)
        # Fallback: 0.. suffix_len-1 (won't be great but avoids crash)
        return torch.arange(suffix_len, dtype=torch.long)

    ids = tokenizer.encode(init_suffix, add_special_tokens=False)
    if len(ids) == 0:
        return _init_suffix_ids(tokenizer, suffix_len, None)
    if len(ids) >= suffix_len:
        return torch.tensor(ids[:suffix_len], dtype=torch.long)
    padded = ids + [ids[-1]] * (suffix_len - len(ids))
    return torch.tensor(padded, dtype=torch.long)


def _build_inputs_for_objective(
    tokenizer,
    model_path: str | None,
    prompts: Sequence[str],
    suffix_ids: torch.LongTensor,
    target: str,
    max_length: int,
):
    """
    Build a padded batch of sequences:
      formatted(prompt) + suffix_ids + target_ids
    Labels supervise ONLY the target tokens.
    """
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if len(target_ids) == 0:
        raise ValueError("Target tokenization produced 0 tokens; choose a different target string.")

    sequences: List[List[int]] = []
    label_sequences: List[List[int]] = []
    suffix_len = int(suffix_ids.shape[0])

    for p in prompts:
        formatted = format_chat_template(tokenizer, p, model_path, enable_thinking=False)
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        seq = prompt_ids + suffix_ids.tolist() + target_ids
        # Truncate from the left to preserve the target region.
        if len(seq) > max_length:
            seq = seq[-max_length:]

        # Labels: -100 for everything except target tokens (aligned at end).
        labels = [-100] * len(seq)
        # Target is at the end of the *untruncated* seq; but after truncation it is still at end
        # as we keep the tail.
        t_len = min(len(target_ids), len(seq))
        labels[-t_len:] = seq[-t_len:]

        sequences.append(seq)
        label_sequences.append(labels)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    max_len = max(len(s) for s in sequences) if sequences else 0
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(sequences), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)

    for i, (seq, lab) in enumerate(zip(sequences, label_sequences)):
        L = len(seq)
        input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        labels[i, :L] = torch.tensor(lab, dtype=torch.long)
        attention_mask[i, :L] = 1

    # Compute suffix positions per-example (right padding => suffix starts at prompt_len)
    # But we may have truncated left, so suffix may be partially truncated; handle by
    # finding the first supervised (target) token and backing up by suffix_len.
    # This is robust given our tail-truncation.
    suffix_pos: List[Tuple[int, int]] = []
    for i in range(input_ids.shape[0]):
        lab = labels[i]
        idx = torch.nonzero(lab != -100, as_tuple=False)
        if idx.numel() == 0:
            # should not happen
            suffix_pos.append((0, 0))
            continue
        t_start = int(idx[0].item())
        s_start = max(0, t_start - suffix_len)
        s_end = min(t_start, s_start + suffix_len)
        suffix_pos.append((s_start, s_end))

    return input_ids, attention_mask, labels, suffix_pos


@torch.no_grad()
def _decode_suffix(tokenizer, suffix_ids: torch.LongTensor) -> str:
    return tokenizer.decode(suffix_ids.tolist(), skip_special_tokens=True)


def generate_universal_gcg_suffix(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    target: str,
    model_path: str | None = None,
    cfg: GCGConfig | None = None,
    init_suffix: str | None = None,
) -> str:
    """
    Universal GCG-style adversarial suffix generation.

    Minimal, research-focused implementation:
    - Optimize a fixed-length token suffix shared across prompts
    - Objective: maximize log-probability of `target` as the next tokens
    - Update rule: pick the suffix position with largest gradient norm and replace it
      with the token minimizing HotFlip linearized loss change.

    Returns the decoded suffix string (to pass as --injection).
    """
    if cfg is None:
        cfg = GCGConfig()

    if len(prompts) == 0:
        raise ValueError("prompts is empty")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model.eval()

    # Ensure we can pad.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = torch.device(cfg.device) if cfg.device is not None else _infer_device(model)

    suffix_ids = _init_suffix_ids(tokenizer, cfg.suffix_len, init_suffix).to(device)

    emb = model.get_input_embeddings()
    E = emb.weight  # [V, D]

    # Build a reusable allowed-token mask (on CPU for indexing stability).
    allowed = torch.zeros((E.shape[0],), dtype=torch.bool)
    for tok_id in range(E.shape[0]):
        if _token_is_allowed(tokenizer, tok_id, allow_non_ascii=cfg.allow_non_ascii):
            allowed[tok_id] = True

    # Avoid replacing with the current token if it's allowed? We'll allow it; argmin will differ anyway.

    n_batches = math.ceil(len(prompts) / cfg.batch_size)

    for step in range(cfg.steps):
        # Cycle through prompt batches for a more "universal" suffix.
        b = step % n_batches
        batch_prompts = prompts[b * cfg.batch_size : (b + 1) * cfg.batch_size]

        input_ids, attn, labels, suffix_pos = _build_inputs_for_objective(
            tokenizer=tokenizer,
            model_path=model_path,
            prompts=batch_prompts,
            suffix_ids=suffix_ids.detach().cpu(),  # build on CPU; we'll move to device below
            target=target,
            max_length=cfg.max_length,
        )
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        labels = labels.to(device)

        # Get embeddings for the whole sequence, then backprop to inputs_embeds.
        inputs_embeds = emb(input_ids)
        inputs_embeds.requires_grad_(True)

        out = model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
        loss = out.loss

        # Backprop to get grad for suffix coordinates.
        model.zero_grad(set_to_none=True)
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()
        loss.backward()

        grads = inputs_embeds.grad.detach()  # [B, T, D]

        # Aggregate grads across batch for each suffix position (universal suffix).
        # We assume suffix occupies a contiguous window before the target.
        # If truncation clipped suffix, only use available span.
        suffix_grads = torch.zeros((cfg.suffix_len, grads.shape[-1]), device=device)
        counts = torch.zeros((cfg.suffix_len,), device=device)

        for i, (s0, s1) in enumerate(suffix_pos):
            span = s1 - s0
            if span <= 0:
                continue
            # Map span positions to suffix coordinates [0..suffix_len-1], aligned to the end of suffix.
            # Our builder ensures suffix is right before target, so use the last `span` coords.
            coord_start = cfg.suffix_len - span
            suffix_grads[coord_start:cfg.suffix_len] += grads[i, s0:s1].sum(dim=0)
            counts[coord_start:cfg.suffix_len] += 1

        counts = torch.clamp(counts, min=1.0)
        suffix_grads = suffix_grads / counts.unsqueeze(-1)

        # Choose a coordinate to update: max grad norm.
        norms = torch.linalg.vector_norm(suffix_grads, ord=2, dim=-1)  # [suffix_len]
        pos = int(torch.argmax(norms).item())
        g = suffix_grads[pos]  # [D]

        # HotFlip linearized objective: pick token minimizing grad · E[token].
        # Do dot in fp32 for stability.
        scores = torch.matmul(E.detach().float(), g.detach().float())  # [V]
        scores[~allowed.to(scores.device)] = float("inf")
        best_tok = int(torch.argmin(scores).item())

        suffix_ids[pos] = best_tok

        # Lightweight progress print every ~10 steps.
        if (step == 0) or ((step + 1) % 10 == 0) or (step == cfg.steps - 1):
            with torch.no_grad():
                cur_suffix = _decode_suffix(tokenizer, suffix_ids.detach().cpu())
            print(f"[GCG] step {step+1}/{cfg.steps}  loss={float(loss.item()):.4f}  pos={pos}  tok={best_tok}")
            print(f"[GCG] suffix: {cur_suffix!r}")

    return _decode_suffix(tokenizer, suffix_ids.detach().cpu())


