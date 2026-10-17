"""
Event-span masking for LV-MAE  (A + C)

  C  change-point events : change score s_i = 1 - cos(e_i, e_{i-1});
                           boundary where s_i > mean(s) + 0.5*std(s), min event length 2
  A  stochastic selection: event distinctiveness = 1 - cos(mean(event), mean(video)),
                           z-scored, sampled with Gumbel-top-k on softmax(score / tau),
                           tau annealed 2.0 -> 0.25 (cosine) over pretraining
  Whole events are masked greedily (each capped at 40% of the video), the remainder
  is topped up uniformly; videos shorter than 8 segments use a per-segment fallback.

Input  x: [L, D] L2-normalised segment embeddings
Output x_masked [len_keep, D], mask [L] (0 keep / 1 masked), ids_restore [L], ids_shuffle [L]
"""
import math
import torch
import torch.nn.functional as F


def _pack(x, keep_idx, mask_idx):
    L, D = x.shape
    ids_shuffle = torch.cat([keep_idx, mask_idx]).long()
    ids_restore = torch.argsort(ids_shuffle)
    x_masked = x[keep_idx]
    mask = torch.ones(L)
    mask[keep_idx] = 0
    return x_masked, mask, ids_restore, ids_shuffle


def _counts(L, mask_ratio):
    len_keep = max(1, int(L * (1 - mask_ratio)))
    return len_keep, L - len_keep


def change_scores(x):
    s = 1 - F.cosine_similarity(x[1:], x[:-1], dim=-1)
    return torch.cat([s.new_zeros(1), s])



def _short_video_fallback(x, mask_ratio, tau):
    """Videos too short to split into events: Gumbel-top-k over single segments by change score."""
    L = x.shape[0]
    _, len_mask = _counts(L, mask_ratio)
    s = change_scores(x)
    s = (s - s.mean()) / (s.std() + 1e-6)
    gumbel = -torch.log(-torch.log(torch.rand(L).clamp_min(1e-9)))
    order = torch.argsort(s / tau + gumbel, descending=True)
    return _pack(x, torch.sort(order[len_mask:]).values, order[:len_mask])


def segment_events(x, min_len=2, z=0.5):
    s = change_scores(x)
    thr = s[1:].mean() + z * s[1:].std() if len(s) > 2 else float("inf")
    bounds, last = [0], 0
    for i in range(1, len(s)):
        if s[i] > thr and i - last >= min_len:
            bounds.append(i)
            last = i
    bounds.append(len(s))
    return [torch.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def event_span_masking(x, mask_ratio, tau=0.5, min_len=2, z=0.5, max_span_frac=0.4, min_video_len=8):
    L = x.shape[0]
    if L < min_video_len:
        return _short_video_fallback(x, mask_ratio, tau=max(tau, 0.05))
    len_keep, len_mask = _counts(L, mask_ratio)
    events = segment_events(x, min_len, z)
    g = F.normalize(x.mean(0), dim=-1)
    sal = torch.stack([1 - F.cosine_similarity(F.normalize(x[e].mean(0), dim=-1), g, dim=-1)
                       for e in events])
    sal = (sal - sal.mean()) / (sal.std() + 1e-6) if len(sal) > 1 else sal
    gumbel = -torch.log(-torch.log(torch.rand(len(events)).clamp_min(1e-9)))
    max_span = max(1, int(round(max_span_frac * L)))
    masked = []
    for ei in torch.argsort(sal / tau + gumbel, descending=True).tolist():
        need = len_mask - len(masked)
        if need <= 0:
            break
        e = events[ei]
        take = min(len(e), need, max_span)
        if len(e) > take:
            start = torch.randint(0, len(e) - take + 1, (1,)).item()
            e = e[start:start + take]
        masked.extend(e.tolist())
    if len(masked) < len_mask:
        rest = torch.tensor(sorted(set(range(L)) - set(masked)), dtype=torch.long)
        masked.extend(rest[torch.randperm(len(rest))[:len_mask - len(masked)]].tolist())
    masked = torch.tensor(sorted(masked), dtype=torch.long)
    keep = torch.tensor(sorted(set(range(L)) - set(masked.tolist())), dtype=torch.long)
    return _pack(x, keep, masked)



def tau_schedule(epoch, total_epochs, tau_start=2.0, tau_end=0.25):
    t = min(max(epoch / max(1, total_epochs - 1), 0.0), 1.0)
    return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * t))
