from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


_EPS = 1.0e-8


def zscore(values: torch.Tensor) -> torch.Tensor:
    """Population-standardized values used by the paper implementation."""
    values = values.float()
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(_EPS)
    return (values - mean) / std


@dataclass
class DisagreementComponents:
    fp_conf: torch.Tensor
    q_conf: torch.Tensor
    fp_margin: torch.Tensor
    q_margin: torch.Tensor
    conf_gap: torch.Tensor
    margin_gap: torch.Tensor
    kl_fp_q: torch.Tensor
    flip: torch.Tensor


def disagreement_components(
    fp_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> DisagreementComponents:
    """Compute the per-sample FP-vs-proxy disagreement signals used by ARCS."""
    fp_probs = F.softmax(fp_logits.float(), dim=1)
    q_probs = F.softmax(q_logits.float(), dim=1)

    fp_conf, fp_pred = fp_probs.max(dim=1)
    q_conf, q_pred = q_probs.max(dim=1)

    fp_top2 = torch.topk(fp_probs, k=2, dim=1).values
    q_top2 = torch.topk(q_probs, k=2, dim=1).values
    fp_margin = fp_top2[:, 0] - fp_top2[:, 1]
    q_margin = q_top2[:, 0] - q_top2[:, 1]

    # The score uses one-sided degradation terms.
    conf_gap = torch.clamp(fp_conf - q_conf, min=0.0)
    margin_gap = torch.clamp(fp_margin - q_margin, min=0.0)

    kl_fp_q = (
        fp_probs
        * (
            fp_probs.clamp_min(_EPS).log()
            - q_probs.clamp_min(_EPS).log()
        )
    ).sum(dim=1)

    flip = (fp_pred != q_pred).float()

    return DisagreementComponents(
        fp_conf=fp_conf,
        q_conf=q_conf,
        fp_margin=fp_margin,
        q_margin=q_margin,
        conf_gap=conf_gap,
        margin_gap=margin_gap,
        kl_fp_q=kl_fp_q,
        flip=flip,
    )


def disagreement_score(
    components: DisagreementComponents,
    kl_weight: float = 0.5,
    flip_bonus: float = 0.25,
) -> torch.Tensor:
    """ARCS disagreement score R(x)."""
    return (
        zscore(components.conf_gap)
        + zscore(components.margin_gap)
        + float(kl_weight) * zscore(components.kl_fp_q)
        + float(flip_bonus) * components.flip
    )


def severity_score(
    components: DisagreementComponents,
    reference_percentile: float = 90.0,
) -> tuple[float, float, float]:
    """Return continuous severity and its confidence/KL reference scales.

    The confidence numerator is the mean one-sided degradation used by ARCS.
    Its reference denominator intentionally uses the percentile of the absolute
    FP-vs-proxy confidence difference, matching the paper implementation.
    """
    q = float(reference_percentile) / 100.0

    abs_conf_gap = torch.abs(components.fp_conf - components.q_conf)
    conf_ref = float(torch.quantile(abs_conf_gap, q).item())
    kl_ref = float(torch.quantile(components.kl_fp_q, q).item())

    conf_ref = max(conf_ref, _EPS)
    kl_ref = max(kl_ref, _EPS)

    mean_conf_degradation = float(components.conf_gap.mean().item())
    mean_kl = float(components.kl_fp_q.mean().item())

    severity = (
        0.5 * mean_conf_degradation / conf_ref
        + 0.5 * mean_kl / kl_ref
    )
    severity = max(0.0, min(1.0, severity))
    return severity, conf_ref, kl_ref
