from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


_EPS = 1.0e-8


def zscore(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(_EPS)
    return (values - mean) / std


@dataclass
class DisagreementComponents:
    fp_probs: torch.Tensor
    q_probs: torch.Tensor
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
    fp_probs = F.softmax(fp_logits.float(), dim=1)
    q_probs = F.softmax(q_logits.float(), dim=1)

    fp_conf, fp_pred = fp_probs.max(dim=1)
    q_conf, q_pred = q_probs.max(dim=1)

    fp_top2 = torch.topk(fp_probs, k=2, dim=1).values
    q_top2 = torch.topk(q_probs, k=2, dim=1).values
    fp_margin = fp_top2[:, 0] - fp_top2[:, 1]
    q_margin = q_top2[:, 0] - q_top2[:, 1]

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
        fp_probs=fp_probs,
        q_probs=q_probs,
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
    conf_reference_values = torch.abs(
        components.fp_conf - components.q_conf
    ).detach().cpu().numpy()

    kl_reference_values = torch.sum(
        components.fp_probs
        * (
            torch.log(components.fp_probs.clamp_min(1.0e-12))
            - torch.log(components.q_probs.clamp_min(1.0e-12))
        ),
        dim=1,
    ).detach().cpu().numpy()

    conf_ref = max(
        float(np.percentile(conf_reference_values, reference_percentile)),
        _EPS,
    )
    kl_ref = max(
        float(np.percentile(kl_reference_values, reference_percentile)),
        _EPS,
    )

    mean_conf_degradation = float(components.conf_gap.mean().item())
    mean_kl = float(components.kl_fp_q.mean().item())

    severity = (
        0.5 * mean_conf_degradation / conf_ref
        + 0.5 * mean_kl / kl_ref
    )
    severity = float(np.clip(severity, 0.0, 1.0))
    return severity, conf_ref, kl_ref
