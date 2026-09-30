from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


@dataclass
class ConfidenceSummary:
    top1: float
    ece: float
    nll: float
    brier: float
    cgap: float
    abs_cgap: float
    coverage: float
    trusted_accuracy: float
    dangerous_error: float
    mistake_capture: float
    tt: int
    tf: int
    ft: int
    ff: int

    def to_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets)

    edges = torch.linspace(
        0.0,
        1.0,
        int(n_bins) + 1,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )

    ece = torch.zeros((), device=probabilities.device, dtype=probabilities.dtype)
    total = max(int(targets.numel()), 1)

    for index in range(int(n_bins)):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum().item())
        if count == 0:
            continue
        bin_accuracy = correct[mask].float().mean()
        bin_confidence = confidence[mask].mean()
        ece = ece + (count / total) * torch.abs(bin_accuracy - bin_confidence)

    return float(ece.item())


def trusted_prediction_counts(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.70,
) -> tuple[int, int, int, int]:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets)
    accepted = confidence >= float(threshold)

    tt = int((correct & accepted).sum().item())
    tf = int((correct & (~accepted)).sum().item())
    ft = int(((~correct) & accepted).sum().item())
    ff = int(((~correct) & (~accepted)).sum().item())
    return tt, tf, ft, ff


def compute_confidence_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_bins: int = 15,
    threshold: float = 0.70,
) -> ConfidenceSummary:
    logits = logits.float()
    targets = targets.long()
    probabilities = F.softmax(logits, dim=1)

    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(targets)

    top1 = 100.0 * float(correct.float().mean().item())
    ece = expected_calibration_error(probabilities, targets, n_bins=n_bins)
    nll = float(F.nll_loss(probabilities.clamp_min(1e-12).log(), targets).item())

    one_hot = F.one_hot(targets, num_classes=probabilities.size(1)).float()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item())

    accuracy_fraction = float(correct.float().mean().item())
    mean_confidence = float(confidence.mean().item())
    cgap = accuracy_fraction - mean_confidence

    tt, tf, ft, ff = trusted_prediction_counts(
        probabilities,
        targets,
        threshold=threshold,
    )
    total = max(tt + tf + ft + ff, 1)
    accepted = tt + ft
    mistakes = ft + ff

    coverage = 100.0 * accepted / total
    trusted_accuracy = 100.0 * tt / accepted if accepted else 0.0
    dangerous_error = 100.0 * ft / total
    mistake_capture = 100.0 * ff / mistakes if mistakes else 0.0

    return ConfidenceSummary(
        top1=top1,
        ece=ece,
        nll=nll,
        brier=brier,
        cgap=cgap,
        abs_cgap=abs(cgap),
        coverage=coverage,
        trusted_accuracy=trusted_accuracy,
        dangerous_error=dangerous_error,
        mistake_capture=mistake_capture,
        tt=tt,
        tf=tf,
        ft=ft,
        ff=ff,
    )
