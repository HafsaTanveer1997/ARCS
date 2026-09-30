import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor([init_temp], dtype=torch.float32))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> float:
        device = val_logits.device
        self.to(device)

        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        value = float(self.temperature.detach().cpu().item())
        if verbose:
            print(f"Temperature: {value:.6f}")
        return value


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(float(temperature), 1e-6)


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, targets).item())
