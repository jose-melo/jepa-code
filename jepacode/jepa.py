import copy

import torch
from torch import nn


def seeded(module_fn, seed: int) -> nn.Module:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    m = module_fn()
    torch.random.set_rng_state(state)
    return m


def mlp(d_in: int, d_h: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Linear(d_h, d_out))


class EncoderE0(nn.Module):
    def __init__(self, d_in: int, d_z: int = 64, d_h: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), mlp(d_h, d_h, d_z))

    def forward(self, x):
        return self.net(x)


class EncoderE1(nn.Module):
    def __init__(self, *_a, **_k):
        super().__init__()
        raise NotImplementedError("TODO(étudiants): small transformer over canonical IR tokens (GPU <= 8 GB)")


class Predictor(nn.Module):
    def __init__(self, d_z: int, n_actions: int, d_h: int = 128):
        super().__init__()
        self.emb, self.net = nn.Embedding(n_actions, d_z), mlp(2 * d_z, d_h, d_z)

    def forward(self, z, a):
        return z + self.net(torch.cat([z, self.emb(a)], -1))


class RewardHead(nn.Module):
    def __init__(self, d_z: int, n_actions: int, d_h: int = 64):
        super().__init__()
        self.emb, self.net = nn.Embedding(n_actions, d_z), mlp(2 * d_z, d_h, 1)

    def forward(self, z, a):
        return self.net(torch.cat([z, self.emb(a)], -1)).squeeze(-1)


class ValueHead(nn.Module):
    def __init__(self, *_a, **_k):
        super().__init__()
        raise NotImplementedError("TODO(étudiants): A2 no-dynamics control, V(IR, sequence), no predictor")


class EMATarget(nn.Module):
    def __init__(self, online: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.m, self.target = momentum, copy.deepcopy(online).requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        for t, o in zip(self.target.parameters(), online.parameters()):
            t.mul_(self.m).add_(o.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def forward(self, x):
        return self.target(x)


def prediction_loss(z_pred, z_next_target):
    return ((z_pred - z_next_target.detach()) ** 2).mean()


def variance_floor(z, gamma: float = 1.0, eps: float = 1e-4):
    return torch.relu(gamma - torch.sqrt(z.var(0) + eps)).mean()


def sigreg(z):
    raise NotImplementedError("TODO(étudiants): SIGReg (LeJEPA) to replace variance_floor")


def effective_rank(z) -> float:
    s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))


def action_sensitivity(pred: Predictor, z, n_actions: int) -> float:
    a = torch.randint(0, n_actions, (z.shape[0],))
    b = (a + torch.randint(1, n_actions, (z.shape[0],))) % n_actions
    with torch.no_grad():
        return float((pred(z, a) - pred(z, b)).norm(dim=-1).mean())


def train(*_a, **_k):
    raise NotImplementedError("TODO(étudiants): training loop")


def plan(*_a, **_k):
    raise NotImplementedError("TODO(étudiants): latent planner")
