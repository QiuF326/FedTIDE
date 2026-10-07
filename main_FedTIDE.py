"""
FedTIDE：面向有限通信轮次的参考状态纠正联邦学习方法
FedTIDE: Reference-State Correction Federated Learning for Limited Communication Rounds

本文件实现论文中的主要实验流程，包括统计初始化、输出尺度校准、
固定参考状态划分、状态条件梯度与曲率统计、受约束状态纠正、
分布式候选评估，以及主要消融与对照实验。

This file implements the main experimental pipeline of FedTIDE, including
statistical initialization, logit-scale calibration, fixed-reference state
partitioning, state-conditional gradient and curvature statistics, constrained
state correction, distributed candidate evaluation, and the main ablation and
comparison settings.

支持模式 / Supported modes:
- full: 完整方法 / full FedTIDE
- global_only: 无状态纠正 / no state-correction direction
- no_projection: 无保护投影 / no protection projection
- no_curvature: 无曲率预处理 / no curvature preconditioning
- init_fedavg: 相同初始化后的分类头 FedAvg /
  classifier-head FedAvg with the same initialization

默认随机种子 / Default seeds: 2025, 2026, 2027

输出文件 / Outputs:
- report.txt: 实验汇总 / experiment summary
- rounds.csv: 逐轮指标 / round-wise metrics
- mechanisms.csv: 机制诊断 / mechanism diagnostics

不使用公开代理数据 / No public proxy data are used.
"""

from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def labels_of(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base = labels_of(dataset.dataset)
        return base[np.asarray(dataset.indices)]
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise TypeError("Dataset must expose targets or labels")

def backbone_input_size(name: str) -> int:
    # Torchvision ViT requires its configured image size.  Swin accepts 224
    # naturally.  CNN encoders can consume native CIFAR/SVHN resolution.
    return 224 if name in {"vit_b_16", "swin_t"} else 32

def get_dataset(
    name: str,
    root: str,
    input_size: int,
    normalization: str,
) -> Tuple[int, Dataset, Dataset]:
    name = name.lower()
    if name not in {"cifar10", "cifar100", "svhn"}:
        raise ValueError("dataset must be cifar10, cifar100, or svhn")

    if normalization == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif name in {"cifar10", "cifar100"}:
        # Keep the protocol used in the user's previous experiments by default.
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    ops = []
    if input_size != 32:
        ops.append(transforms.Resize((input_size, input_size)))
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    tf = transforms.Compose(ops)

    if name == "cifar10":
        return (
            10,
            datasets.CIFAR10(root, train=True, download=True, transform=tf),
            datasets.CIFAR10(root, train=False, download=True, transform=tf),
        )
    if name == "cifar100":
        return (
            100,
            datasets.CIFAR100(root, train=True, download=True, transform=tf),
            datasets.CIFAR100(root, train=False, download=True, transform=tf),
        )

    svhn_root = os.path.join(root, "svhn")
    return (
        10,
        datasets.SVHN(svhn_root, split="train", download=True, transform=tf),
        datasets.SVHN(svhn_root, split="test", download=True, transform=tf),
    )

def dirichlet_partition(
    dataset: Dataset,
    clients: int,
    alpha: float,
    classes: int,
    min_client_size: int,
    seed: int,
) -> List[List[int]]:
    labels = labels_of(dataset)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        result: List[List[int]] = [[] for _ in range(clients)]
        for cls in range(classes):
            indices = np.flatnonzero(labels == cls)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(clients, alpha))
            capacity = np.asarray(
                [len(x) < len(labels) / clients for x in result], dtype=float
            )
            proportions *= capacity
            if proportions.sum() <= 0:
                proportions = np.full(clients, 1.0 / clients)
            else:
                proportions /= proportions.sum()
            cuts = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(indices, cuts)):
                result[client_id].extend(split.tolist())
        if min(map(len, result)) >= min_client_size:
            for part in result:
                rng.shuffle(part)
            return result
    raise RuntimeError("unable to create valid Dirichlet partition")


def load_or_create_partition(
    dataset: Dataset,
    clients: int,
    alpha: float,
    classes: int,
    min_client_size: int,
    seed: int,
    partition_file: str,
) -> List[List[int]]:
    if partition_file:
        path = Path(partition_file)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            splits = [[int(i) for i in part] for part in payload["splits"]]
            if len(splits) != clients:
                raise ValueError("partition file client count mismatch")
            if sum(map(len, splits)) != len(dataset):
                raise ValueError("partition file dataset-size mismatch")
            flat=np.asarray([i for part in splits for i in part],dtype=np.int64)
            if not np.array_equal(np.sort(flat),np.arange(len(dataset))):
                raise ValueError('Partition must cover every training index exactly once')
            return splits

    splits = dirichlet_partition(
        dataset, clients, alpha, classes, min_client_size, seed
    )
    if partition_file:
        path = Path(partition_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "num_clients": clients,
                    "dataset_size": len(dataset),
                    "splits": splits,
                }
            ),
            encoding="utf-8",
        )
    return splits

class ReferenceEncoder(nn.Module):
    def __init__(self, backbone: str, pretrained: bool):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            self.feature_dim = int(net.fc.in_features)
            net.fc = nn.Identity()
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            self.feature_dim = int(net.fc.in_features)
            net.fc = nn.Identity()
        elif backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            self.feature_dim = int(net.classifier[-1].in_features)
            net.classifier = nn.Identity()
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.feature_dim = int(net.classifier[-1].in_features)
            net.classifier = nn.Identity()
        elif backbone == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            net = models.vit_b_16(weights=weights)
            self.feature_dim = int(net.heads.head.in_features)
            net.heads = nn.Identity()
        elif backbone == "swin_t":
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            net = models.swin_t(weights=weights)
            self.feature_dim = int(net.head.in_features)
            net.head = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

        self.encoder = net
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ResidualAdapterBlock(nn.Module):
    """Architecture-independent feature-space residual adapter."""

    def __init__(self, dim: int, bottleneck: int, dropout: float):
        super().__init__()
        hidden = min(max(1, bottleneck), dim)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(hidden, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        # Exact identity at Stage I while still allowing gradients into up.
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.dropout(self.act(self.down(self.norm(x)))))
        return x + delta

class FeatureAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, depth: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualAdapterBlock(dim, bottleneck, dropout) for _ in range(depth)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for block in self.blocks:
            h = block(h)
        return h

class AdaptiveHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        bottleneck: int,
        adapter_depth: int,
        adapter_dropout: float,
    ):
        super().__init__()
        self.adapter = FeatureAdapter(
            feature_dim, bottleneck, adapter_depth, adapter_dropout
        )
        self.fc = nn.Linear(feature_dim, num_classes)
        self.register_buffer("logit_scale", torch.tensor(1.0, dtype=torch.float32))

    def representation(self, z: torch.Tensor) -> torch.Tensor:
        return self.adapter(z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.logit_scale * self.fc(self.representation(z))

class FederatedModel(nn.Module):
    def __init__(self, args: argparse.Namespace, num_classes: int):
        super().__init__()
        self.reference = ReferenceEncoder(args.backbone, args.pretrained)
        self.feature_dim = self.reference.feature_dim
        self.head = AdaptiveHead(
            self.feature_dim,
            num_classes,
            args.adapter_bottleneck,
            args.adapter_depth,
            args.adapter_dropout,
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.reference(x)
        return self.head(z)


def module_state_nbytes(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values()))

STABLE = 0

CORRECTED = 1

PERSISTENT = 2

FORGOTTEN = 3


@dataclass
class Stage1Stats:
    client_id: int
    ridge_A: np.ndarray
    ridge_B: np.ndarray
    class_counts: np.ndarray
    local_samples: int






class FederatedClient:
    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        device: torch.device,
        seed: int,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.seed = seed
        self.features: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.class_counts: Optional[np.ndarray] = None
        # R3 local states are retained locally and reused in R4.  They are never
        # sent sample-by-sample to the server.
        self.r3_states: Optional[torch.Tensor] = None

    def extract_reference_features(
        self,
        model: FederatedModel,
        batch_size: int,
        workers: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.features is not None and self.labels is not None:
            return self.features, self.labels

        model.reference.eval()
        zs, ys = [], []
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
        with torch.no_grad():
            for x, y in loader:
                z = model.features(x.to(self.device))
                zs.append(z.detach().cpu())
                ys.append(y.long().cpu())
        self.features = torch.cat(zs, dim=0)
        self.labels = torch.cat(ys, dim=0)
        return self.features, self.labels

    def build_stage1_stats(
        self,
        model: FederatedModel,
        num_classes: int,
        args: argparse.Namespace,
    ) -> Stage1Stats:
        z, y = self.extract_reference_features(
            model, args.feature_batch_size, args.workers
        )
        z64 = z.double()
        one = torch.ones((len(z64), 1), dtype=torch.float64)
        za = torch.cat([z64, one], dim=1)

        A = za.T @ za
        B = torch.zeros((za.shape[1], num_classes), dtype=torch.float64)
        counts = torch.zeros(num_classes, dtype=torch.float64)
        for cls_tensor in torch.unique(y):
            cls = int(cls_tensor.item())
            mask = y == cls
            B[:, cls] = za[mask].sum(0)
            counts[cls] = float(mask.sum().item())

        self.class_counts = counts.long().numpy()
        return Stage1Stats(
            client_id=self.client_id,
            ridge_A=A.float().numpy(),
            ridge_B=B.float().numpy(),
            class_counts=self.class_counts.copy(),
            local_samples=len(y),
        )

    @torch.no_grad()
    def calibration_loss_sums(
        self,
        head: AdaptiveHead,
        scales: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        assert self.features is not None and self.labels is not None
        head.eval()
        totals = torch.zeros(len(scales), dtype=torch.float64)
        for start in range(0, len(self.features), batch_size):
            z = self.features[start:start + batch_size].to(self.device)
            y = self.labels[start:start + batch_size].to(self.device)
            raw = head.fc(head.representation(z))
            for j, scale in enumerate(scales):
                totals[j] += F.cross_entropy(
                    raw * scale.to(self.device), y, reduction="sum"
                ).double().cpu()
        return totals.numpy()






class FederatedServer:
    def __init__(self, model: FederatedModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.cumulative_mb = 0.0

    def initialize_stage1(self, stats: List[Stage1Stats], ridge_lambda: float) -> None:
        A = np.sum([s.ridge_A.astype(np.float64) for s in stats], axis=0)
        B = np.sum([s.ridge_B.astype(np.float64) for s in stats], axis=0)
        total_n = float(sum(s.local_samples for s in stats))
        A /= max(1.0, total_n)
        B /= max(1.0, total_n)
        reg = np.eye(A.shape[0], dtype=np.float64) * ridge_lambda
        reg[-1, -1] = 0.0
        try:
            solution = np.linalg.solve(A + reg, B)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(A + reg) @ B
        with torch.no_grad():
            self.model.head.fc.weight.copy_(
                torch.from_numpy(solution[:-1].T.astype(np.float32)).to(self.device)
            )
            self.model.head.fc.bias.copy_(
                torch.from_numpy(solution[-1].astype(np.float32)).to(self.device)
            )

    def initialize_stage1_gaussian(
        self,
        stats: List[Stage1Stats],
        cov_shrinkage: float,
        empirical_prior: bool,
    ) -> None:
        """Initialize the linear head as a shared-covariance Gaussian/LDA classifier.

        Stage1Stats already contains all required additive sufficient statistics:
        A[:D,:D] = sum zz^T, B[:D,c] = sum_{y=c} z, B[-1,c] = n_c.
        No extra client upload is needed beyond the existing R1 packet.
        """
        A = np.sum([st.ridge_A.astype(np.float64) for st in stats], axis=0)
        B = np.sum([st.ridge_B.astype(np.float64) for st in stats], axis=0)
        D = A.shape[0] - 1
        C = B.shape[1]
        Q = A[:D, :D]
        sums = B[:D, :].T  # [C, D]
        raw_n = B[-1, :].copy()
        safe_n = np.maximum(raw_n, 1.0)
        means = sums / safe_n[:, None]
        total_n = max(float(raw_n.sum()), 1.0)

        between = np.zeros((D, D), dtype=np.float64)
        for c in range(C):
            if raw_n[c] <= 0:
                continue
            mu = means[c][:, None]
            between += raw_n[c] * (mu @ mu.T)

        cov = (Q - between) / total_n
        cov = 0.5 * (cov + cov.T)
        scale = float(np.trace(cov) / max(D, 1))
        cov = cov + (cov_shrinkage * scale + 1e-8) * np.eye(D, dtype=np.float64)
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)

        if empirical_prior:
            priors = np.maximum(raw_n, 1.0)
            priors = priors / priors.sum()
        else:
            priors = np.full(C, 1.0 / C, dtype=np.float64)

        weight = (precision @ means.T).T
        bias = -0.5 * np.sum(means * (means @ precision.T), axis=1) + np.log(priors)
        with torch.no_grad():
            self.model.head.fc.weight.copy_(
                torch.from_numpy(weight.astype(np.float32)).to(self.device)
            )
            self.model.head.fc.bias.copy_(
                torch.from_numpy(bias.astype(np.float32)).to(self.device)
            )
            self.model.head.logit_scale.fill_(1.0)


    def calibrate_logit_scale(
        self,
        clients: List[FederatedClient],
        args: argparse.Namespace,
    ) -> Tuple[float, float]:
        scales = torch.logspace(
            math.log10(args.cal_scale_min),
            math.log10(args.cal_scale_max),
            steps=args.cal_scale_steps,
        )
        total_loss = np.zeros(args.cal_scale_steps, dtype=np.float64)
        total_n = 0
        for client in clients:
            total_loss += client.calibration_loss_sums(
                self.model.head, scales, args.calibration_batch_size
            )
            total_n += len(client.labels) if client.labels is not None else 0
        idx = int(np.argmin(total_loss))
        best = float(scales[idx].item())
        with torch.no_grad():
            self.model.head.logit_scale.fill_(best)
        return best, float(total_loss[idx] / max(1, total_n))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return 100.0 * correct / max(1, total)







def state_ids(anchor_pred, current_pred, labels):
    a, b = anchor_pred.eq(labels), current_pred.eq(labels)
    out = torch.full_like(labels, PERSISTENT)
    out[a & b] = STABLE
    out[~a & b] = CORRECTED
    out[a & ~b] = FORGOTTEN
    return out


class RepairClient(FederatedClient):
    """All example-level state and cached features stay on the client."""
    @torch.no_grad()
    def packet(self, weight, anchor, batch_size):
        weight, anchor = weight.to(self.device).double(), anchor.to(self.device).double()
        d, c = weight.shape
        grad = torch.zeros(4, d, c, dtype=torch.float64, device=self.device)
        fisher = torch.zeros(c, c, dtype=torch.float64, device=self.device)
        losses = torch.zeros(4, dtype=torch.float64, device=self.device)
        counts = torch.zeros(4, dtype=torch.long, device=self.device)
        states = []
        for lo in range(0, len(self.labels), batch_size):
            z = self.features[lo:lo+batch_size].to(self.device).double()
            x = torch.cat((z, torch.ones(len(z), 1, device=self.device, dtype=z.dtype)), 1)
            y = self.labels[lo:lo+batch_size].to(self.device)
            logits = x @ weight
            p = logits.softmax(1)
            st = state_ids((x @ anchor).argmax(1), logits.argmax(1), y)
            states.append(st.cpu())
            ce = F.cross_entropy(logits, y, reduction='none')
            residual = p - F.one_hot(y, c).to(p.dtype)
            for s in range(4):
                mask = st == s
                grad[s] += x[mask].T @ residual[mask]
                losses[s] += ce[mask].sum()
                counts[s] += mask.sum()
            fisher += torch.diag(p.sum(0)) - p.T @ p
        self.audit_states = torch.cat(states)
        # Float64 upload here is intentional and charged in the communication ledger.
        return dict(gradient=grad.cpu(), fisher=fisher.cpu(), loss=losses.cpu(), count=counts.cpu())

    @torch.no_grad()
    def audit(self, weight, directions, candidates, batch_size):
        weight = weight.to(self.device).double()
        directions = [v.to(self.device).double() for v in directions]
        loss = torch.zeros(len(candidates), 4, dtype=torch.float64, device=self.device)
        correct = torch.zeros(len(candidates), 4, dtype=torch.long, device=self.device)
        for lo in range(0, len(self.labels), batch_size):
            z = self.features[lo:lo+batch_size].to(self.device).double()
            x = torch.cat((z, torch.ones(len(z), 1, device=self.device, dtype=z.dtype)), 1)
            y = self.labels[lo:lo+batch_size].to(self.device)
            st = self.audit_states[lo:lo+batch_size].to(self.device)
            base = x @ weight
            change = [x @ v for v in directions]
            for j, (di, step) in enumerate(candidates):
                logits = base if di < 0 else base + step * change[di]
                ce = F.cross_entropy(logits, y, reduction='none')
                hit = logits.argmax(1).eq(y)
                for s in range(4):
                    mask = st == s
                    loss[j, s] += ce[mask].sum()
                    correct[j, s] += hit[mask].sum()
        return dict(loss=loss.cpu(), correct=correct.cpu())


def add_packets(packets):
    return {k: sum((p[k] for p in packets), torch.zeros_like(packets[0][k])) for k in packets[0]}


def payload_bytes(packet):
    return sum(v.numel() * v.element_size() for v in packet.values())


def positive_inverse(matrix, shrinkage):
    matrix = (matrix + matrix.T).double() / 2
    values, vectors = torch.linalg.eigh(matrix)
    # Same shrinkage coefficient as Stage I; numerical damping, not dataset tuning.
    floor = max(float(values.clamp_min(0).mean()) * shrinkage, 1e-10)
    return (vectors * values.clamp_min(floor).reciprocal()[None, :]) @ vectors.T


def make_directions(packet, gram_inverse, shrinkage, state_aware, mode='full'):
    """Exact aggregate gradients; approximate (Kronecker) curvature, not exact Newton."""
    from scipy.optimize import minimize
    count = packet['count'].double()
    present = count > 0
    total = count.sum().clamp_min(1)
    gradients = packet['gradient'] / count.clamp_min(1)[:, None, None]
    global_gradient = packet['gradient'].sum(0) / total
    output_inverse = positive_inverse(packet['fisher'] / total, shrinkage)
    precondition = (lambda g:g.clone()) if mode=='no_curvature' else (lambda g:gram_inverse @ g @ output_inverse)
    ordinary = -precondition(global_gradient)
    ordinary -= ordinary.mean(1, keepdim=True)  # softmax common-logit gauge
    if not state_aware or not present[PERSISTENT]:
        return [ordinary], [], dict(projected=False, constraint_max=0.)

    # Protect S/C/F and overall CE to first order while minimizing persistent CE.
    protected = [s for s in (STABLE, CORRECTED, FORGOTTEN) if present[s]]
    constraints = [gradients[s] for s in protected] + [global_gradient]
    target = gradients[PERSISTENT]
    mt = precondition(target)
    if mode=='no_projection':
        direction=-mt
        direction-=direction.mean(1,keepdim=True)
        return [ordinary,direction],protected,dict(projected=False,
            constraint_max=max(float((g*direction).sum()) for g in constraints))
    mc = [precondition(g) for g in constraints]
    q = np.array([[float((g * v).sum()) for v in mc] for g in constraints])
    b = np.array([float((g * mt).sum()) for g in constraints])
    scale = max(float(np.abs(q).max()), float(np.abs(b).max()), 1e-12)
    q, b = (q + q.T) / (2*scale), b / scale
    result = minimize(lambda t: .5*t@q@t + b@t, np.zeros(len(b)),
                      jac=lambda t:q@t+b, bounds=[(0,None)]*len(b), method='L-BFGS-B',
                      options=dict(ftol=1e-14, gtol=1e-10, maxiter=500))
    direction = -mt
    for v, lam in zip(mc, result.x):
        direction = direction - float(lam)*v
    direction -= direction.mean(1, keepdim=True)
    violations = [float((g*direction).sum()) for g in constraints]
    tolerance = 1e-7 * max(1., float(target.norm()*direction.norm()))
    ok = (torch.isfinite(direction).all().item() and max(violations) <= tolerance
          and float((target*direction).sum()) < 0)
    # Actual losses are audited even when this numerical first-order check passes.
    dirs = [ordinary, direction] if ok else [ordinary]
    return dirs, protected, dict(projected=bool(ok), constraint_max=max(violations))


def select_candidate(audit, count, protected, mode='full'):
    loss = audit['loss'] / count.clamp_min(1).double()[None, :]
    total = audit['loss'].sum(1) / count.sum().clamp_min(1)
    accuracy_count = audit['correct'].sum(1)
    feasible = torch.isfinite(total) & (total <= total[0] + 1e-10)
    feasible &= accuracy_count >= accuracy_count[0]
    # Do not veto useful updates merely because CE of already-correct samples
    # rises: this was an overly restrictive failure mode in earlier guarded runs.
    # Overall CE and actual global training accuracy provide the safety test.
    feasible[0] = True
    acceptance_count = int(feasible.sum())
    # All modes use the same global training CE / accuracy acceptance rule.
    # State information affects direction construction only.
    idx = int(torch.where(feasible, total, torch.full_like(total, float('inf'))).argmin())
    if total[idx] >= total[0] - 1e-10:
        idx = 0
    select_candidate.acceptance_count=acceptance_count
    return idx, loss, total, feasible


@torch.no_grad()
def install_weight(server, weight):
    server.model.head.fc.weight.copy_(weight[:-1].T.to(server.device).float())
    server.model.head.fc.bias.copy_(weight[-1].to(server.device).float())
    server.model.head.logit_scale.fill_(1.)


@torch.no_grad()
def client_fedavg_head(client, initial_weight, args, communication_round):
    """Client-local SGD on frozen features. Only the float32 head is returned.

    Fresh optimizer each round: SGD without momentum or weight decay.
    The augmented last row is the trainable bias. No state-dependent sampling.
    """
    n = len(client.labels)
    if n == 0:
        raise ValueError('FedAvg requires nonempty clients')
    w = initial_weight.to(device=client.device, dtype=torch.float32).clone()
    rng = torch.Generator(device='cpu').manual_seed(
        int(args.seed) + 1000003 * communication_round + 1009 * client.client_id)
    for _ in range(args.fedavg_local_epochs):
        order = torch.randperm(n, generator=rng)
        for lo in range(0, n, args.fedavg_batch_size):
            ids = order[lo:lo + args.fedavg_batch_size]
            z = client.features[ids].to(device=client.device, dtype=torch.float32)
            y = client.labels[ids].to(client.device)
            x = torch.cat((z, torch.ones(len(z), 1, device=client.device)), dim=1)
            # Exact gradient of mean softmax cross-entropy (ordinary SGD).
            residual = (x @ w).softmax(dim=1)
            residual[torch.arange(len(y), device=client.device), y] -= 1.
            w.add_(x.T @ residual, alpha=-args.fedavg_lr / len(y))
    if not torch.isfinite(w).all():
        raise FloatingPointError('Nonfinite FedAvg head; check the logged SGD settings')
    return w.cpu()


@torch.no_grad()
def fedavg_head_round(clients, weight, args, communication_round):
    # Server aggregation uses only head parameters and public sample counts.
    broadcast = weight.float().cpu()
    total = sum(len(c.dataset) for c in clients)
    aggregate = torch.zeros_like(broadcast, dtype=torch.float64)
    uploaded_bytes = 0
    for client in clients:
        local = client_fedavg_head(client, broadcast, args, communication_round)
        aggregate.add_(local.double(), alpha=len(client.dataset) / total)
        uploaded_bytes += local.numel() * local.element_size()
    # Quantize to the actual float32 global head sent in the next exchange.
    result = aggregate.float().double()
    downloaded_bytes = broadcast.numel() * broadcast.element_size() * len(clients)
    return result, uploaded_bytes, downloaded_bytes


def run(args):
    start=time.perf_counter()
    if not 1<=args.communication_rounds<=10:
        raise ValueError('communication_rounds must be between 1 and 10')
    set_seed(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    nc, train, test = get_dataset(args.dataset, args.data_root, backbone_input_size(args.backbone), args.normalization)
    if not args.partition_file:
        a = str(args.alpha).replace('.', 'p')
        args.partition_file = str(Path(args.partition_dir)/f'{args.dataset}_dir_n{args.num_clients}_a{a}_seed{args.seed}.json')
    splits = load_or_create_partition(train,args.num_clients,args.alpha,nc,args.min_client_size,args.seed,args.partition_file)
    clients = [RepairClient(k, Subset(train,list(map(int,ix))), device,args.seed) for k,ix in enumerate(splits)]
    server = FederatedServer(FederatedModel(args,nc),device)
    test_loader = DataLoader(test,batch_size=args.test_batch_size,shuffle=False,num_workers=args.workers)
    # Initialization is inherited exactly; no changed data/feature preprocessing.
    stats = [c.build_stage1_stats(server.model,nc,args) for c in clients]
    server.initialize_stage1(stats,args.ridge_lambda)
    server.initialize_stage1_gaussian(stats,args.cov_shrinkage,args.empirical_prior)
    weight = torch.cat((server.model.head.fc.weight.detach().cpu().T,
                        server.model.head.fc.bias.detach().cpu()[None,:]),0).double()
    anchor = weight.clone()
    gram = sum(torch.from_numpy(s.ridge_A).double() for s in stats) / len(train)
    gram_inverse = positive_inverse(gram,args.cov_shrinkage)
    reference_bytes = module_state_nbytes(server.model.reference.state_dict())
    rows=[]; cumulative=0;last_time=start
    print(f'device={device} method=FedTIDE-GlobalRepair-v4 mode={args.mode} dataset={args.dataset} '
          f'backbone={args.backbone} clients={list(map(len,splits))} private={len(train)} proxy=0 '
          f'rounds={args.communication_rounds} reference_cached={args.reference_cached} partition={args.partition_file}')

    def record(rnd,phase,up,down,**extras):
        nonlocal cumulative,last_time
        cumulative += up+down
        install_weight(server,weight)
        acc=server.evaluate(test_loader)  # Reporting only; never used for acceptance.
        now=time.perf_counter()
        row=dict(round=rnd,phase=phase,accuracy=acc,up_mib=up/2**20,down_mib=down/2**20,
                 communication_mib=(up+down)/2**20,cumulative_mib=cumulative/2**20,
                 round_seconds=now-last_time,elapsed_seconds=now-start,**extras)
        last_time=now
        rows.append(row)
        print(f'round={rnd:02d} phase={phase} acc={acc:.2f}% '
              f'up={up/2**20:.3f}MiB down={down/2**20:.3f}MiB cumulative={cumulative/2**20:.3f}MiB '
              +' '.join(f'{k}={v}' for k,v in extras.items()),flush=True)
        save_table(args,rows)

    up=sum(s.ridge_A.nbytes+s.ridge_B.nbytes+s.class_counts.nbytes for s in stats)
    record(1,'M-statistical-anchor',up,0 if args.reference_cached else reference_bytes*len(clients))
    if args.communication_rounds==1:
        return rows
    # R2 explicitly counts the previously implicit global scale-calibration exchange.
    server.calibrate_logit_scale(clients,args)
    scale=float(server.model.head.logit_scale)
    weight *= scale; anchor=weight.clone()
    record(2,'E-calibrate-common-coordinate',args.cal_scale_steps*8*len(clients),
           (weight.numel()*8+args.cal_scale_steps*4)*len(clients),selected_scale=scale)
    if args.mode == 'init_fedavg':
        print(f'FedAvg head: lr={args.fedavg_lr} local_epochs={args.fedavg_local_epochs} '
              f'batch_size={args.fedavg_batch_size} momentum=0 weight_decay=0 dtype=float32')
        for rnd in range(3, args.communication_rounds + 1):
            weight, up_bytes, down_bytes = fedavg_head_round(clients, weight, args, rnd)
            record(rnd, 'same-init-fedavg-head', up_bytes, down_bytes)
        return rows
    # Clients cache R2 unscaled anchor; scale + current weight arrive at R3.
    for block in range(4):
        direction_round=3+2*block
        if direction_round>args.communication_rounds:
            break
        phase=('E-global-fit','C-state-correction','X-state-correction','K-global-consolidation')[block]
        diagnostic_anchor=anchor
        packets=[c.packet(weight,diagnostic_anchor,args.calibration_batch_size) for c in clients]
        packet=add_packets(packets)
        aware=(block in (1,2) and args.mode!='global_only')
        directions,protected,info=make_directions(packet,gram_inverse,args.cov_shrinkage,aware,args.mode)
        # Same two direction slots in the two correction blocks for all variants.
        # A duplicate ordinary direction adds no information but controls audit budget.
        if block in (1,2) and len(directions)==1:
            directions.append(directions[0].clone())
        # Kept for diagnostic compatibility; selection has no state-specific veto.
        if block==3:
            protected=[s for s in (STABLE,CORRECTED) if packet['count'][s]>0]
        count=packet['count']
        down=(weight.numel()*8+8)*len(clients)
        record(direction_round,phase+'-propose',sum(payload_bytes(p) for p in packets),down,
               states=count.tolist(),projected=info['projected'],constraint_max=info['constraint_max'])
        if direction_round==args.communication_rounds:
            break
        # Fixed batched line-search rule; all candidates are assessed in ONE exchange.
        # No dataset/seed tuning; zero is always included. Fewer dyadic steps are not hidden rounds.
        candidates=[(-1,0.)]+[(di,2.**(-j)) for di in range(len(directions)) for j in range(8)]
        # Include a data-derived global-CE majorization step, rather than silently
        # failing when an ill-conditioned direction needs less than 1/128.
        # Softmax logit Hessian <= 0.5 I, so this is a conservative step for CE.
        g_all=packet['gradient'].sum(0)/count.sum().clamp_min(1)
        for di,d in enumerate(directions):
            slope=float((g_all*d).sum())
            upper=.5*float((d*(gram@d)).sum())
            if slope<0 and upper>0:
                safe=min(1.,-slope/upper)
                candidates.append((di,safe))
        audits=[c.audit(weight,directions,candidates,args.calibration_batch_size) for c in clients]
        audit=add_packets(audits)
        selected,loss,totals,feasible=select_candidate(audit,count,protected,args.mode)
        di,eta=candidates[selected]
        if di>=0:
            weight = weight + eta*directions[di]
        del packets,audits
        record(direction_round+1,phase+'-audit',payload_bytes(audit)*len(clients),
               (sum(d.numel()*8 for d in directions)+len(candidates)*16)*len(clients),
               accepted=selected!=0,direction=di,step=eta,feasible_candidates=int(feasible.sum()),
               acceptance_candidates=select_candidate.acceptance_count,
               train_ce_before=float(totals[0]),train_ce_after=float(totals[selected]),
               train_acc=100*float(audit['correct'][selected].sum())/int(count.sum()),
               persistent_corrected=int(audit['correct'][selected,PERSISTENT]),
               corrected_retained=int(audit['correct'][selected,CORRECTED]),
               corrected_lost=int(count[CORRECTED]-audit['correct'][selected,CORRECTED]),
               new_forgotten=int(count[STABLE]-audit['correct'][selected,STABLE]),
               net_persistent_repair=int(audit['correct'][selected,PERSISTENT] -
                   (count[STABLE]-audit['correct'][selected,STABLE]) -
                   (count[CORRECTED]-audit['correct'][selected,CORRECTED])),
               state_ce_before=loss[0].tolist(),state_ce_after=loss[selected].tolist())
    return rows


def parse_args():
    p=argparse.ArgumentParser(description='No-proxy experiments; CSV only; default seeds 2025/2026/2027',
        formatter_class=argparse.RawDescriptionHelpFormatter,epilog=MODE_HELP)
    p.add_argument('--dataset',choices=['cifar10','cifar100','svhn'],default='svhn')
    p.add_argument('--backbone',choices=['resnet18','resnet50','mobilenet_v2','efficientnet_b0','vit_b_16','swin_t'],default='resnet18')
    p.add_argument('--normalization',choices=['legacy','imagenet'],default='legacy')
    p.add_argument('--data_root',default='./data')
    p.add_argument('--output_dir',default='./results_globalrepair_v3')
    p.add_argument('--partition_dir',default='./partitions_fedcgs_protocol')
    p.add_argument('--partition_file',default='')
    p.add_argument('--num_clients',type=int,default=30)
    p.add_argument('--alpha',type=float,default=0.1)
    p.add_argument('--seed',type=int,default=None,help='Run a single seed instead of the default three')
    p.add_argument('--communication_rounds',type=int,default=10)
    p.add_argument('--min_client_size',type=int,default=10)
    p.add_argument('--pretrained',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--reference_cached',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--ridge_lambda',type=float,default=.01)
    p.add_argument('--cov_shrinkage',type=float,default=.01)
    p.add_argument('--empirical_prior',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--cal_scale_min',type=float,default=.5)
    p.add_argument('--cal_scale_max',type=float,default=128.)
    p.add_argument('--cal_scale_steps',type=int,default=25)
    p.add_argument('--adapter_bottleneck',type=int,default=128)
    p.add_argument('--adapter_depth',type=int,default=2)
    p.add_argument('--adapter_dropout',type=float,default=0.)
    p.add_argument('--feature_batch_size',type=int,default=1024)
    p.add_argument('--calibration_batch_size',type=int,default=4096)
    p.add_argument('--test_batch_size',type=int,default=256)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--eps',type=float,default=1e-8)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke_test',action='store_true')
    p.add_argument('--mode',choices=MODES,default='full')
    p.add_argument('--fedavg_lr',type=float,default=0.01)
    p.add_argument('--fedavg_local_epochs',type=int,default=3)
    p.add_argument('--fedavg_batch_size',type=int,default=256)
    p.add_argument('--suite',action='store_true',help='Run paired ablations and summarize all seeds')
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--seeds',nargs='+',type=int,default=None)
    return p.parse_args()


def smoke_test(args):
    import tempfile
    torch.set_num_threads(1)
    class ToyDataset(Dataset):
        def __init__(self,n,seed):
            g=torch.Generator().manual_seed(seed)
            self.x=torch.randn(n,8,generator=g)
            y=(self.x[:,0]+.5*self.x[:,1]**2>0).long()+(self.x[:,2]>1).long()
            self.targets=y.tolist()
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):return self.x[i],self.targets[i]
    class ToyEncoder(nn.Module):
        def __init__(self,*unused):super().__init__();self.feature_dim=8
        def forward(self,x):return x
    # Mathematical checks on a synthetic fixture ONLY. Production code never
    # pools client features or uses test labels to compute an update.
    fixture=ToyDataset(90,31)
    w=torch.randn(9,3,dtype=torch.float64,generator=torch.Generator().manual_seed(41))*.1
    anch=w*2
    test_clients=[]
    for k,ids in enumerate(torch.arange(90).chunk(3)):
        c=RepairClient(k,Subset(fixture,ids.tolist()),torch.device('cpu'),2025)
        c.features=fixture.x[ids];c.labels=torch.tensor(fixture.targets)[ids]
        test_clients.append(c)
    packet=add_packets([c.packet(w,anch,17) for c in test_clients])
    x=torch.cat((fixture.x.double(),torch.ones(90,1)),1)
    labels=torch.tensor(fixture.targets)
    probe=w.clone().requires_grad_()
    F.cross_entropy(x@probe,labels).backward()
    assert torch.allclose(packet['gradient'].sum(0)/90,probe.grad,atol=1e-10)
    assert int(packet['count'].sum())==90
    assert torch.allclose(packet['fisher'],packet['fisher'].T,atol=1e-10)
    inv=positive_inverse(x.T@x/90,args.cov_shrinkage)
    dirs,protect,info=make_directions(packet,inv,args.cov_shrinkage,True)
    candidates=[(-1,0.),(0,.25)]
    aud=add_packets([c.audit(w,dirs,candidates,17) for c in test_clients])
    direct=F.cross_entropy(x@(w+.25*dirs[0]),labels)
    assert torch.allclose(aud['loss'][1].sum()/90,direct,atol=1e-10)
    rejected=dict(loss=torch.tensor([[1.,0.,2.,0.],[3.,0.,4.,0.]]),
                  correct=torch.tensor([[1,0,0,0],[0,0,0,0]]))
    assert select_candidate(rejected,torch.tensor([1,0,1,0]),[])[0]==0
    # Verify the new client's update against torch.optim.SGD on a toy batch.
    fa=copy.deepcopy(args);fa.seed=2025;fa.fedavg_lr=.01
    fa.fedavg_local_epochs=1;fa.fedavg_batch_size=1000
    client=test_clients[0]
    got=client_fedavg_head(client,w,fa,3)
    probe=w.float().clone().requires_grad_(True)
    optimizer=torch.optim.SGD([probe],lr=fa.fedavg_lr)
    z=client.features.float()
    xx=torch.cat((z,torch.ones(len(z),1)),1)
    optimizer.zero_grad()
    F.cross_entropy(xx@probe,client.labels).backward();optimizer.step()
    assert torch.allclose(got,probe.detach(),atol=2e-6,rtol=2e-6)
    combined,up,down=fedavg_head_round(test_clients,w,fa,3)
    expected=sum(client_fedavg_head(c,w,fa,3).double()*len(c.dataset)
                 for c in test_clients)/sum(len(c.dataset) for c in test_clients)
    assert torch.equal(combined,expected.float().double())
    assert up==down==len(test_clients)*w.numel()*4
    old=get_dataset,ReferenceEncoder
    globals()['get_dataset']=lambda *unused:(3,ToyDataset(300,11),ToyDataset(90,19))
    globals()['ReferenceEncoder']=ToyEncoder
    a=copy.deepcopy(args);a.cpu=True;a.num_clients=3;a.adapter_bottleneck=4;a.cal_scale_steps=3
    a.seed=2025;a.mode='full';a.communication_rounds=10
    try:
        with tempfile.TemporaryDirectory() as d:
            a.output_dir=d;a.partition_file=d+'/partition.json'
            rows=run(a)
            assert [r['round'] for r in rows]==list(range(1,11))
            for r in rows:
                if 'train_ce_after' in r:
                    assert r['train_ce_after']<=r['train_ce_before']+1e-9
            assert any(r.get('accepted',False) for r in rows)
            a.mode='init_fedavg';a.communication_rounds=5
            avg_rows=run(a)
            assert len(avg_rows)==5
            assert avg_rows[0]['accuracy']==rows[0]['accuracy']
            assert avg_rows[1]['accuracy']==rows[1]['accuracy']
            assert all(r['phase']=='same-init-fedavg-head' for r in avg_rows[2:])
            assert all('accepted' not in r for r in avg_rows)

    finally:
        globals()['get_dataset'],globals()['ReferenceEncoder']=old
    print('SMOKE TEST PASSED: aggregate gradient equals pooled gradient on synthetic fixture; '
          'audit equals direct loss; rejection fallback; ten exchanges; objective descent; '
          'FedAvg equals SGD; weighted aggregation; bytes; five-round FedAvg branch')


MODE_DESCRIPTIONS={
    'full':'完整方法：状态方向+保护投影+曲率+总体候选评估',
    'global_only':'去掉状态方向及其投影，保留全局梯度、曲率和总体候选评估',
    'no_projection':'相对精简完整方法，仅去掉状态方向的保护投影',
    'no_curvature':'相对精简完整方法，仅去掉方向构造的曲率预处理',
    'init_fedavg':'相同统计初始化+相同尺度校准；第3轮起分类头FedAvg（替代优化对照）',
}
MODES=tuple(MODE_DESCRIPTIONS)
MODE_HELP='mode 选项：\n'+'\n'.join(f'  {k:20s} {v}' for k,v in MODE_DESCRIPTIONS.items())




# Three plain-text outputs per experiment: report.txt, rounds.csv, mechanisms.csv.
# Raw metrics are retained without rounding in CSV; report formatting is display only.
_COMPACT = {}


def atomic_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def setting_tag(args):
    alpha=format(args.alpha,'.12g').replace('.','p').replace('-','m')
    return f'{args.dataset}_{args.backbone}_n{args.num_clients}_alpha{alpha}_r{args.communication_rounds}'


def save_table(args, rows):
    if not hasattr(args,'table_root'):
        return
    cache=_COMPACT[args.table_root]
    cache['runs'][(args.mode,args.seed)]=copy.deepcopy(rows)
    cache['partitions'][str(args.seed)]=str(Path(args.partition_file).resolve())
    write_compact(Path(args.table_root))


def write_compact(root):
    cache=_COMPACT[str(root)]
    raw=[]; mech=[]; groups={}
    for (mode,seed),rows in sorted(cache['runs'].items()):
        for r in rows:
            raw.append(dict(mode=mode,seed=seed,round=r['round'],accuracy_pct=r['accuracy'],
                upload_MiB=r['up_mib'],download_MiB=r['down_mib'],
                cumulative_MiB=r['cumulative_mib'],round_time_s=r['round_seconds'],
                elapsed_time_s=r['elapsed_seconds']))
            groups.setdefault((mode,r['round']),[]).append((seed,r))
            if 'accepted' in r:
                mech.append(dict(mode=mode,seed=seed,round=r['round'],accepted=r['accepted'],
                    direction=r.get('direction'),step=r.get('step'),
                    train_ce_before=r.get('train_ce_before'),train_ce_after=r.get('train_ce_after'),
                    train_acc_pct=r.get('train_acc'),
                    persistent_corrected=r.get('persistent_corrected'),
                    corrected_lost=r.get('corrected_lost'),new_forgotten=r.get('new_forgotten')))
    atomic_csv(root/'rounds.csv',raw)
    # A header-only file is valid if no model update has been evaluated yet.
    if mech:
        atomic_csv(root/'mechanisms.csv',mech)
    else:
        tmp=root/'mechanisms.csv.tmp'
        tmp.write_text('mode,seed,round,accepted,direction,step,train_ce_before,train_ce_after,train_acc_pct,persistent_corrected,corrected_lost,new_forgotten\n',encoding='utf-8-sig')
        tmp.replace(root/'mechanisms.csv')
    meta=cache['meta']; cfg=meta['arguments']
    lines=['FedTIDE 实验报告',
        f"数据集={cfg['dataset']} | 模型={cfg['backbone']} | 客户端={cfg['num_clients']} | alpha={cfg['alpha']} | 最大轮次={cfg['communication_rounds']}",
        '状态='+cache['status'], '计划种子='+','.join(map(str,meta['seeds'])),
        '模式='+','.join(meta['modes']),
        '说明：准确率单位%；通信单位MiB；时间单位秒；±为样本标准差；n为实际种子数。',
        '仅记录实际轮次，不填补未运行数据。单个种子标准差记为NA。',
        '', '一、逐轮汇总（所有实际轮次）',
        'mode\tround\tn\taccuracy_mean±std\tcumulative_MiB_mean\telapsed_s_mean']
    def summary(mode,rnd,entries):
        values=np.array([float(r['accuracy']) for _,r in entries])
        sd=f'{values.std(ddof=1):.3f}' if len(values)>1 else 'NA'
        return '\t'.join([mode,str(rnd),str(len(entries)),f'{values.mean():.3f}±{sd}',
            f"{np.mean([r['cumulative_mib'] for _,r in entries]):.3f}",
            f"{np.mean([r['elapsed_seconds'] for _,r in entries]):.3f}"])
    for (mode,rnd),entries in sorted(groups.items()):
        lines.append(summary(mode,rnd,entries))
    lines+=['','二、已完成运行的末轮汇总（不取最佳准确率）',
            'mode\tround\tn\taccuracy_mean±std\tcumulative_MiB_mean\telapsed_s_mean']
    for (mode,rnd),entries in sorted(groups.items()):
        expected=cfg['communication_rounds']
        if rnd==expected:
            lines.append(summary(mode,rnd,entries))
    lines+=['','三、各随机种子的逐轮原始结果',
        'mode\tseed\tround\taccuracy_pct\tupload_MiB\tdownload_MiB\tcumulative_MiB\tround_time_s\telapsed_time_s']
    for r in raw:
        lines.append('\t'.join(str(v) if isinstance(v,(str,int)) else f'{v:.6f}' for v in r.values()))
    lines+=['','四、候选更新诊断',
        'direction: -1=保留模型，0=全局方向，1=第二候选槽位；global_only中的第二槽位可能为重复全局方向。',
        'init_fedavg无候选审核，因此无该模式的候选诊断；SGD设置见配置。',
        '纠错数和遗忘数针对更新前状态分组，不能直接替代全体样本净正确数变化。']
    if mech:
        lines.append('\t'.join(mech[0]))
        for r in mech:
            lines.append('\t'.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in r.values()))
    else:
        lines.append('尚无候选评估记录。')
    lines+=['','五、配置与追溯信息（每次实验仅记录一次）',
            json.dumps(dict(**meta,partition_files=cache['partitions']),ensure_ascii=False,indent=2,default=str)]
    if cache.get('error'):
        lines+=['错误信息：'+cache['error']]
    tmp=root/'report.txt.tmp'
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tmp.replace(root/'report.txt')


def run_experiments(args):
    import hashlib
    import platform
    import sys
    from datetime import datetime, timezone
    if args.seed is not None and args.seeds is not None:
        raise ValueError('Use --seed OR --seeds')
    seeds=args.seeds if args.seeds is not None else ([args.seed] if args.seed is not None else [2025,2026,2027])
    modes=args.modes if args.suite else [args.mode]
    if not seeds or not modes or len(set(seeds))!=len(seeds) or len(set(modes))!=len(modes):
        raise ValueError('Seeds and modes must be nonempty and unique')
    if args.partition_file and len(seeds)>1:
        raise ValueError('For multiple seeds omit --partition_file')
    if not 1<=args.communication_rounds<=10:
        raise ValueError('Use 1 to 10 communication rounds')
    if not math.isfinite(args.fedavg_lr) or args.fedavg_lr<=0 or args.fedavg_local_epochs<1 or args.fedavg_batch_size<1:
        raise ValueError('FedAvg lr must be finite and positive; local epochs and batch size must be positive')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=Path(args.output_dir)/f"{setting_tag(args)}_{'-'.join(modes)}_{stamp}_{time.time_ns()}"
    root.mkdir(parents=True,exist_ok=False)
    _COMPACT[str(root)]=dict(runs={},partitions={},status='running',meta=dict(
        method='FedTIDE-GlobalRepair-v4',arguments=vars(args).copy(),seeds=seeds,
        modes={m:MODE_DESCRIPTIONS[m] for m in modes},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
            cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')))
    write_compact(root)
    print(f'Output: {root.resolve()}\nSeeds: {seeds}\nModes: {modes}',flush=True)
    try:
        for seed in seeds:
            for mode in modes:
                a=copy.deepcopy(args);a.seed=seed;a.mode=mode;a.table_root=str(root)
                _COMPACT[str(root)]['status']=f'running mode={mode} seed={seed}'
                write_compact(root)
                print(f'RUN mode={mode} seed={seed}: {MODE_DESCRIPTIONS[mode]}',flush=True)
                run(a)
    except BaseException as exc:
        _COMPACT[str(root)]['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        _COMPACT[str(root)]['error']=f'{type(exc).__name__}: {exc}'
        write_compact(root)
        raise
    _COMPACT[str(root)]['status']='completed'
    write_compact(root)
    del _COMPACT[str(root)]
    print(f'Saved 3 files: {root.resolve()}\nCopy report.txt for analysis.',flush=True)
    return root


if __name__=='__main__':
    args=parse_args()
    smoke_test(args) if args.smoke_test else run_experiments(args)
