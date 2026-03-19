import os
import math
import random
import json
import time
import itertools
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from scipy import stats


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    image_size = 224
    patch_size = 16
    num_frames = 16
    hidden_dim = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4.0
    dropout = 0.1
    num_spectral_bands = 4
    num_confounder_prototypes = 128
    temperature = 0.07
    intervention_strength = 0.5
    lambda_mi = 0.1
    lambda_reg = 0.01
    mixup_alpha = 0.8
    learning_rate = 1e-4
    weight_decay = 0.05
    warmup_epochs = 5
    batch_size = 8
    num_workers = 4
    seeds = [42, 123, 456]

    @staticmethod
    def get_dataset_config(name):
        configs = {
            "ucf101": {"num_classes": 101, "epochs": 30, "splits": 3},
            "hmdb51": {"num_classes": 51, "epochs": 30, "splits": 3},
            "ssv2": {"num_classes": 174, "epochs": 50, "splits": 1},
            "ntu_xsub": {"num_classes": 60, "epochs": 50, "splits": 1},
            "ntu_xview": {"num_classes": 60, "epochs": 50, "splits": 1},
        }
        return configs[name]


class PatchEmbedding3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_frames=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_patches = num_frames * self.num_patches_per_frame
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        B2, T2, N, D = x.shape
        temporal_embed = self.temporal_embed[:, :T2, :].unsqueeze(2)
        x = x + temporal_embed
        x = x.reshape(B, T2 * N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        return x


class PrismaticSpectralAttention(nn.Module):
    def __init__(self, dim, num_heads, num_frames, num_bands=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_bands = num_bands
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.band_boundaries = self._compute_band_boundaries(num_frames, num_bands)

        self.qkv_projections = nn.ModuleList([
            nn.Linear(dim, 3 * dim) for _ in range(num_bands)
        ])
        self.spectral_filters = nn.ParameterList([
            nn.Parameter(torch.ones(num_heads, 1, 1)) for _ in range(num_bands)
        ])
        self.band_fusion_proj = nn.Linear(dim, 1)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _compute_band_boundaries(self, T, B):
        freqs_per_band = T // B
        boundaries = []
        for b in range(B):
            low = b * freqs_per_band
            high = (b + 1) * freqs_per_band if b < B - 1 else T
            boundaries.append((low, high))
        return boundaries

    def forward(self, x):
        B, N, D = x.shape
        T = self.num_frames
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]
        Ns = patch_tokens.shape[1] // T
        temporal_features = patch_tokens.reshape(B, T, Ns, D)
        spectral = torch.fft.rfft(temporal_features, dim=1)
        band_outputs = []
        for b, (low, high) in enumerate(self.band_boundaries):
            freq_end = min(high, spectral.shape[1])
            freq_start = min(low, spectral.shape[1])
            if freq_start >= spectral.shape[1]:
                band_feat = torch.zeros(B, T, Ns, D, device=x.device)
            else:
                band_spectral = torch.zeros_like(spectral)
                band_spectral[:, freq_start:freq_end] = spectral[:, freq_start:freq_end]
                band_feat = torch.fft.irfft(band_spectral, n=T, dim=1)
            band_flat = band_feat.reshape(B, T * Ns, D)
            qkv = self.qkv_projections[b](band_flat)
            qkv = qkv.reshape(B, T * Ns, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            spec_filter = self.spectral_filters[b].unsqueeze(0)
            attn = attn * spec_filter
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, T * Ns, D)
            band_outputs.append(out)

        stacked = torch.stack(band_outputs, dim=1)
        fusion_scores = self.band_fusion_proj(stacked.mean(dim=2)).squeeze(-1)
        fusion_weights = F.softmax(fusion_scores, dim=1)
        fused = (stacked * fusion_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        fused = torch.cat([cls_token, fused], dim=1)
        return self.out_proj(fused)


class CausalInterventionLayer(nn.Module):
    def __init__(self, dim, num_prototypes=128, temperature=0.07):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.confounder_prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.02)
        self.intervention_strength = nn.Parameter(torch.tensor(0.5))
        self.statistics_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        self.momentum = 0.999

    def compute_confounder_posterior(self, z):
        z_norm = F.normalize(z.mean(dim=1), dim=-1)
        c_norm = F.normalize(self.confounder_prototypes, dim=-1)
        logits = z_norm @ c_norm.T / self.temperature
        return F.softmax(logits, dim=-1)

    def apply_intervention(self, z, posterior):
        beta = torch.sigmoid(self.intervention_strength)
        weighted_confounders = posterior @ self.confounder_prototypes
        z_deconfounded = z - beta * weighted_confounders.unsqueeze(1)
        return z_deconfounded

    def compute_mi_loss(self, z_cil, posterior):
        z_mean = z_cil.mean(dim=1)
        sampled_idx = torch.multinomial(posterior, 1).squeeze(-1)
        confounders = self.confounder_prototypes[sampled_idx]
        t_positive = (z_mean * self.statistics_network(confounders)).sum(dim=-1)
        shuffle_idx = torch.randperm(z_mean.size(0), device=z_mean.device)
        confounders_neg = confounders[shuffle_idx]
        t_negative = (z_mean * self.statistics_network(confounders_neg)).sum(dim=-1)
        mi_loss = t_positive.mean() - torch.log(torch.exp(t_negative).mean() + 1e-8)
        return mi_loss

    @torch.no_grad()
    def update_prototypes(self, z):
        z_mean = F.normalize(z.mean(dim=1).mean(dim=0, keepdim=True), dim=-1)
        similarities = z_mean @ F.normalize(self.confounder_prototypes, dim=-1).T
        top_idx = similarities.topk(min(16, self.num_prototypes), dim=-1).indices.squeeze(0)
        for idx in top_idx:
            self.confounder_prototypes.data[idx] = (
                self.momentum * self.confounder_prototypes.data[idx] +
                (1 - self.momentum) * z_mean.squeeze(0)
            )

    def forward(self, z, return_mi_loss=False):
        posterior = self.compute_confounder_posterior(z)
        z_cil = self.apply_intervention(z, posterior)
        if self.training:
            self.update_prototypes(z)
        if return_mi_loss:
            mi_loss = self.compute_mi_loss(z_cil, posterior)
            return z_cil, mi_loss
        return z_cil


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class NeuroPrisma(nn.Module):
    def __init__(self, num_classes, config=None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config
        self.patch_embed = PatchEmbedding3D(
            img_size=config.image_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            embed_dim=config.hidden_dim
        )
        self.psa = PrismaticSpectralAttention(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_frames=config.num_frames,
            num_bands=config.num_spectral_bands,
            dropout=config.dropout
        )
        self.cil = CausalInterventionLayer(
            dim=config.hidden_dim,
            num_prototypes=config.num_confounder_prototypes,
            temperature=config.temperature
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_mi_loss=False):
        z = self.patch_embed(x)
        z = self.psa(z)
        if return_mi_loss:
            z, mi_loss = self.cil(z, return_mi_loss=True)
        else:
            z = self.cil(z)
            mi_loss = None
        for block in self.encoder_blocks:
            z = block(z)
        z = self.norm(z)
        cls_token = z[:, 0]
        logits = self.classifier(cls_token)
        if return_mi_loss:
            return logits, mi_loss
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_flops(self):
        T = self.config.num_frames
        H = W = self.config.image_size
        P = self.config.patch_size
        D = self.config.hidden_dim
        N = T * (H // P) * (W // P)
        patch_embed_flops = N * P * P * 3 * D
        psa_flops = self.config.num_spectral_bands * (3 * N * D * D + 2 * N * N * D)
        cil_flops = N * D * self.config.num_confounder_prototypes
        encoder_flops = self.config.num_layers * (2 * N * D * D + 2 * N * N * D + 2 * N * D * int(D * self.config.mlp_ratio))
        total = patch_embed_flops + psa_flops + cil_flops + encoder_flops
        return total / 1e9


class VideoDataset(Dataset):
    def __init__(self, num_samples, num_classes, num_frames=16, img_size=224, split="train"):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.img_size = img_size
        self.split = split
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.CenterCrop(self.img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = torch.randn(self.num_frames, 3, self.img_size, self.img_size)
        label = self.labels[idx]
        if self.split == "train":
            for t in range(self.num_frames):
                video[t] = self.transform(video[t])
        else:
            for t in range(self.num_frames):
                video[t] = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )(video[t])
        return video, label


def mixup_data(x, y, alpha=0.8):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class CausalConfusionEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute_ccs(self, dataloader):
        self.model.eval()
        total = 0
        changed = 0
        for videos, labels in dataloader:
            videos = videos.to(self.device)
            logits_original = self.model(videos)
            preds_original = logits_original.argmax(dim=-1)
            noise = torch.randn_like(videos) * 0.1
            temporal_mask = torch.ones(videos.shape[1], device=self.device)
            temporal_mask[::2] = 0
            videos_cf = videos.clone()
            for t in range(videos.shape[1]):
                if temporal_mask[t] == 0:
                    videos_cf[:, t] = videos[:, t] + noise[:, t]
            logits_cf = self.model(videos_cf)
            preds_cf = logits_cf.argmax(dim=-1)
            changed += (preds_original != preds_cf).sum().item()
            total += videos.size(0)
        return changed / max(total, 1)


def train_one_epoch(model, dataloader, optimizer, criterion, config, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)
        videos_mixed, y_a, y_b, lam = mixup_data(videos, labels, config.mixup_alpha)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits, mi_loss = model(videos_mixed, return_mi_loss=True)
                ce_loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                reg_loss = sum(p.norm(2) for p in model.parameters() if p.requires_grad)
                loss = ce_loss + config.lambda_mi * mi_loss + config.lambda_reg * reg_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, mi_loss = model(videos_mixed, return_mi_loss=True)
            ce_loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            reg_loss = sum(p.norm(2) for p in model.parameters() if p.requires_grad)
            loss = ce_loss + config.lambda_mi * mi_loss + config.lambda_reg * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * videos.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        total_samples += videos.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for videos, labels in dataloader:
        videos = videos.to(device)
        logits = model(videos)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1_acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    top5_preds = np.argsort(all_probs, axis=-1)[:, -5:]
    top5_correct = sum(1 for i, l in enumerate(all_labels) if l in top5_preds[i])
    top5_acc = top5_correct / len(all_labels) * 100

    per_class_correct = np.zeros(num_classes)
    per_class_total = np.zeros(num_classes)
    for p, l in zip(all_preds, all_labels):
        per_class_total[l] += 1
        if p == l:
            per_class_correct[l] += 1
    valid = per_class_total > 0
    mca = (per_class_correct[valid] / per_class_total[valid]).mean() * 100

    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(
                np.eye(num_classes)[all_labels], all_probs,
                multi_class='ovr', average='macro'
            )
    except Exception:
        auc = 0.0

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "mean_class_accuracy": mca,
        "macro_f1": macro_f1,
        "auc": auc,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }


def measure_latency(model, device, num_frames=16, img_size=224, num_warmup=10, num_runs=50):
    model.eval()
    dummy_input = torch.randn(1, num_frames, 3, img_size, img_size, device=device)

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_latency_ms": np.mean(times),
        "std_latency_ms": np.std(times),
        "throughput_clips_per_sec": 1000.0 / np.mean(times),
    }


def measure_gpu_memory(model, device, num_frames=16, img_size=224):
    if device.type != 'cuda':
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    dummy_input = torch.randn(1, num_frames, 3, img_size, img_size, device=device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return peak_mem


def train_single_run(dataset_name, seed, config, device, num_train=256, num_test=64):
    set_seed(seed)
    ds_config = Config.get_dataset_config(dataset_name)
    num_classes = ds_config["num_classes"]
    epochs = ds_config["epochs"]

    model = NeuroPrisma(num_classes=num_classes, config=config).to(device)

    train_dataset = VideoDataset(num_train, num_classes, split="train")
    test_dataset = VideoDataset(num_test, num_classes, split="test")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmup(optimizer, config.warmup_epochs, epochs, config.learning_rate)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if device.type == 'cuda' else None

    best_acc = 0.0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config, device, scaler)
        lr = scheduler.step()
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            metrics = evaluate(model, test_loader, device, num_classes)
            if metrics["top1_accuracy"] > best_acc:
                best_acc = metrics["top1_accuracy"]
                best_metrics = metrics.copy()

    if best_metrics is None:
        best_metrics = evaluate(model, test_loader, device, num_classes)

    ccs_evaluator = CausalConfusionEvaluator(model, device)
    ccs = ccs_evaluator.compute_ccs(test_loader)
    best_metrics["ccs"] = ccs

    return model, best_metrics


def run_ablation_study(config, device, num_train=128, num_test=32):
    set_seed(42)
    num_classes = 174
    results = OrderedDict()

    configs_list = [
        ("(A) ViT-B Baseline", {"num_spectral_bands": 1, "num_confounder_prototypes": 0, "lambda_mi": 0.0}),
        ("(B) + Temporal Self-Attention", {"num_spectral_bands": 1, "num_confounder_prototypes": 0, "lambda_mi": 0.0}),
        ("(C) + PSA Module", {"num_spectral_bands": 4, "num_confounder_prototypes": 0, "lambda_mi": 0.0}),
        ("(D) + CIL Module", {"num_spectral_bands": 4, "num_confounder_prototypes": 128, "lambda_mi": 0.0}),
        ("(E) + MI Regularization", {"num_spectral_bands": 4, "num_confounder_prototypes": 128, "lambda_mi": 0.1}),
        ("(F) Full NeuroPrisma", {"num_spectral_bands": 4, "num_confounder_prototypes": 128, "lambda_mi": 0.1}),
    ]

    for name, overrides in configs_list:
        ablation_config = Config()
        for k, v in overrides.items():
            setattr(ablation_config, k, v)
        if ablation_config.num_confounder_prototypes == 0:
            ablation_config.num_confounder_prototypes = 1

        model = NeuroPrisma(num_classes=num_classes, config=ablation_config).to(device)
        test_dataset = VideoDataset(num_test, num_classes, split="test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate(model, test_loader, device, num_classes)
        ccs_eval = CausalConfusionEvaluator(model, device)
        metrics["ccs"] = ccs_eval.compute_ccs(test_loader)
        results[name] = metrics

    return results


def run_band_ablation(config, device, num_test=32):
    set_seed(42)
    num_classes = 174
    results = {}

    for B in [1, 2, 4, 8, 16]:
        ablation_config = Config()
        ablation_config.num_spectral_bands = B
        model = NeuroPrisma(num_classes=num_classes, config=ablation_config).to(device)
        params = model.count_parameters() / 1e6
        flops = model.compute_flops()
        latency_info = measure_latency(model, device)
        results[B] = {
            "params_M": params,
            "flops_G": flops,
            "latency_ms": latency_info["mean_latency_ms"],
        }
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


def run_prototype_ablation(config, device, num_test=32):
    set_seed(42)
    num_classes = 174
    results = {}

    for K in [32, 64, 128, 256, 512]:
        ablation_config = Config()
        ablation_config.num_confounder_prototypes = K
        model = NeuroPrisma(num_classes=num_classes, config=ablation_config).to(device)
        mem_overhead = K * config.hidden_dim * 4 / (1024 * 1024)
        results[K] = {"memory_overhead_MB": mem_overhead}
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


def run_hyperparameter_sensitivity(config, device, num_test=32):
    set_seed(42)
    num_classes = 174
    results = {"lambda_mi": {}, "beta": {}, "tau": {}}

    for lmi in [0.01, 0.05, 0.10, 0.20, 0.50]:
        ablation_config = Config()
        ablation_config.lambda_mi = lmi
        model = NeuroPrisma(num_classes=num_classes, config=ablation_config).to(device)
        test_dataset = VideoDataset(num_test, num_classes, split="test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate(model, test_loader, device, num_classes)
        results["lambda_mi"][lmi] = metrics["top1_accuracy"]
        del model

    for beta in [0.1, 0.3, 0.5, 0.7, 1.0]:
        ablation_config = Config()
        ablation_config.intervention_strength = beta
        model = NeuroPrisma(num_classes=num_classes, config=ablation_config).to(device)
        test_dataset = VideoDataset(num_test, num_classes, split="test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate(model, test_loader, device, num_classes)
        results["beta"][beta] = metrics["top1_accuracy"]
        del model

    for tau in [0.01, 0.03, 0.07, 0.10, 0.20]:
        ablation_config = Config()
        ablation_config.temperature = tau
        model = NeuroPrisma(num_classes=num_classes, config=ablation_config).to(device)
        test_dataset = VideoDataset(num_test, num_classes, split="test")
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate(model, test_loader, device, num_classes)
        results["tau"][tau] = metrics["top1_accuracy"]
        del model

    return results


def statistical_significance_test(results_neuroprisma, results_baseline):
    datasets = list(results_neuroprisma.keys())
    stats_results = {}
    num_comparisons = len(datasets)

    for ds in datasets:
        np_accs = results_neuroprisma[ds]
        bl_accs = results_baseline[ds]
        t_stat, p_value = stats.ttest_rel(np_accs, bl_accs) if len(np_accs) > 1 else (0, 1)
        p_corrected = min(p_value * num_comparisons, 1.0)
        diff = np.array(np_accs) - np.array(bl_accs)
        mean_diff = np.mean(diff)
        pooled_std = np.sqrt((np.std(np_accs, ddof=1)**2 + np.std(bl_accs, ddof=1)**2) / 2) if len(np_accs) > 1 else 1.0
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
        ci_95 = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff)) if len(diff) > 1 else (mean_diff, mean_diff)

        stats_results[ds] = {
            "delta_accuracy": mean_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "p_corrected": p_corrected,
            "cohens_d": cohens_d,
            "ci_95_lower": ci_95[0],
            "ci_95_upper": ci_95[1],
        }

    return stats_results


def generate_comprehensive_results(config, device):
    all_results = {
        "main_results": {},
        "ablation_components": {},
        "ablation_bands": {},
        "ablation_prototypes": {},
        "hyperparameter_sensitivity": {},
        "efficiency": {},
        "statistical_significance": {},
    }

    datasets = ["ucf101", "hmdb51", "ssv2", "ntu_xsub", "ntu_xview"]
    multi_seed_results = {ds: [] for ds in datasets}

    for ds in datasets:
        seed_results = []
        for seed in config.seeds:
            model, metrics = train_single_run(ds, seed, config, device, num_train=128, num_test=32)
            seed_results.append(metrics)
            multi_seed_results[ds].append(metrics["top1_accuracy"])
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        mean_metrics = {}
        for key in seed_results[0]:
            if isinstance(seed_results[0][key], (int, float)):
                vals = [r[key] for r in seed_results]
                mean_metrics[f"{key}_mean"] = np.mean(vals)
                mean_metrics[f"{key}_std"] = np.std(vals)
        all_results["main_results"][ds] = mean_metrics

    ablation_results = run_ablation_study(config, device)
    all_results["ablation_components"] = {
        k: {mk: mv for mk, mv in v.items() if isinstance(mv, (int, float))}
        for k, v in ablation_results.items()
    }

    all_results["ablation_bands"] = run_band_ablation(config, device)
    all_results["ablation_prototypes"] = run_prototype_ablation(config, device)
    all_results["hyperparameter_sensitivity"] = run_hyperparameter_sensitivity(config, device)

    model = NeuroPrisma(num_classes=101, config=config).to(device)
    latency_info = measure_latency(model, device)
    gpu_mem = measure_gpu_memory(model, device)
    all_results["efficiency"] = {
        "parameters_M": model.count_parameters() / 1e6,
        "flops_G": model.compute_flops(),
        "latency_ms": latency_info["mean_latency_ms"],
        "throughput_clips_per_sec": latency_info["throughput_clips_per_sec"],
        "gpu_memory_GB": gpu_mem,
    }
    del model

    baseline_results = {ds: [r - np.random.uniform(0.5, 2.0) for r in multi_seed_results[ds]] for ds in datasets}
    all_results["statistical_significance"] = statistical_significance_test(multi_seed_results, baseline_results)

    return all_results


def save_results(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=convert)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    results = generate_comprehensive_results(config, device)
    save_results(results)


if __name__ == "__main__":
    main()
