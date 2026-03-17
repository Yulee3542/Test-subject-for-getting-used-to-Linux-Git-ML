#!/usr/bin/env python
# coding: utf-8
# Face-Mask Classification: S²TDPT vs Spikformer vs Vanilla Transformer vs ResNet34
# 환경: CPU-only (i5-6500T, 24GB RAM)

import matplotlib
matplotlib.use('Agg')

import os, time, json, random, sys, gc
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from tqdm import tqdm

# ── 재현성 ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
print(f'Torch : {torch.__version__}', flush=True)

# ════════════════
#  하이퍼파라미터
# ════════════════
CFG = dict(
    data_dir       = './data',
    img_size       = 64,
    crop_size      = 56,
    batch_size     = 16,             # ANN
    snn_batch_size = 8,              # SNN (T=4배 메모리)
    train_ratio    = 0.8,
    num_workers    = 0,              # CPU 학습 시 0
    num_classes    = 2,

    # 공통 학습
    epochs         = 10,
    lr             = 1e-3,
    weight_decay   = 1e-3,

    # Transformer 공통
    embed_dim      = 256,
    num_heads      = 4,
    depth          = 4,
    patch_size     = 8,
    mlp_ratio      = 2,

    # SNN 전용
    T              = 4,
    tau            = 2.0,
    v_threshold    = 1.0,
    v_reset        = 0.0,

    # STDP 전용
    A_stdp         = 0.9,
    tau_stdp       = 0.5,
    w_offset       = 0.9,

    # Early stopping
    early_stop_patience = 4,        # N 에포크 동안 val_acc 개선 없으면 중단

    checkpoint_dir = './checkpoints',
    results_dir    = './results',
)

os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
os.makedirs(CFG['results_dir'],    exist_ok=True)
print('CFG loaded.', flush=True)

# ── 런 로그 파일 ──
RUN_LOG = os.path.join(CFG['results_dir'], 'run_log.txt')
def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(RUN_LOG, 'a') as f:
        f.write(line + '\n')

# ════════════════
#  Dataset
# ════════════════
MEAN = [0.48235, 0.45882, 0.40784]
STD  = [1.0/255.0, 1.0/255.0, 1.0/255.0]

transform = transforms.Compose([
    transforms.Resize((CFG['img_size'], CFG['img_size'])),
    transforms.CenterCrop(CFG['crop_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

dataset   = ImageFolder(CFG['data_dir'], transform=transform)
n_train   = int(CFG['train_ratio'] * len(dataset))
n_test    = len(dataset) - n_train
train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                  generator=torch.Generator().manual_seed(SEED))

print(f'Classes : {dataset.classes}', flush=True)
print(f'Total   : {len(dataset)}  |  Train: {n_train}  |  Test: {n_test}', flush=True)


def denormalize(t):
    img = t.clone().permute(1,2,0).numpy()
    for c in range(img.shape[2]):
        img[:,:,c] = img[:,:,c] * STD[c] + MEAN[c]
    return img.clip(0,1)


# ════════════════
#  LIF / TTFS
# ════════════════
class LIFNode(nn.Module):
    def __init__(self, tau=2.0, v_th=1.0, v_reset=0.0):
        super().__init__()
        self.tau     = tau
        self.v_th    = v_th
        self.v_reset = v_reset
        self.mem     = None

    def reset(self):
        self.mem = None

    @staticmethod
    def surrogate(x):
        return torch.sigmoid(4.0 * x)

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        beta = 1.0 - 1.0 / self.tau
        self.mem = beta * self.mem + x
        spike = self.surrogate(self.mem - self.v_th)
        self.mem = self.v_reset * spike + self.mem * (1.0 - spike)
        return spike


class TTFSEncoder(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, x):
        t_spike = ((1.0 - x) * (self.T - 1)).round().long()
        spikes  = torch.zeros(self.T, *x.shape, device=x.device)
        for t in range(self.T):
            spikes[t] = (t_spike == t).float()
        return spikes


# ════════════════
#  학습 / 평가 루프
# ════════════════
def reset_lif(model):
    for m in model.modules():
        if isinstance(m, LIFNode):
            m.reset()


def train_one_epoch(model, loader, optimizer, criterion, is_snn):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in tqdm(loader, desc='  train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if is_snn:
            reset_lif(model)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, is_snn):
    """loss, acc, precision, recall, f1, all_preds, all_labels 반환"""
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc='  eval ', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if is_snn:
            reset_lif(model)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        preds       = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    return total_loss / len(loader), acc, prec*100, rec*100, f1*100, all_preds, all_labels


# ════════════════
#  시각화
# ════════════════
PALETTE = {
    'ResNet34':            '#E74C3C',
    'Vanilla Transformer': '#3498DB',
    'Spikformer':          '#2ECC71',
    'S2TDPT':              '#1ABC9C',
}


def _live_plot(history, name, color, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.5))
    fig.suptitle(f'Live Training  —  {name}', fontsize=12, fontweight='bold')
    ep = range(1, len(history['train_loss'])+1)

    axes[0].plot(ep, history['train_loss'], '--', color=color, alpha=0.6, label='train')
    axes[0].plot(ep, history['val_loss'],         color=color, label='val')
    axes[0].set(title='Loss', xlabel='Epoch')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history['train_acc'], '--', color=color, alpha=0.6, label='train acc')
    axes[1].plot(ep, history['val_acc'],         color=color, label='val acc')
    axes[1].set(title='Accuracy (%)', xlabel='Epoch')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, history['val_precision'], 'o-', color=color, alpha=0.7, label='precision')
    axes[2].plot(ep, history['val_recall'],    's-', color=color, alpha=0.5, label='recall')
    axes[2].plot(ep, history['val_f1'],        '^-', color=color, alpha=0.9, label='F1', lw=2)
    axes[2].set(title='Val Precision / Recall / F1 (%)', xlabel='Epoch')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('²', '2')
    plt.savefig(os.path.join(results_dir, f'live_{safe_name}.png'), bbox_inches='tight')
    plt.close(fig)


def _save_confusion_matrix(preds, labels, name, results_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['with_mask','without_mask'],
                yticklabels=['with_mask','without_mask'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('²', '2')
    plt.savefig(os.path.join(results_dir, f'cm_{safe_name}.png'), bbox_inches='tight')
    plt.close(fig)


# ════════════════
#  run_training
# ════════════════
MODEL_BAR  = None
MODEL_NAMES = ['ResNet34', 'Vanilla Transformer', 'Spikformer', 'S2TDPT']
ALL_RESULTS = {}


def run_training(model, name, is_snn=False, epochs=None):
    global MODEL_BAR
    if MODEL_BAR is None:
        MODEL_BAR = tqdm(total=len(MODEL_NAMES),
                         desc='전체 학습 진행',
                         bar_format='{desc}: {bar:40} {n}/{total}  [{elapsed}<{remaining}]',
                         leave=True, file=sys.stdout)

    epochs     = epochs or CFG['epochs']
    results_dir = CFG['results_dir']
    safe_name  = name.replace(' ', '_').replace('²', '2')
    ckpt_best  = os.path.join(CFG['checkpoint_dir'], f'{safe_name}_best.pth')
    ckpt_epoch_dir = os.path.join(CFG['checkpoint_dir'], safe_name)
    os.makedirs(ckpt_epoch_dir, exist_ok=True)

    batch_size = CFG['snn_batch_size'] if is_snn else CFG['batch_size']
    train_loader_local = DataLoader(train_ds, batch_size=batch_size,
                                    shuffle=True,  num_workers=CFG['num_workers'])
    test_loader_local  = DataLoader(test_ds,  batch_size=batch_size,
                                    shuffle=False, num_workers=CFG['num_workers'])

    model = model.to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched = CosineAnnealingLR(opt, T_max=epochs)

    # ── 이어 학습: 마지막 epoch 체크포인트 탐색 ──
    history    = {'train_loss':[], 'train_acc':[],
                  'val_loss':[], 'val_acc':[],
                  'val_precision':[], 'val_recall':[], 'val_f1':[]}
    best_acc   = 0.
    best_state = None
    start_ep   = 1

    existing_epochs = sorted([
        int(f.replace('epoch_','').replace('.pth',''))
        for f in os.listdir(ckpt_epoch_dir)
        if f.startswith('epoch_') and f.endswith('.pth')
    ])
    if existing_epochs:
        last_ep  = existing_epochs[-1]
        resume   = torch.load(os.path.join(ckpt_epoch_dir, f'epoch_{last_ep}.pth'),
                              map_location='cpu')
        model.load_state_dict(resume['model'])
        opt.load_state_dict(resume['optimizer'])
        sched.load_state_dict(resume['scheduler'])
        history  = resume['history']
        best_acc = resume['best_acc']
        start_ep = last_ep + 1
        log(f'[RESUME] {name} — epoch {last_ep}부터 이어 학습')

    color   = PALETTE.get(name, '#95A5A6')
    n_params = sum(p.numel() for p in model.parameters())
    log(f'{"="*55}')
    log(f'  Training: {name}  |  params: {n_params:,}  |  start_ep: {start_ep}')
    log(f'{"="*55}')

    t_start      = time.time()
    no_improve   = 0
    patience     = CFG['early_stop_patience']
    last_preds   = []
    last_labels  = []

    for ep in range(start_ep, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader_local, opt, crit, is_snn)
        va_loss, va_acc, va_prec, va_rec, va_f1, ep_preds, ep_labels = \
            evaluate(model, test_loader_local, crit, is_snn)
        sched.step()
        gc.collect()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        history['val_precision'].append(va_prec)
        history['val_recall'].append(va_rec)
        history['val_f1'].append(va_f1)

        improved = va_acc > best_acc
        if improved:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            last_preds  = ep_preds
            last_labels = ep_labels
            no_improve  = 0
        else:
            no_improve += 1

        # 에포크 체크포인트 저장
        torch.save({
            'model':     model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'history':   history,
            'best_acc':  best_acc,
            'epoch':     ep,
        }, os.path.join(ckpt_epoch_dir, f'epoch_{ep}.pth'))

        elapsed = time.time() - t0
        eta_h   = elapsed * (epochs - ep) / 3600
        log(f'  Ep {ep:2d}/{epochs} | tr={tr_loss:.3f} va_acc={va_acc:.1f}% '
            f'f1={va_f1:.1f}% best={best_acc:.1f}% ETA={eta_h:.1f}h'
            + (' ★' if improved else ''))
        _live_plot(history, name, color, results_dir)

        # Early stopping
        if no_improve >= patience:
            log(f'  [Early Stop] {name} — {patience} 에포크 개선 없음. 중단.')
            break

    total_min = (time.time() - t_start) / 60
    log(f'  Best val acc: {best_acc:.2f}%  |  Total: {total_min:.1f} min')

    # best 체크포인트 저장
    torch.save({'model': best_state, 'history': history,
                'best_acc': best_acc, 'name': name,
                'train_mins': total_min}, ckpt_best)

    # Confusion matrix
    if last_preds:
        _save_confusion_matrix(last_preds, last_labels, name, results_dir)

    return history, best_acc, total_min


# ════════════════
#  should_skip
# ════════════════
def should_skip(model_name):
    safe_name = model_name.replace(' ', '_').replace('²', '2')
    file_path = os.path.join(CFG['checkpoint_dir'], f'{safe_name}_best.pth')
    if not os.path.exists(file_path):
        return False
    try:
        torch.load(file_path, map_location='cpu', weights_only=True)
        log(f'--- [SKIP] {model_name} 이미 완료됨 (정상 파일 확인) ---')
        return True
    except Exception as e:
        log(f'--- [WARN] {model_name} 체크포인트 손상됨: {e}. 다시 학습합니다. ---')
        return False


# ════════════════
#  최종 시각화
# ════════════════
def save_final_plots(all_results, results_dir):
    names      = list(all_results.keys())
    best_accs  = [all_results[n][1] for n in names]
    best_f1s   = [max(all_results[n][0]['val_f1']) if all_results[n][0]['val_f1'] else 0 for n in names]
    best_precs = [max(all_results[n][0]['val_precision']) if all_results[n][0]['val_precision'] else 0 for n in names]
    best_recs  = [max(all_results[n][0]['val_recall']) if all_results[n][0]['val_recall'] else 0 for n in names]
    train_mins = [all_results[n][2] for n in names]
    colors     = [PALETTE.get(n, '#95A5A6') for n in names]

    # 1. 학습 커브 (loss + acc + f1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Curves — Face-Mask Dataset', fontsize=13, fontweight='bold')
    axes[0].set_title('Loss'); axes[1].set_title('Accuracy (%)'); axes[2].set_title('Val F1 (%)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
    for name, (h, best_acc, _) in all_results.items():
        c  = PALETTE.get(name, '#95A5A6')
        ep = range(1, len(h['train_loss'])+1)
        axes[0].plot(ep, h['train_loss'], '--', color=c, lw=1.5, alpha=0.5)
        axes[0].plot(ep, h['val_loss'],         color=c, lw=2, label=name)
        axes[1].plot(ep, h['train_acc'], '--', color=c, lw=1.5, alpha=0.5)
        axes[1].plot(ep, h['val_acc'],         color=c, lw=2, label=f'{name} ({best_acc:.1f}%)')
        if h['val_f1']:
            axes[2].plot(ep, h['val_f1'], color=c, lw=2, label=name)
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'result_curves.png'), bbox_inches='tight')
    plt.close()
    log('result_curves.png 저장 완료')

    # 2. 성능 비교 바 차트 (Acc / F1 / Precision / Recall / Time)
    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 5))
    fig2.suptitle('Final Comparison — Face-Mask Dataset', fontsize=12, fontweight='bold')
    metrics = [
        (best_accs,  'Best Val Accuracy (%)', 'Accuracy'),
        (best_f1s,   'Best Val F1 (%)',        'F1 Score'),
        (best_precs, 'Best Val Precision (%)', 'Precision'),
        (best_recs,  'Best Val Recall (%)',    'Recall'),
        (train_mins, 'Training Time (min)',    'Time'),
    ]
    for ax, (vals, ylabel, title) in zip(axes2, metrics):
        ax.set(ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3)
        for idx, (v, c) in enumerate(zip(vals, colors)):
            ax.bar(idx, v, color=c, width=0.5, edgecolor='white', linewidth=1.5)
            ax.text(idx, v + max(vals)*0.01, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=8)
        if title != 'Time':
            ax.set_ylim(max(0, min(vals)-5), 101)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'result_comparison.png'), bbox_inches='tight')
    plt.close()
    log('result_comparison.png 저장 완료')

    # 3. summary.json
    summary = {}
    for name in names:
        h, acc, mins = all_results[name]
        summary[name] = {
            'best_acc':      round(acc, 4),
            'best_f1':       round(max(h['val_f1'])   if h['val_f1']   else 0, 4),
            'best_precision':round(max(h['val_precision']) if h['val_precision'] else 0, 4),
            'best_recall':   round(max(h['val_recall']) if h['val_recall'] else 0, 4),
            'train_mins':    round(mins, 2),
            'epochs_run':    len(h['train_loss']),
        }
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log('summary.json 저장 완료')


# ════════════════
#  모델 클래스
# ════════════════
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        self.in_ch = 64
        self.stem  = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.stage1 = self._make(block, 64,  layers[0], 1)
        self.stage2 = self._make(block, 128, layers[1], 2)
        self.stage3 = self._make(block, 256, layers[2], 2)
        self.stage4 = self._make(block, 512, layers[3], 2)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Linear(512, num_classes)

    def _make(self, block, out_ch, n, stride):
        layers = [block(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(n-1):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.fc(torch.flatten(self.pool(x), 1))


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size),
            nn.LayerNorm([embed_dim, img_size // patch_size, img_size // patch_size])
        )

    def forward(self, x):
        x = self.proj(x)
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1,2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0),-1,-1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:,0]))


class SpikingPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim//2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            LIFNode(CFG['tau'], CFG['v_threshold'], CFG['v_reset']),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            LIFNode(CFG['tau'], CFG['v_threshold'], CFG['v_reset']),
        )

    def forward(self, x):
        x = self.proj(x)
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1,2)


class SSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q_conv = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), LIFNode(CFG['tau']))
        self.k_conv = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), LIFNode(CFG['tau']))
        self.v_conv = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), LIFNode(CFG['tau']))
        self.proj   = nn.Linear(dim, dim)

    def _reshape(self, x):
        B, N, D = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)

    def _apply(self, seq, x):
        B, N, D = x.shape
        out = seq[0](x)
        out = seq[1](out.reshape(B*N,D)).reshape(B,N,D)
        return seq[2](out)

    def forward(self, x):
        B, N, D = x.shape
        q = self._apply(self.q_conv, x)
        k = self._apply(self.k_conv, x)
        v = self._apply(self.v_conv, x)
        qk  = self._reshape(q * k) * self.scale
        v_h = self._reshape(v)
        out = (qk @ v_h.transpose(-2,-1)) @ v_h
        return self.proj(out.transpose(1,2).reshape(B, N, D))


class SpikingMLP(nn.Module):
    def __init__(self, dim, ratio=2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*ratio, bias=False)
        self.bn1 = nn.BatchNorm1d(dim*ratio)
        self.lif1= LIFNode(CFG['tau'])
        self.fc2 = nn.Linear(dim*ratio, dim, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lif2= LIFNode(CFG['tau'])

    def forward(self, x):
        B, N, D = x.shape
        out = self.lif1(self.bn1(self.fc1(x).reshape(B*N,-1)).reshape(B,N,-1))
        out = self.lif2(self.bn2(self.fc2(out).reshape(B*N,-1)).reshape(B,N,-1))
        return out


class SpikformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = SSA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = SpikingMLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Spikformer(nn.Module):
    def __init__(self, img_size, embed_dim, depth, num_heads, mlp_ratio, num_classes, T):
        super().__init__()
        self.T      = T
        self.encoder= TTFSEncoder(T)
        self.sps    = SpikingPatchEmbed(3, embed_dim)
        self.blocks = nn.ModuleList([SpikformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.head   = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        spikes = self.encoder(x)
        out    = 0.
        for t in range(self.T):
            xt = self.sps(spikes[t])
            for blk in self.blocks:
                xt = blk(xt)
            out = out + xt
        return self.head((out / self.T).mean(dim=1))


class SPS(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim//2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            LIFNode(CFG['tau'], CFG['v_threshold'], CFG['v_reset']),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            LIFNode(CFG['tau'], CFG['v_threshold'], CFG['v_reset']),
        )

    def forward(self, x):
        return self.proj(x)


class S2TDPSA(nn.Module):
    def __init__(self, dim, num_heads, A_stdp, tau_stdp, w_offset, T_max=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.A_stdp    = A_stdp
        self.tau_stdp  = tau_stdp
        self.w_offset  = w_offset
        self.T_max     = T_max
        self.q_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), LIFNode(CFG['tau']))
        self.k_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), LIFNode(CFG['tau']))
        self.v_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), LIFNode(CFG['tau']))
        self.out_proj = nn.Linear(dim, dim)

    def _apply_proj(self, proj_seq, x):
        B, N, D = x.shape
        out = proj_seq[0](x)
        out = proj_seq[1](out.reshape(B*N, D)).reshape(B, N, D)
        return proj_seq[2](out)

    def _latency(self, spikes):
        r = spikes.mean(dim=-1, keepdim=True)
        return self.T_max * (1.0 - r)

    def forward(self, x):
        B, N, D = x.shape
        Q = self._apply_proj(self.q_proj, x)
        K = self._apply_proj(self.k_proj, x)
        V = self._apply_proj(self.v_proj, x)
        t_Q   = self._latency(Q)
        t_K   = self._latency(K)
        delta_t = t_Q - t_K.transpose(1,2)
        f_stdp  = self.A_stdp * torch.exp(-delta_t.abs() / self.tau_stdp)
        delta_w = torch.where(delta_t < 0, f_stdp, -f_stdp)
        A       = delta_w + self.w_offset
        return self.out_proj(A @ V)


class S2TDPTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, A_stdp, tau_stdp, w_offset):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = S2TDPSA(dim, num_heads, A_stdp, tau_stdp, w_offset)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = SpikingMLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class S2TDPT(nn.Module):
    def __init__(self, img_size, embed_dim, depth, num_heads, mlp_ratio, num_classes,
                 T, A_stdp, tau_stdp, w_offset):
        super().__init__()
        self.T      = T
        self.encoder= TTFSEncoder(T)
        self.sps    = SPS(3, embed_dim, patch_size=2)
        self.blocks = nn.ModuleList([
            S2TDPTBlock(embed_dim, num_heads, mlp_ratio, A_stdp, tau_stdp, w_offset)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        spikes   = self.encoder(x)
        feat_sum = 0.
        for t in range(self.T):
            xt = self.sps(spikes[t])
            B, D, h, w = xt.shape
            xt = xt.flatten(2).transpose(1,2)
            for blk in self.blocks:
                xt = blk(xt)
            feat_sum = feat_sum + xt
        feat = (feat_sum / self.T).mean(dim=1)
        return self.head(feat)


# ════════════════
#  main
# ════════════════
if __name__ == '__main__':
    log('===== 학습 시작 =====')

    models_to_run = [
        ('ResNet34', lambda: ResNet(BasicBlock, [3,4,6,3], CFG['num_classes']), False),
        ('Vanilla Transformer', lambda: VanillaTransformer(
            img_size=CFG['crop_size'], patch_size=CFG['patch_size'],
            embed_dim=CFG['embed_dim'], depth=CFG['depth'],
            num_heads=CFG['num_heads'], mlp_ratio=CFG['mlp_ratio'],
            num_classes=CFG['num_classes']), False),
        ('Spikformer', lambda: Spikformer(
            img_size=CFG['crop_size'], embed_dim=CFG['embed_dim'],
            depth=CFG['depth'], num_heads=CFG['num_heads'],
            mlp_ratio=CFG['mlp_ratio'], num_classes=CFG['num_classes'],
            T=CFG['T']), True),
        ('S2TDPT', lambda: S2TDPT(
            img_size=CFG['crop_size'], embed_dim=CFG['embed_dim'],
            depth=CFG['depth'], num_heads=CFG['num_heads'],
            mlp_ratio=CFG['mlp_ratio'], num_classes=CFG['num_classes'],
            T=CFG['T'], A_stdp=CFG['A_stdp'],
            tau_stdp=CFG['tau_stdp'], w_offset=CFG['w_offset']), True),
    ]

    for model_name, model_fn, is_snn in models_to_run:
        if should_skip(model_name):
            safe = model_name.replace(' ', '_').replace('²', '2')
            ckpt = torch.load(os.path.join(CFG['checkpoint_dir'], f'{safe}_best.pth'),
                              map_location='cpu')
            ALL_RESULTS[model_name] = (ckpt['history'], ckpt['best_acc'], ckpt.get('train_mins', 0.))
            if MODEL_BAR is None:
                MODEL_BAR = tqdm(total=len(MODEL_NAMES),
                                 desc='전체 학습 진행',
                                 bar_format='{desc}: {bar:40} {n}/{total}  [{elapsed}<{remaining}]',
                                 leave=True, file=sys.stdout)
            MODEL_BAR.update(1)
            continue

        log(f'--- [START] {model_name} ---')
        model = model_fn()
        h, acc, mins = run_training(model, model_name, is_snn=is_snn)
        ALL_RESULTS[model_name] = (h, acc, mins)
        if MODEL_BAR:
            MODEL_BAR.update(1)
            MODEL_BAR.set_postfix(last=f'{model_name} {acc:.1f}%')
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if MODEL_BAR:
        MODEL_BAR.close()

    log('🎉 전체 학습 완료!')
    save_final_plots(ALL_RESULTS, CFG['results_dir'])

    log('\n=== 최종 결과 요약 ===')
    for n in ALL_RESULTS:
        h, acc, mins = ALL_RESULTS[n]
        f1 = max(h['val_f1']) if h['val_f1'] else 0
        log(f'  {n:25s}  acc={acc:.2f}%  f1={f1:.2f}%  time={mins:.0f}min')
