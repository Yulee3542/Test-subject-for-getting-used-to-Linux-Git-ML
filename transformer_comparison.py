#!/usr/bin/env python
# coding: utf-8

# # Face-Mask Classification: S²TDPT vs Spikformer vs Vanilla Transformer vs ResNet34
# 
# **환경**: CPU-only (i5-6500T, 24GB RAM)  
# **데이터**: face-mask dataset (with_mask / without_mask)  
# **이미지 크기**: 64×64, **배치**: 16
# 
# | 모델 | 구현 근거 |
# |------|----------|
# | ResNet34 | 친구 코드(udyann.ipynb) 그대로 |
# | Vanilla Transformer | ANN Transformer (MHSA + softmax) |
# | Spikformer | SSA (Spiking Self-Attention, Zhou 2022) |
# | **S²TDPT** | **논문 완전 재현** — SPS + S²TDPSA(STDP) + Spiking MLP |
# 
# > 경량화 설정 (CPU 밤샘 기준): L=2, D=128, T=4 timesteps, 10 epochs

# ## 0. 환경 설정 & 공통 유틸

# In[19]:


# !pip install torch torchvision tqdm matplotlib

import os, time, json, random, sys, gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use('Agg')  # .py 실행 시 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm, trange

# ── 재현성 ──
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'Torch : {torch.__version__}')

# ════════════════
#  하이퍼파라미터 
# ═══════════════
CFG = dict(
    data_dir      = './data',       # ImageFolder 루트
    img_size      = 64,             # 요청: 64
    crop_size     = 56,
    batch_size    = 16,             # ANN (ResNet, Vanilla Transformer)
    snn_batch_size= 4,              # SNN (Spikformer, S²TDPT) — T=4배 메모리
    train_ratio   = 0.8,
    num_workers   = 4,              # i5-6500T 4코어 풀 활용
    num_classes   = 2,

    # 공통 학습
    epochs        = 10,
    lr            = 1e-3,
    weight_decay  = 1e-4,

    # Transformer 공통
    embed_dim     = 256,            # 경량화 (논문=384)
    num_heads     = 4,
    depth         = 4,              # 레이어 수 (논문=4)
    patch_size    = 8,              # 64/8 = 8×8 = 64 패치
    mlp_ratio     = 2,

    # SNN 전용
    T             = 4,              # timesteps (논문=4)
    tau           = 2.0,            # LIF 시정수
    v_threshold   = 1.0,
    v_reset       = 0.0,

    # STDP 전용
    A_stdp        = 0.9,
    tau_stdp      = 0.5,
    w_offset      = 0.9,

    checkpoint_dir = './checkpoints',
)

os.makedirs(CFG['checkpoint_dir'], exist_ok=True)
print('CFG loaded.')


# ## 1. Dataset (udyann.ipynb 참고)

# In[21]:


# VGG16 mean/std — udyann.ipynb 그대로
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

# ── udyann.ipynb 버그 수정: split된 subset을 DataLoader에 전달 ──
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          shuffle=True,  num_workers=CFG['num_workers'], pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'],
                          shuffle=False, num_workers=CFG['num_workers'], drop_last=False)
mini_loader  = DataLoader(train_ds, batch_size=4, shuffle=True)

print(f'Classes : {dataset.classes}')
print(f'Total   : {len(dataset)}  |  Train: {n_train}  |  Test: {n_test}')
print(f'Batches : train={len(train_loader)}, test={len(test_loader)}')


# In[22]:


def denormalize(t):
    """udyann.ipynb Cell 36 — Normalize 역변환"""
    img = t.clone().permute(1,2,0).numpy()
    for c in range(img.shape[2]):
        img[:,:,c] = img[:,:,c] * STD[c] + MEAN[c]
    return img.clip(0,1)


# ## 2. 공통 모듈 — LIF 뉴런 & 학습/평가 루프

# In[23]:


# ── Leaky Integrate-and-Fire (논문 Eq. 1-3) ──
class LIFNode(nn.Module):
    """단순화 LIF: surrogate gradient (Sigmoid)"""
    def __init__(self, tau=2.0, v_th=1.0, v_reset=0.0):
        super().__init__()
        self.tau      = tau
        self.v_th     = v_th
        self.v_reset  = v_reset
        self.mem      = None

    def reset(self):
        self.mem = None

    @staticmethod
    def surrogate(x):
        # Sigmoid surrogate gradient
        return torch.sigmoid(4.0 * x)

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        # U[t] = beta * H[t-1] + X[t]
        beta = 1.0 - 1.0 / self.tau
        self.mem = beta * self.mem + x
        # S[t] = Θ(U - v_th)  via surrogate
        spike = self.surrogate(self.mem - self.v_th)
        # H[t] = v_reset * S + U * (1-S)
        self.mem = self.v_reset * spike + self.mem * (1.0 - spike)
        return spike


# ── TTFS 인코더 (논문 Eq.17, S2TDPT_test.ipynb 참고) ──
class TTFSEncoder(nn.Module):
    """픽셀값 → spike latency: t = T*(1 - x)"""
    def __init__(self, T=4):
        super().__init__()
        self.T = T

    def forward(self, x):
        # x: [B, C, H, W]  ∈ [0,1]
        # 반환: binary spikes [T, B, C, H, W]
        t_spike = ((1.0 - x) * (self.T - 1)).round().long()  # [B,C,H,W]
        spikes = torch.zeros(self.T, *x.shape, device=x.device)
        for t in range(self.T):
            spikes[t] = (t_spike == t).float()
        return spikes  # [T, B, C, H, W]


print('LIFNode & TTFSEncoder 정의 완료')


# In[30]:


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
    model.eval()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in tqdm(loader, desc='  eval ', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if is_snn:
            reset_lif(model)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total * 100

def _live_plot(history, name, color):
    """에포크마다 커브를 파일로 저장"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.suptitle(f'Training  —  {name}', fontsize=12, fontweight='bold')
    ep = range(1, len(history['train_loss'])+1)
    axes[0].plot(ep, history['train_loss'], '--', color=color, alpha=0.6, label='train')
    axes[0].plot(ep, history['val_loss'],         color=color, label='val')
    axes[0].set(title='Loss', xlabel='Epoch')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, history['train_acc'], '--', color=color, alpha=0.6, label='train')
    axes[1].plot(ep, history['val_acc'],         color=color, label='val')
    axes[1].set(title='Accuracy (%)', xlabel='Epoch')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('²', '2')
    plt.savefig(f'live_{safe_name}.png', bbox_inches='tight')
    plt.close(fig)


PALETTE = {
    'ResNet34':            '#E74C3C',
    'Vanilla Transformer': '#3498DB',
    'Spikformer':          '#2ECC71',
    'S2TDPT':              '#1ABC9C',
}


def run_training(model, name, is_snn=False, epochs=None):
    """
    학습 루프:
      [1] 배치 tqdm bar
      [2] 에포크마다 print + 커브 파일 저장
    """
    global MODEL_BAR
    if MODEL_BAR is None:
        MODEL_BAR = tqdm(total=len(MODEL_NAMES),
                         desc='전체 학습 진행',
                         bar_format='{desc}: {bar:40} {n}/{total}  [{elapsed}<{remaining}]',
                         leave=True, file=sys.stdout)

    epochs = epochs or CFG['epochs']
    batch_size = CFG['snn_batch_size'] if is_snn else CFG['batch_size']
    train_loader_local = DataLoader(train_ds, batch_size=batch_size,
                                    shuffle=True, num_workers=CFG['num_workers'], pin_memory=False)
    test_loader_local  = DataLoader(test_ds,  batch_size=batch_size,
                                    shuffle=False, num_workers=CFG['num_workers'], drop_last=False)
    model  = model.to(device)
    crit   = nn.CrossEntropyLoss()
    opt    = optim.AdamW(model.parameters(),
                         lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    sched  = CosineAnnealingLR(opt, T_max=epochs)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    best_acc, best_state = 0., None
    t_start = time.time()
    color   = PALETTE.get(name, '#95A5A6')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*55}', flush=True)
    print(f'  Training: {name}  |  params: {n_params:,}', flush=True)
    print(f'{"="*55}', flush=True)

    for ep in range(1, epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader_local, opt, crit, is_snn)
        va_loss, va_acc = evaluate(model, test_loader_local, crit, is_snn)
        sched.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        eta_h   = elapsed * (epochs - ep) / 3600
        print(f'  Ep {ep:2d}/{epochs} | tr={tr_loss:.3f} va_acc={va_acc:.1f}% best={best_acc:.1f}% ETA={eta_h:.1f}h', flush=True)
        _live_plot(history, name, color)

    total_min = (time.time() - t_start) / 60
    print(f'  Best val acc: {best_acc:.2f}%  |  Total: {total_min:.1f} min', flush=True)

    ckpt_path = os.path.join(CFG['checkpoint_dir'],
                              f'{name.replace(" ","_")}_best.pth')
    torch.save({'model': best_state, 'history': history,
                'best_acc': best_acc, 'name': name,
                'train_mins': total_min}, ckpt_path)

    return history, best_acc, total_min


ALL_RESULTS = {}
# 전체 모델 진행바 — run_training() 첫 호출 시 자동 초기화
MODEL_BAR = None
MODEL_NAMES = ['ResNet34', 'Vanilla Transformer', 'Spikformer', 'S²TDPT']

print('학습 유틸 + 프로그레스바 준비 완료')


# ## 3. Model A — ResNet34 (베이스라인)
# *udyann.ipynb 코드 그대로 사용*

# In[31]:


# ── udyann.ipynb Cell 12-13 ──
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
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
    """이미지를 패치 시퀀스로 변환 (CNN stem)"""
    def __init__(self, img_size, patch_size, in_ch=3, embed_dim=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size),
            nn.LayerNorm([embed_dim,
                          img_size // patch_size,
                          img_size // patch_size])
        )

    def forward(self, x):  # [B,C,H,W] → [B,N,D]
        x = self.proj(x)              # [B,D,h,w]
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1,2)  # [B,N,D]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # [B,N,D]
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, depth,
                 num_heads, mlp_ratio, num_classes):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # [B,3,H,W]
        x   = self.patch_embed(x)             # [B,N,D]
        cls = self.cls_token.expand(x.size(0),-1,-1)
        x   = torch.cat([cls, x], dim=1)      # [B,N+1,D]
        x   = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:,0]))   # CLS token


vanilla = VanillaTransformer(
    img_size   = CFG['crop_size'],
    patch_size = CFG['patch_size'],
    embed_dim  = CFG['embed_dim'],
    depth      = CFG['depth'],
    num_heads  = CFG['num_heads'],
    mlp_ratio  = CFG['mlp_ratio'],
    num_classes= CFG['num_classes'],
)
print(f'VanillaTransformer params: {sum(p.numel() for p in vanilla.parameters()):,}')


# In[ ]:


h, acc, mins = run_training(vanilla, 'Vanilla Transformer', is_snn=False)
ALL_RESULTS['Vanilla Transformer'] = (h, acc, mins)
MODEL_BAR.update(1); MODEL_BAR.set_postfix(last=f'VanillaTF {acc:.1f}%')


# ## 5. Model C — Spikformer (SSA)
# *Zhou et al. 2022 — Spiking Self-Attention: Q·K elementwise multiply, no softmax*

# In[ ]:


class SpikingPatchEmbed(nn.Module):
    """Spiking Patch Splitting — Conv+BN+LIF 스택"""
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

    def forward(self, x):   # [B,C,H,W] → [B,N,D]
        x = self.proj(x)    # [B,D,H/2,W/2]
        B, D, h, w = x.shape
        return x.flatten(2).transpose(1,2)  # [B,N,D]


class SSA(nn.Module):
    """
    Spiking Self-Attention (Zhou 2022, Eq.6)
    A = SN(Q) ⊙ SN(K) · V  — no softmax, elementwise multiply
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_conv = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                     nn.BatchNorm1d(dim),
                                     LIFNode(CFG['tau']))
        self.k_conv = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                     nn.BatchNorm1d(dim),
                                     LIFNode(CFG['tau']))
        self.v_conv = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                     nn.BatchNorm1d(dim),
                                     LIFNode(CFG['tau']))
        self.proj   = nn.Linear(dim, dim)

    def _reshape(self, x):  # [B,N,D] → [B,H,N,d]
        B, N, D = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).transpose(1,2)

    def forward(self, x):   # x: [B,N,D]
        B, N, D = x.shape
        # BN expects [B,D] — apply over sequence
        q = self.q_conv[0](x)
        q = self.q_conv[1](q.reshape(B*N,D)).reshape(B,N,D)
        q = self.q_conv[2](q)

        k = self.k_conv[0](x)
        k = self.k_conv[1](k.reshape(B*N,D)).reshape(B,N,D)
        k = self.k_conv[2](k)

        v = self.v_conv[0](x)
        v = self.v_conv[1](v.reshape(B*N,D)).reshape(B,N,D)
        v = self.v_conv[2](v)

        # SSA: A = (Q⊙K) @ V  (논문 Eq.6)
        qk = self._reshape(q * k) * self.scale  # [B,H,N,d]
        v_h = self._reshape(v)                   # [B,H,N,d]
        # [B,H,N,d] @ [B,H,d,N] → [B,H,N,N] → [B,H,N,d]
        out = (qk @ v_h.transpose(-2,-1)) @ v_h
        out = out.transpose(1,2).reshape(B, N, D)  # [B,N,D]
        return self.proj(out)


class SpikingMLP(nn.Module):
    def __init__(self, dim, ratio=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*ratio, bias=False),
            nn.BatchNorm1d(dim*ratio),
            LIFNode(CFG['tau']),
            nn.Linear(dim*ratio, dim, bias=False),
            nn.BatchNorm1d(dim),
            LIFNode(CFG['tau']),
        )

    def forward(self, x):   # [B,N,D]
        B, N, D = x.shape
        out = self.net[0](x)
        out = self.net[1](out.reshape(B*N,-1)).reshape(B,N,-1)
        out = self.net[2](out)
        out = self.net[3](out)
        out = self.net[4](out.reshape(B*N,-1)).reshape(B,N,-1)
        out = self.net[5](out)
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
        self.T       = T
        self.encoder = TTFSEncoder(T)
        self.sps     = SpikingPatchEmbed(3, embed_dim)  # Spiking Patch Splitting
        self.blocks  = nn.ModuleList([
            SpikformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # [B,3,H,W]
        spikes = self.encoder(x)    # [T,B,3,H,W]
        out    = 0.
        for t in range(self.T):
            xt = self.sps(spikes[t])          # [B,N,D]
            for blk in self.blocks:
                xt = blk(xt)
            out = out + xt
        out = out / self.T                    # temporal mean
        out = out.mean(dim=1)                 # spatial mean (GAP)
        return self.head(out)


spikformer = Spikformer(
    img_size   = CFG['crop_size'],
    embed_dim  = CFG['embed_dim'],
    depth      = CFG['depth'],
    num_heads  = CFG['num_heads'],
    mlp_ratio  = CFG['mlp_ratio'],
    num_classes= CFG['num_classes'],
    T          = CFG['T'],
)
print(f'Spikformer params: {sum(p.numel() for p in spikformer.parameters()):,}')


# In[ ]:


h, acc, mins = run_training(spikformer, 'Spikformer', is_snn=True)
ALL_RESULTS['Spikformer'] = (h, acc, mins)
MODEL_BAR.update(1); MODEL_BAR.set_postfix(last=f'Spikformer {acc:.1f}%')


# ## 6. Model D — S²TDPT (논문 완전 재현)
# *Mondal & Kumar 2025 — SPS + S²TDPSA (STDP Self-Attention) + Spiking MLP*

# In[ ]:


class SPS(nn.Module):
    """
    Spiking Patch Splitting (논문 Fig.3)
    Conv+BN+LIF+Conv+BN+LIF+MaxPool × (depth//2)
    CPU 경량화: 1 SPS block
    """
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Sequential(
            # Block 1
            nn.Conv2d(in_ch, embed_dim//2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            LIFNode(CFG['tau'], CFG['v_threshold'], CFG['v_reset']),
            # Block 2  (stride=2 → downscale)
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            LIFNode(CFG['tau'], CFG['v_threshold'], CFG['v_reset']),
        )

    def forward(self, x):   # [B,C,H,W] → [B,D,H/2,W/2]
        return self.proj(x)


class S2TDPSA(nn.Module):
    """
    S²TDPT Self-Attention (논문 Eq.17-22)
    1. Spike rates → TTFS latency: t = T*(1 - r/D_H)
    2. Δt = t_Q - t_K
    3. ΔW = A_stdp * exp(-|Δt|/τ_stdp)  (LTP if Δt<0, LTD otherwise)
    4. A_ij = ΔW + w_offset   ∈ (0, 1)  — softmax 불필요
    5. Out = A @ V
    """
    def __init__(self, dim, num_heads,
                 A_stdp, tau_stdp, w_offset, T_max=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.A_stdp    = A_stdp
        self.tau_stdp  = tau_stdp
        self.w_offset  = w_offset
        self.T_max     = T_max

        # Q, K, V: Conv1d (논문은 Conv2d, 여기선 패치 시퀀스 처리)
        self.q_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            LIFNode(CFG['tau']))
        self.k_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            LIFNode(CFG['tau']))
        self.v_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            LIFNode(CFG['tau']))
        self.out_proj = nn.Linear(dim, dim)

    def _apply_proj(self, proj_seq, x):   # x: [B,N,D]
        B, N, D = x.shape
        out = proj_seq[0](x)              # Linear
        out = proj_seq[1](out.reshape(B*N, D)).reshape(B, N, D)  # BN
        out = proj_seq[2](out)            # LIF
        return out

    def _spike_rate_to_latency(self, spikes):  # spikes: [B,N,D] ∈{0,1}
        # 논문 Eq.18: t = T*(1 - r/D_H),  r = sum of spikes / D_H
        r = spikes.mean(dim=-1, keepdim=True)   # [B,N,1]
        return self.T_max * (1.0 - r)           # [B,N,1]

    def forward(self, x):  # x: [B,N,D]
        B, N, D = x.shape

        Q_spike = self._apply_proj(self.q_proj, x)  # [B,N,D] binary
        K_spike = self._apply_proj(self.k_proj, x)
        V_spike = self._apply_proj(self.v_proj, x)

        # TTFS latency (논문 Eq.18)
        t_Q = self._spike_rate_to_latency(Q_spike)  # [B,N,1]
        t_K = self._spike_rate_to_latency(K_spike)  # [B,N,1]

        # Δt 행렬 (논문 Eq.19)
        delta_t = t_Q - t_K.transpose(1, 2)        # [B,N,N]

        # STDP 커널 (논문 Eq.20-21)
        f_stdp  = self.A_stdp * torch.exp(-delta_t.abs() / self.tau_stdp)
        delta_w = torch.where(delta_t < 0, f_stdp, -f_stdp)  # LTP/LTD

        # Attention score (논문 Eq.22): (0,1) bounded — softmax 불필요
        A = delta_w + self.w_offset                 # [B,N,N]

        # Output
        out = A @ V_spike                           # [B,N,D]
        return self.out_proj(out)


class S2TDPTBlock(nn.Module):
    """S²TDPT Encoder Block: S²TDPSA + Spiking MLP + residual"""
    def __init__(self, dim, num_heads, mlp_ratio,
                 A_stdp, tau_stdp, w_offset):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = S2TDPSA(dim, num_heads, A_stdp, tau_stdp, w_offset)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = SpikingMLP(dim, mlp_ratio)   # Spikformer와 동일 Spiking MLP

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # residual (membrane potential 누적)
        x = x + self.mlp(self.norm2(x))
        return x


class S2TDPT(nn.Module):
    """
    S²TDPT 완전 재현 (논문 Fig.3)
    Input → TTFS Encode → SPS × L → S²TDPSA+MLP × L → GTMP → GAP → FC
    """
    def __init__(self, img_size, embed_dim, depth, num_heads,
                 mlp_ratio, num_classes, T,
                 A_stdp, tau_stdp, w_offset):
        super().__init__()
        self.T       = T
        self.encoder = TTFSEncoder(T)
        self.sps     = SPS(3, embed_dim, patch_size=2)
        self.blocks  = nn.ModuleList([
            S2TDPTBlock(embed_dim, num_heads, mlp_ratio, A_stdp, tau_stdp, w_offset)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):   # [B,3,H,W]
        spikes = self.encoder(x)            # [T,B,3,H,W]
        feat_sum = 0.
        for t in range(self.T):
            xt = self.sps(spikes[t])        # [B,D,h,w]
            B, D, h, w = xt.shape
            xt = xt.flatten(2).transpose(1,2)   # [B,N,D]
            for blk in self.blocks:
                xt = blk(xt)
            feat_sum = feat_sum + xt

        # GTMP: temporal mean (논문 — T dim collapse)
        feat = feat_sum / self.T            # [B,N,D]
        # GAP: spatial mean
        feat = feat.mean(dim=1)             # [B,D]
        return self.head(feat)

# ── should_skip: 이미 학습된 모델 건너뛰기 ──
def should_skip(model_name):
    safe_name = model_name.replace(' ', '_')
    file_path = os.path.join(CFG['checkpoint_dir'], f'{safe_name}_best.pth')
    if not os.path.exists(file_path):
        return False
    try:
        torch.load(file_path, map_location='cpu', weights_only=True)
        print(f'--- [SKIP] {model_name} 이미 완료됨 (정상 파일 확인) ---', flush=True)
        return True
    except Exception as e:
        print(f'--- [WARN] {model_name} 체크포인트 손상됨: {e}. 다시 학습합니다. ---', flush=True)
        return False


if __name__ == '__main__':

    # ── 모델 학습 ──
    models_to_run = [
        ('ResNet34',           lambda: ResNet(BasicBlock, [3,4,6,3], CFG['num_classes']), False),
        ('Vanilla Transformer', lambda: VanillaTransformer(
            img_size=CFG['crop_size'], patch_size=CFG['patch_size'],
            embed_dim=CFG['embed_dim'], depth=CFG['depth'],
            num_heads=CFG['num_heads'], mlp_ratio=CFG['mlp_ratio'],
            num_classes=CFG['num_classes']), False),
        ('Spikformer',         lambda: Spikformer(
            img_size=CFG['crop_size'], embed_dim=CFG['embed_dim'],
            depth=CFG['depth'], num_heads=CFG['num_heads'],
            mlp_ratio=CFG['mlp_ratio'], num_classes=CFG['num_classes'],
            T=CFG['T']), True),
        ('S2TDPT',             lambda: S2TDPT(
            img_size=CFG['crop_size'], embed_dim=CFG['embed_dim'],
            depth=CFG['depth'], num_heads=CFG['num_heads'],
            mlp_ratio=CFG['mlp_ratio'], num_classes=CFG['num_classes'],
            T=CFG['T'], A_stdp=CFG['A_stdp'],
            tau_stdp=CFG['tau_stdp'], w_offset=CFG['w_offset']), True),
    ]

    for model_name, model_fn, is_snn in models_to_run:
        if should_skip(model_name):
            # 체크포인트에서 결과 복원
            ckpt = torch.load(os.path.join(CFG['checkpoint_dir'], f'{model_name.replace(" ", "_")}_best.pth'),
                              map_location='cpu')
            ALL_RESULTS[model_name] = (ckpt['history'], ckpt['best_acc'], 0.)
            continue
        print(f'--- [START] {model_name} 학습을 시작합니다. ---', flush=True)
        model = model_fn()
        h, acc, mins = run_training(model, model_name, is_snn=is_snn)
        ALL_RESULTS[model_name] = (h, acc, mins)
        if MODEL_BAR:
            MODEL_BAR.update(1)
            MODEL_BAR.set_postfix(last=f'{model_name} {acc:.1f}%')
        # OOM 방지: 모델을 CPU로 내리고 명시적 메모리 해제
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if MODEL_BAR:
        MODEL_BAR.close()
    print('\n🎉 전체 학습 완료!', flush=True)

    # ── 최종 시각화 ──
    names     = list(ALL_RESULTS.keys())
    best_accs = [ALL_RESULTS[n][1] for n in names]
    train_mins= [ALL_RESULTS[n][2] for n in names]
    colors    = [PALETTE.get(n, '#95A5A6') for n in names]

    # 학습 커브
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — Face-Mask Dataset', fontsize=13, fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    axes[0].set_title('Loss (dashed=train / solid=val)')
    axes[1].set_title('Accuracy % (dashed=train / solid=val)')
    for name, (h, best_acc, _) in ALL_RESULTS.items():
        c  = PALETTE.get(name, '#95A5A6')
        ep = range(1, len(h['train_loss'])+1)
        axes[0].plot(ep, h['train_loss'], '--', color=c, lw=1.5, alpha=0.5)
        axes[0].plot(ep, h['val_loss'],         color=c, lw=2,   label=name)
        axes[1].plot(ep, h['train_acc'], '--', color=c, lw=1.5, alpha=0.5)
        axes[1].plot(ep, h['val_acc'],         color=c, lw=2,
                     label=f'{name}  (best {best_acc:.1f}%)')
    for ax in axes:
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('result_curves.png', bbox_inches='tight')
    plt.close()
    print('result_curves.png 저장 완료', flush=True)

    # 성능 바 차트
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle('Final Comparison — Face-Mask Dataset (CPU, 10 epochs)',
                  fontsize=12, fontweight='bold')
    axes2[0].set(ylabel='Best Val Accuracy (%)', title='Accuracy')
    axes2[1].set(ylabel='Training Time (min)',   title='Training Time')
    for ax in axes2:
        ax.grid(True, alpha=0.3)
    for idx, (name, acc, mins, c) in enumerate(zip(names, best_accs, train_mins, colors)):
        axes2[0].bar(idx, acc, color=c, width=0.5, edgecolor='white', linewidth=1.5)
        axes2[0].text(idx, acc + 0.3, f'{acc:.2f}%', ha='center', fontsize=10, fontweight='bold')
        axes2[1].bar(idx, mins, color=c, width=0.5, edgecolor='white', linewidth=1.5)
        axes2[1].text(idx, mins + 0.5, f'{mins:.0f}m', ha='center', fontsize=10, fontweight='bold')
    for ax in axes2:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=10)
    axes2[0].set_ylim(max(0, min(best_accs)-5), 101)
    plt.tight_layout()
    plt.savefig('result_comparison.png', bbox_inches='tight')
    plt.close()
    print('result_comparison.png 저장 완료', flush=True)

    print('\n=== 최종 결과 요약 ===', flush=True)
    for n in names:
        _, acc, mins = ALL_RESULTS[n]
        print(f'  {n:25s}  acc={acc:.2f}%   time={mins:.0f}min', flush=True)

