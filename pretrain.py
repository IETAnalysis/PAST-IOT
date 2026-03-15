import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import os
import glob
from tqdm import tqdm
from model01 import IoTAnomalyModel
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 配置 ---
DATA_PATH = ""
SAVE_PATH = ""

BATCH_SIZE = 0000
EPOCHS = 0000
LR = 0000
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEQ_LEN = 0000


class PhaseAwareReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, preds, target):
        rec_cont, rec_disc_logits = preds
        target_cont = target[:, :, :2]
        loss_val = F.mse_loss(rec_cont, target_cont)
        target_disc = target[:, :, 2].long().clamp(0, 0000)
        loss_phase = self.loss_ce(
            rec_disc_logits.reshape(-1, rec_disc_logits.size(-1)),
            target_disc.reshape(-1)
        )
        return loss_val + 2.0 * loss_phase


class DissimilarityLoss(torch.nn.Module):
    def __init__(self, margin=10.0):
        super().__init__()
        self.margin = margin

    def forward(self, z_pos, z_neg):
        # 拉大正负样本距离
        dist = F.pairwise_distance(z_pos, z_neg)
        loss = torch.mean(torch.clamp(self.margin - dist, min=0.0))
        return loss


class UncertaintyLoss(torch.nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        final_loss = 0
        for i, loss in enumerate(losses):
            v = self.log_vars[i].clamp(-4.5, 4.5)
            precision = torch.exp(-v)
            final_loss += 0.5 * precision * loss + 0.5 * v
        return final_loss


# --- 2. 物理感知异常注入 ---
def inject_anomaly_complex(x):
    B, L, C = x.shape
    x_aug = x.clone()
    labels = torch.zeros(B, dtype=torch.long, device=x.device)

    for i in range(B):
        if np.random.rand() < 0.1: continue

        type_rnd = np.random.choice([
            'burst', 'scan_tcp', 'dos_mock', 'noise',
            'flag_mutation', 'drop', 'jitter', 'freeze',
            'scan_udp', 'stealth_scan'
        ])

        # 1. Burst: 流量突发 (Size 激增)
        if type_rnd == 'burst':
            idx = np.random.randint(0, L)
            val = x_aug[i, idx, 1]
            sign = torch.sign(val) if torch.abs(val) > 1e-6 else 1.0
            x_aug[i, idx, 1] += (3.0 * sign)  # 增大包大小
            labels[i] = 1

        # 2. Scan TCP (模拟 TCP 扫描)
        elif type_rnd == 'scan_tcp':
            seg = np.random.randint(2, 6)
            s = np.random.randint(0, L - seg)

            scan_flag = np.random.choice([2.0, 4.0, 18.0])
            x_aug[i, s:s + seg, 2] = scan_flag

            # 扫描通常 IAT 极短且包很小
            raw_iat = 0.001
            x_aug[i, s:s + seg, 0] = np.log1p(raw_iat * 1000)
            x_aug[i, s:s + seg, 1] = 1.0  # 小包
            labels[i] = 2

        # 3. DoS Mock (高频流量)
        elif type_rnd == 'dos_mock':
            x_aug[i, :, 0] = 0.05  # 极短 IAT
            direction = 1.0 if np.random.rand() > 0.5 else -1.0
            x_aug[i, :, 1] = 5.0 * direction  # 大包 DoS
            labels[i] = 3

        # 4. Noise (仅对连续变量 IAT/Size 加噪)
        elif type_rnd == 'noise':
            # 只对前两个通道(IAT, Size)加噪，不碰 Flag
            noise = torch.randn_like(x_aug[i, :, :2]) * 0.1
            x_aug[i, :, :2] += noise
            labels[i] = 4

        # 5. Flag Mutation
        elif type_rnd == 'flag_mutation':
            seg = np.random.randint(1, L)
            rnd_flags = torch.randint(0, 300, (seg,), device=x.device).float()
            # 随机替换一段
            s = np.random.randint(0, L - seg + 1)
            x_aug[i, s:s + seg, 2] = rnd_flags
            labels[i] = 5

        # 6. Drop (丢包)
        elif type_rnd == 'drop':
            idx = np.random.choice(L, size=np.random.randint(1, 5), replace=False)
            x_aug[i, idx, :] = 0
            labels[i] = 6

        # 7. Jitter (仅 IAT 抖动)
        elif type_rnd == 'jitter':
            noise = torch.randn(L, device=x.device) * 0.5
            x_aug[i, :, 0] = torch.clamp(x_aug[i, :, 0] + noise, min=0)
            labels[i] = 7

        # 8. Freeze (通信僵死)
        elif type_rnd == 'freeze':
            s = np.random.randint(0, L - 1)
            x_aug[i, s:, 0] = 10.0  # 巨大 IAT
            x_aug[i, s:, 1] = 0  # Size 0
            x_aug[i, s:, 2] = 0  # No Flag
            labels[i] = 8

        # 9. Scan UDP (模拟 UDP 扫描)
        elif type_rnd == 'scan_udp':
            seg = np.random.randint(4, 16)
            s = np.random.randint(0, L - seg)
            rand_val = np.random.rand()

            if rand_val < 0.25:
                x_aug[i, s:s + seg, 2] = 267.0  # Empty payload
                x_aug[i, s:s + seg, 1] = 0.0
            elif rand_val < 0.50:
                x_aug[i, s:s + seg, 2] = 264.0  # Sequential ports
                x_aug[i, s:s + seg, 1] = 1.0
            else:
                x_aug[i, s:s + seg, 2] = 268.0  # High Entropy
                x_aug[i, s:s + seg, 1] = 1.0

            raw_iat_seconds = np.random.uniform(0.00001, 0.001)
            x_aug[i, s:s + seg, 0] = np.log1p(raw_iat_seconds * 1000)
            labels[i] = 9

        # 10. Stealth Scan (低速扫描)
        elif type_rnd == 'stealth_scan':
            seg = np.random.randint(8, 20)
            s = np.random.randint(0, L - seg)

            if np.random.rand() > 0.5:
                x_aug[i, s:s + seg, 2] = 268.0  # UDP Random Port
            else:
                x_aug[i, s:s + seg, 2] = 2.0  # TCP SYN

            # IAT 稳定且较大
            fixed_t = np.random.choice([0.05, 0.1, 0.2, 0.5, 1.0])
            val = np.log1p(fixed_t * 1000)
            x_aug[i, s:s + seg, 0] = val
            x_aug[i, s:s + seg, 1] = 1.0

            labels[i] = 10

    return x_aug, labels


def load_data(path):
    files = glob.glob(os.path.join(path, "*.pt"))
    if not files: raise ValueError(f"No data files in {path}")
    all_d = []
    print("Loading Pretrain Data...")
    for f in tqdm(files):
        all_d.extend(torch.load(f, weights_only=False))

    valid_feats = []
    for x in all_d:
        if 'feature' in x and x['feature'].shape == (SEQ_LEN, 3):
            valid_feats.append(x['feature'])

    if not valid_feats:
        raise ValueError("No valid 3-channel features found. Run feature02.py first.")
    return torch.stack(valid_feats)


def main():
    data = load_data(DATA_PATH)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 初始化模型
    model = IoTAnomalyModel(seq_len=SEQ_LEN).to(DEVICE)

    # 实例化 Losses
    loss_cont = DissimilarityLoss(margin=1.5).to(DEVICE)
    loss_rec = PhaseAwareReconstructionLoss().to(DEVICE)
    mt_loss = UncertaintyLoss(num_tasks=4).to(DEVICE)

    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': mt_loss.parameters(), 'lr': 0000}
    ], lr=LR, weight_decay=0000)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Start Pre-training (4-Task Enhanced)...")
    model.train()

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        total_l = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", ncols=150)

        for (x_pos,) in pbar:
            x_pos = x_pos.to(DEVICE)
            x_pos = torch.nan_to_num(x_pos)
            B = x_pos.shape[0]
            # 生成负样本
            x_neg, aux_labels = inject_anomaly_complex(x_pos)

            # 生成 Masked Phase 样本
            mask_bool = torch.rand(x_pos.shape[0], x_pos.shape[1], device=DEVICE) < 0.5
            x_masked = x_pos.clone()
            x_masked[:, :, 2][mask_bool] = 0.0  # Phase 置零
            target_phase = x_pos[:, :, 2][mask_bool]  # 预测目标

            optimizer.zero_grad()

            # Forward Pass
            z_pos, rec_pos, _, log_pos_full = model(x_pos, mode='pretrain')
            z_neg, _, _, log_neg = model(x_neg, mode='pretrain')
            _, _, pred_phase_masked, _ = model(x_masked, mode='pretrain')

            # --- Loss Calculation ---
            # 1. Contrastive Loss
            l1 = loss_cont(z_pos, z_neg)
            # 2. Reconstruction Loss
            l2 = loss_rec(rec_pos, x_pos)
            # 3. Classification Loss
            l3 = F.cross_entropy(log_pos_full, torch.zeros(B, dtype=torch.long, device=DEVICE)) + \
                 F.cross_entropy(log_neg, aux_labels)
            # 4. MPP (Masked Phase Prediction)
            mpp_logits = pred_phase_masked[mask_bool]
            mpp_targets = target_phase.long()
            l_mpp = F.cross_entropy(mpp_logits, mpp_targets)

            # Multi-task Weighting
            loss = mt_loss([l1, l2, l3, l_mpp])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 监控距离
            dist = F.pairwise_distance(z_pos, z_neg).mean().item()

            total_l += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'Cont': l1.item(),
                'Rec': l2.item(),
                'Cls': l3.item(),
                'MPP': l_mpp.item(),
                'Dist': dist
            })

        scheduler.step()
        avg_loss = total_l / len(loader)
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved Best Model to {SAVE_PATH}")


if __name__ == "__main__":
    main()