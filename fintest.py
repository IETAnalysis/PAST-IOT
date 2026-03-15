import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, WeightedRandomSampler
import os
import torch.nn as nn
import random
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix
from model01 import IoTAnomalyModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings

warnings.filterwarnings("ignore")

# --- 配置 ---
#PROCESSED_DATA_B = "/home/njust/data/slj/时间序列/时序特征/MQTT-IoT-IDS2020(CSV207)"
#PROCESSED_DATA_B = "/home/njust/data/slj/时间序列/时序特征/BoT_IoT(207)"
PROCESSED_DATA_B = "/home/njust/data/slj/时间序列/时序特征/IOT_NI(207-1)"
#PROCESSED_DATA_B = "/home/njust/data/slj/时间序列/时序特征/Edgello Tset"
PRETRAINED_PATH = "/home/njust/data/slj/时间序列/训练权重/pretrained_encoder_best207-fenjiechuangxin.pth"
SAVE_FINE_TUNED = "finetuned_IOT_NI_ldam_full.pth"

LABEL_RATIO = 0000
EPOCHS = 0000
BATCH_SIZE = 0000
PATIENCE = 0000
DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'

# --- 相位感知重构损失 ---
class PhaseAwareReconstructionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, preds, target):
        rec_cont, rec_disc_logits = preds
        target_cont = target[:, :, :2]
        loss_val = F.mse_loss(rec_cont, target_cont)
        target_disc = target[:, :, 2].long().clamp(0, 0000)
        rec_disc_logits = rec_disc_logits.view(-1, 0000)
        target_disc = target_disc.view(-1)
        loss_phase = self.loss_ce(rec_disc_logits, target_disc)

        return loss_val + 0.1*loss_phase


def compute_uaff(y_true, y_pred):
    """Uncertainty-Aware F1 Score"""
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    bias = np.mean(y_true)  # 真实异常率
    if 1 - bias == 0:
        u_prec = p
    else:
        u_prec = (p - bias) / (1 - bias)
    u_prec = max(0, u_prec)
    if u_prec + r == 0: return 0.0
    return 2 * (u_prec * r) / (u_prec + r)


def compute_pa_k(y_true, y_scores, k=20):
    """Point Adjustment %K F1"""
    if len(y_scores) == 0: return 0.0
    threshold = np.percentile(y_scores, 100 - k)
    y_pred_adjusted = (y_scores >= threshold).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred_adjusted, average='binary', pos_label=1, zero_division=0)
    return f1


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    print("\n[Textual Confusion Matrix (Rows=True, Cols=Pred)]")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(cm_df)
    print("-" * 50)


def validate(model, loader):
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits, _, _ = model(x, mode='finetune')
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            y_prob_all.extend(probs.cpu().numpy())

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_probs = np.array(y_prob_all)

    # 1. 多分类指标
    acc = accuracy_score(y_true, y_pred)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics = {'acc': acc, 'f1': f1_macro}

    # 2. 二分类指标 (0是正常, 其他是异常)
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)

    bin_acc = accuracy_score(y_true_bin, y_pred_bin)
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='binary', pos_label=1, zero_division=0
    )
    metrics.update({'bin_acc': bin_acc, 'bin_p': bin_p, 'bin_r': bin_r, 'bin_f1': bin_f1})

    # 3. 高级异常检测指标 (UAff, PA%K)
    # 异常分数 = 1 - P(Benign)
    anomaly_scores = 1.0 - y_probs[:, 0]

    metrics['uaff'] = compute_uaff(y_true_bin, y_pred_bin)

    # 动态计算 K 值 (基于真实异常比例)
    total_anomalies = np.sum(y_true_bin)
    real_ratio = (total_anomalies / len(y_true_bin)) * 100 if len(y_true_bin) > 0 else 0
    metrics['pak'] = compute_pa_k(y_true_bin, anomaly_scores, k=max(real_ratio, 0.01))

    return metrics, y_true, y_pred


# --- 数据加载与采样 ---
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def load_data(path):
    files = glob.glob(os.path.join(path, "*.pt"))
    data = []
    for f in files: data.extend(torch.load(f, weights_only=False))

    # 校验维度 [32, 3]
    valid_data = [x for x in data if x['feature'].shape == (32, 3)]
    if len(valid_data) == 0: raise ValueError("No valid 3-channel data found.")

    feats = torch.stack([x['feature'] for x in valid_data])
    labels = torch.tensor([x['label'] for x in valid_data]).long()
    id2name = {}
    for x in valid_data:
        if x['label'] not in id2name: id2name[x['label']] = x['class_name']
    return TensorDataset(feats, labels), len(id2name), id2name


def main():
    dataset, num_classes, id2name = load_data(PROCESSED_DATA_B)
    target_names = [id2name[i] for i in range(num_classes)]
    print(f"Classes Map: {id2name}")

    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len
    train_pool, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    # 下采样构建训练集
    indices = []
    all_indices = train_pool.indices
    all_labels = dataset.tensors[1][all_indices]
    for lbl in torch.unique(all_labels):
        idx = (all_labels == lbl).nonzero(as_tuple=True)[0]
        real_idx = [all_indices[i] for i in idx.tolist()]
        k = int(len(real_idx) * LABEL_RATIO)
        k = max(k, 10)
        indices.extend(random.sample(real_idx, k))
    train_set = Subset(dataset, indices)

    # 训练集统计 & 采样器
    train_labels = [dataset.tensors[1][i].item() for i in train_set.indices]
    class_counts = [0] * num_classes
    for lbl in train_labels: class_counts[lbl] += 1
    print(f"Training Class Distribution: {class_counts}")

    class_weights = []
    for count in class_counts:
        if count > 0:
            w = 1.0 / np.sqrt(count)
            class_weights.append(w)
        else:
            class_weights.append(0.0)

    # 归一化权重便于观察
    w_min = min([x for x in class_weights if x > 0])
    class_weights = [x / w_min for x in class_weights]
    print(f"Smoothed Class Weights: {[f'{w:.2f}' for w in class_weights]}")

    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # 5. 模型加载
    model = IoTAnomalyModel(seq_len=0000, num_classes=num_classes).to(DEVICE)
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading Pretrained Weights: {PRETRAINED_PATH}")
        state = torch.load(PRETRAINED_PATH, weights_only=True)
        # 兼容性加载
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
        print(f"Skipped layers: {[k for k in state.keys() if k not in pretrained]}")
    else:
        print("Warning: Training from scratch!")

    # 6. Loss 配置
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    criterion_rec = PhaseAwareReconstructionLoss().to(DEVICE)
    REC_WEIGHT = 0000

    optimizer = optim.AdamW(model.parameters(), lr=0000, weight_decay=0000)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    stopper = EarlyStopping(patience=PATIENCE)

    print("\nStarting Fine-tuning...")
    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_cls = 0
        total_rec = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            outputs, (rec_cont, rec_disc), aux_out = model(x, mode='finetune')

            l_cls_val = criterion_cls(outputs, y)
            l_rec_val = criterion_rec((rec_cont, rec_disc), x)
            loss = l_cls_val + REC_WEIGHT * l_rec_val

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += l_cls_val.item()
            total_rec += l_rec_val.item()

        scheduler.step()
        met, _, _ = validate(model, val_loader)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} (Cls: {total_cls / len(train_loader):.4f}, Rec: {total_rec / len(train_loader):.4f}) | "
              f"Macro-F1: {met['f1']:.4f} | UAff: {met['uaff']:.4f}")

        if met['f1'] > best_val_f1:
            best_val_f1 = met['f1']
            torch.save(model.state_dict(), SAVE_FINE_TUNED)

        stopper(met['f1'])
        if stopper.early_stop:
            print("Early Stopping.")
            break

    # 7. Final Test
    print("\n" + "=" * 40)
    print("       FINAL TEST REPORT       ")
    print("=" * 40)
    model.load_state_dict(torch.load(SAVE_FINE_TUNED, weights_only=True))
    met, y_true, y_pred = validate(model, test_loader)

    print("\n[1] Multi-class Detailed Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("\n[2] Binary Classification Metrics:")
    print(f"    * Accuracy:  {met['bin_acc']:.4f}")
    print(f"    * Precision: {met['bin_p']:.4f}")
    print(f"    * Recall:    {met['bin_r']:.4f}")
    print(f"    * F1-Score:  {met['bin_f1']:.4f}")

    print("\n[3] Advanced Anomaly Metrics:")
    print(f"    * UAff: {met['uaff']:.4f} | PA%K: {met['pak']:.4f}")

    plot_confusion_matrix(y_true, y_pred, target_names)


if __name__ == "__main__":
    main()