import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import Config


# ==================== 早停机制 ====================
class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ==================== 评估函数 ====================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ==================== 报告生成函数 ====================
def generate_report(model, dataloader, device, config, label_map):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    target_names = [lbl for lbl, _ in sorted(label_map.items(), key=lambda x: x[1])]

    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\n分类报告:\n", report)

    os.makedirs(os.path.dirname(config.REPORT_SAVE_PATH), exist_ok=True)
    with open(config.REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names,
                cmap="Blues")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("混淆矩阵")
    plt.tight_layout()
    plt.savefig(config.CM_SAVE_PATH)
    plt.show()


# ==================== 自定义数据集 ====================
class FeatureDataset(Dataset):
    def __init__(self, data, labels):
        self.features = data
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ==================== 读取 JSONL 数据 ====================
def load_jsonl(path):
    features, labels = [], []
    raw_labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if "token_vectors" in item:
                    token_vecs = np.array(item["token_vectors"])
                    if token_vecs.size == 0:
                        continue
                    doc_feature = np.mean(token_vecs, axis=0)
                elif "vector" in item:
                    doc_feature = np.array(item["vector"])
                else:
                    print(f"跳过无向量字段的记录: {item.get('filename', '未知文件')}")
                    continue
                features.append(doc_feature)
                raw_labels.append(str(item["label"]))
            except Exception as e:
                print(f"读取异常，跳过一条记录，错误: {e}")
                continue

    # 构建 label_map（字符串 -> 数字）
    unique_labels = sorted(set(raw_labels))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels = [label_map[lbl] for lbl in raw_labels]

    print(f"总共有效样本数: {len(features)}，类别数: {len(label_map)} -> {unique_labels}")
    return np.array(features), np.array(labels), label_map


# ==================== MLP模型 ====================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


# ==================== 单轮训练 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


# ==================== 主训练函数 ====================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    features, labels, label_map = load_jsonl(Config.DATA_PATH)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=Config.SEED, stratify=labels
    )

    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FeatureDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)

    input_dim = features.shape[1]
    num_classes = len(label_map)
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    early_stopper = EarlyStopping(Config.EARLY_STOPPING_PATIENCE, Config.BERT_MODEL_PATH)

    print("开始训练...")

    # ====== loss日志文件 ======
    loss_log_path = os.path.join(os.path.dirname(Config.REPORT_SAVE_PATH), "bert_loss_log.txt")
    os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)

    with open(loss_log_path, "w", encoding="utf-8") as loss_file:

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{Config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            # 写入日志
            loss_file.write(f"Epoch {epoch+1}: "
                            f"Train Loss={train_loss:.4f}, "
                            f"Val Loss={val_loss:.4f}, "
                            f"Val Acc={val_acc:.4f}\n")
            loss_file.flush()

            if early_stopper.step(val_loss, model):
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("加载最佳模型进行评估...")
    model.load_state_dict(torch.load(Config.BERT_MODEL_PATH))

    # 生成并保存评估报告和混淆矩阵
    generate_report(model, val_loader, device, Config, label_map)


if __name__ == "__main__":
    train()
