import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "sans-serif"]
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 路径配置 ====================
FEATURE_DIR = "/home/xlc/BERTAST/data/compare_features"
OUTPUT_DIR  = "/home/xlc/BERTAST/results/compare"

# ==================== 超参（与原 Config 保持一致）====================
BATCH_SIZE              = 32
EPOCHS                  = 50
LR                      = 1e-3
EARLY_STOPPING_PATIENCE = 5
SEED                    = 42

# 要对比的模型（须与 extract_features_compare.py 中 MODEL_CONFIGS 的 key 一致）
MODEL_NAMES = [
    "codebert",
    "bert-base",
    "distilbert",
    "tinybert",
    "graphcodebert",
    "longcoder",
]


# ==================== 早停 ====================
class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience  = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        return self.counter >= self.patience


# ==================== 数据集 ====================
class FeatureDataset(Dataset):
    def __init__(self, data, labels):
        self.features = data
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],   dtype=torch.long),
        )


# ==================== 读取 JSONL ====================
def load_jsonl(path):
    """
    复用原脚本逻辑，兼容 token_vectors / vector 两种字段。
    返回: features(ndarray), labels(ndarray), label_map(dict)
    """
    features, raw_labels, filenames = [], [], []

    with open(path, "r", encoding="utf-8") as f:
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
                    print(f"  跳过无向量字段: {item.get('filename', '?')}")
                    continue

                features.append(doc_feature)
                raw_labels.append(str(item["label"]))
                filenames.append(item.get("filename", "unknown"))

            except Exception as e:
                print(f"  读取异常，跳过，错误: {e}")

    # 按文件名排序，与原脚本保持一致
    sorted_data             = sorted(zip(features, raw_labels, filenames), key=lambda x: x[2])
    features, raw_labels, _ = zip(*sorted_data)

    features   = np.array(features)
    raw_labels = list(raw_labels)

    unique_labels = sorted(set(raw_labels))
    label_map     = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    labels        = np.array([label_map[lbl] for lbl in raw_labels])

    return features, labels, label_map


# ==================== MLP ====================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, 256)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# ==================== 训练单 epoch ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples = 0, 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)
    return total_loss / total_samples


# ==================== 评估 ====================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs        = model(batch_x)
            total_loss    += criterion(outputs, batch_y).item() * batch_x.size(0)
            correct       += (outputs.argmax(dim=1) == batch_y).sum().item()
            total_samples += batch_x.size(0)
    return total_loss / total_samples, correct / total_samples


# ==================== 分类报告 + 混淆矩阵 ====================
def generate_report(model, dataloader, device, label_map, report_path, cm_path, model_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            preds = model(batch_x.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    target_names = [lbl for lbl, _ in sorted(label_map.items(), key=lambda x: x[1])]
    report_str   = classification_report(all_labels, all_preds, target_names=target_names)
    report_dict  = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)

    print(f"\n  [{model_name}] 分类报告:\n{report_str}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"模型: {model_name}\n\n{report_str}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    return report_dict


# ==================== MLP 推理延迟 ====================
def measure_mlp_latency(model, dataloader, device, n_warmup=5):
    """
    单独测量 MLP 分类头推理延迟（不含特征提取阶段）。
    返回: (avg_latency_ms_per_sample, throughput_per_s)
    """
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, (batch_x, _) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _  = model(batch_x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= n_warmup:
                latencies.append((t1 - t0) * 1000 / batch_x.size(0))

    if not latencies:
        return float("nan"), float("nan")
    avg_ms = round(sum(latencies) / len(latencies), 4)
    thr    = round(1000.0 / avg_ms, 2)
    return avg_ms, thr


# ==================== 单模型完整训练 ====================
def train_one_model(model_name, device):
    jsonl_path = os.path.join(FEATURE_DIR, f"{model_name}_mean.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"\n⚠️  [{model_name}] 找不到特征文件：{jsonl_path}，跳过。")
        return None

    print(f"\n{'='*60}")
    print(f"  开始训练: {model_name}")
    print(f"{'='*60}")

    features, labels, label_map = load_jsonl(jsonl_path)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(FeatureDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    input_dim   = features.shape[1]
    num_classes = len(label_map)
    model       = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.pt")
    early_stopper   = EarlyStopping(EARLY_STOPPING_PATIENCE, model_save_path)

    loss_log_path = os.path.join(OUTPUT_DIR, f"{model_name}_loss_log.txt")
    best_val_acc  = 0.0

    with open(loss_log_path, "w", encoding="utf-8") as lf:
        for epoch in range(EPOCHS):
            train_loss        = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            best_val_acc      = max(best_val_acc, val_acc)

            print(f"  Epoch {epoch+1:>3}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            lf.write(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                     f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}\n")
            lf.flush()

            if early_stopper.step(val_loss, model):
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))

    # 分类报告 + 混淆矩阵
    report_path = os.path.join(OUTPUT_DIR, f"{model_name}_report.txt")
    cm_path     = os.path.join(OUTPUT_DIR, f"{model_name}_cm.png")
    report_dict = generate_report(
        model, val_loader, device, label_map,
        report_path, cm_path, model_name
    )

    # MLP 推理延迟
    mlp_latency_ms, mlp_throughput = measure_mlp_latency(model, val_loader, device)

    result = {
        "model":                model_name,
        "input_dim":            input_dim,
        "val_accuracy":         round(float(report_dict.get("accuracy",  float("nan"))), 4),
        "macro_precision":      round(float(report_dict.get("macro avg", {}).get("precision", float("nan"))), 4),
        "macro_recall":         round(float(report_dict.get("macro avg", {}).get("recall",    float("nan"))), 4),
        "macro_f1":             round(float(report_dict.get("macro avg", {}).get("f1-score",  float("nan"))), 4),
        "mlp_latency_ms":       mlp_latency_ms,
        "mlp_throughput_per_s": mlp_throughput,
    }

    print(f"\n  [{model_name}] 汇总: "
          f"Acc={result['val_accuracy']:.4f}  "
          f"Macro-F1={result['macro_f1']:.4f}  "
          f"MLP延迟={mlp_latency_ms} ms/sample")

    return result


# ==================== 横向对比图 ====================
def plot_comparison(all_results, output_dir):
    models  = [r["model"]          for r in all_results]
    acc     = [r["val_accuracy"]   for r in all_results]
    f1      = [r["macro_f1"]       for r in all_results]
    latency = [r["mlp_latency_ms"] for r in all_results]

    x   = range(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def _bar(ax, values, title, ylabel, color):
        bars = ax.bar(x, values, color=color, width=0.5)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, values):
            label = f"{val:.4f}" if val < 10 else f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                label, ha="center", va="bottom", fontsize=9,
            )

    _bar(axes[0], acc,     "Validation Accuracy",    "Accuracy",   "#4C8EDA")
    _bar(axes[1], f1,      "Macro F1",                "F1 Score",   "#5DBB8A")
    _bar(axes[2], latency, "MLP Latency (ms/sample)", "ms/sample",  "#E88040")

    plt.suptitle("Multi-Model Comparison — MLP on Malware Behavior Dataset", fontsize=13, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  对比图已保存: {save_path}")


# ==================== 主入口 ====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取特征提取阶段的推理统计（若存在）
    inference_stats = {}
    stats_path = os.path.join(FEATURE_DIR, "inference_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            inference_stats = {s["model"]: s for s in json.load(f)}
        print(f"已加载特征提取推理统计: {stats_path}")

    all_results = []
    for model_name in MODEL_NAMES:
        result = train_one_model(model_name, device)
        if result is None:
            continue

        # 合并特征提取阶段延迟
        s = inference_stats.get(model_name, {})
        result["extract_latency_ms"]       = s.get("avg_latency_ms",   float("nan"))
        result["extract_throughput_per_s"] = s.get("throughput_per_s", float("nan"))

        all_results.append(result)

    if not all_results:
        print("\n❌ 没有可汇总的结果，请先运行 extract_features_compare.py。")
        return

    # 保存汇总 JSON
    summary_path = os.path.join(OUTPUT_DIR, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n汇总 JSON: {summary_path}")

    # 打印汇总表
    print(f"\n{'='*100}")
    print("  横向对比汇总")
    print(f"{'='*100}")
    header = (
        f"{'模型':<16} {'dim':>5} {'Acc':>7} {'Macro-P':>9} "
        f"{'Macro-R':>9} {'Macro-F1':>9} {'MLP延迟ms':>10} {'提取延迟ms':>11}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        el = f"{r['extract_latency_ms']:.3f}" if r["extract_latency_ms"] == r["extract_latency_ms"] else "N/A"
        ml = f"{r['mlp_latency_ms']:.4f}"     if r["mlp_latency_ms"]    == r["mlp_latency_ms"]    else "N/A"
        print(
            f"{r['model']:<16} {r['input_dim']:>5}"
            f" {r['val_accuracy']:>7.4f}"
            f" {r['macro_precision']:>9.4f}"
            f" {r['macro_recall']:>9.4f}"
            f" {r['macro_f1']:>9.4f}"
            f" {ml:>10}"
            f" {el:>11}"
        )

    # 绘图
    plot_comparison(all_results, OUTPUT_DIR)
    print(f"\n✅ 全部完成，结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()