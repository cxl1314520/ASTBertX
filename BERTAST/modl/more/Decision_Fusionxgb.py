
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch, Data
from train_bert_mlp import load_jsonl as load_bert_data
from AST import ASTGraphDataset, collate_fn, ASTOnlyModel
from utils.config import Config
import csv
import joblib
from xgboost import XGBClassifier


# ===== XGBoost 替代 BERTModel(MLP) =====
class XGBoostBERTModel:
    """用 XGBoost 替代原来的 MLP 分类器，对外接口与原 BERTModel 保持一致。"""

    def __init__(self, num_classes, **xgb_kwargs):
        self.num_classes = num_classes
        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=Config.SEED,
            n_jobs=-1,
        )
        params.update(xgb_kwargs)
        self.model = XGBClassifier(**params)
        self._fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self._fitted = True

    def predict_logits(self, X):
        """返回 torch.FloatTensor [N, num_classes]，对应原 bert_model(batch) 的输出。"""
        if not self._fitted:
            raise RuntimeError("XGBoostBERTModel 未训练，请先调用 fit()")
        proba = self.model.predict_proba(X)          # [N, C]  numpy
        logits = np.log(proba + 1e-8)               # log-prob 作为 logits
        return torch.tensor(logits, dtype=torch.float32)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"✅ XGBoost 已保存: {path}")

    @classmethod
    def load(cls, path, num_classes):
        obj = cls(num_classes=num_classes)
        obj.model = joblib.load(path)
        obj._fitted = True
        print(f"✅ XGBoost 已加载: {path}")
        return obj


# ===== 优化的特征融合模型 =====
class EnhancedFusionModel(nn.Module):
    def __init__(self, input_dim_bert, input_dim_ast, hidden_dim, num_classes,
                 fusion_type='concat', dropout_rate=0.3):
        super().__init__()
        self.fusion_type = fusion_type

        self.bert_proj = nn.Sequential(
            nn.Linear(input_dim_bert, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.ast_proj = nn.Sequential(
            nn.Linear(input_dim_ast, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        if fusion_type == 'concat':
            fusion_input_dim = hidden_dim * 2
        elif fusion_type == 'add':
            fusion_input_dim = hidden_dim
        elif fusion_type == 'attention':
            fusion_input_dim = hidden_dim
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        else:
            fusion_input_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        self.alignment_loss_fn = nn.MSELoss()

    def forward(self, bert_vec, ast_vec):
        bert_proj = self.bert_proj(bert_vec)
        ast_proj = self.ast_proj(ast_vec)

        if self.fusion_type == 'concat':
            fused = torch.cat([bert_proj, ast_proj], dim=1)
        elif self.fusion_type == 'add':
            fused = bert_proj + ast_proj
        elif self.fusion_type == 'attention':
            features = torch.stack([bert_proj, ast_proj], dim=1)
            attended, _ = self.attention(features, features, features)
            fused = attended.mean(dim=1)
        else:
            fused = torch.cat([bert_proj, ast_proj], dim=1)

        logits = self.classifier(fused)
        alignment_loss = self.alignment_loss_fn(
            F.normalize(bert_proj, p=2, dim=1),
            F.normalize(ast_proj, p=2, dim=1)
        )
        return logits, alignment_loss, fused


# ===== 加权融合模型 =====
class WeightedFusionModel(nn.Module):
    def __init__(self, input_dim_bert, input_dim_ast, hidden_dim, num_classes):
        super().__init__()
        self.bert_classifier = nn.Sequential(
            nn.Linear(input_dim_bert, hidden_dim), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes)
        )
        self.ast_classifier = nn.Sequential(
            nn.Linear(input_dim_ast, hidden_dim), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes)
        )
        self.weight_bert = nn.Parameter(torch.tensor(0.5))
        self.weight_ast = nn.Parameter(torch.tensor(0.5))

    def forward(self, bert_vec, ast_vec):
        bert_logits = self.bert_classifier(bert_vec)
        ast_logits = self.ast_classifier(ast_vec)
        weights = torch.softmax(torch.stack([self.weight_bert, self.weight_ast]), dim=0)
        fused_logits = weights[0] * bert_logits + weights[1] * ast_logits
        return fused_logits, weights, torch.cat(
            [bert_logits.unsqueeze(1), ast_logits.unsqueeze(1)], dim=1)


# ===== 模型工厂 =====
class ModelFactory:
    @staticmethod
    def create_model(model_type, input_dim_bert, input_dim_ast, hidden_dim, num_classes, **kwargs):
        if model_type == 'simple_concat':
            return SimpleConcatenationModel(input_dim_bert, input_dim_ast, hidden_dim, num_classes)
        elif model_type == 'enhanced_concat':
            return EnhancedFusionModel(input_dim_bert, input_dim_ast, hidden_dim, num_classes,
                                       fusion_type='concat', **kwargs)
        elif model_type == 'weighted_fusion':
            return WeightedFusionModel(input_dim_bert, input_dim_ast, hidden_dim, num_classes)
        elif model_type == 'attention_fusion':
            return EnhancedFusionModel(input_dim_bert, input_dim_ast, hidden_dim, num_classes,
                                       fusion_type='attention', **kwargs)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")


# ===== 简单的特征拼接模型 =====
class SimpleConcatenationModel(nn.Module):
    def __init__(self, input_dim_bert, input_dim_ast, hidden_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim_bert + input_dim_ast, hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, bert_vec, ast_vec):
        fused = torch.cat([bert_vec, ast_vec], dim=1)
        out = self.classifier(fused)
        return out, torch.tensor(0.0), None


# ===== 数据对齐函数 =====
def align_data_simple(bert_features, bert_labels_raw, bert_filenames, ast_dataset):
    print("🔄 数据对齐...")
    bert_count = len(bert_filenames)
    ast_count = len(ast_dataset)
    aligned_count = min(bert_count, ast_count)
    print(f"BERT样本数: {bert_count}, AST样本数: {ast_count}, 对齐样本数: {aligned_count}")

    unified_label_map = ast_dataset.label_map
    print(f"统一标签映射: {unified_label_map}")

    bert_labels_unified = []
    for label in bert_labels_raw[:aligned_count]:
        if isinstance(label, (int, np.integer)):
            bert_labels_unified.append(label)
        else:
            bert_labels_unified.append(unified_label_map.get(label, 0))

    aligned_bert_features = bert_features[:aligned_count]
    aligned_bert_labels = np.array(bert_labels_unified)
    aligned_ast_data = [ast_dataset[i] for i in range(aligned_count)]
    aligned_filenames = bert_filenames[:aligned_count]

    print(f"✅ 成功对齐 {aligned_count} 个样本")
    return aligned_bert_features, aligned_bert_labels, aligned_ast_data, aligned_filenames, unified_label_map


# ===== 数据预处理函数 =====
def convert_ast_data_to_pyg_format(ast_data):
    if isinstance(ast_data, Data):
        return ast_data
    elif isinstance(ast_data, dict):
        pyg_data = Data()
        for key, value in ast_data.items():
            if key in ['x', 'edge_index', 'y', 'batch']:
                if isinstance(value, np.ndarray):
                    setattr(pyg_data, key, torch.from_numpy(value))
                elif torch.is_tensor(value):
                    setattr(pyg_data, key, value)
                else:
                    setattr(pyg_data, key, value)
            else:
                setattr(pyg_data, key, value)
        return pyg_data
    return ast_data


def process_batch_graph(batch_graph):
    if isinstance(batch_graph, tuple):
        if len(batch_graph) >= 2:
            graph_data, labels = batch_graph[0], batch_graph[1]
            if not isinstance(graph_data, Data):
                graph_data = convert_ast_data_to_pyg_format(graph_data)
            return graph_data, labels
        else:
            graph_data = batch_graph[0]
            if not isinstance(graph_data, Data):
                graph_data = convert_ast_data_to_pyg_format(graph_data)
            return graph_data, None
    elif isinstance(batch_graph, Data):
        return batch_graph, None
    else:
        try:
            return convert_ast_data_to_pyg_format(batch_graph), None
        except Exception:
            raise ValueError(f"无法处理的批处理图格式: {type(batch_graph)}")


# ===== 从 AST 样本中安全提取 language 字段 =====
def get_language_from_sample(sample):
    """兼容 dict 和 torch_geometric.data.Data 两种格式。"""
    if isinstance(sample, dict):
        return sample.get("language", "unknown")
    # Data 对象：.language 可能是字符串或不存在
    lang = getattr(sample, "language", None)
    if lang is None:
        return "unknown"
    # torch_geometric 有时把非张量属性存成列表，保险起见转 str
    if isinstance(lang, (list, tuple)):
        return str(lang[0]) if lang else "unknown"
    return str(lang)


# ===== 模型验证函数 =====
def verify_model_structure():
    print("🔍 验证模型结构...")
    xgb_path = Config.BERT_MODEL_PATH.replace('.pt', '_xgb.pkl').replace('.pth', '_xgb.pkl')
    if os.path.exists(xgb_path):
        print(f"✅ XGBoost 模型文件存在: {xgb_path}")
    else:
        print(f"⚠️  XGBoost 模型不存在，将在训练时自动生成: {xgb_path}")
    return True


# ===== 核心训练函数 =====
def train_optimized_fusion(model_type='enhanced_concat'):
    print(f"🚀 开始训练优化融合模型: {model_type}")

    # ---------- 1. 加载数据 ----------
    print("📊 加载数据...")
    bert_features, bert_labels_raw, bert_label_map, bert_filenames = load_bert_data(Config.DATA_PATH)
    ast_dataset = ASTGraphDataset(Config.AST_GRAPH_DIR)

    # ---------- 2. 数据对齐 ----------
    aligned_bert_features, aligned_bert_labels, aligned_ast_data, aligned_filenames, label_map = \
        align_data_simple(bert_features, bert_labels_raw, bert_filenames, ast_dataset)

    if aligned_bert_features is None or len(aligned_bert_features) == 0:
        print("❌ 数据对齐失败")
        return

    label_set = list(label_map.keys())
    num_classes = len(label_set)
    print(f"✅ 对齐后数据: {len(aligned_bert_features)} 个样本, {num_classes} 个类别")

    # ---------- 3. XGBoost 替代 MLP ----------
    print("🤖 处理 XGBoost BERT 分类器...")
    xgb_path = Config.BERT_MODEL_PATH.replace('.pt', '_xgb.pkl').replace('.pth', '_xgb.pkl')

    if os.path.exists(xgb_path):
        xgb_model = XGBoostBERTModel.load(xgb_path, num_classes=num_classes)
    else:
        print("  未找到已保存的 XGBoost，开始训练...")
        xgb_model = XGBoostBERTModel(num_classes=num_classes)
        xgb_model.fit(aligned_bert_features, aligned_bert_labels)
        xgb_model.save(xgb_path)

    # ---------- 4. 加载 AST 预训练模型 ----------
    print("🤖 加载 AST 预训练模型...")
    use_gat = getattr(Config, "USE_GAT", True)
    conv_type = "gat" if use_gat else "sage"
    ast_model = ASTOnlyModel(
        node_type_num=len(ast_dataset.node_type_map),
        node_emb_dim=getattr(Config, "NODE_EMB_DIM", 64),
        hidden_dim=getattr(Config, "GAT_HIDDEN_DIM", 128),
        heads=getattr(Config, "GAT_HEADS", 4),
        num_classes=num_classes,
        conv_type=conv_type
    ).to(Config.DEVICE)
    ast_model.load_state_dict(torch.load(Config.AST_MODEL_PATH))
    ast_model.eval()

    # ---------- 5. 提取特征（同步收集 language 元数据）----------
    print("📊 提取特征...")

    # XGBoost 一次性全量推理
    print("  提取 XGBoost/BERT logits...")
    all_logits_bert = xgb_model.predict_logits(aligned_bert_features)  # [N, C]

    # AST 特征
    print("  提取 AST 特征...")
    all_logits_ast = []
    all_labels     = []
    all_languages  = []   # ← 每个样本的编程语言

    with torch.no_grad():
        for i in range(0, len(aligned_ast_data), Config.BATCH_SIZE):
            batch_indices = list(range(i, min(i + Config.BATCH_SIZE, len(aligned_ast_data))))
            batch_data    = [aligned_ast_data[idx] for idx in batch_indices]
            batch_labels  = [aligned_bert_labels[idx] for idx in batch_indices]
            # 提取每个样本的语言
            batch_languages = [get_language_from_sample(s) for s in batch_data]

            if batch_data:
                batch_data_pyg = [convert_ast_data_to_pyg_format(data) for data in batch_data]
                try:
                    batch_result = collate_fn(batch_data_pyg)
                    batch_graph, _ = process_batch_graph(batch_result)
                except Exception:
                    batch_graph = Batch.from_data_list(batch_data_pyg)

                batch_graph = batch_graph.to(Config.DEVICE)
                outputs = ast_model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
                all_logits_ast.append(outputs.cpu())
                all_labels.extend(batch_labels)
                all_languages.extend(batch_languages)

    all_logits_ast = torch.cat(all_logits_ast, dim=0)
    all_labels     = torch.tensor(all_labels)
    all_languages  = np.array(all_languages)   # shape [N]

    print(f"✅ 特征提取完成: XGBoost {all_logits_bert.shape}, AST {all_logits_ast.shape}")
    unique_langs, lang_counts = np.unique(all_languages, return_counts=True)
    print(f"   语言分布: { {l: c for l, c in zip(unique_langs, lang_counts)} }")

    # ---------- 6. 数据划分 ----------
    train_idx, val_idx = train_test_split(
        np.arange(len(all_labels)), test_size=0.2,
        stratify=all_labels.numpy(), random_state=Config.SEED
    )
    val_languages = all_languages[val_idx]   # 验证集对应的语言标签
    print(f"📊 数据划分: 训练集 {len(train_idx)}, 验证集 {len(val_idx)}")

    # ---------- 7. 构建融合模型 ----------
    model = ModelFactory.create_model(
        model_type=model_type,
        input_dim_bert=num_classes,
        input_dim_ast=num_classes,
        hidden_dim=256,
        num_classes=num_classes,
        dropout_rate=0.4
    ).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    best_val_acc    = 0
    patience_counter = 0

    log_path = os.path.join(os.path.dirname(Config.FUSION_REPORT_PATH), f"fusion_{model_type}_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"🎯 开始训练 {model_type} 模型...")

    # ---------- 8. 训练循环 ----------
    with open(log_path, "w", newline='', encoding="utf-8") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Epoch", "Train Loss", "Align Loss", "Train Acc", "Val Acc", "Learning Rate"])

        for epoch in range(Config.EPOCHS):
            model.train()
            total_loss, total_align_loss, correct_train, total_samples = 0, 0, 0, 0
            np.random.shuffle(train_idx)

            for i in range(0, len(train_idx), Config.BATCH_SIZE):
                batch_indices = train_idx[i:i + Config.BATCH_SIZE]
                b_bert   = all_logits_bert[batch_indices].to(Config.DEVICE)
                b_ast    = all_logits_ast[batch_indices].to(Config.DEVICE)
                b_labels = all_labels[batch_indices].to(Config.DEVICE)

                outputs, align_loss, _ = model(b_bert, b_ast)
                cls_loss        = F.cross_entropy(outputs, b_labels)
                total_batch_loss = cls_loss + 0.1 * align_loss

                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss       += cls_loss.item() * len(batch_indices)
                total_align_loss += align_loss.item() * len(batch_indices)
                preds             = torch.argmax(outputs, dim=1)
                correct_train    += (preds == b_labels).sum().item()
                total_samples    += len(batch_indices)

            avg_train_loss = total_loss / total_samples
            avg_align_loss = total_align_loss / total_samples
            train_acc      = correct_train / total_samples
            val_acc        = evaluate_model(model,
                                            all_logits_bert[val_idx],
                                            all_logits_ast[val_idx],
                                            all_labels[val_idx],
                                            Config.DEVICE)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{Config.EPOCHS}: "
                  f"Loss: {avg_train_loss:.4f} | Align: {avg_align_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            log_writer.writerow([epoch+1, f"{avg_train_loss:.4f}", f"{avg_align_loss:.4f}",
                                  f"{train_acc:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])
            log_file.flush()

            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                patience_counter = 0
                model_path = Config.FUSION_MODEL_PATH.replace('.pth', f'_{model_type}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"✅ 验证集提升，模型已保存: {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("⛔ 早停触发")
                    break

    # ---------- 9. 最终评估（传入语言标签）----------
    final_evaluation(model,
                     all_logits_bert[val_idx],
                     all_logits_ast[val_idx],
                     all_labels[val_idx],
                     label_map,
                     model_type,
                     Config.DEVICE,
                     val_languages)


# ===== 单轮验证准确率 =====
def evaluate_model(model, bert_features, ast_features, labels, device):
    model.eval()
    with torch.no_grad():
        outputs, _, _ = model(bert_features.to(device), ast_features.to(device))
        preds = torch.argmax(outputs, dim=1)
        acc   = (preds == labels.to(device)).sum().item() / len(labels)
    return acc


# ===== 最终评估：分类报告 + 语言×类别准确率矩阵 =====
def final_evaluation(model, bert_features, ast_features, labels, label_map,
                     model_type, device, languages=None):
    model.eval()
    with torch.no_grad():
        outputs, _, fused_features = model(bert_features.to(device), ast_features.to(device))
        preds       = torch.argmax(outputs, dim=1)
        true_labels = labels.to(device)

    preds_np = preds.cpu().numpy()
    true_np  = true_labels.cpu().numpy()

    acc          = accuracy_score(true_np, preds_np)
    idx_to_label = {idx: label for label, idx in label_map.items()}
    target_names = [idx_to_label[i] for i in range(len(label_map))]

    report = classification_report(true_np, preds_np,
                                   target_names=target_names, digits=4)
    print(f"\n🎉 {model_type} 模型最终评估结果:")
    print(f"准确率: {acc:.4f}")
    print("分类报告:")
    print(report)

    # ===== 语言 × 类别 准确率矩阵 =====
    lang_type_lines = []

    if languages is not None and len(languages) > 0:
        unique_languages  = sorted(set(languages))
        unique_label_ids  = sorted(set(true_np))
        col_width         = 18          # 每列宽度
        lang_col_width    = 14          # 语言列宽度
        total_width       = lang_col_width + col_width * (len(unique_label_ids) + 1)

        # ── 表头 ──
        header = f"\n{'  语言 × 类别  准确率统计（acc(样本数)）':=^{total_width}}"
        col_header = f"{'语言':<{lang_col_width}}"
        for lid in unique_label_ids:
            col_header += f"{idx_to_label[lid]:>{col_width}}"
        col_header += f"{'[总体]':>{col_width}}"

        lang_type_lines.append(header)
        lang_type_lines.append(col_header)
        lang_type_lines.append("─" * total_width)

        # ── 每种语言一行 ──
        for lang in unique_languages:
            lang_mask   = (languages == lang)
            row         = f"{lang:<{lang_col_width}}"
            lang_correct = 0
            lang_total   = 0

            for lid in unique_label_ids:
                mask  = lang_mask & (true_np == lid)
                count = int(mask.sum())
                if count == 0:
                    row += f"{'N/A':>{col_width}}"
                else:
                    correct      = int((preds_np[mask] == true_np[mask]).sum())
                    cell_acc     = correct / count
                    cell_str     = f"{cell_acc:.4f}({count})"
                    row         += f"{cell_str:>{col_width}}"
                    lang_correct += correct
                    lang_total   += count

            # 该语言整体
            if lang_total > 0:
                overall_str = f"{lang_correct/lang_total:.4f}({lang_total})"
                row += f"{overall_str:>{col_width}}"
            else:
                row += f"{'N/A':>{col_width}}"

            lang_type_lines.append(row)

        # ── 各类别总体（末行）──
        lang_type_lines.append("─" * total_width)
        total_row = f"{'[全部]':<{lang_col_width}}"
        for lid in unique_label_ids:
            mask  = (true_np == lid)
            count = int(mask.sum())
            if count == 0:
                total_row += f"{'N/A':>{col_width}}"
            else:
                correct   = int((preds_np[mask] == true_np[mask]).sum())
                cell_str  = f"{correct/count:.4f}({count})"
                total_row += f"{cell_str:>{col_width}}"
        overall_str = f"{acc:.4f}({len(true_np)})"
        total_row += f"{overall_str:>{col_width}}"
        lang_type_lines.append(total_row)
        lang_type_lines.append("─" * total_width)

        lang_type_str = "\n".join(lang_type_lines)
        print(lang_type_str)

        # ── 同时保存为单独 CSV ──
        report_dir = os.path.dirname(Config.FUSION_REPORT_PATH)
        csv_path   = os.path.join(report_dir, f"fusion_{model_type}_lang_type_acc.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            cw = csv.writer(cf)
            # 表头行
            cw.writerow(["language"] + [idx_to_label[i] for i in unique_label_ids] + ["overall"])
            # 数据行（每种语言）
            for lang in unique_languages:
                lang_mask    = (languages == lang)
                row_data     = [lang]
                lang_correct = 0
                lang_total   = 0
                for lid in unique_label_ids:
                    mask  = lang_mask & (true_np == lid)
                    count = int(mask.sum())
                    if count == 0:
                        row_data.append("N/A")
                    else:
                        correct      = int((preds_np[mask] == true_np[mask]).sum())
                        row_data.append(f"{correct/count:.4f}({count})")
                        lang_correct += correct
                        lang_total   += count
                row_data.append(f"{lang_correct/lang_total:.4f}({lang_total})"
                                if lang_total > 0 else "N/A")
                cw.writerow(row_data)
            # 总体行
            total_row_data = ["[全部]"]
            for lid in unique_label_ids:
                mask  = (true_np == lid)
                count = int(mask.sum())
                if count == 0:
                    total_row_data.append("N/A")
                else:
                    correct = int((preds_np[mask] == true_np[mask]).sum())
                    total_row_data.append(f"{correct/count:.4f}({count})")
            total_row_data.append(f"{acc:.4f}({len(true_np)})")
            cw.writerow(total_row_data)
        print(f"📋 语言×类别准确率 CSV 已保存: {csv_path}")

    # ===== 保存文本报告 =====
    report_dir  = os.path.dirname(Config.FUSION_REPORT_PATH)
    report_path = os.path.join(report_dir, f"fusion_{model_type}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_type} Fusion Model Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write(report)
        if lang_type_lines:
            f.write("\n\n")
            f.write("\n".join(lang_type_lines))
            f.write("\n")

    # ===== 混淆矩阵 =====
    cm = confusion_matrix(true_np, preds_np)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names, yticklabels=target_names,
                cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_type} Fusion Model Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_dir  = os.path.dirname(Config.FUSION_CM_PATH)
    cm_path = os.path.join(cm_dir, f"fusion_{model_type}_cm.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 分类报告已保存: {report_path}")
    print(f"📈 混淆矩阵已保存: {cm_path}")


# ===== 模型比较函数 =====
def compare_models():
    model_types = ['simple_concat', 'enhanced_concat', 'weighted_fusion', 'attention_fusion']
    results = {}
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"训练 {model_type} 模型")
        print(f"{'='*60}")
        try:
            train_optimized_fusion(model_type)
            results[model_type] = "完成"
        except Exception as e:
            print(f"❌ {model_type} 训练失败: {e}")
            results[model_type] = f"失败: {e}"

    print(f"\n{'='*60}")
    print("模型比较总结")
    print(f"{'='*60}")
    for model_type, status in results.items():
        print(f"{model_type:20} : {status}")


if __name__ == '__main__':
    model_type = 'enhanced_concat'

    if verify_model_structure():
        train_optimized_fusion(model_type)
        # compare_models()
    else:
        print("❌ 模型结构验证失败")