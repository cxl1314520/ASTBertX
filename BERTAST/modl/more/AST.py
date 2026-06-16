import os
import json
import time
import random
import logging
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch_geometric.data import Data as GeoData, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool, SAGEConv

# 你的 Config，请确保包含下面这些字段或在 Config 中添加： 
# AST_GRAPH_DIR, BATCH_SIZE, LR, EPOCHS, AST_MODEL_PATH, REPORT_SAVE_PATH, SEED,
# EARLY_STOPPING_PATIENCE, CACHE_DIR (可选), VALIDATE_EVERY_N_EPOCHS (可选),
# GRAD_ACCUM_STEPS (可选), DEVICE (可选)
from utils.config import Config

# ========== Logging ========== 
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========== 设备选择 ========== 
def get_device():
    if torch.cuda.is_available():
        logger.info(f"🚀 使用 GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    else:
        logger.info("🖥️ 使用 CPU")
        return torch.device("cpu")

Config.DEVICE = getattr(Config, "DEVICE", get_device())
DEVICE = Config.DEVICE

# ========== 随机种子 ========== 
SEED = getattr(Config, "SEED", 42)
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ========== 缓存 / 预处理工具 ========== 
CACHE_DIR = Path(getattr(Config, "CACHE_DIR", os.path.join(os.path.dirname(getattr(Config, "AST_GRAPH_DIR", ".")), "cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
NODE_TYPE_MAP_PATH = CACHE_DIR / "node_type_map.json"
LABEL_MAP_PATH = CACHE_DIR / "label_map.json"
PT_CACHE_DIR = CACHE_DIR / "pt_graphs"
PT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def build_or_load_maps(json_files):
    if NODE_TYPE_MAP_PATH.exists() and LABEL_MAP_PATH.exists():
        with open(NODE_TYPE_MAP_PATH, 'r', encoding='utf-8') as f:
            node_type_map = json.load(f)
        with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        logger.info("已加载 node_type_map 与 label_map（来自缓存）")
        return node_type_map, label_map

    node_types = set()
    labels = set()
    for p in tqdm(json_files, desc="扫描 JSON 文件以建立映射"):
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        for node in data.get("nodes", []):
            node_types.add(node.get("type", "UNK"))
        labels.add(data.get("label", "unknown"))
    node_type_map = {t: i for i, t in enumerate(sorted(node_types))}
    label_map = {lbl: i for i, lbl in enumerate(sorted(labels))}
    with open(NODE_TYPE_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(node_type_map, f, ensure_ascii=False, indent=2)
    with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info("已构建并缓存 node_type_map 与 label_map")
    return node_type_map, label_map

def json_list_in_dir(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.json'):
                files.append(os.path.join(dirpath, fname))
    files.sort()
    return files

# ========== Dataset ========== 
class ASTGraphDataset(Dataset):
    def __init__(self, json_dir, force_recache=False):
        self.json_files = json_list_in_dir(json_dir)  # 获取目录下所有 JSON 文件
        self.node_type_map, self.label_map = build_or_load_maps(self.json_files)  # 获取映射
        self.pt_paths = self.cache_jsons_to_pt(self.json_files, self.node_type_map, self.label_map, force_recache)
        
    def cache_jsons_to_pt(self, json_files, node_type_map, label_map, force_recache=False):
        pt_paths = []
        invalid_count = 0
        for jf in tqdm(json_files, desc="缓存 JSON -> PT"):
            basename = Path(jf).stem
            pt_path = PT_CACHE_DIR / f"{basename}.pt"
            pt_paths.append(str(pt_path))
            if pt_path.exists() and not force_recache:
                continue
            with open(jf, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            # 处理节点特征 - 只使用节点类型
            node_types = [node.get("type", "UNK") for node in data.get("nodes", [])]
            num_nodes = len(node_types)
            
            # 创建节点特征张量 - 1D 张量 [num_nodes]，每个节点一个类型ID
            x = torch.tensor([node_type_map.get(t, 0) for t in node_types], dtype=torch.long)
            
            # 处理边索引 - 确保索引在有效范围内
            edges = data.get("edges", [])
            if edges:
                # 过滤掉超出节点范围的边
                valid_edges = []
                for edge in edges:
                    if (0 <= edge[0] < num_nodes and 0 <= edge[1] < num_nodes):
                        valid_edges.append(edge)
                    else:
                        invalid_count += 1
                
                if valid_edges:
                    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
                    # 再次检查边索引是否有效
                    if edge_index.max() >= num_nodes:
                        logger.warning(f"边索引超出范围: {edge_index.max()} >= {num_nodes}")
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            raw_label = data.get("label", "unknown")
            y = torch.tensor(label_map.get(raw_label, 0), dtype=torch.long)
            raw_filename = data.get("filename", None)
            if raw_filename is None:
                filename = os.path.basename(jf)
            else:
                # 去掉多余路径，仅保留真实文件名
                filename = os.path.basename(raw_filename.strip())
            sample = {
                "x": x,
                "edge_index": edge_index,
                "y": y,
                "language": data.get("language", "unknown"),
                "filename": filename,
                "num_nodes": num_nodes
            }
            torch.save(sample, str(pt_path))
        
        if invalid_count > 0:
            logger.warning(f"跳过了 {invalid_count} 个无效边")
        
        return pt_paths

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.pt_paths[idx])
        return sample

# ========== collate_fn ========== 
def collate_fn(batch):
    ast_graphs = []
    labels = []
    for item in batch:
        # 确保 x 是 1D 张量 [num_nodes]
        x = item['x']
        if x.dim() == 2:
            x = x.squeeze(1)  # 如果是 2D，压缩成 1D
        
        # 验证数据
        num_nodes = x.size(0)
        if item['edge_index'].size(1) > 0:
            max_edge_idx = item['edge_index'].max().item()
            if max_edge_idx >= num_nodes:
                logger.warning(f"边索引 {max_edge_idx} 超出节点范围 {num_nodes}，跳过此图")
                continue
        
        data = GeoData(x=x, edge_index=item['edge_index'], y=item['y'])
        ast_graphs.append(data)
        labels.append(item['y'])
    
    if len(ast_graphs) == 0:
        # 如果所有图都被跳过，创建一个空图
        logger.warning("批次中所有图都被跳过，创建空图")
        data = GeoData(x=torch.zeros(1, dtype=torch.long), 
                      edge_index=torch.empty((2, 0), dtype=torch.long), 
                      y=torch.tensor(0, dtype=torch.long))
        ast_graphs.append(data)
        labels.append(torch.tensor(0, dtype=torch.long))
    
    batch_graph = Batch.from_data_list(ast_graphs)
    labels = torch.stack(labels).squeeze()
    return batch_graph, labels

# ========== DataLoader helper ========== 
def get_dataloader(dataset, batch_size, shuffle):
    num_workers = getattr(Config, "NUM_WORKERS", max(1, (os.cpu_count() or 4) // 2))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True
    )

# ========== 模型 ========== 
class ASTOnlyModel(nn.Module):
    def __init__(self, node_type_num, node_emb_dim, hidden_dim, heads, num_classes, conv_type="gat"):
        super().__init__()
        self.node_embedding = nn.Embedding(node_type_num, node_emb_dim)
        self.conv_type = conv_type.lower()
        if self.conv_type == "gat":
            self.conv1 = GATv2Conv(node_emb_dim, hidden_dim, heads=heads, concat=True, dropout=0.3)
            self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=True, dropout=0.3)
            final_dim = hidden_dim
        else:
            self.conv1 = SAGEConv(node_emb_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            final_dim = hidden_dim

        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # 调试信息
        logger.debug(f"输入 x shape: {x.shape}, dtype: {x.dtype}")
        logger.debug(f"边索引 shape: {edge_index.shape}, max: {edge_index.max().item()}")
        logger.debug(f"批次信息: {batch.shape}")
        
        # 确保 x 是 Long 类型且是 1D
        if x.dtype != torch.long:
            x = x.long()
        
        # 如果 x 是 2D，压缩成 1D 用于 embedding
        if x.dim() == 2:
            x = x.squeeze(1)
        
        # embedding 层
        x = self.node_embedding(x)  # [num_nodes, node_emb_dim]
        logger.debug(f"Embedding 后 x shape: {x.shape}")
        
        # 图卷积层
        if self.conv_type == "gat":
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        
        # 全局池化
        x = self.pool(x, batch)
        logger.debug(f"池化后 x shape: {x.shape}")
        
        # 分类器
        return self.classifier(x)

# ========== 数据集划分 ==========
def split_dataset(dataset, test_size=0.2, random_state=SEED):
    """划分数据集为训练集和测试集"""
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=[dataset[i]['y'].item() for i in indices]
    )
    
    # 创建训练集和测试集的子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    logger.info(f"数据集划分: 训练集 {len(train_dataset)} 样本, 测试集 {len(test_dataset)} 样本")
    
    # 检查类别分布
    train_labels = [dataset[i]['y'].item() for i in train_indices]
    test_labels = [dataset[i]['y'].item() for i in test_indices]
    
    logger.info(f"训练集类别分布: {dict(Counter(train_labels))}")
    logger.info(f"测试集类别分布: {dict(Counter(test_labels))}")
    
    return train_dataset, test_dataset

# ========== 训练 & 验证 ========== 
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_graph, labels in loader:
            batch_graph = batch_graph.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc, all_preds, all_labels

def train():
    ast_dir = getattr(Config, "AST_GRAPH_DIR")
    dataset = ASTGraphDataset(json_dir=ast_dir, force_recache=True)  # 强制重新缓存
    
    # 划分数据集为训练集和测试集
    train_dataset, test_dataset = split_dataset(dataset, test_size=0.2)
    
    # 创建数据加载器
    train_loader = get_dataloader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 测试第一个批次
    for batch_graph, labels in train_loader:
        print(f"训练批次图: x shape: {batch_graph.x.shape}, 边索引 shape: {batch_graph.edge_index.shape}")
        print(f"训练批次信息: {batch_graph.batch.shape}, 标签: {labels.shape}")
        break
    
    # 模型创建
    use_gat = getattr(Config, "USE_GAT", True)
    conv_type = "gat" if use_gat else "sage"
    model = ASTOnlyModel(
        node_type_num=len(dataset.node_type_map),
        node_emb_dim=getattr(Config, "NODE_EMB_DIM", 64),
        hidden_dim=getattr(Config, "GAT_HIDDEN_DIM", 128),
        heads=getattr(Config, "GAT_HEADS", 4),
        num_classes=len(dataset.label_map),
        conv_type=conv_type
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(Config, "LR", 1e-3), eps=1e-08)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # 训练过程
    best_val_loss = float('inf')
    no_improve_count = 0
    best_val_preds, best_val_labels = None, None
    best_test_preds, best_test_labels = None, None

    EPOCHS = getattr(Config, "EPOCHS", 50)
    grad_accum_steps = getattr(Config, "GRAD_ACCUM_STEPS", 1)
    validate_every = getattr(Config, "VALIDATE_EVERY_N_EPOCHS", 1)
    early_stop_patience = getattr(Config, "EARLY_STOPPING_PATIENCE", 8)

    logger.info("开始训练 AST-only 多分类模型...")
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)
        for i, (batch_graph, labels) in pbar:
            batch_graph = batch_graph.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
                loss = criterion(outputs, labels) / grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss.item() * labels.size(0) * grad_accum_steps
            global_step += 1
            pbar.set_postfix({"loss": f"{(total_train_loss / ((i + 1) * train_loader.batch_size)):.4f}"})

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        if (epoch + 1) % validate_every == 0:
            # 在测试集上验证
            avg_test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}")

            if avg_test_loss < best_val_loss:
                best_val_loss = avg_test_loss
                no_improve_count = 0
                save_path = getattr(Config, "AST_MODEL_PATH", "best_ast_model.pt")
                torch.save(model.to("cpu").state_dict(), save_path)
                model.to(DEVICE)
                best_val_preds = test_preds.copy()
                best_val_labels = test_labels.copy()
                logger.info(f"✅ 新的最佳模型已保存: {save_path}")
            else:
                no_improve_count += 1
                logger.info(f"早停计数: {no_improve_count}/{early_stop_patience}")
                if no_improve_count >= early_stop_patience:
                    logger.info("⏹️ 早停触发，终止训练")
                    break

    # 最终测试
    if best_val_preds is None or best_val_labels is None:
        avg_test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
        best_val_preds = test_preds
        best_val_labels = test_labels

    logger.info("\n📝 最优模型测试集报告：")
    target_names = list(sorted(dataset.label_map.keys(), key=lambda k: dataset.label_map[k]))
    report = classification_report(best_val_labels, best_val_preds, target_names=target_names, digits=4)
    cm = confusion_matrix(best_val_labels, best_val_preds)
    logger.info("分类报告：\n" + report)
    logger.info("混淆矩阵：\n" + str(cm))

    # 保存报告
    report_save_path = getattr(Config, "REPORT_SAVE_PATH", os.path.join(str(CACHE_DIR), "test_report.txt"))
    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write("Test Set Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Test Set Confusion Matrix:\n")
        f.write(str(cm))
        f.write(f"\n\nDataset Info:\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Total samples: {len(dataset)}\n")
    logger.info(f"✅ 测试报告已保存: {report_save_path}")

    # 绘图并保存混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title("测试集混淆矩阵")
    plt.tight_layout()
    cm_image_path = os.path.join(os.path.dirname(report_save_path), "test_confusion_matrix.png")
    plt.savefig(cm_image_path, dpi=300)
    logger.info(f"✅ 混淆矩阵图已保存: {cm_image_path}")

if __name__ == "__main__":
    start_time = time.time()
    train()
    logger.info(f"总耗时: {time.time() - start_time:.1f}s")