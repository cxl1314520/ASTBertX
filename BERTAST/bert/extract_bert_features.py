import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ========== 配置 ==========
PRETRAINED_MODEL = r"/home/cxl/BERTAST/LocalModel/code_bert"
FILE_PATH_PREFIX = "/home/cxl/BERTAST/data/"
CSV_FILE_PATH = "/home/cxl/BERTAST/data/1.csv"
OUTPUT_DIR = r"/home/cxl/BERTAST/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出文件
OUTPUT_FILES = {
    "cls": os.path.join(OUTPUT_DIR, "sentence_vectors_cls.jsonl"),
    "mean": os.path.join(OUTPUT_DIR, "sentence_vectors_mean.jsonl"),
    "hybrid": os.path.join(OUTPUT_DIR, "sentence_vectors_hybrid.jsonl"),
}

# ========== 初始化模型 ==========
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, local_files_only=True)
model = AutoModel.from_pretrained(PRETRAINED_MODEL, local_files_only=True)
model.to(DEVICE)
model.eval()

# ========== 特征提取函数 ==========
def extract_sentence_vectors(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

        # CLS 向量
        cls_vec = hidden_states[:, 0, :]  # [1, 768]

        # Mean Pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        denom = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_vec = masked_hidden.sum(dim=1) / denom

        # Hybrid 向量
        hybrid_vec = torch.cat([cls_vec, mean_vec], dim=-1)  # [1, 1536]

    return (
        cls_vec.squeeze(0).cpu().numpy(),
        mean_vec.squeeze(0).cpu().numpy(),
        hybrid_vec.squeeze(0).cpu().numpy(),
    )

# ========== 主流程 ==========
def sanitize_file_path(file_path):
    """统一路径分隔符"""
    sanitized_path = file_path.replace("\\", "/")
    return sanitized_path

def main():
    total_count = 0
    writers = {mode: open(path, "w", encoding="utf-8") for mode, path in OUTPUT_FILES.items()}

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        file_paths = df['file'].tolist()
        filenames = df['filename'].tolist()
        categories = df['categories'].tolist()

        for file_path, filename, category in tqdm(zip(file_paths, filenames, categories), desc="提取特征（多分类）"):
            full_file_path = sanitize_file_path(os.path.join(FILE_PATH_PREFIX, file_path))

            if not os.path.exists(full_file_path):
                print(f"❌ 文件不存在：{full_file_path}")
                continue

            try:
                with open(full_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()

                cls_vec, mean_vec, hybrid_vec = extract_sentence_vectors(code)

                # 直接保留原始多分类标签
                item_base = {"filename": filename, "label": str(category)}

                writers["cls"].write(json.dumps({**item_base, "vector": cls_vec.tolist()}) + "\n")
                writers["mean"].write(json.dumps({**item_base, "vector": mean_vec.tolist()}) + "\n")
                writers["hybrid"].write(json.dumps({**item_base, "vector": hybrid_vec.tolist()}) + "\n")

                total_count += 1

            except Exception as e:
                print(f"❌ 跳过 {full_file_path}，错误：{e}")

    finally:
        for f in writers.values():
            f.close()

    print(f"\n✅ 特征提取完成，总样本数：{total_count}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("fatal_error.log", "w", encoding="utf-8") as f:
            f.write(f"主程序崩溃：{str(e)}\n")
        raise
