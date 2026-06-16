import os
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==================== 路径配置 ====================
DATA_ROOT  = "/home/xlc/BERTAST/data"
OUTPUT_DIR = "/home/xlc/BERTAST/data/compare_features"

# 5个类别子文件夹名，即为标签（与实际目录名完全一致）
CATEGORIES = ["benign", "denial of service", "file path", "injection", "overflow"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 模型配置 ====================
# local=True  → 本地路径加载
# local=False → HuggingFace 自动下载
MODEL_CONFIGS = {
    "codebert": {
        "path":  "/home/xlc/BERTAST/LocalModel/code_bert",
        "local": True,
    },
    "bert-base": {
        "path":  "/home/xlc/BERTAST/LocalModel/bert-base-uncased",
        "local": True,
    },
    "distilbert": {
        "path":  "/home/xlc/BERTAST/LocalModel/distilbert-base-uncased",
        "local": True,
    },
    "tinybert": {
        "path":  "/home/xlc/BERTAST/LocalModel/TinyBERT_General_4L_312D",
        "local": True,
    },
    "graphcodebert": {
        "path":  "/home/xlc/BERTAST/LocalModel/graphcodebert-base",
        "local": True,
    },
    "longcoder": {
        "path":  "/home/xlc/BERTAST/LocalModel/longcoder-base",
        "local": True,
    },
}


# ==================== 扫描数据目录 ====================
def scan_dataset(data_root, categories):
    """
    扫描 data_root 下各子文件夹，文件夹名即标签。
    返回 [(full_path, label, filename), ...]
    """
    from collections import Counter
    samples = []

    for cat in categories:
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir):
            print(f"⚠️  目录不存在，跳过: {cat_dir}")
            continue
        for fname in sorted(os.listdir(cat_dir)):
            fpath = os.path.join(cat_dir, fname)
            if os.path.isfile(fpath):
                samples.append((fpath, cat, fname))

    print(f"\n共扫描到 {len(samples)} 个文件，类别分布：")
    dist = Counter(s[1] for s in samples)
    for cat, cnt in sorted(dist.items()):
        print(f"  {cat:<28} {cnt} 个文件")

    return samples


# ==================== Mean Pooling 特征提取 ====================
def extract_vector(text, tokenizer, model, max_length=512):
    """
    与原 CodeBERT 脚本 mean pooling 逻辑完全一致。
    返回: numpy array shape=[hidden_dim]
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs       = model(**inputs)
        hidden_states = outputs.last_hidden_state            # [1, seq_len, dim]
        attn_mask     = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden = hidden_states * attn_mask
        denom         = attn_mask.sum(dim=1).clamp(min=1e-9)
        mean_vec      = masked_hidden.sum(dim=1) / denom     # [1, dim]

    return mean_vec.squeeze(0).cpu().numpy()


# ==================== 单模型提取流程 ====================
def run_extraction_for_model(model_name, model_cfg, samples):
    print(f"\n{'='*60}")
    print(f"  模型: {model_name}")
    print(f"  路径: {model_cfg['path']}")
    print(f"{'='*60}")

    load_kwargs = {"local_files_only": True} if model_cfg["local"] else {}
    tokenizer   = AutoTokenizer.from_pretrained(model_cfg["path"], **load_kwargs)
    bert_model  = AutoModel.from_pretrained(model_cfg["path"], **load_kwargs)
    bert_model.to(DEVICE)
    bert_model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_mean.jsonl")

    latencies   = []
    total_count = 0
    skipped     = 0

    with open(out_path, "w", encoding="utf-8") as writer:
        for full_path, label, filename in tqdm(samples, desc=f"[{model_name}]"):
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()

                if not code.strip():
                    skipped += 1
                    continue

                # 计时：单样本推理延迟
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                vec = extract_vector(code, tokenizer, bert_model)

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)

                writer.write(json.dumps({
                    "filename": filename,
                    "label":    label,
                    "vector":   vec.tolist(),
                }) + "\n")
                total_count += 1

            except Exception as e:
                print(f"\n  ❌ 跳过 {filename}，错误：{e}")
                skipped += 1

    # 推理统计
    if latencies:
        avg_latency_ms   = round(sum(latencies) / len(latencies), 3)
        throughput_per_s = round(1000.0 / avg_latency_ms, 2)
    else:
        avg_latency_ms   = float("nan")
        throughput_per_s = float("nan")

    stats = {
        "model":            model_name,
        "total_samples":    total_count,
        "skipped":          skipped,
        "avg_latency_ms":   avg_latency_ms,
        "throughput_per_s": throughput_per_s,
        "output_jsonl":     out_path,
    }

    print(f"\n  ✅ 完成：样本={total_count}，跳过={skipped}")
    print(f"     均值延迟: {avg_latency_ms:.3f} ms/sample")
    print(f"     推理吞吐: {throughput_per_s:.2f} samples/s")

    # 释放显存，防止多模型连跑 OOM
    del bert_model, tokenizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return stats


# ==================== 主入口 ====================
def main():
    print(f"使用设备: {DEVICE}")

    samples = scan_dataset(DATA_ROOT, CATEGORIES)
    if not samples:
        print("❌ 未找到任何样本，请检查 DATA_ROOT 和 CATEGORIES 配置。")
        return

    all_stats = []

    for model_name, model_cfg in MODEL_CONFIGS.items():
        try:
            stats = run_extraction_for_model(model_name, model_cfg, samples)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n⚠️  [{model_name}] 失败，跳过。错误：{e}")
            all_stats.append({
                "model":            model_name,
                "total_samples":    0,
                "skipped":          len(samples),
                "avg_latency_ms":   float("nan"),
                "throughput_per_s": float("nan"),
                "output_jsonl":     "N/A",
                "error":            str(e),
            })

    # 保存推理统计
    stats_path = os.path.join(OUTPUT_DIR, "inference_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # 打印汇总表
    print(f"\n{'='*65}")
    print("  推理效率汇总")
    print(f"{'='*65}")
    print(f"{'模型':<20} {'样本数':>8} {'均值延迟(ms)':>14} {'吞吐(samples/s)':>16}")
    print("-" * 65)
    for s in all_stats:
        lat = f"{s['avg_latency_ms']:.3f}"   if s["avg_latency_ms"]   == s["avg_latency_ms"]   else "N/A"
        thr = f"{s['throughput_per_s']:.2f}" if s["throughput_per_s"] == s["throughput_per_s"] else "N/A"
        print(f"{s['model']:<20} {s['total_samples']:>8} {lat:>14} {thr:>16}")

    print(f"\n推理统计已保存: {stats_path}")
    print(f"JSONL 特征文件目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()