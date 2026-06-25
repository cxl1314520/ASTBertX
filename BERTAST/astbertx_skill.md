# ASTBertX Skill

You are helping the user operate the **ASTBertX** framework — a multilingual exploit-script detection system that fuses GraphCodeBERT sequence semantics with AST-GATv2 structural semantics, then classifies via XGBoost.

## Project structure

```
BERTAST/
├── data/
│   ├── 1.csv                     # Dataset index (columns: file, filename, categories, extension)
│   └── 结构树.py                 # AST graph generator (tree-sitter)
├── bert/
│   ├── extract_bert_features.py  # BERT feature extractor (CodeBERT / GraphCodeBERT)
│   └── train.py                  # Multi-BERT comparison trainer (codebert/graphcodebert/…)
├── modl/
│   ├── ast/
│   │   ├── AST.py                # AST-GATv2 model (trains on ast/ graph JSON files)
│   │   └── utils/config.py       # AST module config
│   ├── bert/
│   │   └── utils/config.py       # BERT module config
│   └── more/
│       ├── AST.py                # AST dataset helpers (reused by fusion)
│       ├── train_bert_mlp.py     # GraphCodeBERT + MLP trainer
│       ├── Decision_Fusionxgb.py # XGBoost fusion trainer (main fusion entry)
│       └── utils/config.py       # Fusion module config  ← primary config to edit
└── output/                       # All trained models + reports go here
```

## Five-class exploit categories

| ID | Label |
|----|-------|
| 0  | Benign |
| 1  | Overflow |
| 2  | Injection |
| 3  | Denial of Service |
| 4  | File Path |

Supported languages: Python · Perl · C · Ruby · HTML · PHP

---

## Step-by-step pipeline

Before running any step, open `BERTAST/modl/more/utils/config.py` and set `BASE_PATH` to the absolute path of the `BERTAST/` folder. All other paths derive from it.

### Step 0 — Install dependencies

```bash
cd BERTAST
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio
pip install transformers torch-geometric xgboost scikit-learn
pip install tree-sitter tree-sitter-languages matplotlib seaborn tqdm joblib pandas
```

Download pretrained models:
```python
from huggingface_hub import snapshot_download
snapshot_download("microsoft/graphcodebert-base", local_dir="LocalModel/graphcodebert-base")
snapshot_download("microsoft/codebert-base",      local_dir="LocalModel/code_bert")
```

### Step 1 — Prepare the dataset

Option A – HuggingFace:
```python
from datasets import load_dataset
ds = load_dataset("wwe123/Exploit-DB-EX")
```

Option B – Exploit-DB manual download. The CSV (`data/1.csv`) must have columns:
- `file` — relative path to the script file under `data/`
- `filename` — bare filename
- `categories` — one of the five labels above
- `extension` — file extension (py / pl / c / rb / html / php)

### Step 2 — Generate AST graphs

Parses every script in `data/1.csv` with tree-sitter and writes one JSON per file under `ast/`.

```bash
cd BERTAST
python data/结构树.py
```

Each JSON has keys: `filename`, `label`, `language`, `nodes` (list of node dicts), `edges` (list of [src,dst] pairs).

### Step 3 — Extract GraphCodeBERT features

Reads scripts listed in `data/1.csv`, runs GraphCodeBERT, writes mean-pooled embeddings to `bert/graphcodebert_mean.jsonl`.

```bash
cd BERTAST
python bert/extract_bert_features.py
```

Each JSONL line: `{"filename": "…", "label": "…", "vector": […768 floats…]}`.

### Step 4 — Train the AST-GATv2 model

Reads graph JSONs from `ast/`, trains a GATv2 graph network, saves best checkpoint to `output/ast_only_model.pt`.

```bash
cd BERTAST/modl/ast
python AST.py
```

Key config fields in `modl/ast/utils/config.py`:
- `AST_GRAPH_DIR` — path to the `ast/` folder
- `AST_MODEL_PATH` — save path for the trained model
- `EPOCHS`, `BATCH_SIZE`, `LR`, `EARLY_STOPPING_PATIENCE`

### Step 5 — Train the XGBoost fusion model

Loads GraphCodeBERT logits + AST-GATv2 logits, projects them into a shared space, trains an `EnhancedFusionModel`, and produces an XGBoost classifier on top of the fused features.

```bash
cd BERTAST/modl/more
python Decision_Fusionxgb.py
```

Outputs written to `output/`:
- `fusion_enhanced_concat_xgb.pkl` — final XGBoost classifier
- `fusion_model.pt` (or `fusion_enhanced_concat_xgb_enhanced_concat.pth`) — fusion projection network
- `fusion_enhanced_concat_report.txt` — classification report
- `fusion_enhanced_concat_cm.png` — confusion matrix
- `fusion_enhanced_concat_lang_type_acc.csv` — per-language × per-category accuracy

### Step 6 — Evaluate / compare BERT backbones (optional)

Trains MLP classifiers on top of several BERT-family encoders and plots a comparison chart.

```bash
cd BERTAST/bert
python train.py
```

---

## Running inference on a new code sample

No dedicated inference script exists yet. Use this pattern:

```python
import torch, json, joblib, numpy as np
from transformers import AutoTokenizer, AutoModel
from modl.more.AST import ASTOnlyModel
from modl.more.Decision_Fusionxgb import EnhancedFusionModel
from modl.more.utils.config import Config

LABEL_MAP = {0: "Benign", 1: "Overflow", 2: "Injection",
             3: "Denial of Service", 4: "File Path"}
NUM_CLASSES = 5

# 1. BERT feature
tokenizer = AutoTokenizer.from_pretrained(Config.PRETRAINED_MODEL, local_files_only=True)
bert_model = AutoModel.from_pretrained(Config.PRETRAINED_MODEL, local_files_only=True).to(Config.DEVICE)
bert_model.eval()

with open("your_script.py") as f:
    code = f.read()

inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(Config.DEVICE)
with torch.no_grad():
    out = bert_model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    bert_vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)  # [1, 768]

# 2. XGBoost logits from BERT vec
xgb = joblib.load(Config.BERT_MODEL_PATH.replace(".pt", "_xgb.pkl"))
bert_logits = torch.tensor(np.log(xgb.predict_proba(bert_vec.cpu().numpy()) + 1e-8))

# 3. AST feature (requires AST JSON generated by 结构树.py for this file)
# ast_logits = ... (run AST.py inference on the file's graph JSON)

# 4. Fusion
fusion = EnhancedFusionModel(NUM_CLASSES, NUM_CLASSES, 256, NUM_CLASSES).to(Config.DEVICE)
fusion.load_state_dict(torch.load(Config.FUSION_MODEL_PATH.replace(".pt", "_enhanced_concat.pth")))
fusion.eval()
with torch.no_grad():
    logits, _, _ = fusion(bert_logits.unsqueeze(0).to(Config.DEVICE),
                          ast_logits.unsqueeze(0).to(Config.DEVICE))
    pred = torch.argmax(logits, dim=1).item()
print("Predicted:", LABEL_MAP[pred])
```

---

## Common issues

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: tree_sitter_languages` | `pip install tree-sitter-languages` |
| `FileNotFoundError` on model paths | Update `BASE_PATH` in `modl/more/utils/config.py` |
| Edge index out of range | Normal for malformed scripts; they are auto-filtered |
| CUDA OOM | Reduce `BATCH_SIZE` in config |
| `XGBoostBERTModel 未训练` | Run Step 5 first; XGBoost is trained and saved automatically |
| Data alignment mismatch | Ensure `1.csv` row order matches between BERT and AST runs |

---

## What to do when invoked

1. Ask the user which step they need help with (or if they want the full pipeline).
2. Check whether `modl/more/utils/config.py` has the correct `BASE_PATH`.
3. Verify prerequisite outputs exist before running a later step.
4. Run the appropriate command(s) and report results.
5. If errors occur, diagnose using the common-issues table above.
