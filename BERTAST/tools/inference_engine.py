"""
ASTBertX Inference Engine
=========================
Shared prediction backend used by all three defense layers.

Fast mode  (BERT-only):  GraphCodeBERT → mean-pool → XGBoost   (~3-5 s first call, <0.5 s cached)
Full mode  (BERT+AST) :  GraphCodeBERT + AST-GATv2 → Fusion → XGBoost  (higher accuracy)

Usage
-----
from tools.inference_engine import ASTBertXInference

engine = ASTBertXInference(base_path="/path/to/BERTAST")
result = engine.predict_fast("exploit.py")
result = engine.predict_full("exploit.py")
# result = {"file": "...", "label": "Injection", "label_id": 2,
#            "confidence": 0.91, "probabilities": {...}, "mode": "fast"}
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F

BERTAST_ROOT = str(Path(__file__).resolve().parent.parent)
if BERTAST_ROOT not in sys.path:
    sys.path.insert(0, BERTAST_ROOT)

logger = logging.getLogger(__name__)

LABEL_MAP: Dict[int, str] = {
    0: "Benign",
    1: "Overflow",
    2: "Injection",
    3: "Denial of Service",
    4: "File Path",
}
THREAT_IDS = {1, 2, 3, 4}

EXTENSION_TO_LANGUAGE = {
    "c": "c", "py": "python", "rb": "ruby",
    "pl": "perl", "php": "php", "html": "html",
}
SUPPORTED_EXTENSIONS = set(EXTENSION_TO_LANGUAGE.keys())

# ─── AST helpers (mirrors data/结构树.py) ────────────────────────────────────

MAX_DEPTH = 500
SNIPPET_MAX = 30
LOOP_TYPES = {
    "for_statement", "while_statement", "do_statement",
    "for_each_statement", "foreach_statement",
}
PATH_TOKENS = ["../", "..\\", "%2e%2e", "%252e%252e"]


def _simple_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(s)
    return -sum((v / len(s)) * math.log2(v / len(s)) for v in cnt.values())


def _has_path_token(raw: str) -> bool:
    try:
        decoded = urllib.parse.unquote(raw)
    except Exception:
        decoded = raw
    lower = (raw + " " + decoded).lower()
    return any(t in lower for t in PATH_TOKENS)


def _traverse(node, nodes, edges, code, parent_id=None, depth=0, anc_script=None):
    if anc_script is None:
        anc_script = []
    if depth > MAX_DEPTH:
        return
    nid = len(nodes)
    snippet = code[node.start_byte:node.end_byte].strip()
    sl = snippet.lower()
    in_script = anc_script[-1] if anc_script else ("<script" in sl or "</script" in sl)
    anc_script.append(in_script)
    nodes.append({
        "type": node.type,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "depth": depth,
        "code_snippet": snippet[:SNIPPET_MAX],
        "num_children": len(node.children),
        "node_degree": (1 if parent_id is not None else 0) + len(node.children),
        "contains_path_traversal_tokens": _has_path_token(snippet),
        "is_loop": node.type in LOOP_TYPES,
        "is_in_script_context": bool(in_script),
    })
    if parent_id is not None:
        edges.append((parent_id, nid))
    for child in node.children:
        _traverse(child, nodes, edges, code, nid, depth + 1, anc_script)
    anc_script.pop()


def _parse_ast(filepath: str) -> Optional[dict]:
    ext = Path(filepath).suffix.lstrip(".")
    lang_key = EXTENSION_TO_LANGUAGE.get(ext)
    if lang_key is None:
        return None
    try:
        from tree_sitter import Parser
        from tree_sitter_languages import get_language
        lang = get_language(lang_key)
        parser = Parser()
        parser.set_language(lang)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read(1024 * 1024)
        tree = parser.parse(code.encode("utf-8"))
        nodes: list = []
        edges: list = []
        _traverse(tree.root_node, nodes, edges, code)
        return {"nodes": nodes, "edges": edges}
    except Exception as exc:
        logger.warning("AST parse failed for %s: %s", filepath, exc)
        return None


# ─── Model cache (singleton per process) ────────────────────────────────────

class _Cache:
    tokenizer = None
    bert_model = None
    xgb_bert = None
    ast_model = None
    fusion_model = None
    node_type_map: Optional[dict] = None
    num_node_types: int = 0


# ─── Main engine ─────────────────────────────────────────────────────────────

class ASTBertXInference:
    """
    Unified inference engine for ASTBertX.

    Parameters
    ----------
    base_path : str
        Absolute path to the BERTAST/ directory.
    device : str | None
        "cuda" / "cpu" / None (auto-detect).
    confidence_threshold : float
        Minimum confidence to report a threat (used by callers, not enforced here).
    """

    def __init__(
        self,
        base_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ):
        self.base_path = Path(base_path)
        self.confidence_threshold = confidence_threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Resolve model paths relative to base_path
        self._model_dir = self.base_path / "output"
        self._local_model = str(self.base_path / "LocalModel" / "graphcodebert-base")
        self._xgb_path = str(self._model_dir / "graphcodebert_best_xgb.pkl")
        self._ast_model_path = str(self._model_dir / "ast_only_model.pt")
        self._fusion_path = str(self._model_dir / "fusion_model_enhanced_concat.pth")
        self._node_type_map_path = str(self.base_path / "cache" / "node_type_map.json")

    # ── private loaders ──────────────────────────────────────────────────────

    def _get_bert(self):
        if _Cache.tokenizer is None:
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading GraphCodeBERT from %s …", self._local_model)
            _Cache.tokenizer = AutoTokenizer.from_pretrained(
                self._local_model, local_files_only=True)
            _Cache.bert_model = AutoModel.from_pretrained(
                self._local_model, local_files_only=True).to(self.device)
            _Cache.bert_model.eval()
        return _Cache.tokenizer, _Cache.bert_model

    def _get_xgb(self):
        if _Cache.xgb_bert is None:
            logger.info("Loading XGBoost classifier from %s …", self._xgb_path)
            _Cache.xgb_bert = joblib.load(self._xgb_path)
        return _Cache.xgb_bert

    def _get_ast_model(self):
        if _Cache.ast_model is None:
            from modl.more.AST import ASTOnlyModel
            if _Cache.node_type_map is None:
                with open(self._node_type_map_path, "r", encoding="utf-8") as f:
                    _Cache.node_type_map = json.load(f)
            _Cache.num_node_types = len(_Cache.node_type_map)
            model = ASTOnlyModel(
                node_type_num=_Cache.num_node_types,
                node_emb_dim=64,
                hidden_dim=128,
                heads=4,
                num_classes=len(LABEL_MAP),
                conv_type="gat",
            ).to(self.device)
            model.load_state_dict(
                torch.load(self._ast_model_path, map_location=self.device))
            model.eval()
            _Cache.ast_model = model
        return _Cache.ast_model, _Cache.node_type_map

    def _get_fusion(self):
        if _Cache.fusion_model is None:
            from modl.more.Decision_Fusionxgb import EnhancedFusionModel
            n = len(LABEL_MAP)
            model = EnhancedFusionModel(n, n, 256, n).to(self.device)
            model.load_state_dict(
                torch.load(self._fusion_path, map_location=self.device))
            model.eval()
            _Cache.fusion_model = model
        return _Cache.fusion_model

    # ── BERT embedding ───────────────────────────────────────────────────────

    def _bert_vec(self, code: str) -> np.ndarray:
        tokenizer, bert_model = self._get_bert()
        inputs = tokenizer(
            code, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(self.device)
        with torch.no_grad():
            out = bert_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return vec.cpu().numpy()  # [1, 768]

    # ── AST logits ───────────────────────────────────────────────────────────

    def _ast_logits(self, filepath: str) -> Optional[torch.Tensor]:
        ast_data = _parse_ast(filepath)
        if ast_data is None:
            return None
        ast_model, node_type_map = self._get_ast_model()

        node_types = [n.get("type", "UNK") for n in ast_data["nodes"]]
        x = torch.tensor(
            [node_type_map.get(t, 0) for t in node_types], dtype=torch.long)

        edges = ast_data.get("edges", [])
        num_nodes = len(node_types)
        valid = [(s, d) for s, d in edges if 0 <= s < num_nodes and 0 <= d < num_nodes]
        if valid:
            ei = torch.tensor(valid, dtype=torch.long).t().contiguous()
        else:
            ei = torch.empty((2, 0), dtype=torch.long)

        from torch_geometric.data import Data as GeoData, Batch
        graph = Batch.from_data_list(
            [GeoData(x=x, edge_index=ei,
                     y=torch.tensor(0, dtype=torch.long))]
        ).to(self.device)

        with torch.no_grad():
            logits = ast_model(graph.x, graph.edge_index, graph.batch)
        return logits.cpu()  # [1, NUM_CLASSES]

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_fast(self, filepath: str) -> dict:
        """BERT-only prediction. ~3-5 s first call, <0.5 s cached."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        bert_vec = self._bert_vec(code)
        xgb = self._get_xgb()
        proba = xgb.predict_proba(bert_vec)[0]
        label_id = int(np.argmax(proba))
        return {
            "file": filepath,
            "label": LABEL_MAP[label_id],
            "label_id": label_id,
            "confidence": float(proba[label_id]),
            "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(proba)},
            "is_threat": label_id in THREAT_IDS,
            "mode": "fast",
        }

    def predict_full(self, filepath: str) -> dict:
        """BERT + AST fusion prediction. Falls back to fast if AST fails."""
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        bert_vec = self._bert_vec(code)
        xgb = self._get_xgb()
        bert_proba = xgb.predict_proba(bert_vec)[0]
        bert_logits = torch.tensor(
            np.log(bert_proba + 1e-8), dtype=torch.float32).unsqueeze(0)

        ast_logits = self._ast_logits(filepath)
        if ast_logits is None:
            logger.warning("%s: AST parse failed, falling back to fast mode", filepath)
            label_id = int(np.argmax(bert_proba))
            return {
                "file": filepath,
                "label": LABEL_MAP[label_id],
                "label_id": label_id,
                "confidence": float(bert_proba[label_id]),
                "probabilities": {LABEL_MAP[i]: float(p)
                                  for i, p in enumerate(bert_proba)},
                "is_threat": label_id in THREAT_IDS,
                "mode": "fast_fallback",
            }

        fusion = self._get_fusion()
        with torch.no_grad():
            logits, _, _ = fusion(
                bert_logits.to(self.device), ast_logits.to(self.device))
            proba = F.softmax(logits, dim=1).cpu().numpy()[0]
        label_id = int(np.argmax(proba))
        return {
            "file": filepath,
            "label": LABEL_MAP[label_id],
            "label_id": label_id,
            "confidence": float(proba[label_id]),
            "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(proba)},
            "is_threat": label_id in THREAT_IDS,
            "mode": "full",
        }

    def is_supported(self, filepath: str) -> bool:
        return Path(filepath).suffix.lstrip(".").lower() in SUPPORTED_EXTENSIONS
