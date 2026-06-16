import os
import json
import logging
from tree_sitter import Language, Parser
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import pandas as pd
import urllib.parse
import math
from collections import Counter

MAX_DEPTH = 500
SNIPPET_MAX_LEN = 30

SRC_DIR = r"/home/cxl/malware_behavior_classification/data"
OUT_DIR = r"/home/cxl/malware_behavior_classification/ast"
SO_PATH = r"/home/cxl/malware_behavior_classification/old/astgeneration/tree-sitter-languages/my-languages.so"
CSV_PATH = r"/home/cxl/malware_behavior_classification/data/1.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("ast_processing.log", maxBytes=10 * 1024 * 1024, backupCount=5),
        logging.StreamHandler()
    ]
)

EXTENSION_TO_LANGUAGE = {
    "c": "c", "py": "python", "rb": "ruby", "pl": "perl",
    "php": "php", "html": "html"
}

from tree_sitter_languages import get_language
SUPPORTED_LANGUAGES = {
    lang: get_language(lang) or Language(SO_PATH, lang)
    for lang in EXTENSION_TO_LANGUAGE.values()
    if get_language(lang) or Language(SO_PATH, lang)
}

parser_cache = {}
def get_parser(lang_key):
    if lang_key not in parser_cache:
        parser = Parser()
        parser.set_language(SUPPORTED_LANGUAGES[lang_key])
        parser_cache[lang_key] = parser
    return parser_cache[lang_key]

LOOP_NODE_TYPES = {
    "for_statement", "while_statement", "do_statement", "for_each_statement",
    "enhanced_for_statement", "range_for_statement", "foreach_statement",
    "for", "while", "do", "repeat_statement"
}

PATH_TRAVERSAL_TOKENS = ["../", "..\\", "%2e%2e", "%252e%252e", "%2e%2e%2f", "%2e%2e%5c"]

def simple_entropy(s: str) -> float:
    if not s:
        return 0.0
    cnt = Counter(s)
    probs = [v/len(s) for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs)

def contains_path_tokens(raw: str) -> bool:
    if not raw:
        return False
    try:
        decoded = urllib.parse.unquote(raw)
    except Exception:
        decoded = raw
    lower = (raw + " " + decoded).lower()
    return any(tok in lower for tok in PATH_TRAVERSAL_TOKENS)

def traverse_ast(node, node_list, edge_list, code_str, parent_id=None, depth=0, parent_node=None, ancestor_script_flags=None):
    if ancestor_script_flags is None:
        ancestor_script_flags = []

    if depth > MAX_DEPTH:
        return

    node_id = len(node_list)
    snippet = code_str[node.start_byte:node.end_byte].strip()
    snippet_lower = snippet.lower()
    num_children = len(node.children)

    contains_path_traversal = contains_path_tokens(snippet)
    is_loop_flag = node.type in LOOP_NODE_TYPES

    in_script_context = ancestor_script_flags[-1] if ancestor_script_flags else ("<script" in snippet_lower or "</script" in snippet_lower)
    ancestor_script_flags.append(in_script_context)

    node_entry = {
        "type": node.type,
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "depth": depth,
        "code_snippet": snippet[:SNIPPET_MAX_LEN],
        "num_children": num_children,
        "node_degree": (1 if parent_node else 0) + num_children,
        "contains_path_traversal_tokens": contains_path_traversal,
        "is_loop": bool(is_loop_flag),
        "is_in_script_context": bool(in_script_context),
    }
    node_list.append(node_entry)

    if parent_id is not None:
        edge_list.append((parent_id, node_id))

    for child in node.children:
        traverse_ast(child, node_list, edge_list, code_str, parent_id=node_id, depth=depth+1, parent_node=node, ancestor_script_flags=ancestor_script_flags)
    ancestor_script_flags.pop()

def safe_write_json(data, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(path), delete=False) as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
            tmp_name = tmp_file.name
        os.replace(tmp_name, path)
        return True
    except Exception as e:
        logging.error(f"写入JSON失败 {path}: {e}", exc_info=True)
        return False

def process_file(args):
    fpath, rel_path, label, lang_key = args
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read(1024*1024)
        parser = get_parser(lang_key)
        tree = parser.parse(code.encode("utf-8"))
        root = tree.root_node

        node_list, edge_list = [], []
        traverse_ast(root, node_list, edge_list, code)

        if not node_list:
            logging.warning(f"空AST节点，跳过文件: {fpath}")
            return None

        graph = {
            "filename": rel_path,
            "label": label,
            "language": lang_key,
            "nodes": node_list,
            "edges": edge_list,
            "node_count": len(node_list),
            "edge_count": len(edge_list)
        }
        return graph, rel_path
    except Exception as e:
        logging.error(f"处理文件失败 {fpath}: {e}", exc_info=True)
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total_count = 0
    chunk_size = 1000

    for chunk in pd.read_csv(CSV_PATH, chunksize=chunk_size):
        tasks = []
        for _, row in chunk.iterrows():
            rel_path = str(row["file"]).replace("\\","/")
            fpath = os.path.join(SRC_DIR, rel_path)
            label = row.get("categories", None)
            language = str(row.get("extension","")).lower()
            lang_key = EXTENSION_TO_LANGUAGE.get(language)
            if lang_key in SUPPORTED_LANGUAGES:
                tasks.append((fpath, rel_path, label, lang_key))

        if not tasks:
            continue

        count = 0
        max_workers = max(1, os.cpu_count())
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="生成AST图"):
                result = future.result()
                if result:
                    graph, rel_path = result
                    out_name = rel_path + ".json"
                    out_path = os.path.join(OUT_DIR, out_name)
                    if safe_write_json(graph, out_path):
                        count += 1
        total_count += count
        logging.info(f"当前批处理完成，生成图文件数：{count}")

    logging.info(f"✅ 全部处理完成，共生成图文件数：{total_count}")

if __name__ == "__main__":
    main()
