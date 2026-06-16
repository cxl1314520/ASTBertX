import torch
import numpy as np
import os

class Config:
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_PATH = "/home/cxl/malware_behavior_classification"
    DATA_PATH = os.path.join(BASE_PATH, "bert/sentence_vectors_hybrid.jsonl")
    AST_GRAPH_DIR = os.path.join(BASE_PATH, "ast")
    SCRIPT_FOLDER = os.path.join(BASE_PATH, "data")
    PRETRAINED_MODEL = os.path.join(BASE_PATH, "models/LocalModel/bert-base-uncased")

    BERT_MODEL_PATH = os.path.join(BASE_PATH, "modl/bert/bert_mlp.pt")
    AST_MODEL_PATH = os.path.join(BASE_PATH, "modl/ast/ast_only_model.pt")
    FUSION_MODEL_PATH = os.path.join(BASE_PATH, "modl/fusion_model.pt")

    REPORT_SAVE_PATH = os.path.join(BASE_PATH, "output/classification_report.txt")
    CM_SAVE_PATH = os.path.join(BASE_PATH, "output/confusion_matrix.png")
    FUSION_REPORT_PATH = os.path.join(BASE_PATH, "output/fusion_classification_report.txt")
    FUSION_CM_PATH = os.path.join(BASE_PATH, "output/fusion_confusion_matrix.png")
    FUSION_LOSS_LOG= os.path.join(BASE_PATH, "output/fusion_confusion.log")
    EPOCHS = 200
    BATCH_SIZE = 8
    LR = 1e-4
    PATIENCE = 10
    EARLY_STOPPING_PATIENCE = 10