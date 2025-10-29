# -*- coding: utf-8 -*-
"""
faiss_index_fastapi.py
API FastAPI pour construire un index FAISS à partir d'empreintes audio (.npy)
Basé sur la version script CLI d'origine.
"""

import os
import sys
import json
import faiss
import numpy as np
from tqdm import tqdm
from fastapi import FastAPI, HTTPException # pyright: ignore[reportMissingImports]
from pydantic import BaseModel

# ============================================================
# =============== CHEMIN DU PROJET ============================
# ============================================================

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from utils.utils import crawl_directory


# ============================================================
# =============== FONCTION DE CONSTRUCTION ====================
# ============================================================

def build_faiss_index(config_path: str):
    """
    Construit un index FAISS à partir d'un fichier de configuration JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Charger le fichier JSON
    with open(config_path, "r") as f:
        args = json.load(f)

    input_dir = os.path.join(project_path, args["input_dir"])
    output_dir = os.path.join(project_path, args["output_dir"])
    name = args["name"]
    index_str = args["index"]
    d = args["d"]

    os.makedirs(output_dir, exist_ok=True)

    # Choix du type d'index
    if index_str == "IVF":
        quantizer = faiss.IndexFlatIP(128)
        nlist, M, nbits = 256, 64, 8
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)
    else:
        index = faiss.index_factory(d, index_str, faiss.METRIC_INNER_PRODUCT)

    # Lecture des empreintes (.npy)
    fingerprints = crawl_directory(input_dir, extension="npy")
    if not fingerprints:
        raise ValueError(f"Aucune empreinte trouvée dans {input_dir}")

    total_fingerprints = 0
    for f in tqdm(fingerprints, desc="Comptage des empreintes"):
        x = np.load(f)
        total_fingerprints += x.shape[0]

    print(f"Total fingerprints: {total_fingerprints}\nCreating index...")

    xb = np.zeros(shape=(total_fingerprints, d), dtype=np.float32)
    json_correspondence = {}

    i = 0
    for f in tqdm(fingerprints, desc="Chargement des fichiers .npy"):
        x = np.load(f)
        size = x.shape[0]
        xb[i:i + size] = x
        json_correspondence[i] = os.path.basename(f).removesuffix(".npy")
        i += size

    print("Training index...")
    index.train(xb)
    index.add(xb)

    # Sauvegarde
    json_path = os.path.join(output_dir, name + ".json")
    index_path = os.path.join(output_dir, name + ".index")

    with open(json_path, "w") as f:
        json.dump(json_correspondence, f, indent=2)
    faiss.write_index(index, index_path)

    print("✅ Index terminé et sauvegardé.")
    return {
        "total_vectors": total_fingerprints,
        "index_file": index_path,
        "json_file": json_path,
        "dimension": d
    }


# ============================================================
# ======================= API FASTAPI =========================
# ============================================================

app = FastAPI(
    title="FAISS Index Builder API",
    description="API pour construire un index FAISS à partir de fichiers .npy (empreintes audio)",
    version="1.0"
)

class ConfigRequest(BaseModel):
    config_path: str


@app.post("/build_index")
def build_index(request: ConfigRequest):
    """
    Endpoint principal : construit l'index FAISS à partir d'un fichier config JSON.
    """
    try:
        result = build_faiss_index(request.config_path)
        return {
            "message": "Index FAISS construit avec succès 🎯",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de construction d'index FAISS 🧠"}


# ============================================================
# ==================== LANCEMENT SERVEUR =====================
# ============================================================

# À exécuter avec :
# uvicorn faiss_fastapi:app --host 0.0.0.0 --port 8001 --reload
