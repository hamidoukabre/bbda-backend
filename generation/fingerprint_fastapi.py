# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.utils.data import Dataset, DataLoader
from fastapi import UploadFile, File
from zipfile import ZipFile



# ============================================================
# =============== CONFIGURATION & INITIALISATION ============
# ============================================================

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

from utils.utils import crawl_directory, extract_mel_spectrogram
from models.neural_fingerprinter import Neural_Fingerprinter
from generation.faiss_fastapi import build_faiss_index

CONFIG_PATH = os.path.join(project_path, "generation", "config_fingerprint.json")
CONFIG_FAISS_PATH = os.path.join(project_path, "generation", "config_faiss.json")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

SR = config["SR"]
HOP_SIZE = config["HOP SIZE"]
BATCH_SIZE = config["batch size"]
ATTENTION = config["attention"]
OUTPUT_DIR = os.path.join(project_path, config["output dir"])

# ✅ Chemin absolu du fichier de poids
PT_FILE = os.path.normpath(os.path.join(project_path, config["weights"]))
if not os.path.exists(PT_FILE):
    raise FileNotFoundError(f"⚠️ Le fichier du modèle n'existe pas : {PT_FILE}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Neural_Fingerprinter(attention=ATTENTION).to(device)
model.load_state_dict(torch.load(PT_FILE, map_location=device))
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ======================= DATASET ===========================
# ============================================================

class FileDataset(Dataset):
    def __init__(self, file, sr, hop_size):
        self.y, self.F = librosa.load(file, sr=sr)
        self.H = hop_size
        self.dur = self.y.size // self.F
        self._get_spectrograms()

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return torch.from_numpy(self.spectrograms[idx])

    def _get_spectrograms(self):
        self.spectrograms = []
        J = int(np.floor((self.y.size - self.F) / self.H)) + 1
        for j in range(J):
            S = extract_mel_spectrogram(signal=self.y[j * self.H:j * self.H + self.F])
            self.spectrograms.append(S.reshape(1, *S.shape))

# ============================================================
# ======================== API FASTAPI =======================
# ============================================================

app = FastAPI(
    title="Audio Fingerprint + FAISS Auto-Index API",
    description="API pour générer les empreintes audio et mettre à jour automatiquement l'index FAISS.",
    version="3.0"
)

class FolderRequest(BaseModel):
    input_dir: str

class FileRequest(BaseModel):
    input_file: str

# ============================================================
# ==================== ROUTES ===============================
# ============================================================

@app.post("/")
def generate_fingerprints(request: FolderRequest):
    input_dir = request.input_dir
    if not os.path.isdir(input_dir):
        raise HTTPException(status_code=400, detail=f"Dossier non trouvé: {input_dir}")

    all_songs = crawl_directory(input_dir, extension='wav')
    if not all_songs:
        raise HTTPException(status_code=400, detail=f"Aucun fichier .wav trouvé dans {input_dir}")

    to_discard = [
        os.path.basename(song).removesuffix('.npy') + '.wav'
        for song in crawl_directory(OUTPUT_DIR)
    ]
    songs_to_process = [s for s in all_songs if os.path.basename(s) not in to_discard]

    if not songs_to_process:
        return {"message": "Aucun nouveau fichier à fingerprint."}

    fails = 0
    totals = len(songs_to_process)
    for file in tqdm(songs_to_process, desc="Extraction des empreintes", total=totals):
        try:
            process_single_file(file)
        except Exception as e:
            print(f"Erreur avec {file}: {e}")
            fails += 1

    try:
        result = build_faiss_index(CONFIG_FAISS_PATH)
        return {
            "message": "Empreintes générées et index FAISS mis à jour ✅",
            "fingerprint_summary": {"success": totals - fails, "failed": fails},
            "index_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la mise à jour FAISS: {e}")


@app.post("/file")
def fingerprint_single_file(request: FileRequest):
    input_file = request.input_file
    if not os.path.isfile(input_file):
        raise HTTPException(status_code=400, detail=f"Fichier non trouvé: {input_file}")
    if not input_file.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format .wav")

    try:
        output_path = process_single_file(input_file)
        result = build_faiss_index(CONFIG_FAISS_PATH)
        return {
            "message": "Empreinte générée et index FAISS mis à jour ✅",
            "output_file": output_path,
            "index_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")




@app.post("/upload_file")
def upload_fingerprint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format .wav")

    # Sauvegarde temporaire
    tmp_path = os.path.join(OUTPUT_DIR, file.filename)
    try:
        with open(tmp_path, "wb") as f:
            f.write(file.file.read())

        # Générer l'empreinte
        output_path = process_single_file(tmp_path)
        
        # Mettre à jour l'index FAISS
        result = build_faiss_index(CONFIG_FAISS_PATH)

        return {
            "message": "Empreinte générée et index FAISS mis à jour ✅",
            "output_file": output_path,
            "index_result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    finally:
        # ✅ SUPPRIMER le fichier audio temporaire après traitement
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"🗑️ Fichier temporaire supprimé : {tmp_path}")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer {tmp_path}: {e}")





@app.post("/upload_folder")
def upload_folder(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Le fichier doit être au format .zip")

    # Sauvegarde temporaire
    tmp_zip_path = os.path.join(OUTPUT_DIR, file.filename)
    folder_path = tmp_zip_path.replace(".zip", "")
    
    try:
        with open(tmp_zip_path, "wb") as f:
            f.write(file.file.read())

        # Extraction
        os.makedirs(folder_path, exist_ok=True)
        with ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

        # Générer les empreintes pour tous les wav
        all_songs = crawl_directory(folder_path, extension="wav")
        fails = 0
        for wav_file in all_songs:
            try:
                process_single_file(wav_file)
            except Exception as e:
                print(f"Erreur avec {wav_file}: {e}")
                fails += 1

        result = build_faiss_index(CONFIG_FAISS_PATH)
        
        return {
            "message": "Empreintes générées et index FAISS mis à jour ✅",
            "fingerprint_summary": {"success": len(all_songs)-fails, "failed": fails},
            "index_result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    finally:
        # ✅ SUPPRIMER le fichier ZIP et le dossier extrait
        if os.path.exists(tmp_zip_path):
            try:
                os.remove(tmp_zip_path)
                print(f"🗑️ Fichier ZIP supprimé : {tmp_zip_path}")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer {tmp_zip_path}: {e}")
        
        if os.path.exists(folder_path):
            try:
                import shutil
                shutil.rmtree(folder_path)
                print(f"🗑️ Dossier temporaire supprimé : {folder_path}")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer {folder_path}: {e}")

# ============================================================
# =============== FONCTION UTILITAIRE =======================
# ============================================================

def process_single_file(file_path):
    dataset = FileDataset(file=file_path, sr=SR, hop_size=HOP_SIZE)
    if dataset.dur < 1:
        raise ValueError("Durée du fichier trop courte pour générer une empreinte.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    fingerprints = []
    with torch.no_grad():
        for X in loader:
            X = model(X.to(device))
            fingerprints.append(X.cpu().numpy())

    fingerprints = np.vstack(fingerprints)
    output_path = os.path.join(OUTPUT_DIR, os.path.basename(file_path).replace('.wav', '.npy'))
    np.save(output_path, fingerprints)
    return output_path


@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de génération et indexation automatique 🎶🧠"}
