# -*- coding: utf-8 -*-
"""
🎶 Unified Audio Monitoring API
Combinaison de :
 - /fingerprint → génération & indexation d’empreintes audio
 - /monitoring → détection & identification audio en temps réel
 - /radios → surveillance automatique des flux radio
"""

import uvicorn
import logging
import threading
import requests
import time
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ======================================================
# CONFIGURATION LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("UnifiedAPI")

# ======================================================
# IMPORT DES MODULES
# ======================================================
from generation.fingerprint_fastapi import app as fingerprint_app
from api.app_fastapi import app as monitoring_app
from api.radio_fastapi import app as radio_surveillance_app, MonitoringManager

# ======================================================
# APPLICATION PRINCIPALE
# ======================================================
app = FastAPI(
    title="🎧 Unified Audio Monitoring API",
    description=(
        "API unifiée pour la surveillance audio :\n"
        "- Génération d’empreintes audio (/fingerprint)\n"
        "- Détection audio temps réel (/monitoring)\n"
        "- Surveillance radio automatique (/radios)"
    ),
    version="3.0.0"
)

# ------------------------------------------------------
# Middleware global (CORS)
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Montage des sous-applications
# ------------------------------------------------------
app.mount("/fingerprint", fingerprint_app)
app.mount("/monitoring", monitoring_app)
app.mount("/radios", radio_surveillance_app)

# ------------------------------------------------------
# Route racine
# ------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🚀 Unified Audio Monitoring API is running!",
        "services": {
            "Fingerprint API": "/fingerprint/",
            "Monitoring API": "/monitoring/",
            "Radio Surveillance API": "/radios/"
        }
    }

# ======================================================
# TEST AUTOMATIQUE DES SOUS-API
# ======================================================
def test_subapis(base_url="http://127.0.0.1:8000"):
    """Vérifie que les sous-API répondent correctement."""
    time.sleep(5)
    endpoints = {
        "Fingerprint API": f"{base_url}/fingerprint/",
        "Monitoring API": f"{base_url}/monitoring/",
        "Radio Surveillance API": f"{base_url}/radios/"
    }

    logger.info("🧪 Vérification automatique des sous-API...")
    for name, url in endpoints.items():
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                logger.info(f"✅ {name} fonctionne ({url})")
            else:
                logger.warning(f"⚠️ {name} a répondu avec le code {r.status_code}")
        except Exception as e:
            logger.error(f"❌ {name} inaccessible : {e}")
    logger.info("🔎 Vérification des sous-API terminée.\n")

# ======================================================
# LANCEMENT AUTOMATIQUE DE LA SURVEILLANCE RADIO
# ======================================================
def start_radio_surveillance():
    """Démarre automatiquement la surveillance multi-radio."""
    try:
        time.sleep(8)
        logger.info("📡 Démarrage automatique de la surveillance radio...")
        MonitoringManager.start_all()
        logger.info("✅ Surveillance radio initialisée avec succès.")
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage de la surveillance radio : {e}")

# ======================================================
# FONCTION PRINCIPALE DE LANCEMENT
# ======================================================
def run():
    logger.info("🎧 Lancement de l’API unifiée...")

    # Port dynamique pour Render ou local
    port = int(os.environ.get("PORT", 8000))

    # Lancer les threads secondaires
    threading.Thread(target=test_subapis, daemon=True).start()
    threading.Thread(target=start_radio_surveillance, daemon=True).start()

    # Lancement du serveur Uvicorn
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=port, reload=False)

# ======================================================
# POINT D’ENTRÉE PRINCIPAL
# ======================================================
if __name__ == "__main__":
    run()
