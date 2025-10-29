

# # uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload


# -*- coding: utf-8 -*-

import uvicorn
import logging
import threading
import requests
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("UnifiedAPI")

# ------------------------------------------------------
# Import des sous-applications
# ------------------------------------------------------
from generation.fingerprint_fastapi import app as fingerprint_app
from api.app_fastapi import app as monitoring_app

# ------------------------------------------------------
# Import du système de surveillance multi-radio
# ------------------------------------------------------
from api.radio_fastapi import app as radio_surveillance_app, MonitoringManager

# ------------------------------------------------------
# Application principale
# ------------------------------------------------------
app = FastAPI(
    title="🎶 Unified Audio Monitoring API",
    description=(
        "API unifiée combinant :\n"
        " - /fingerprint → génération & indexation automatique d’empreintes audio\n"
        " - /monitoring → détection et identification audio en temps réel\n"
        " - /radios → surveillance automatique des flux radio"
    ),
    version="3.0.0"
)

# Middleware global
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des sous-applications
app.mount("/fingerprint", fingerprint_app)
app.mount("/monitoring", monitoring_app)
app.mount("/radios", radio_surveillance_app)

# ------------------------------------------------------
# Route racine
# ------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🚀 Unified Audio Monitoring API ready!",
        "services": {
            "Fingerprint API": "/fingerprint/",
            "Monitoring API": "/monitoring/",
            "Radio Surveillance API": "/radios/"
        }
    }

# ------------------------------------------------------
# Vérification automatique des sous-API
# ------------------------------------------------------
def test_subapis(base_url="http://127.0.0.1:8000"):
    """Test automatique pour s'assurer que les sous-API répondent."""
    time.sleep(5)
    endpoints = {
        "Fingerprint API": f"{base_url}/fingerprint/",
        "Monitoring API": f"{base_url}/monitoring/",
        "Radio Surveillance API": f"{base_url}/radios/"
    }
    logger.info("🧪 Vérification des sous-API...")
    for name, url in endpoints.items():
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                logger.info(f"✅ {name} fonctionne ({url})")
            else:
                logger.warning(f"⚠️ {name} répond avec code {r.status_code}")
        except Exception as e:
            logger.error(f"❌ Impossible d'accéder à {name}: {e}")
    logger.info("🔎 Vérification terminée.\n")

# ------------------------------------------------------
# Démarrage automatique de la surveillance radio
# ------------------------------------------------------
def start_radio_surveillance():
    """Lance la surveillance multi-radio automatiquement au démarrage."""
    try:
        time.sleep(8)  # petit délai pour laisser l'API démarrer
        logger.info("📡 Lancement automatique de la surveillance radio...")
        # Appel direct du manager interne si tu veux éviter les requêtes HTTP
        MonitoringManager.start_all()  # méthode à ajuster selon ton fichier radio_surveillance_fastapi
        logger.info("✅ Surveillance radio initialisée automatiquement.")
    except Exception as e:
        logger.error(f"❌ Erreur au démarrage de la surveillance radio : {e}")

# ------------------------------------------------------
# Fonction principale
# ------------------------------------------------------
def run():
    logger.info("🎧 Démarrage de l’API unifiée...")
    # Thread pour vérifier les sous-API
    threading.Thread(target=test_subapis, daemon=True).start()
    # Thread pour lancer la surveillance radio
    threading.Thread(target=start_radio_surveillance, daemon=True).start()
    # Lancement serveur principal
    uvicorn.run("main_fastapi:app", host="0.0.0.0", port=8000, reload=True)

# ------------------------------------------------------
# Entrée principale
# ------------------------------------------------------
if __name__ == "__main__":
    run()


# uvicorn main_fastapi:app --host 0.0.0.0 --port 8000 --reload
