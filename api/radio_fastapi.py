
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import datetime
import threading
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
from uuid import uuid4
from collections import deque
from contextlib import contextmanager

# Imports pour audio et ML
import pandas as pd
import numpy as np
import torch
import faiss
import librosa
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import soundfile as sf
from io import BytesIO, StringIO

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse

# ----------------- LOGGING AVANCÉ -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('radio_monitoring.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ----------------- PROJECT PATH -----------------
project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

# ----------------- IMPORTS DES MODULES EXISTANTS -----------------
from utils.utils import query_sequence_search, get_winner, extract_mel_spectrogram, search_index
from models.neural_fingerprinter import Neural_Fingerprinter

# ----------------- CONFIGURATION -----------------
class Config:
    """Configuration centralisée"""
    MAX_RECONNECT_ATTEMPTS = 999999  # Essentiellement infini
    RECONNECT_DELAY = 5  # secondes
    CHUNK_DURATION = 5  # secondes d'audio à analyser
    STREAM_TIMEOUT = 30
    STREAM_CHUNK_SIZE = 8192
    MAX_BUFFER_SIZE = 500  # chunks max en mémoire
    HEALTH_CHECK_INTERVAL = 60  # secondes
    MIN_CONFIDENCE = 0.44
    WEBSOCKET_PING_INTERVAL = 30
    MAX_AUDIO_QUEUE_SIZE = 10
    CONNECTION_POOL_SIZE = 10
    
config = Config()

# ----------------- MODELS PYDANTIC -----------------
class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    model_loaded: bool
    active_radios: int
    total_detections: int

class DetectionEvent(BaseModel):
    id: str
    radio: str
    track: str
    artist: str
    full_name: str
    start_time: str
    end_time: Optional[str] = None
    duration: float = 0.0
    confidence: float
    score: float
    session_id: str

# class RadioStatus(BaseModel):
#     name: str
#     is_active: bool
#     is_connected: bool
#     current_track: Optional[str]
#     total_detections: int
#     last_detection_time: Optional[str]
#     reconnect_count: int
#     error_count: int

class RadioStatus(BaseModel):
    name: str
    is_active: bool
    is_connected: bool
    current_track: Optional[str]
    total_detections: int
    last_detection_time: Optional[str]
    reconnect_count: int
    error_count: int
    played_duration: Optional[float] = 0.0



class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.datetime.now().isoformat()
        super().__init__(**data)

# ----------------- SESSION STORAGE -----------------
class SessionStorage:
    """Stockage persistant avec gestion d'erreurs"""
    
    def __init__(self, storage_path: str = "monitoring_sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        
    def save_session(self, session_data: Dict):
        """Sauvegarde avec retry"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with self._lock:
                    filename = f"session_{session_data['session_id']}.json"
                    filepath = self.storage_path / filename
                    
                    # Sauvegarde atomique
                    temp_file = filepath.with_suffix('.tmp')
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(session_data, f, indent=2, ensure_ascii=False)
                    temp_file.replace(filepath)
                    
                logger.info(f"✅ Session sauvegardée: {filename}")
                return True
            except Exception as e:
                logger.error(f"Tentative {attempt+1}/{max_attempts} - Erreur sauvegarde: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)
        return False
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """Charge une session avec gestion d'erreur"""
        try:
            filepath = self.storage_path / f"session_{session_id}.json"
            if not filepath.exists():
                return None
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur chargement session {session_id}: {e}")
            return None

# ----------------- AUDIO BUFFER MANAGER -----------------
class AudioBufferManager:
    """Gestion optimisée des buffers audio"""
    
    def __init__(self, max_size: int = 500):
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self.total_received = 0
        self.total_processed = 0
    
    def add_chunk(self, chunk: bytes):
        with self._lock:
            self.buffer.append(chunk)
            self.total_received += 1
    
    def get_audio_data(self, min_chunks: int = 300) -> Optional[bytes]:
        """Récupère des données audio si suffisamment disponibles"""
        with self._lock:
            if len(self.buffer) < min_chunks:
                return None
            
            # Extraire les chunks
            chunks = []
            for _ in range(min(min_chunks, len(self.buffer))):
                if self.buffer:
                    chunks.append(self.buffer.popleft())
            
            self.total_processed += len(chunks)
            return b"".join(chunks) if chunks else None
    
    def clear(self):
        with self._lock:
            self.buffer.clear()

# ----------------- CONNECTION MANAGER -----------------
class ConnectionManager:
    """Gestion robuste des connexions HTTP"""
    
    def __init__(self):
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Crée une session avec retry automatique"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=config.CONNECTION_POOL_SIZE,
            pool_maxsize=config.CONNECTION_POOL_SIZE
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get_stream(self, url: str, timeout: int = 30):
        """Obtient un stream avec gestion d'erreur"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": "https://zeno.fm/"
        }
        
        return self.session.get(
            url,
            stream=True,
            timeout=timeout,
            headers=headers,
            allow_redirects=True
        )

# ----------------- RADIO MONITOR -----------------
class RadioMonitor:
    """Moniteur ultra-robuste pour une radio"""
    
    def __init__(self, name: str, url: str, monitoring_manager):
        self.name = name
        self.url = url
        self.monitoring_manager = monitoring_manager
        
        # État
        self.is_active = True
        self.is_connected = False
        self.should_stop = threading.Event()
        
        # Statistiques
        self.current_track: Optional[str] = None
        self.detections: List[Dict] = []
        self.total_detections = 0
        self.reconnect_count = 0
        self.error_count = 0
        self.last_detection_time: Optional[datetime.datetime] = None
        self.start_time = time.time()
        
        # Composants
        self.buffer_manager = AudioBufferManager()
        self.connection_manager = ConnectionManager()
        self._lock = threading.Lock()
        
        # Thread
        self.monitor_thread: Optional[threading.Thread] = None

        self.total_played_by_track = {}  # nouveau dictionnaire : { "Artiste - Titre": durée_totale_en_sec }

    
    def start(self):
        """Démarre le monitoring"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning(f"{self.name}: Déjà démarré")
            return
        
        self.should_stop.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"Radio-{self.name}",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"🚀 {self.name}: Monitoring démarré")
    
    def stop(self):
        """Arrête le monitoring"""
        self.should_stop.set()
        self.is_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info(f"🛑 {self.name}: Monitoring arrêté")
    
    def _monitoring_loop(self):
        """Boucle principale avec reconnexion automatique"""
        attempt = 0
        
        while not self.should_stop.is_set() and self.is_active:
            attempt += 1
            
            try:
                logger.info(f"🔌 {self.name}: Tentative connexion #{attempt}")
                self._connect_and_monitor()
                
            except requests.exceptions.ChunkedEncodingError as e:
                logger.warning(f"⚠️ {self.name}: Connexion interrompue (ChunkedEncoding)")
                self.is_connected = False
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"⚠️ {self.name}: Timeout")
                self.is_connected = False
                self.error_count += 1
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"⚠️ {self.name}: Erreur connexion")
                self.is_connected = False
                self.error_count += 1
                
            except Exception as e:
                logger.error(f"❌ {self.name}: Erreur inattendue: {e}")
                logger.error(traceback.format_exc())
                self.error_count += 1
            
            finally:
                self.is_connected = False
                self.buffer_manager.clear()
            
            # Reconnexion automatique
            if not self.should_stop.is_set() and self.is_active:
                self.reconnect_count += 1
                delay = min(config.RECONNECT_DELAY * (1 + attempt * 0.1), 60)
                logger.info(f"🔄 {self.name}: Reconnexion dans {delay:.1f}s...")
                self.should_stop.wait(delay)
    
    def _connect_and_monitor(self):
        """Connexion et monitoring du stream"""
        response = self.connection_manager.get_stream(
            self.url,
            timeout=config.STREAM_TIMEOUT
        )
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        logger.info(f"✅ {self.name}: Connecté (HTTP {response.status_code})")
        self.is_connected = True
        
        # Traitement du stream
        for chunk in response.iter_content(chunk_size=config.STREAM_CHUNK_SIZE):
            if self.should_stop.is_set() or not self.is_active:
                break
            
            if chunk:
                self.buffer_manager.add_chunk(chunk)
                
                # Analyse périodique
                audio_data = self.buffer_manager.get_audio_data(min_chunks=300)
                if audio_data:
                    self._process_audio(audio_data)
    
    def _process_audio(self, audio_data: bytes):
        """Traite les données audio"""
        try:
            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp.write(audio_data)
                temp_path = tmp.name
            
            try:
                # Charger avec librosa
                y, sr = librosa.load(
                    temp_path,
                    sr=self.monitoring_manager.config['SR'],
                    mono=True,
                    duration=config.CHUNK_DURATION
                )
                
                # Identification
                if len(y) > 0:
                    result = self._identify_track(y)
                    if result:
                        self._handle_detection(result)
            
            finally:
                # Nettoyage
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"{self.name}: Erreur traitement audio: {e}")
    
    def _identify_track(self, audio_array: np.ndarray) -> Optional[Dict]:
        """Identification par empreinte"""
        try:
            F = self.monitoring_manager.config['SR']
            H = self.monitoring_manager.config['Hop size']
            k = self.monitoring_manager.config['neighbors']
            
            # Extraction des features
            J = max(1, int(np.floor((audio_array.size - F) / H)) + 1)
            if J <= 0:
                return None
            
            xq = np.stack([
                extract_mel_spectrogram(audio_array[j*H:j*H+F]).reshape(1, 256, 32)
                for j in range(J)
            ])
            
            # Inférence
            with torch.no_grad():
                out = self.monitoring_manager.model(
                    torch.from_numpy(xq).to(self.monitoring_manager.device)
                )
                D, I = self.monitoring_manager.index.search(out.cpu().numpy(), k)
            
            # Vote majoritaire
            counts = {}
            dists_per_song = {}
            
            for j in range(J):
                top_idx = int(I[j, 0])
                top_dist = float(D[j, 0])
                song_idx = search_index(top_idx, self.monitoring_manager.sorted_arr)
                song_id = self.monitoring_manager.json_correspondence[str(song_idx)]
                
                counts[song_id] = counts.get(song_id, 0) + 1
                dists_per_song.setdefault(song_id, []).append(top_dist)
            
            if not counts:
                return None
            
            # Meilleur match
            winner = max(counts.items(), key=lambda kv: kv[1])[0]
            confidence = counts[winner] / float(J)
            
            if confidence < config.MIN_CONFIDENCE:
                return None
            
            mean_dist = float(np.mean(dists_per_song[winner]))
            score = 1.0 / (1.0 + mean_dist)
            
            # Parser le nom
            parts = winner.split(' - ', 1)
            artist = parts[0] if len(parts) > 1 else "Unknown"
            title = parts[1] if len(parts) > 1 else winner
            
            return {
                'track': title,
                'artist': artist,
                'full_name': winner,
                'confidence': confidence,
                'score': score
            }
        
        except Exception as e:
            logger.error(f"{self.name}: Erreur identification: {e}")
            return None
    
    def _handle_detection(self, result: Dict):
        """Gère une nouvelle détection"""
        with self._lock:
            now = datetime.datetime.now()
            
            # Vérifier si nouvelle piste
            if self.current_track != result['full_name']:
                
                # Finaliser l'ancienne
                if self.current_track and self.detections:
                    last = self.detections[-1]
                    last['end_time'] = now.isoformat()
                    duration = (now - datetime.datetime.fromisoformat(
                        last['start_time']
                    )).total_seconds()
                    last['duration'] = duration

                    # ✅ Ajout : cumuler la durée totale par chanson
                    key = f"{last['artist']} - {last['track']}"
                    self.total_played_by_track[key] = self.total_played_by_track.get(key, 0.0) + duration        
                            
                # Nouvelle détection
                detection = {
                    'id': str(uuid4())[:8],
                    'radio': self.name,
                    'track': result['track'],
                    'artist': result['artist'],
                    'start_time': now.isoformat(),
                    'end_time': None,
                    'duration': 0.0,
                    'confidence': result['confidence'],
                    'score': result['score'],
                    'session_id': self.monitoring_manager.session_id
                }
                
                self.detections.append(detection)
                self.current_track = result['full_name']
                self.total_detections += 1
                self.last_detection_time = now
                
                # Notifier via WebSocket
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.monitoring_manager.broadcast_detection(detection),
                        self.monitoring_manager.loop
                    )
                except Exception as e:
                    logger.error(f"Erreur notification WebSocket: {e}")
                
                logger.info(
                    f"🎵 {self.name}: {result['artist']} - {result['track']} "
                    f"({result['confidence']*100:.1f}%)"
                )
    

    def _get_current_played_duration(self) -> float:
        """Retourne la durée écoulée pour la piste en cours."""
        if not self.current_track or not self.detections:
            return 0.0
        last = self.detections[-1]
        if last['end_time'] is not None:
            return last['duration']
        start = datetime.datetime.fromisoformat(last['start_time'])
        return (datetime.datetime.now() - start).total_seconds()



    def get_status(self):
        return RadioStatus(
            name=self.name,
            is_active=self.is_active,
            is_connected=self.is_connected,
            current_track=self.current_track,
            total_detections=len(self.detections),
            last_detection_time=self.detections[-1]['start_time'] if self.detections else None,
            reconnect_count=self.reconnect_count,
            error_count=self.error_count,
            played_duration=self._get_current_played_duration()
        )


    # def get_status(self) -> RadioStatus:
    #     """Retourne le statut actuel"""
    #     with self._lock:
    #         return RadioStatus(
    #             name=self.name,
    #             is_active=self.is_active,
    #             is_connected=self.is_connected,
    #             current_track=self.current_track,
    #             total_detections=self.total_detections,
    #             last_detection_time=self.last_detection_time.isoformat() if self.last_detection_time else None,
    #             reconnect_count=self.reconnect_count,
    #             error_count=self.error_count
    #         )

# ----------------- MONITORING MANAGER -----------------
class MonitoringManager:
    """Gestionnaire principal ultra-robuste"""
    
    def __init__(self):
        self.config: Optional[Dict] = None
        self.model: Optional[Neural_Fingerprinter] = None
        self.index: Optional[faiss.Index] = None
        self.json_correspondence: Optional[Dict] = None
        self.sorted_arr: Optional[np.ndarray] = None
        self.device: str = 'cpu'
        
        self.session_id: Optional[str] = None
        self.session_storage = SessionStorage()
        self.start_time: Optional[float] = None
        
        self.radio_monitors: Dict[str, RadioMonitor] = {}
        self.websocket_clients: List[WebSocket] = []
        self._ws_lock = threading.Lock()
        
        # Event loop pour async
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        
        # Health check périodique
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
    
    def load_configuration(self, config_path: str):
        """Charge la configuration"""
        try:
            config_file = Path(project_path) / config_path
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"✅ Configuration chargée: {config_file}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement config: {e}")
            raise
    
    def initialize_model(self):
        """Initialise le modèle"""
        try:
            attention = self.config.get('attention', False)
            self.device = 'cuda' if torch.cuda.is_available() and self.config.get('device')=='cuda' else 'cpu'
            self.model = Neural_Fingerprinter(attention=attention).to(self.device)
            self.model.load_state_dict(
                torch.load(Path(project_path)/self.config['weights'], map_location=self.device)
            )
            self.model.eval()
            logger.info(f"✅ Modèle chargé sur {self.device}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def initialize_index(self):
        """Initialise l'index FAISS"""
        try:
            self.index = faiss.read_index(str(Path(project_path)/self.config['index']))
            self.index.nprobe = self.config.get('nprobes', 1)
            
            with open(Path(project_path)/self.config['json'], 'r') as f:
                self.json_correspondence = json.load(f)
                self.sorted_arr = np.sort(np.array(list(map(int, self.json_correspondence.keys()))))
            
            logger.info("✅ Index FAISS initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation index: {e}")
            raise
    
    def initialize_radios(self):
        """Initialise les radios du Burkina Faso"""
        RADIOS = {
            "Optima FM": "https://stream.zeno.fm/ktrcvrs2dy8uv",
            "RADIO LEGENDE Ouaga": "https://stream.zeno.fm/p1e2466d6ehvv",
            "POGNERE FM POUYTENGA": "https://stream.zeno.fm/74514wb9yy8uv",
            "BASSY FM ZINIARE": "https://stream.zeno.fm/bbgurdr7py8uv",
            "SAVANE FM": "https://stream-153.zeno.fm/zf9ga7y9grquv",
            "Horizon Fm": "https://stream.zeno.fm/7861hrn4bf9uv",
            "RADIO PALABRE": "https://stream-178.zeno.fm/581q2b6cay8uv",
            "TILGRE FM": "https://stream-175.zeno.fm/18ydf6w8by8uv"
            
        }

        
        for name, url in RADIOS.items():
            self.radio_monitors[name] = RadioMonitor(name, url, self)
        
        logger.info(f"✅ {len(self.radio_monitors)} radios initialisées")
    
    def start_monitoring(self) -> str:
        """Démarre la surveillance"""
        self.session_id = str(uuid4())[:8]
        self.start_time = time.time()
        
        for monitor in self.radio_monitors.values():
            monitor.start()
        
        logger.info(f"🚀 Surveillance démarrée - Session: {self.session_id}")
        return self.session_id
    
    def stop_monitoring(self):
        """Arrête la surveillance"""
        for monitor in self.radio_monitors.values():
            monitor.stop()
        
        # Sauvegarder la session
        self._save_current_session()
        
        logger.info("🛑 Surveillance arrêtée")
    
    async def broadcast_detection(self, detection: Dict):
        """Envoie une détection à tous les clients WebSocket"""
        with self._ws_lock:
            disconnected = []
            
            for ws in self.websocket_clients:
                try:
                    await ws.send_json({
                        'event': 'detection',
                        'data': detection
                    })
                except Exception as e:
                    logger.warning(f"Client WebSocket déconnecté: {e}")
                    disconnected.append(ws)
            
            # Nettoyer les clients déconnectés
            for ws in disconnected:
                self.websocket_clients.remove(ws)
    
    def register_websocket(self, websocket: WebSocket):
        """Enregistre un client WebSocket"""
        with self._ws_lock:
            self.websocket_clients.append(websocket)
            logger.info(f"WebSocket connecté (total: {len(self.websocket_clients)})")
    
    def unregister_websocket(self, websocket: WebSocket):
        """Désenregistre un client WebSocket"""
        with self._ws_lock:
            if websocket in self.websocket_clients:
                self.websocket_clients.remove(websocket)
                logger.info(f"WebSocket déconnecté (restants: {len(self.websocket_clients)})")
    

    def get_consolidated_report(self) -> List[Dict]:
        """Génère le rapport consolidé"""
        report = []

        for name, monitor in self.radio_monitors.items():
            # Boucle sur les détections individuelles
            for detection in monitor.detections:
                dt = datetime.datetime.fromisoformat(detection['start_time'])
                duration = detection['duration']

                report.append({
                    'heure': dt.strftime('%H:%M:%S'),
                    'date': dt.strftime('%d/%m/%Y'),
                    'radio': detection['radio'],
                    'titre': detection['track'],
                    'artiste': detection['artist'],
                    'duree': f"{int(duration // 60)}:{int(duration % 60):02d}",
                    'confidence': f"{detection['confidence'] * 100:.1f}%"
                })
            
            # ---------- AJOUT : cumuls par chanson ----------
            for track, seconds in monitor.total_played_by_track.items():
                report.append({
                    'radio': name,
                    'titre': track.split(' - ')[1] if ' - ' in track else track,
                    'artiste': track.split(' - ')[0] if ' - ' in track else 'Unknown',
                    'duree_totale': f"{int(seconds // 60)}:{int(seconds % 60):02d}"
                })

        # Trier par date et heure
        report.sort(key=lambda x: (x.get('date', ''), x.get('heure', '')), reverse=True)
        return report


    def get_summary(self) -> Dict[str, RadioStatus]:
        """Résumé de toutes les radios"""
        return {
            name: monitor.get_status()
            for name, monitor in self.radio_monitors.items()
        }
    
    def _save_current_session(self):
        """Sauvegarde la session en cours"""
        if not self.session_id:
            return

        session_data = {
            'session_id': self.session_id,
            'start_time': datetime.datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.datetime.now().isoformat(),
            'total_duration': time.time() - self.start_time,
            'radios': {
                name: {
                    'total_detections': monitor.total_detections,
                    'detections': monitor.detections,
                    # ✅ Ajout de la durée totale par chanson (format lisible)
                    'total_played_by_track': {
                        k: f"{int(v // 60)}:{int(v % 60):02d}" for k, v in monitor.total_played_by_track.items()
                    }
                }
                for name, monitor in self.radio_monitors.items()
            }
        }
        
        self.session_storage.save_session(session_data)
    
    def _health_check_loop(self):
        """Vérifie périodiquement l'état des radios"""
        while True:
            try:
                time.sleep(config.HEALTH_CHECK_INTERVAL)
                
                for name, monitor in self.radio_monitors.items():
                    if monitor.is_active and not monitor.is_connected:
                        logger.warning(f"⚠️ {name}: Non connecté (tentatives: {monitor.reconnect_count})")
                
            except Exception as e:
                logger.error(f"Erreur health check: {e}")

# ----------------- FASTAPI APP -----------------
monitoring_manager = MonitoringManager()

app = FastAPI(
    title="Multi-Radio Monitoring API Ultra-Robuste",
    version="3.0.0",
    description="API de surveillance radio avec reconnexion automatique et gestion d'erreurs complète"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------- EXCEPTION HANDLERS -----------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestion globale des exceptions"""
    logger.error(f"Exception non gérée: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content=ApiResponse(
            success=False,
            message=f"Erreur serveur: {str(exc)}"
        ).dict()
    )

# ----------------- ENDPOINTS -----------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Status de santé de l'API"""
    total_detections = sum(
        monitor.total_detections 
        for monitor in monitoring_manager.radio_monitors.values()
    )
    
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        uptime=time.time() - monitoring_manager.start_time if monitoring_manager.start_time else 0,
        model_loaded=monitoring_manager.model is not None,
        active_radios=sum(1 for m in monitoring_manager.radio_monitors.values() if m.is_active),
        total_detections=total_detections
    )

@app.post("/radios/start", response_model=ApiResponse)
async def start_monitoring(background_tasks: BackgroundTasks):
    """Démarre la surveillance multi-radios"""
    try:
        # Initialiser si nécessaire
        if monitoring_manager.config is None:
            monitoring_manager.load_configuration("api/config_online.json")
            monitoring_manager.initialize_model()
            monitoring_manager.initialize_index()
            monitoring_manager.initialize_radios()
        
        # Démarrer
        session_id = monitoring_manager.start_monitoring()
        
        return ApiResponse(
            success=True,
            message="Surveillance multi-radios démarrée avec succès",
            data={
                "session_id": session_id,
                "radios_count": len(monitoring_manager.radio_monitors),
                "radios": list(monitoring_manager.radio_monitors.keys())
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur démarrage: {e}")
        logger.error(traceback.format_exc())
        return ApiResponse(
            success=False,
            message=f"Erreur lors du démarrage: {str(e)}"
        )

@app.post("/radios/stop", response_model=ApiResponse)
async def stop_monitoring():
    """Arrête la surveillance multi-radios"""
    try:
        monitoring_manager.stop_monitoring()
        
        return ApiResponse(
            success=True,
            message="Surveillance arrêtée avec succès"
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur arrêt: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur lors de l'arrêt: {str(e)}"
        )

@app.get("/radios/status", response_model=ApiResponse)
async def get_status():
    """Status détaillé de toutes les radios"""
    try:
        summary = monitoring_manager.get_summary()
        
        return ApiResponse(
            success=True,
            message="Status récupéré",
            data={
                "session_id": monitoring_manager.session_id,
                "uptime": time.time() - monitoring_manager.start_time if monitoring_manager.start_time else 0,
                "radios": {name: status.dict() for name, status in summary.items()}
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur status: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )





@app.get("/radios/report", response_model=ApiResponse)
async def get_report(
    date_filter: Optional[str] = Query(None, description="Filtre date (YYYY-MM-DD)"),
    radio_filter: Optional[str] = Query(None, description="Filtre radio"),
    limit: int = Query(1000, description="Nombre max de résultats")
):
    """Rapport consolidé avec filtres"""
    try:
        report = monitoring_manager.get_consolidated_report()
        
        # Appliquer filtres
        if date_filter:
            report = [r for r in report if r['date'] == datetime.datetime.strptime(date_filter, '%Y-%m-%d').strftime('%d/%m/%Y')]
        
        if radio_filter:
            report = [r for r in report if r['radio'] == radio_filter]
        
        # Limiter
        report = report[:limit]
        
        return ApiResponse(
            success=True,
            message=f"{len(report)} détections trouvées",
            data={
                "count": len(report),
                "detections": report
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur rapport: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )

@app.get("/radios/export")
async def export_report(
    date_filter: Optional[str] = Query(None),
    radio_filter: Optional[str] = Query(None)
):
    """Export CSV du rapport"""
    try:
        report = monitoring_manager.get_consolidated_report()
        
        # Appliquer filtres
        if date_filter:
            report = [r for r in report if r['date'] == datetime.datetime.strptime(date_filter, '%Y-%m-%d').strftime('%d/%m/%Y')]
        
        if radio_filter:
            report = [r for r in report if r['radio'] == radio_filter]
        
        if not report:
            raise HTTPException(status_code=404, detail="Aucune donnée disponible")
        
        # Créer CSV
        df = pd.DataFrame(report)
        output = StringIO()
        output.write('\ufeff')  # BOM UTF-8
        df.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        # Nom du fichier
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rapport_radios_{timestamp}.csv"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/radios/statistics", response_model=ApiResponse)
async def get_statistics():
    """Statistiques globales"""
    try:
        stats = {
            "total_radios": len(monitoring_manager.radio_monitors),
            "active_radios": sum(1 for m in monitoring_manager.radio_monitors.values() if m.is_active),
            "connected_radios": sum(1 for m in monitoring_manager.radio_monitors.values() if m.is_connected),
            "total_detections": sum(m.total_detections for m in monitoring_manager.radio_monitors.values()),
            "total_reconnects": sum(m.reconnect_count for m in monitoring_manager.radio_monitors.values()),
            "total_errors": sum(m.error_count for m in monitoring_manager.radio_monitors.values()),
            "uptime_seconds": time.time() - monitoring_manager.start_time if monitoring_manager.start_time else 0,
        }
        
        # Top radios
        top_radios = sorted(
            [(name, m.total_detections) for name, m in monitoring_manager.radio_monitors.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        stats["top_radios"] = [{"name": name, "detections": count} for name, count in top_radios]
        
        return ApiResponse(
            success=True,
            message="Statistiques récupérées",
            data=stats
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur statistiques: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )

@app.post("/radios/{radio_name}/pause", response_model=ApiResponse)
async def pause_radio(radio_name: str):
    """Met en pause une radio spécifique"""
    try:
        if radio_name not in monitoring_manager.radio_monitors:
            raise HTTPException(status_code=404, detail="Radio non trouvée")
        
        monitor = monitoring_manager.radio_monitors[radio_name]
        monitor.is_active = False
        
        return ApiResponse(
            success=True,
            message=f"Radio {radio_name} mise en pause"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur pause: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )

@app.post("/radios/{radio_name}/resume", response_model=ApiResponse)
async def resume_radio(radio_name: str):
    """Reprend une radio en pause"""
    try:
        if radio_name not in monitoring_manager.radio_monitors:
            raise HTTPException(status_code=404, detail="Radio non trouvée")
        
        monitor = monitoring_manager.radio_monitors[radio_name]
        monitor.is_active = True
        monitor.start()
        
        return ApiResponse(
            success=True,
            message=f"Radio {radio_name} reprise"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur reprise: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )

@app.websocket("/ws/radios")
async def websocket_radios(websocket: WebSocket):
    """WebSocket pour événements temps réel"""
    await websocket.accept()
    monitoring_manager.register_websocket(websocket)
    
    try:
        # Envoyer le status initial
        await websocket.send_json({
            'event': 'connected',
            'data': {
                'session_id': monitoring_manager.session_id,
                'radios': list(monitoring_manager.radio_monitors.keys())
            }
        })
        
        # Maintenir la connexion avec ping
        while True:
            try:
                # Attendre un message du client (avec timeout)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=config.WEBSOCKET_PING_INTERVAL
                )
                
                # Traiter les commandes si nécessaire
                if message == "ping":
                    await websocket.send_json({'event': 'pong'})
            
            except asyncio.TimeoutError:
                # Envoyer un ping pour maintenir la connexion
                try:
                    await websocket.send_json({
                        'event': 'ping',
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                except:
                    break
    
    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté normalement")
    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")
    finally:
        monitoring_manager.unregister_websocket(websocket)

@app.get("/sessions", response_model=ApiResponse)
async def list_sessions(
    limit: int = Query(50, description="Nombre de sessions à retourner")
):
    """Liste les sessions sauvegardées"""
    try:
        sessions = monitoring_manager.session_storage.list_sessions(limit=limit)
        
        return ApiResponse(
            success=True,
            message=f"{len(sessions)} sessions trouvées",
            data={"sessions": sessions}
        )
    
    except Exception as e:
        logger.error(f"❌ Erreur liste sessions: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )

@app.get("/sessions/{session_id}", response_model=ApiResponse)
async def get_session(session_id: str):
    """Récupère une session spécifique"""
    try:
        session = monitoring_manager.session_storage.load_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session non trouvée")
        
        return ApiResponse(
            success=True,
            message="Session récupérée",
            data=session
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération session: {e}")
        return ApiResponse(
            success=False,
            message=f"Erreur: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Actions au démarrage"""
    logger.info("=" * 60)
    logger.info("🚀 API Multi-Radio Monitoring Ultra-Robuste v3.0.0")
    logger.info("=" * 60)
    logger.info("✅ Serveur démarré")
    logger.info(f"📍 Device: {torch.cuda.is_available() and 'CUDA' or 'CPU'}")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Actions à l'arrêt"""
    logger.info("🛑 Arrêt du serveur...")
    
    try:
        # Arrêter toutes les radios
        monitoring_manager.stop_monitoring()
        
        # Fermer l'event loop
        monitoring_manager.loop.call_soon_threadsafe(monitoring_manager.loop.stop)
        
        logger.info("✅ Arrêt propre effectué")
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt: {e}")

# ----------------- LANCEMENT -----------------
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        workers=1  # Important: un seul worker pour les threads
    )




    # uvicorn radio_fastapi:app --host 0.0.0.0 --port 8000 --workers 1