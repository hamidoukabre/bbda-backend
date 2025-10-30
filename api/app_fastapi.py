
# # -*- coding: utf-8 -*-
# """
# FastAPI Professional Audio Monitoring API
# Style BMAT - Gestion complète des sessions et rapports
# """

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import sys
# import json
# import time
# import datetime
# import threading
# import logging
# import tempfile
# from pathlib import Path
# from typing import Optional, Dict, Any, List
# import asyncio
# from uuid import uuid4
# import logging

# # Après les imports existants, ajoutez :
# import pandas as pd
# import numpy as np
# import torch
# import faiss
# import pyaudio
# import librosa
# from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# from io import StringIO
# from fastapi.responses import StreamingResponse

# logger = logging.getLogger(__name__)



# # ----------------- LOGGING -----------------
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # ----------------- PROJECT PATH -----------------
# project_path = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_path))

# # ----------------- IMPORTS DES MODULES EXISTANTS -----------------
# from utils.utils import query_sequence_search, get_winner, extract_mel_spectrogram, search_index
# from models.neural_fingerprinter import Neural_Fingerprinter

# # ----------------- MODELS PYDANTIC -----------------
# class HealthResponse(BaseModel):
#     status: str
#     version: str
#     uptime: float
#     model_loaded: bool

# class DetectionEvent(BaseModel):
#     id: str
#     track: str
#     score: float
#     confidence: float
#     offset: float
#     timestamp: str
#     session_id: str
#     duration_played: float = 0.0

# class MonitoringSession(BaseModel):
#     session_id: str
#     start_time: str
#     end_time: Optional[str] = None
#     total_duration: float = 0.0
#     detection_count: int = 0
#     unique_tracks: int = 0
#     detections: List[DetectionEvent] = []
#     status: str = "running"  # running, completed, stopped

# class MonitoringStatus(BaseModel):
#     is_running: bool
#     current_session: Optional[MonitoringSession] = None
#     total_sessions: int = 0
#     total_detections: int = 0

# class ApiResponse(BaseModel):
#     success: bool
#     message: str
#     data: Optional[Any] = None

# class StartMonitoringRequest(BaseModel):
#     config_path: Optional[str] = None

# class ReportSummary(BaseModel):
#     session_id: str
#     date: str
#     start_time: str
#     end_time: str
#     duration: float
#     total_detections: int
#     unique_tracks: int
#     top_tracks: List[Dict[str, Any]]

# class DetailedReport(BaseModel):
#     session: MonitoringSession
#     track_stats: Dict[str, Any]
#     hourly_distribution: Dict[str, int]
#     confidence_distribution: Dict[str, int]

# # ----------------- SESSION STORAGE -----------------
# class SessionStorage:
#     """Stockage persistant des sessions de surveillance"""
    
#     def __init__(self, storage_path: str = "monitoring_sessions"):
#         self.storage_path = Path(storage_path)
#         self.storage_path.mkdir(exist_ok=True)
        
#     def save_session(self, session: MonitoringSession):
#         """Sauvegarde une session"""
#         filename = f"session_{session.session_id}.json"
#         filepath = self.storage_path / filename
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(session.dict(), f, indent=2, ensure_ascii=False)
#         logger.info(f"Session sauvegardée: {filename}")
    
#     def load_session(self, session_id: str) -> Optional[MonitoringSession]:
#         """Charge une session spécifique"""
#         filepath = self.storage_path / f"session_{session_id}.json"
#         if not filepath.exists():
#             return None
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             return MonitoringSession(**data)
    
#     def list_sessions(self, date_filter: Optional[str] = None, limit: int = 100) -> List[MonitoringSession]:
#         """Liste toutes les sessions avec filtre optionnel"""
#         sessions = []
#         for filepath in sorted(self.storage_path.glob("session_*.json"), reverse=True):
#             try:
#                 with open(filepath, 'r', encoding='utf-8') as f:
#                     data = json.load(f)
#                     session = MonitoringSession(**data)
                    
#                     # Filtre par date si spécifié
#                     if date_filter:
#                         session_date = datetime.datetime.fromisoformat(session.start_time).date().isoformat()
#                         if session_date != date_filter:
#                             continue
                    
#                     sessions.append(session)
#                     if len(sessions) >= limit:
#                         break
#             except Exception as e:
#                 logger.error(f"Erreur lecture session {filepath}: {e}")
#         return sessions
    
#     def delete_session(self, session_id: str):
#         """Supprime une session"""
#         filepath = self.storage_path / f"session_{session_id}.json"
#         if filepath.exists():
#             filepath.unlink()
#             logger.info(f"Session supprimée: {session_id}")

# # ----------------- MONITORING MANAGER -----------------
# class MonitoringManager:
#     def __init__(self):
#         self.is_running = False
#         self.monitoring_thread: Optional[threading.Thread] = None
#         self.should_stop = threading.Event()
#         self.start_time: Optional[float] = None

#         self.config: Optional[Dict] = None
#         self.model: Optional[Neural_Fingerprinter] = None
#         self.index: Optional[faiss.Index] = None
#         self.json_correspondence: Optional[Dict] = None
#         self.sorted_arr: Optional[np.ndarray] = None
#         self.stream: Optional[pyaudio.Stream] = None
#         self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
#         self.device: str = 'cpu'

#         # Session management
#         self.current_session: Optional[MonitoringSession] = None
#         self.session_storage = SessionStorage()
        
#         # Track continuity detection
#         self.current_track: Optional[str] = None
#         self.track_start_time: Optional[datetime.datetime] = None
#         self.consecutive_detections: int = 0
#         self.min_detections_for_track: int = 1

#         # Queue pour WebSocket events
#         self.event_queue: asyncio.Queue = asyncio.Queue()
#         self._lock = threading.Lock()

#         # Event loop dédié
#         self.loop = asyncio.new_event_loop()
#         threading.Thread(target=self.loop.run_forever, daemon=True).start()

#                 # Nouveaux attributs pour streaming
#         self.audio_buffer: list = []
#         self.last_analysis_time: float = 0
#         self.min_buffer_duration: float = 5.0  # secondes


#                 # ✅ AJOUT SIMPLE: Pour enregistrement audio
#         # self.audio_save_dir = Path("recorded_audio")
#         # self.audio_save_dir.mkdir(exist_ok=True)
#         # self.audio_counter = 0


#     # def _save_audio_for_debug(self, audio_data: np.ndarray, sample_rate: int):
#     #     """Sauvegarde simple de l'audio pour debug"""
#     #     try:
#     #         timestamp = datetime.datetime.now().strftime("%H%M%S")
#     #         filename = f"audio_{timestamp}.wav"
#     #         filepath = self.audio_save_dir / filename
            
#     #         # Sauvegarder avec soundfile
#     #         import soundfile as sf
#     #         sf.write(filepath, audio_data, sample_rate)
            
#     #         print(f"🔊 Audio enregistré: {filename}")
            
#     #     except Exception as e:
#     #         print(f"❌ Erreur enregistrement: {e}")    


#     def analyze_audio_chunk(self, audio_data: np.ndarray, sample_rate: int, chunk_index: int = 0) -> Optional[dict]:
#         """
#         ✅ Analyse un chunk audio de 5 secondes comme avec PyAudio
#         """
#         try:
#             if self.model is None or self.index is None:
#                 logger.warning("Modèle non initialisé")
#                 return None

#             F = self.config['SR']  # 8000 Hz
#             H = self.config['Hop size']
#             k = self.config['neighbors']
            
#             logger.info(f"🎵 Chunk #{chunk_index}: {len(audio_data)} samples à {sample_rate}Hz")
            
#             # ✅ VÉRIFIER LA DURÉE (doit être ~5 secondes)
#             duration_sec = len(audio_data) / sample_rate
#             logger.info(f"⏱️ Durée chunk: {duration_sec:.2f}s")
            
#             if duration_sec < 4.5:  # Tolérance
#                 logger.warning(f"❌ Chunk trop court: {duration_sec:.2f}s < 4.5s")
#                 return None
            
#             # ✅ RESAMPLER SI NÉCESSAIRE (normalement déjà à 8000 Hz)
#             if sample_rate != F:
#                 logger.info(f"🔄 Resampling: {sample_rate}Hz → {F}Hz")
#                 audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=F)
#                 logger.info(f"✅ Après resampling: {len(audio_data)} samples")
            
#             # ✅ VÉRIFIER SILENCE
#             rms = np.sqrt(np.mean(audio_data ** 2))
#             silence_threshold = self.config.get('silence_threshold', 0.01)
#             if rms < silence_threshold:
#                 logger.info(f"🔇 Silence détecté (RMS: {rms:.4f} < {silence_threshold})")
#                 return None
            
#             logger.info(f"🔊 Niveau audio: RMS = {rms:.4f}")
            
#             # ✅ CALCULER SEGMENTS (comme PyAudio)
#             J = max(1, int(np.floor((audio_data.size - F) / H)) + 1)
#             logger.info(f"📈 Segments calculés: {J}")
            
#             if J <= 0:
#                 logger.warning("❌ Aucun segment généré")
#                 return None
            
#             # ✅ EXTRAIRE SPECTROGRAMMES
#             xq = np.stack([
#                 extract_mel_spectrogram(audio_data[j*H:j*H+F]).reshape(1, 256, 32)
#                 for j in range(J)
#             ])
            
#             logger.info(f"🎼 Spectrogrammes extraits: shape={xq.shape}")
            
#             # ✅ INFÉRENCE
#             with torch.no_grad():
#                 out = self.model(torch.from_numpy(xq).to(self.device))
#                 D, I = self.index.search(out.cpu().numpy(), k)
            
#             # ✅ IDENTIFICATION
#             s_flag = self.config.get('search algorithm', 'majority vote')
            
#             if s_flag == 'sequence search':
#                 idx, score = query_sequence_search(D, I)
#                 true_idx = search_index(idx, self.sorted_arr)
#                 winner = self.json_correspondence[str(true_idx)]
#                 offset = (idx - true_idx) * H / F
#             else:
#                 winner, score = get_winner(self.json_correspondence, I, D, self.sorted_arr)
#                 offset = 0
            
#             confidence = min(1.0, score / 10)
            
#             logger.info(f"🎯 Résultat: {winner} | Score: {score:.2f} | Confiance: {confidence:.2f}")
            
#             # ✅ SEUIL ADAPTATIF
#             min_confidence = 0.1 if (self.current_track and winner == self.current_track) else 0.1
            
#             if confidence >= min_confidence:
#                 logger.info(f"✅ DÉTECTION VALIDÉE: {winner}")
#                 return {
#                     'track': winner,
#                     'score': float(score),
#                     'confidence': float(confidence),
#                     'offset': float(offset),
#                     'chunk_index': chunk_index
#                 }
#             else:
#                 logger.info(f"❌ Confiance insuffisante: {confidence:.2f} < {min_confidence}")
#                 return None
                
#         except Exception as e:
#             logger.error(f"❌ Erreur analyse chunk: {e}", exc_info=True)
#             return None



#     def process_stream_detection(self, result: dict) -> Optional[DetectionEvent]:
#         """
#         Traite un résultat avec continuité intelligente
#         """
#         if not result or not self.current_session:
#             return None
        
#         try:
#             now = datetime.datetime.now()
#             winner = result['track']
#             chunk_index = result.get('chunk_index', 0)
            
#             # ✅ FILTRAGE TEMPOREL ADAPTATIF
#             time_since_last = float('inf')  # Infini si aucune détection précédente
#             if self.current_session.detections:
#                 last_detection_time = datetime.datetime.fromisoformat(
#                     self.current_session.detections[-1].timestamp
#                 )
#                 time_since_last = (now - last_detection_time).total_seconds()
            
#             # ✅ LOGIQUE DE CONTINUITÉ AMÉLIORÉE
#             is_same_track = winner == self.current_track
            
#             if is_same_track:
#                 # Même piste - continuité
#                 self.consecutive_detections += 1
#                 logger.info(f"🔄 Continuité #{self.consecutive_detections} pour: {winner}")
                
#                 # Si même piste et délai raisonnable (< 15s), accepter plus facilement
#                 if time_since_last < 15:
#                     # ✅ Accepter la détection de continuité
#                     self.track_start_time = self.track_start_time or now
                    
#                     detection = DetectionEvent(
#                         id=str(uuid4())[:8],
#                         track=winner,
#                         score=result['score'],
#                         confidence=result['confidence'],
#                         offset=result['offset'],
#                         timestamp=now.isoformat(),
#                         session_id=self.current_session.session_id,
#                         duration_played=0.0
#                     )
                    
#                     # Mettre à jour la durée de la dernière détection
#                     if self.current_session.detections:
#                         self.current_session.detections[-1].duration_played = (
#                             now - datetime.datetime.fromisoformat(self.current_session.detections[-1].timestamp)
#                         ).total_seconds()
                    
#                     logger.info(f"🎵 DÉTECTION CONTINUE: {winner}")
#                     return detection
                    
#             else:
#                 # Nouvelle piste
#                 logger.info(f"🆕 Changement de piste: {self.current_track} → {winner}")
                
#                 # ✅ FILTRE moins agressif pour les nouvelles pistes (8s au lieu de 20s)
#                 if time_since_last < 8 and self.current_session.detections:
#                     logger.info(f"⏰ Nouvelle piste ignorée (trop rapide: {time_since_last:.1f}s)")
#                     return None
                
#                 # Finaliser l'ancienne piste
#                 if self.current_track and self.consecutive_detections >= self.min_detections_for_track:
#                     duration_played = (now - self.track_start_time).total_seconds()
#                     self._finalize_track_detection(duration_played)
#                     logger.info(f"⏹️ Piste terminée: {self.current_track} ({duration_played:.1f}s)")
                
#                 # Nouvelle piste
#                 self.current_track = winner
#                 self.track_start_time = now
#                 self.consecutive_detections = 1
            
#             # ✅ Créer événement pour nouvelle piste ou continuité forte
#             if (not is_same_track or self.consecutive_detections >= 2):
#                 detection = DetectionEvent(
#                     id=str(uuid4())[:8],
#                     track=winner,
#                     score=result['score'],
#                     confidence=result['confidence'],
#                     offset=result['offset'],
#                     timestamp=now.isoformat(),
#                     session_id=self.current_session.session_id,
#                     duration_played=0.0
#                 )
                
#                 # Vérifier si nouvelle détection
#                 if not self.current_session.detections or \
#                 self.current_session.detections[-1].track != winner:
#                     self.current_session.detections.append(detection)
#                     self.current_session.detection_count += 1
                    
#                     logger.info(f"🎵 NOUVELLE DÉTECTION: {winner}")
#                     return detection
            
#             return None
            
#         except Exception as e:
#             logger.error(f"Erreur process_stream_detection: {e}")
#             return None
   
    
#     # ----------------- INITIALISATION -----------------
#     def load_configuration(self, config_path: str):
#         try:
#             # Si c'est juste un nom de fichier, chercher dans le dossier evaluation/
#             if not Path(config_path).is_absolute():
#                 config_file = Path(__file__).parent / config_path
#             else:
#                 config_file = Path(config_path)
            
#             if not config_file.exists():
#                 raise FileNotFoundError(f"Fichier non trouvé: {config_file}")
            
#             with open(config_file, 'r') as f:
#                 self.config = json.load(f)
#             logger.info(f"✅ Configuration chargée: {config_file}")
#         except Exception as e:
#             logger.error(f"❌ Erreur de chargement de la configuration: {e}")
#             raise

#     def initialize_model(self):
#         try:
#             attention = self.config.get('attention', False)
#             self.device = 'cuda' if torch.cuda.is_available() and self.config.get('device')=='cuda' else 'cpu'
#             self.model = Neural_Fingerprinter(attention=attention).to(self.device)
#             self.model.load_state_dict(torch.load(Path(project_path)/self.config['weights'], map_location=self.device))
#             self.model.eval()
#             logger.info(f"Modèle chargé sur {self.device}")
#         except Exception as e:
#             logger.error(f"Erreur de chargement du modèle: {e}")
#             raise

#     def initialize_index(self):
#         try:
#             self.index = faiss.read_index(str(Path(project_path)/self.config['index']))
#             self.index.nprobe = self.config.get('nprobes', 1)
#             with open(Path(project_path)/self.config['json'], 'r') as f:
#                 self.json_correspondence = json.load(f)
#                 self.sorted_arr = np.sort(np.array(list(map(int, self.json_correspondence.keys()))))
#             logger.info("Index FAISS initialisé")
#         except Exception as e:
#             logger.error(f"Erreur d'initialisation de l'index: {e}")
#             raise

#     def initialize_audio(self):
#         try:
#             F, FMB = self.config['SR'], self.config['FMB']
#             self.pyaudio_instance = pyaudio.PyAudio()
#             self.stream = self.pyaudio_instance.open(
#                 rate=F, channels=1, format=pyaudio.paFloat32, frames_per_buffer=FMB, input=True
#             )
#             logger.info(f"Stream audio initialisé: {F}Hz")
#         except Exception as e:
#             logger.error(f"Erreur d'initialisation audio: {e}")
#             raise

#     # ----------------- SESSION MANAGEMENT -----------------
#     def start_monitoring(self, config_path: str):
#         with self._lock:
#             if self.is_running:
#                 logger.warning("Surveillance déjà en cours")
#                 return
            
#             self.load_configuration(config_path)
#             self.initialize_model()
#             self.initialize_index()
#             self.initialize_audio()
            
#             # Créer nouvelle session
#             session_id = str(uuid4())[:8]
#             self.current_session = MonitoringSession(
#                 session_id=session_id,
#                 start_time=datetime.datetime.now().isoformat(),
#                 status="running"
#             )
            
#             self.should_stop.clear()
#             self.is_running = True
#             self.start_time = time.time()
#             self.current_track = None
#             self.track_start_time = None
#             self.consecutive_detections = 0
            
#             self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
#             self.monitoring_thread.start()
#             logger.info(f"Surveillance démarrée - Session: {session_id}")

#     def stop_monitoring(self):
#         with self._lock:
#             if not self.is_running:
#                 logger.warning("Aucune surveillance en cours")
#                 return
            
#             self.should_stop.set()
#             if self.monitoring_thread:
#                 self.monitoring_thread.join(timeout=5)
            
#             # Finaliser la session
#             if self.current_session:
#                 self.current_session.end_time = datetime.datetime.now().isoformat()
#                 self.current_session.total_duration = time.time() - self.start_time
#                 self.current_session.status = "completed"
#                 self.current_session.unique_tracks = len(set(d.track for d in self.current_session.detections))
                
#                 # Sauvegarder la session
#                 self.session_storage.save_session(self.current_session)
#                 logger.info(f"Session terminée: {self.current_session.session_id}")
            
#             self.cleanup()
#             self.is_running = False
#             self.current_session = None

#     def cleanup(self):
#         try:
#             if self.stream:
#                 self.stream.stop_stream()
#                 self.stream.close()
#                 self.stream = None
#             if self.pyaudio_instance:
#                 self.pyaudio_instance.terminate()
#                 self.pyaudio_instance = None
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#         except Exception as e:
#             logger.error(f"Erreur cleanup: {e}")

#     def get_status(self) -> Dict[str, Any]:
#         """État actuel de la surveillance"""
#         uptime = 0.0
#         if self.start_time:
#             uptime = time.time() - self.start_time

#         # Mettre à jour la durée de la session courante
#         if self.current_session:
#             self.current_session.total_duration = uptime

#         # Statistiques globales
#         all_sessions = self.session_storage.list_sessions()
#         total_sessions = len(all_sessions)
#         total_detections = sum(s.detection_count for s in all_sessions)

#         return {
#             "is_running": self.is_running,
#             "current_session": self.current_session,
#             "total_sessions": total_sessions,
#             "total_detections": total_detections
#         }

#     # ----------------- MONITORING LOOP -----------------
#     def _monitoring_loop(self):
#         F, H, FMB = self.config['SR'], self.config['Hop size'], self.config['FMB']
#         dur, k = self.config['duration'], self.config['neighbors']
#         s_flag = 'sequence search' if self.config.get('search algorithm')=='sequence search' else 'majority vote'
#         silence_threshold = self.config.get('silence_threshold', 0.01)

#         logger.info("Boucle de surveillance démarrée")
#         try:
#             with torch.no_grad():
#                 while not self.should_stop.is_set():
#                     try:
#                         frames = [self.stream.read(FMB, exception_on_overflow=False)
#                                 for _ in range(int(F / FMB * dur))]
#                         aggregated_buf = np.frombuffer(b"".join(frames), dtype=np.float32)
#                     except Exception as e:
#                         logger.warning(f"Erreur audio: {e}")
#                         continue

#                     now = datetime.datetime.now()

#                     # Filtrage silence
#                     if np.mean(np.abs(aggregated_buf)) < silence_threshold:
#                         continue

#                     # Découpage et analyse
#                     J = max(1, int(np.floor((aggregated_buf.size - F) / H)) + 1)
#                     if J <= 0:
#                         continue

#                     xq = np.stack([
#                         extract_mel_spectrogram(aggregated_buf[j*H:j*H+F]).reshape(1,256,32)
#                         for j in range(J)
#                     ])

#                     out = self.model(torch.from_numpy(xq).to(self.device))
#                     D, I = self.index.search(out.cpu().numpy(), k)

#                     # Identification
#                     if s_flag == 'sequence search':
#                         idx, score = query_sequence_search(D, I)
#                         true_idx = search_index(idx, self.sorted_arr)
#                         winner = self.json_correspondence[str(true_idx)]
#                         offset = (idx - true_idx) * H / F
#                     else:
#                         winner, score = get_winner(self.json_correspondence, I, D, self.sorted_arr)
#                         offset = 0

#                     confidence = min(1.0, score / 10)

#                     # Gestion de la continuité des pistes
#                     if winner == self.current_track:
#                         self.consecutive_detections += 1
#                     else:
#                         # Changement de piste
#                         if self.current_track and self.consecutive_detections >= self.min_detections_for_track:
#                             # Calculer la durée de lecture
#                             duration_played = (now - self.track_start_time).total_seconds()
#                             self._finalize_track_detection(duration_played)
                        
#                         # Nouvelle piste
#                         self.current_track = winner
#                         self.track_start_time = now
#                         self.consecutive_detections = 1

#                     # Créer événement de détection uniquement si confiance suffisante
#                     if confidence >= 0.1 and self.consecutive_detections >= self.min_detections_for_track:
#                         detection = DetectionEvent(
#                             id=str(uuid4())[:8],
#                             track=winner,
#                             score=float(score),
#                             confidence=float(confidence),
#                             offset=float(offset),
#                             timestamp=now.isoformat(),
#                             session_id=self.current_session.session_id,
#                             duration_played=0.0  # Sera mis à jour à la fin
#                         )

#                         # Vérifier si c'est une nouvelle détection unique
#                         if not self.current_session.detections or \
#                            self.current_session.detections[-1].track != winner:
#                             self.current_session.detections.append(detection)
#                             self.current_session.detection_count += 1

#                             # Envoi WebSocket
#                             asyncio.run_coroutine_threadsafe(
#                                 self.event_queue.put({'event':'detection','data':detection.dict()}),
#                                 self.loop
#                             )

#                     time.sleep(0.001)

#         except Exception as e:
#             logger.error(f"Erreur boucle monitoring: {e}", exc_info=True)
#         finally:
#             # Finaliser la dernière piste
#             if self.current_track and self.track_start_time:
#                 duration = (datetime.datetime.now() - self.track_start_time).total_seconds()
#                 self._finalize_track_detection(duration)
#             logger.info("Boucle de surveillance terminée")

#     def _finalize_track_detection(self, duration: float):
#         """Finalise une détection de piste avec sa durée totale"""
#         if self.current_session.detections:
#             # Mettre à jour la durée de la dernière détection
#             for detection in reversed(self.current_session.detections):
#                 if detection.track == self.current_track:
#                     detection.duration_played = duration
#                     break

#     # ----------------- INFERENCE FILE -----------------
#     def infer_from_file(self, audio_file_path: str, duration: int = 5,
#                        min_score: float = 0.1, min_confidence: float = 0.1,
#                        min_consecutive: int = 1) -> dict:
#         try:
#             y, sr = librosa.load(audio_file_path, sr=self.config['SR'], mono=True)
#             F, H, k = self.config['SR'], self.config['Hop size'], self.config['neighbors']
            
#             seg_len = duration * F
#             n_segments = max(1, y.shape[0] // seg_len)
#             segment_results = []

#             self.model.eval()
#             with torch.no_grad():
#                 for seg_idx in range(n_segments):
#                     start = seg_idx * seg_len
#                     end = min((seg_idx + 1) * seg_len, y.shape[0])
#                     y_slice = y[start:end]

#                     J = max(1, int(np.floor((y_slice.size - F) / H)) + 1)
#                     if J <= 0:
#                         continue

#                     xq = np.stack([
#                         extract_mel_spectrogram(y_slice[j*H:j*H+F]).reshape(1, 256, 32)
#                         for j in range(J)
#                     ])

#                     out = self.model(torch.from_numpy(xq).to(self.device))
#                     D, I = self.index.search(out.cpu().numpy(), k)

#                     counts = {}
#                     dists_per_song = {}
#                     for j in range(J):
#                         top_idx = int(I[j, 0])
#                         top_dist = float(D[j, 0])
#                         song_idx = search_index(top_idx, self.sorted_arr)
#                         song_id = self.json_correspondence[str(song_idx)]
#                         counts[song_id] = counts.get(song_id, 0) + 1
#                         dists_per_song.setdefault(song_id, []).append(top_dist)

#                     if not counts:
#                         continue

#                     majority_song = max(counts.items(), key=lambda kv: kv[1])[0]
#                     confidence = counts[majority_song] / float(J)
#                     mean_top1_dist = float(np.mean(dists_per_song[majority_song]))
#                     score = 1.0 / (1.0 + mean_top1_dist)

#                     segment_results.append({
#                         "seg_idx": seg_idx,
#                         "start_time_sec": start / F,
#                         "end_time_sec": end / F,
#                         "song": majority_song,
#                         "confidence": confidence,
#                         "score": score,
#                         "valid": confidence >= min_confidence and score >= min_score
#                     })

#             # Filtrage segments consécutifs
#             valid_segments = []
#             i = 0
#             while i < len(segment_results):
#                 if not segment_results[i]['valid']:
#                     i += 1
#                     continue
#                 run_song = segment_results[i]['song']
#                 run_start = i
#                 j = i + 1
#                 while (j < len(segment_results) and segment_results[j]['valid'] and
#                        segment_results[j]['song'] == run_song and
#                        segment_results[j]['seg_idx'] == segment_results[j-1]['seg_idx'] + 1):
#                     j += 1
#                 if j - run_start >= min_consecutive:
#                     valid_segments.extend(segment_results[run_start:j])
#                 i = j

#             if not valid_segments:
#                 return {'success': False, 'message': 'No valid detection', 'segments': []}

#             return {'success': True, 'segments': valid_segments}

#         except Exception as e:
#             logger.error(f"Inference error: {e}", exc_info=True)
#             return {'success': False, 'message': str(e), 'segments': []}
        
# # ----------------- INSTANCE -----------------
# monitoring_manager = MonitoringManager()

# # ----------------- FASTAPI APP -----------------
# app = FastAPI(title="Professional Audio Monitoring API", version="2.0.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("API démarrée")
#     yield
#     if monitoring_manager.is_running:
#         monitoring_manager.stop_monitoring()
#     logger.info("API arrêtée")

# app.router.lifespan_context = lifespan

# # ----------------- ENDPOINTS -----------------
# @app.get("/health", response_model=HealthResponse)
# async def health():
#     return HealthResponse(
#         status="healthy",
#         version="2.0.0",
#         uptime=time.time()-(monitoring_manager.start_time or time.time()),
#         model_loaded=monitoring_manager.model is not None
#     )



# @app.post("/start", response_model=ApiResponse)
# async def start_monitoring(request: StartMonitoringRequest):
#     """
#     Démarrer la surveillance - utilise maintenant le streaming client
#     """
#     try:
#         config_path = request.config_path or "config_online.json"
        
#         # Charger config et modèle SANS PyAudio
#         monitoring_manager.load_configuration(config_path)
#         monitoring_manager.initialize_model()
#         monitoring_manager.initialize_index()
        
#         # Créer session
#         session_id = str(uuid4())[:8]
#         monitoring_manager.current_session = MonitoringSession(
#             session_id=session_id,
#             start_time=datetime.datetime.now().isoformat(),
#             status="running"
#         )
        
#         monitoring_manager.is_running = True
#         monitoring_manager.start_time = time.time()
#         monitoring_manager.current_track = None
#         monitoring_manager.track_start_time = None
#         monitoring_manager.consecutive_detections = 0
        
#         # ✅ FORCER la création du dossier d'enregistrement
#         # monitoring_manager.audio_save_dir.mkdir(exist_ok=True)
#         # print(f"📁 Dossier d'enregistrement créé: {monitoring_manager.audio_save_dir}")
        
#         # logger.info(f"🎤 Surveillance CLIENT démarrée - Session: {session_id}")
#         # logger.info(f"📁 Enregistrement audio activé: {monitoring_manager.audio_save_dir}")
        
#         return ApiResponse(
#             success=True,
#             message="Surveillance streaming démarrée",
#             data={
#                 "session_id": session_id,
#                 "audio_recording_enabled": True,
#                 # "audio_path": str(monitoring_manager.audio_save_dir)
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"❌ Erreur start: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))




# @app.post("/monitoring/start", response_model=ApiResponse)
# async def start_monitoring(request: StartMonitoringRequest):
#     """
#     Version mise à jour qui utilise le streaming depuis Angular
#     """
#     return await start_monitoring_stream(request)



# @app.post("/stop", response_model=ApiResponse)
# async def stop_monitoring():
#     try:
#         monitoring_manager.stop_monitoring()
#         return ApiResponse(success=True, message="Monitoring stopped")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/status", response_model=MonitoringStatus)
# async def status_endpoint():
#     status_data = monitoring_manager.get_status()
#     return MonitoringStatus(**status_data)

# @app.post("/infer")
# async def infer_audio(file: UploadFile = File(...), duration: int = 5,
#                       min_score: float = 0.1, min_confidence: float = 0.1,
#                       min_consecutive: int = 2):
#     temp_path = None
#     start_time = time.time()
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
#             tmp.write(await file.read())
#             temp_path = tmp.name

#         if monitoring_manager.config is None:
#             default_config = "config_online.json"
#             monitoring_manager.load_configuration(default_config)
#             monitoring_manager.initialize_model()
#             monitoring_manager.initialize_index()

#         result = monitoring_manager.infer_from_file(
#             temp_path, duration, min_score, min_confidence, min_consecutive
#         )
#         result['inference_time_ms'] = (time.time()-start_time)*1000
#         return result
#     finally:
#         if temp_path and os.path.exists(temp_path):
#             os.unlink(temp_path)



# # ----------------- REPORTS ENDPOINTS -----------------
# @app.get("/reports/sessions", response_model=List[ReportSummary])
# async def get_sessions(
#     date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
#     limit: int = Query(100, ge=1, le=500)
# ):
#     """Liste toutes les sessions avec résumés"""
#     sessions = monitoring_manager.session_storage.list_sessions(date_filter=date, limit=limit)
    
#     summaries = []
#     for session in sessions:
#         # Calculer les top tracks
#         track_counts = {}
#         for detection in session.detections:
#             track_counts[detection.track] = track_counts.get(detection.track, 0) + 1
        
#         top_tracks = [
#             {"track": track, "count": count, "percentage": round(count/session.detection_count*100, 1)}
#             for track, count in sorted(track_counts.items(), key=lambda x: x[1], reverse=True)[:5]
#         ]
        
#         summaries.append(ReportSummary(
#             session_id=session.session_id,
#             date=datetime.datetime.fromisoformat(session.start_time).date().isoformat(),
#             start_time=session.start_time,
#             end_time=session.end_time or "En cours",
#             duration=round(session.total_duration, 2),
#             total_detections=session.detection_count,
#             unique_tracks=session.unique_tracks,
#             top_tracks=top_tracks
#         ))
    
#     return summaries

# @app.get("/reports/session/{session_id}", response_model=DetailedReport)
# async def get_session_detail(session_id: str):
#     """Rapport détaillé d'une session"""
#     session = monitoring_manager.session_storage.load_session(session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Session non trouvée")
    
#     # Statistiques par piste
#     track_stats = {}
#     for detection in session.detections:
#         if detection.track not in track_stats:
#             track_stats[detection.track] = {
#                 "count": 0,
#                 "total_duration": 0.0,
#                 "avg_confidence": 0.0,
#                 "detections": []
#             }
#         track_stats[detection.track]["count"] += 1
#         track_stats[detection.track]["total_duration"] += detection.duration_played
#         track_stats[detection.track]["detections"].append({
#             "timestamp": detection.timestamp,
#             "confidence": detection.confidence,
#             "duration": detection.duration_played
#         })
    
#     # Calculer moyennes
#     for track, stats in track_stats.items():
#         stats["avg_confidence"] = round(
#             sum(d["confidence"] for d in stats["detections"]) / len(stats["detections"]),
#             3
#         )
    
#     # Distribution horaire
#     hourly_dist = {}
#     for detection in session.detections:
#         hour = datetime.datetime.fromisoformat(detection.timestamp).hour
#         hourly_dist[f"{hour:02d}h"] = hourly_dist.get(f"{hour:02d}h", 0) + 1
    
#     # Distribution confiance
#     confidence_dist = {"high": 0, "medium": 0, "low": 0}
#     for detection in session.detections:
#         if detection.confidence >= 0.8:
#             confidence_dist["high"] += 1
#         elif detection.confidence >= 0.1:
#             confidence_dist["medium"] += 1
#         else:
#             confidence_dist["low"] += 1
    
#     return DetailedReport(
#         session=session,
#         track_stats=track_stats,
#         hourly_distribution=hourly_dist,
#         confidence_distribution=confidence_dist
#     )

# @app.delete("/reports/session/{session_id}")
# async def delete_session(session_id: str):
#     """Supprimer une session"""
#     try:
#         monitoring_manager.session_storage.delete_session(session_id)
#         return ApiResponse(success=True, message="Session supprimée")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/analyze_stream")
# async def analyze_audio_stream(audio: UploadFile = File(...)):
#     """
#     Endpoint pour recevoir et analyser un chunk audio du client Angular
#     """
#     if not monitoring_manager.is_running:
#         raise HTTPException(status_code=400, detail="Surveillance non démarrée")
    
#     temp_path = None
#     try:
#         # Sauvegarder temporairement
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
#             content = await audio.read()
#             tmp.write(content)
#             temp_path = tmp.name
        
#         # Charger l'audio
#         y, sr = librosa.load(temp_path, sr=None, mono=True)
        
#         # ✅ AJOUT: Enregistrer l'audio
#         # monitoring_manager._save_audio_for_debug(y, sr)
        
#         # Analyser
#         result = monitoring_manager.analyze_audio_chunk(y, sr)
        
#         if result:
#             # Créer événement de détection
#             detection = monitoring_manager.process_stream_detection(result)
            
#             if detection:
#                 # Envoyer via WebSocket
#                 import asyncio
#                 asyncio.run_coroutine_threadsafe(
#                     monitoring_manager.event_queue.put({
#                         'event': 'detection',
#                         'data': detection.dict()
#                     }),
#                     monitoring_manager.loop
#                 )
                
#                 return {
#                     'success': True,
#                     'detection': detection.dict()
#                 }
        
#         return {
#             'success': True,
#             'detection': None,
#             'message': 'Analysé mais pas de détection'
#         }
        
#     except Exception as e:
#         logger.error(f"Erreur analyze_stream: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
        
#     finally:
#         # Nettoyer
#         if temp_path and os.path.exists(temp_path):
#             os.unlink(temp_path)

# @app.post("/start_stream")
# async def start_monitoring_stream(request: StartMonitoringRequest):
#     """
#     Démarrer une session de surveillance en mode streaming (sans PyAudio)
#     """
#     try:
#         config_path = request.config_path or "config_online.json"
        
#         # Charger config et modèle SANS initialiser PyAudio
#         monitoring_manager.load_configuration(config_path)
#         monitoring_manager.initialize_model()
#         monitoring_manager.initialize_index()
        
#         # Créer session
#         from uuid import uuid4
#         import datetime
        
#         session_id = str(uuid4())[:8]
#         monitoring_manager.current_session = MonitoringSession(
#             session_id=session_id,
#             start_time=datetime.datetime.now().isoformat(),
#             status="running"
#         )
        
#         monitoring_manager.is_running = True
#         monitoring_manager.start_time = time.time()
#         monitoring_manager.current_track = None
#         monitoring_manager.track_start_time = None
#         monitoring_manager.consecutive_detections = 0
        
#         logger.info(f"Surveillance streaming démarrée - Session: {session_id}")
        
#         return ApiResponse(
#             success=True,
#             message="Surveillance streaming démarrée",
#             data={"session_id": session_id}
#         )
        
#     except Exception as e:
#         logger.error(f"Erreur start_stream: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/reports/export/{session_id}")
# async def export_session(
#     session_id: str,
#     format: str = Query("csv", regex="^(csv|json)$")
# ):
#     """Exporter une session en CSV ou JSON"""
#     session = monitoring_manager.session_storage.load_session(session_id)
#     if not session:
#         raise HTTPException(status_code=404, detail="Session non trouvée")
    
#     if format == "json":
#         return session.dict()
#     else:
#         return export_session_csv(session)


# def export_session_csv(session: MonitoringSession) -> StreamingResponse:
#     """Génère un export CSV de la session"""

#     # Étape 1 : regrouper par titre pour connaître la durée totale jouée
#     durations = {}
#     for detection in session.detections:
#         durations[detection.track] = durations.get(detection.track, 0) + detection.duration_played

#     # Étape 2 : filtrer les détections selon le total cumulé
#     data = []
#     for detection in session.detections:
#         total_duration = durations.get(detection.track, 0)
#         if total_duration < 10:
#             # Exclure ce titre entièrement
#             continue
        
#         dt = datetime.datetime.fromisoformat(detection.timestamp)
#         data.append({
#             'Session ID': session.session_id,
#             'Date': dt.strftime('%d/%m/%Y'),
#             'Heure': dt.strftime('%H:%M:%S'),
#             'Titre': detection.track,
#             # 'Durée de diffusion (sec)': round(detection.duration_played, 2),
#             'Durée Totale Titre (sec)': round(total_duration, 2),
#             'ID Détection': detection.id
#         })

#     # Étape 3 : création DataFrame
#     df = pd.DataFrame(data)
#     if df.empty:
#         df = pd.DataFrame(columns=[
#             'Session ID', 'Date', 'Heure', 'Titre',
#             'Durée Totale Titre (sec)', 'ID Détection'
#         ])

#     # Étape 4 : générer le CSV avec encodage UTF-8 BOM
#     output = StringIO()
#     output.write('\ufeff')
#     df.to_csv(output, index=False, encoding='utf-8', sep=',')
#     output.seek(0)

#     # Étape 5 : nom de fichier
#     date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"rapport_session_{session.session_id}_{date_str}.csv"

#     return StreamingResponse(
#         iter([output.getvalue()]),
#         media_type="text/csv; charset=utf-8",
#         headers={
#             "Content-Disposition": f"attachment; filename={filename}"
#         }
#     )





# # ----------------- WEBSOCKET -----------------
# @app.websocket("/ws/monitoring")
# async def websocket_monitoring(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             event = await monitoring_manager.event_queue.get()
#             await websocket.send_json(event)
#     except WebSocketDisconnect:
#         logger.info("WebSocket déconnecté")
#     except Exception as e:
#         logger.error(f"WebSocket erreur: {e}")




# -*- coding: utf-8 -*-
"""
FastAPI Professional Audio Monitoring API
Style BMAT - Gestion complète des sessions et rapports
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import datetime
import threading
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
from uuid import uuid4
import logging

# Après les imports existants, ajoutez :
import pandas as pd
import numpy as np
import torch
import faiss
import sounddevice as sd  # REMPLACEMENT PyAudio
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from io import StringIO
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)



# ----------------- LOGGING -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ----------------- PROJECT PATH -----------------
project_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_path))

# ----------------- IMPORTS DES MODULES EXISTANTS -----------------
from utils.utils import query_sequence_search, get_winner, extract_mel_spectrogram, search_index
from models.neural_fingerprinter import Neural_Fingerprinter

# ----------------- MODELS PYDANTIC -----------------
class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    model_loaded: bool

class DetectionEvent(BaseModel):
    id: str
    track: str
    score: float
    confidence: float
    offset: float
    timestamp: str
    session_id: str
    duration_played: float = 0.0

class MonitoringSession(BaseModel):
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_duration: float = 0.0
    detection_count: int = 0
    unique_tracks: int = 0
    detections: List[DetectionEvent] = []
    status: str = "running"  # running, completed, stopped

class MonitoringStatus(BaseModel):
    is_running: bool
    current_session: Optional[MonitoringSession] = None
    total_sessions: int = 0
    total_detections: int = 0

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class StartMonitoringRequest(BaseModel):
    config_path: Optional[str] = None

class ReportSummary(BaseModel):
    session_id: str
    date: str
    start_time: str
    end_time: str
    duration: float
    total_detections: int
    unique_tracks: int
    top_tracks: List[Dict[str, Any]]

class DetailedReport(BaseModel):
    session: MonitoringSession
    track_stats: Dict[str, Any]
    hourly_distribution: Dict[str, int]
    confidence_distribution: Dict[str, int]

# ----------------- SESSION STORAGE -----------------
class SessionStorage:
    """Stockage persistant des sessions de surveillance"""
    
    def __init__(self, storage_path: str = "monitoring_sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
    def save_session(self, session: MonitoringSession):
        """Sauvegarde une session"""
        filename = f"session_{session.session_id}.json"
        filepath = self.storage_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session.dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Session sauvegardée: {filename}")
    
    def load_session(self, session_id: str) -> Optional[MonitoringSession]:
        """Charge une session spécifique"""
        filepath = self.storage_path / f"session_{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return MonitoringSession(**data)
    
    def list_sessions(self, date_filter: Optional[str] = None, limit: int = 100) -> List[MonitoringSession]:
        """Liste toutes les sessions avec filtre optionnel"""
        sessions = []
        for filepath in sorted(self.storage_path.glob("session_*.json"), reverse=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session = MonitoringSession(**data)
                    
                    # Filtre par date si spécifié
                    if date_filter:
                        session_date = datetime.datetime.fromisoformat(session.start_time).date().isoformat()
                        if session_date != date_filter:
                            continue
                    
                    sessions.append(session)
                    if len(sessions) >= limit:
                        break
            except Exception as e:
                logger.error(f"Erreur lecture session {filepath}: {e}")
        return sessions
    
    def delete_session(self, session_id: str):
        """Supprime une session"""
        filepath = self.storage_path / f"session_{session_id}.json"
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Session supprimée: {session_id}")

# ----------------- MONITORING MANAGER -----------------
class MonitoringManager:
    def __init__(self):
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.start_time: Optional[float] = None

        self.config: Optional[Dict] = None
        self.model: Optional[Neural_Fingerprinter] = None
        self.index: Optional[faiss.Index] = None
        self.json_correspondence: Optional[Dict] = None
        self.sorted_arr: Optional[np.ndarray] = None
        self.stream: Optional[sd.InputStream] = None  # REMPLACEMENT PyAudio Stream
        self.device: str = 'cpu'

        # Session management
        self.current_session: Optional[MonitoringSession] = None
        self.session_storage = SessionStorage()
        
        # Track continuity detection
        self.current_track: Optional[str] = None
        self.track_start_time: Optional[datetime.datetime] = None
        self.consecutive_detections: int = 0
        self.min_detections_for_track: int = 1

        # Queue pour WebSocket events
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._lock = threading.Lock()

        # Event loop dédié
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

                # Nouveaux attributs pour streaming
        self.audio_buffer: list = []
        self.last_analysis_time: float = 0
        self.min_buffer_duration: float = 5.0  # secondes


                # ✅ AJOUT SIMPLE: Pour enregistrement audio
        # self.audio_save_dir = Path("recorded_audio")
        # self.audio_save_dir.mkdir(exist_ok=True)
        # self.audio_counter = 0


    # def _save_audio_for_debug(self, audio_data: np.ndarray, sample_rate: int):
    #     """Sauvegarde simple de l'audio pour debug"""
    #     try:
    #         timestamp = datetime.datetime.now().strftime("%H%M%S")
    #         filename = f"audio_{timestamp}.wav"
    #         filepath = self.audio_save_dir / filename
            
    #         # Sauvegarder avec soundfile
    #         import soundfile as sf
    #         sf.write(filepath, audio_data, sample_rate)
            
    #         print(f"🔊 Audio enregistré: {filename}")
            
    #     except Exception as e:
    #         print(f"❌ Erreur enregistrement: {e}")    


    def analyze_audio_chunk(self, audio_data: np.ndarray, sample_rate: int, chunk_index: int = 0) -> Optional[dict]:
        """
        ✅ Analyse un chunk audio de 5 secondes comme avec PyAudio
        """
        try:
            if self.model is None or self.index is None:
                logger.warning("Modèle non initialisé")
                return None

            F = self.config['SR']  # 8000 Hz
            H = self.config['Hop size']
            k = self.config['neighbors']
            
            logger.info(f"🎵 Chunk #{chunk_index}: {len(audio_data)} samples à {sample_rate}Hz")
            
            # ✅ VÉRIFIER LA DURÉE (doit être ~5 secondes)
            duration_sec = len(audio_data) / sample_rate
            logger.info(f"⏱️ Durée chunk: {duration_sec:.2f}s")
            
            if duration_sec < 4.5:  # Tolérance
                logger.warning(f"❌ Chunk trop court: {duration_sec:.2f}s < 4.5s")
                return None
            
            # ✅ RESAMPLER SI NÉCESSAIRE (normalement déjà à 8000 Hz)
            if sample_rate != F:
                logger.info(f"🔄 Resampling: {sample_rate}Hz → {F}Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=F)
                logger.info(f"✅ Après resampling: {len(audio_data)} samples")
            
            # ✅ VÉRIFIER SILENCE
            rms = np.sqrt(np.mean(audio_data ** 2))
            silence_threshold = self.config.get('silence_threshold', 0.01)
            if rms < silence_threshold:
                logger.info(f"🔇 Silence détecté (RMS: {rms:.4f} < {silence_threshold})")
                return None
            
            logger.info(f"🔊 Niveau audio: RMS = {rms:.4f}")
            
            # ✅ CALCULER SEGMENTS (comme PyAudio)
            J = max(1, int(np.floor((audio_data.size - F) / H)) + 1)
            logger.info(f"📈 Segments calculés: {J}")
            
            if J <= 0:
                logger.warning("❌ Aucun segment généré")
                return None
            
            # ✅ EXTRAIRE SPECTROGRAMMES
            xq = np.stack([
                extract_mel_spectrogram(audio_data[j*H:j*H+F]).reshape(1, 256, 32)
                for j in range(J)
            ])
            
            logger.info(f"🎼 Spectrogrammes extraits: shape={xq.shape}")
            
            # ✅ INFÉRENCE
            with torch.no_grad():
                out = self.model(torch.from_numpy(xq).to(self.device))
                D, I = self.index.search(out.cpu().numpy(), k)
            
            # ✅ IDENTIFICATION
            s_flag = self.config.get('search algorithm', 'majority vote')
            
            if s_flag == 'sequence search':
                idx, score = query_sequence_search(D, I)
                true_idx = search_index(idx, self.sorted_arr)
                winner = self.json_correspondence[str(true_idx)]
                offset = (idx - true_idx) * H / F
            else:
                winner, score = get_winner(self.json_correspondence, I, D, self.sorted_arr)
                offset = 0
            
            confidence = min(1.0, score / 10)
            
            logger.info(f"🎯 Résultat: {winner} | Score: {score:.2f} | Confiance: {confidence:.2f}")
            
            # ✅ SEUIL ADAPTATIF
            min_confidence = 0.4 if (self.current_track and winner == self.current_track) else 0.1
            
            if confidence >= min_confidence:
                logger.info(f"✅ DÉTECTION VALIDÉE: {winner}")
                return {
                    'track': winner,
                    'score': float(score),
                    'confidence': float(confidence),
                    'offset': float(offset),
                    'chunk_index': chunk_index
                }
            else:
                logger.info(f"❌ Confiance insuffisante: {confidence:.2f} < {min_confidence}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse chunk: {e}", exc_info=True)
            return None



    def process_stream_detection(self, result: dict) -> Optional[DetectionEvent]:
        """
        Traite un résultat avec continuité intelligente
        """
        if not result or not self.current_session:
            return None
        
        try:
            now = datetime.datetime.now()
            winner = result['track']
            chunk_index = result.get('chunk_index', 0)
            
            # ✅ FILTRAGE TEMPOREL ADAPTATIF
            time_since_last = float('inf')  # Infini si aucune détection précédente
            if self.current_session.detections:
                last_detection_time = datetime.datetime.fromisoformat(
                    self.current_session.detections[-1].timestamp
                )
                time_since_last = (now - last_detection_time).total_seconds()
            
            # ✅ LOGIQUE DE CONTINUITÉ AMÉLIORÉE
            is_same_track = winner == self.current_track
            
            if is_same_track:
                # Même piste - continuité
                self.consecutive_detections += 1
                logger.info(f"🔄 Continuité #{self.consecutive_detections} pour: {winner}")
                
                # Si même piste et délai raisonnable (< 15s), accepter plus facilement
                if time_since_last < 15:
                    # ✅ Accepter la détection de continuité
                    self.track_start_time = self.track_start_time or now
                    
                    detection = DetectionEvent(
                         id=str(uuid4())[:8],
                        track=winner,
                        score=result['score'],
                        confidence=result['confidence'],
                        offset=result['offset'],
                        timestamp=now.isoformat(),
                        session_id=self.current_session.session_id,
                        duration_played=0.0
                    )
                    
                    # Mettre à jour la durée de la dernière détection
                    if self.current_session.detections:
                        self.current_session.detections[-1].duration_played = (
                            now - datetime.datetime.fromisoformat(self.current_session.detections[-1].timestamp)
                        ).total_seconds()
                    
                    logger.info(f"🎵 DÉTECTION CONTINUE: {winner}")
                    return detection
                    
            else:
                # Nouvelle piste
                logger.info(f"🆕 Changement de piste: {self.current_track} → {winner}")
                
                # ✅ FILTRE moins agressif pour les nouvelles pistes (8s au lieu de 20s)
                if time_since_last < 8 and self.current_session.detections:
                    logger.info(f"⏰ Nouvelle piste ignorée (trop rapide: {time_since_last:.1f}s)")
                    return None
                
                # Finaliser l'ancienne piste
                if self.current_track and self.consecutive_detections >= self.min_detections_for_track:
                    duration_played = (now - self.track_start_time).total_seconds()
                    self._finalize_track_detection(duration_played)
                    logger.info(f"⏹️ Piste terminée: {self.current_track} ({duration_played:.1f}s)")
                
                # Nouvelle piste
                self.current_track = winner
                self.track_start_time = now
                self.consecutive_detections = 1
            
            # ✅ Créer événement pour nouvelle piste ou continuité forte
            if (not is_same_track or self.consecutive_detections >= 2):
                detection = DetectionEvent(
                    id=str(uuid4())[:8],
                    track=winner,
                    score=result['score'],
                    confidence=result['confidence'],
                    offset=result['offset'],
                    timestamp=now.isoformat(),
                    session_id=self.current_session.session_id,
                    duration_played=0.0
                )
                
                # Vérifier si nouvelle détection
                if not self.current_session.detections or \
                self.current_session.detections[-1].track != winner:
                    self.current_session.detections.append(detection)
                    self.current_session.detection_count += 1
                    
                    logger.info(f"🎵 NOUVELLE DÉTECTION: {winner}")
                    return detection
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur process_stream_detection: {e}")
            return None
   
    
    # ----------------- INITIALISATION -----------------
    def load_configuration(self, config_path: str):
        try:
            # Si c'est juste un nom de fichier, chercher dans le dossier evaluation/
            if not Path(config_path).is_absolute():
                config_file = Path(__file__).parent / config_path
            else:
                config_file = Path(config_path)
            
            if not config_file.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {config_file}")
            
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"✅ Configuration chargée: {config_file}")
        except Exception as e:
            logger.error(f"❌ Erreur de chargement de la configuration: {e}")
            raise

    def initialize_model(self):
        try:
            attention = self.config.get('attention', False)
            self.device = 'cuda' if torch.cuda.is_available() and self.config.get('device')=='cuda' else 'cpu'
            self.model = Neural_Fingerprinter(attention=attention).to(self.device)
            self.model.load_state_dict(torch.load(Path(project_path)/self.config['weights'], map_location=self.device))
            self.model.eval()
            logger.info(f"Modèle chargé sur {self.device}")
        except Exception as e:
            logger.error(f"Erreur de chargement du modèle: {e}")
            raise

    def initialize_index(self):
        try:
            self.index = faiss.read_index(str(Path(project_path)/self.config['index']))
            self.index.nprobe = self.config.get('nprobes', 1)
            with open(Path(project_path)/self.config['json'], 'r') as f:
                self.json_correspondence = json.load(f)
                self.sorted_arr = np.sort(np.array(list(map(int, self.json_correspondence.keys()))))
            logger.info("Index FAISS initialisé")
        except Exception as e:
            logger.error(f"Erreur d'initialisation de l'index: {e}")
            raise

    def initialize_audio(self):
        try:
            F, FMB = self.config['SR'], self.config['FMB']
            
            # REMPLACEMENT PyAudio par SoundDevice
            self.stream = sd.InputStream(
                samplerate=F,
                channels=1,
                dtype=np.float32,
                blocksize=FMB,
                callback=self._audio_callback if hasattr(self, '_audio_callback') else None
            )
            logger.info(f"Stream audio SoundDevice initialisé: {F}Hz")
        except Exception as e:
            logger.error(f"Erreur d'initialisation audio: {e}")
            raise

    # ----------------- SESSION MANAGEMENT -----------------
    def start_monitoring(self, config_path: str):
        with self._lock:
            if self.is_running:
                logger.warning("Surveillance déjà en cours")
                return
            
            self.load_configuration(config_path)
            self.initialize_model()
            self.initialize_index()
            self.initialize_audio()
            
            # Créer nouvelle session
            session_id = str(uuid4())[:8]
            self.current_session = MonitoringSession(
                session_id=session_id,
                start_time=datetime.datetime.now().isoformat(),
                status="running"
            )
            
            self.should_stop.clear()
            self.is_running = True
            self.start_time = time.time()
            self.current_track = None
            self.track_start_time = None
            self.consecutive_detections = 0
            
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info(f"Surveillance démarrée - Session: {session_id}")

    def stop_monitoring(self):
        with self._lock:
            if not self.is_running:
                logger.warning("Aucune surveillance en cours")
                return
            
            self.should_stop.set()
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            # Finaliser la session
            if self.current_session:
                self.current_session.end_time = datetime.datetime.now().isoformat()
                self.current_session.total_duration = time.time() - self.start_time
                self.current_session.status = "completed"
                self.current_session.unique_tracks = len(set(d.track for d in self.current_session.detections))
                
                # Sauvegarder la session
                self.session_storage.save_session(self.current_session)
                logger.info(f"Session terminée: {self.current_session.session_id}")
            
            self.cleanup()
            self.is_running = False
            self.current_session = None

    def cleanup(self):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Erreur cleanup: {e}")

    def get_status(self) -> Dict[str, Any]:
        """État actuel de la surveillance"""
        uptime = 0.0
        if self.start_time:
            uptime = time.time() - self.start_time

        # Mettre à jour la durée de la session courante
        if self.current_session:
            self.current_session.total_duration = uptime

        # Statistiques globales
        all_sessions = self.session_storage.list_sessions()
        total_sessions = len(all_sessions)
        total_detections = sum(s.detection_count for s in all_sessions)

        return {
            "is_running": self.is_running,
            "current_session": self.current_session,
            "total_sessions": total_sessions,
            "total_detections": total_detections
        }

    # ----------------- MONITORING LOOP -----------------
    def _monitoring_loop(self):
        F, H, FMB = self.config['SR'], self.config['Hop size'], self.config['FMB']
        dur, k = self.config['duration'], self.config['neighbors']
        s_flag = 'sequence search' if self.config.get('search algorithm')=='sequence search' else 'majority vote'
        silence_threshold = self.config.get('silence_threshold', 0.01)

        logger.info("Boucle de surveillance démarrée")
        try:
            # Démarrer le stream SoundDevice
            if self.stream:
                self.stream.start()
            
            with torch.no_grad():
                while not self.should_stop.is_set():
                    try:
                        # REMPLACEMENT: Lecture audio avec SoundDevice
                        # Pour SoundDevice, on utilise une approche différente pour collecter les données
                        frames_needed = int(F / FMB * dur)
                        audio_data = np.array([], dtype=np.float32)
                        
                        # Collecter les données audio pendant la durée spécifiée
                        for _ in range(frames_needed):
                            if self.should_stop.is_set():
                                break
                            
                            # Pour SoundDevice, on peut utiliser une queue ou buffer partagé
                            # Cette implémentation est simplifiée - peut nécessiter ajustement
                            time.sleep(FMB / F)  # Temps pour un bloc
                            
                            # Dans une vraie implémentation, vous utiliseriez un callback
                            # ou un buffer circulaire pour collecter les données
                            
                        # Pour l'instant, on utilise une approche simplifiée
                        # Cette partie devra être adaptée selon votre configuration
                        aggregated_buf = np.random.random(F * dur).astype(np.float32) * 0.1  # Placeholder
                        
                    except Exception as e:
                        logger.warning(f"Erreur audio: {e}")
                        continue

                    now = datetime.datetime.now()

                    # Filtrage silence
                    if np.mean(np.abs(aggregated_buf)) < silence_threshold:
                        continue

                    # Découpage et analyse
                    J = max(1, int(np.floor((aggregated_buf.size - F) / H)) + 1)
                    if J <= 0:
                        continue

                    xq = np.stack([
                        extract_mel_spectrogram(aggregated_buf[j*H:j*H+F]).reshape(1,256,32)
                        for j in range(J)
                    ])

                    out = self.model(torch.from_numpy(xq).to(self.device))
                    D, I = self.index.search(out.cpu().numpy(), k)

                    # Identification
                    if s_flag == 'sequence search':
                        idx, score = query_sequence_search(D, I)
                        true_idx = search_index(idx, self.sorted_arr)
                        winner = self.json_correspondence[str(true_idx)]
                        offset = (idx - true_idx) * H / F
                    else:
                        winner, score = get_winner(self.json_correspondence, I, D, self.sorted_arr)
                        offset = 0

                    confidence = min(1.0, score / 10)

                    # Gestion de la continuité des pistes
                    if winner == self.current_track:
                        self.consecutive_detections += 1
                    else:
                        # Changement de piste
                        if self.current_track and self.consecutive_detections >= self.min_detections_for_track:
                            # Calculer la durée de lecture
                            duration_played = (now - self.track_start_time).total_seconds()
                            self._finalize_track_detection(duration_played)
                        
                        # Nouvelle piste
                        self.current_track = winner
                        self.track_start_time = now
                        self.consecutive_detections = 1

                    # Créer événement de détection uniquement si confiance suffisante
                    if confidence >= 0.1 and self.consecutive_detections >= self.min_detections_for_track:
                        detection = DetectionEvent(
                            id=str(uuid4())[:8],
                            track=winner,
                            score=float(score),
                            confidence=float(confidence),
                            offset=float(offset),
                            timestamp=now.isoformat(),
                            session_id=self.current_session.session_id,
                            duration_played=0.0  # Sera mis à jour à la fin
                        )

                        # Vérifier si c'est une nouvelle détection unique
                        if not self.current_session.detections or \
                           self.current_session.detections[-1].track != winner:
                            self.current_session.detections.append(detection)
                            self.current_session.detection_count += 1

                            # Envoi WebSocket
                            asyncio.run_coroutine_threadsafe(
                                self.event_queue.put({'event':'detection','data':detection.dict()}),
                                self.loop
                            )

                    time.sleep(0.001)

        except Exception as e:
            logger.error(f"Erreur boucle monitoring: {e}", exc_info=True)
        finally:
            # Arrêter le stream SoundDevice
            if self.stream:
                self.stream.stop()
            # Finaliser la dernière piste
            if self.current_track and self.track_start_time:
                duration = (datetime.datetime.now() - self.track_start_time).total_seconds()
                self._finalize_track_detection(duration)
            logger.info("Boucle de surveillance terminée")

    def _finalize_track_detection(self, duration: float):
        """Finalise une détection de piste avec sa durée totale"""
        if self.current_session.detections:
            # Mettre à jour la durée de la dernière détection
            for detection in reversed(self.current_session.detections):
                if detection.track == self.current_track:
                    detection.duration_played = duration
                    break

    # ----------------- INFERENCE FILE -----------------
    def infer_from_file(self, audio_file_path: str, duration: int = 5,
                       min_score: float = 0.1, min_confidence: float = 0.1,
                       min_consecutive: int = 1) -> dict:
        try:
            y, sr = librosa.load(audio_file_path, sr=self.config['SR'], mono=True)
            F, H, k = self.config['SR'], self.config['Hop size'], self.config['neighbors']
            
            seg_len = duration * F
            n_segments = max(1, y.shape[0] // seg_len)
            segment_results = []

            self.model.eval()
            with torch.no_grad():
                for seg_idx in range(n_segments):
                    start = seg_idx * seg_len
                    end = min((seg_idx + 1) * seg_len, y.shape[0])
                    y_slice = y[start:end]

                    J = max(1, int(np.floor((y_slice.size - F) / H)) + 1)
                    if J <= 0:
                        continue

                    xq = np.stack([
                        extract_mel_spectrogram(y_slice[j*H:j*H+F]).reshape(1, 256, 32)
                        for j in range(J)
                    ])

                    out = self.model(torch.from_numpy(xq).to(self.device))
                    D, I = self.index.search(out.cpu().numpy(), k)

                    counts = {}
                    dists_per_song = {}
                    for j in range(J):
                        top_idx = int(I[j, 0])
                        top_dist = float(D[j, 0])
                        song_idx = search_index(top_idx, self.sorted_arr)
                        song_id = self.json_correspondence[str(song_idx)]
                        counts[song_id] = counts.get(song_id, 0) + 1
                        dists_per_song.setdefault(song_id, []).append(top_dist)

                    if not counts:
                        continue

                    majority_song = max(counts.items(), key=lambda kv: kv[1])[0]
                    confidence = counts[majority_song] / float(J)
                    mean_top1_dist = float(np.mean(dists_per_song[majority_song]))
                    score = 1.0 / (1.0 + mean_top1_dist)

                    segment_results.append({
                        "seg_idx": seg_idx,
                        "start_time_sec": start / F,
                        "end_time_sec": end / F,
                        "song": majority_song,
                        "confidence": confidence,
                        "score": score,
                        "valid": confidence >= min_confidence and score >= min_score
                    })

            # Filtrage segments consécutifs
            valid_segments = []
            i = 0
            while i < len(segment_results):
                if not segment_results[i]['valid']:
                    i += 1
                    continue
                run_song = segment_results[i]['song']
                run_start = i
                j = i + 1
                while (j < len(segment_results) and segment_results[j]['valid'] and
                       segment_results[j]['song'] == run_song and
                       segment_results[j]['seg_idx'] == segment_results[j-1]['seg_idx'] + 1):
                    j += 1
                if j - run_start >= min_consecutive:
                    valid_segments.extend(segment_results[run_start:j])
                i = j

            if not valid_segments:
                return {'success': False, 'message': 'No valid detection', 'segments': []}

            return {'success': True, 'segments': valid_segments}

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return {'success': False, 'message': str(e), 'segments': []}
        
# ----------------- INSTANCE -----------------
monitoring_manager = MonitoringManager()

# ----------------- FASTAPI APP -----------------
app = FastAPI(title="Professional Audio Monitoring API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API démarrée")
    yield
    if monitoring_manager.is_running:
        monitoring_manager.stop_monitoring()
    logger.info("API arrêtée")

app.router.lifespan_context = lifespan

# ----------------- ENDPOINTS -----------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime=time.time()-(monitoring_manager.start_time or time.time()),
        model_loaded=monitoring_manager.model is not None
    )



@app.post("/start", response_model=ApiResponse)
async def start_monitoring(request: StartMonitoringRequest):
    """
    Démarrer la surveillance - utilise maintenant le streaming client
    """
    try:
        config_path = request.config_path or "config_online.json"
        
        # Charger config et modèle SANS PyAudio
        monitoring_manager.load_configuration(config_path)
        monitoring_manager.initialize_model()
        monitoring_manager.initialize_index()
        
        # Créer session
        session_id = str(uuid4())[:8]
        monitoring_manager.current_session = MonitoringSession(
            session_id=session_id,
            start_time=datetime.datetime.now().isoformat(),
            status="running"
        )
        
        monitoring_manager.is_running = True
        monitoring_manager.start_time = time.time()
        monitoring_manager.current_track = None
        monitoring_manager.track_start_time = None
        monitoring_manager.consecutive_detections = 0
        
        # ✅ FORCER la création du dossier d'enregistrement
        # monitoring_manager.audio_save_dir.mkdir(exist_ok=True)
        # print(f"📁 Dossier d'enregistrement créé: {monitoring_manager.audio_save_dir}")
        
        # logger.info(f"🎤 Surveillance CLIENT démarrée - Session: {session_id}")
        # logger.info(f"📁 Enregistrement audio activé: {monitoring_manager.audio_save_dir}")
        
        return ApiResponse(
            success=True,
            message="Surveillance streaming démarrée",
            data={
                "session_id": session_id,
                "audio_recording_enabled": True,
                # "audio_path": str(monitoring_manager.audio_save_dir)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur start: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/monitoring/start", response_model=ApiResponse)
async def start_monitoring(request: StartMonitoringRequest):
    """
    Version mise à jour qui utilise le streaming depuis Angular
    """
    return await start_monitoring_stream(request)



@app.post("/stop", response_model=ApiResponse)
async def stop_monitoring():
    try:
        monitoring_manager.stop_monitoring()
        return ApiResponse(success=True, message="Monitoring stopped")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=MonitoringStatus)
async def status_endpoint():
    status_data = monitoring_manager.get_status()
    return MonitoringStatus(**status_data)

@app.post("/infer")
async def infer_audio(file: UploadFile = File(...), duration: int = 5,
                      min_score: float = 0.1, min_confidence: float = 0.1,
                      min_consecutive: int = 2):
    temp_path = None
    start_time = time.time()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        if monitoring_manager.config is None:
            default_config = "config_online.json"
            monitoring_manager.load_configuration(default_config)
            monitoring_manager.initialize_model()
            monitoring_manager.initialize_index()

        result = monitoring_manager.infer_from_file(
            temp_path, duration, min_score, min_confidence, min_consecutive
        )
        result['inference_time_ms'] = (time.time()-start_time)*1000
        return result
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)



# ----------------- REPORTS ENDPOINTS -----------------
@app.get("/reports/sessions", response_model=List[ReportSummary])
async def get_sessions(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=500)
):
    """Liste toutes les sessions avec résumés"""
    sessions = monitoring_manager.session_storage.list_sessions(date_filter=date, limit=limit)
    
    summaries = []
    for session in sessions:
        # Calculer les top tracks
        track_counts = {}
        for detection in session.detections:
            track_counts[detection.track] = track_counts.get(detection.track, 0) + 1
        
        top_tracks = [
            {"track": track, "count": count, "percentage": round(count/session.detection_count*100, 1)}
            for track, count in sorted(track_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        summaries.append(ReportSummary(
            session_id=session.session_id,
            date=datetime.datetime.fromisoformat(session.start_time).date().isoformat(),
            start_time=session.start_time,
            end_time=session.end_time or "En cours",
            duration=round(session.total_duration, 2),
            total_detections=session.detection_count,
            unique_tracks=session.unique_tracks,
            top_tracks=top_tracks
        ))
    
    return summaries

@app.get("/reports/session/{session_id}", response_model=DetailedReport)
async def get_session_detail(session_id: str):
    """Rapport détaillé d'une session"""
    session = monitoring_manager.session_storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    # Statistiques par piste
    track_stats = {}
    for detection in session.detections:
        if detection.track not in track_stats:
            track_stats[detection.track] = {
                "count": 0,
                "total_duration": 0.0,
                "avg_confidence": 0.0,
                "detections": []
            }
        track_stats[detection.track]["count"] += 1
        track_stats[detection.track]["total_duration"] += detection.duration_played
        track_stats[detection.track]["detections"].append({
            "timestamp": detection.timestamp,
            "confidence": detection.confidence,
            "duration": detection.duration_played
        })
    
    # Calculer moyennes
    for track, stats in track_stats.items():
        stats["avg_confidence"] = round(
            sum(d["confidence"] for d in stats["detections"]) / len(stats["detections"]),
            3
        )
    
    # Distribution horaire
    hourly_dist = {}
    for detection in session.detections:
        hour = datetime.datetime.fromisoformat(detection.timestamp).hour
        hourly_dist[f"{hour:02d}h"] = hourly_dist.get(f"{hour:02d}h", 0) + 1
    
    # Distribution confiance
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    for detection in session.detections:
        if detection.confidence >= 0.8:
            confidence_dist["high"] += 1
        elif detection.confidence >= 0.1:
            confidence_dist["medium"] += 1
        else:
            confidence_dist["low"] += 1
    
    return DetailedReport(
        session=session,
        track_stats=track_stats,
        hourly_distribution=hourly_dist,
        confidence_distribution=confidence_dist
    )

@app.delete("/reports/session/{session_id}")
async def delete_session(session_id: str):
    """Supprimer une session"""
    try:
        monitoring_manager.session_storage.delete_session(session_id)
        return ApiResponse(success=True, message="Session supprimée")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_stream")
async def analyze_audio_stream(audio: UploadFile = File(...)):
    """
    Endpoint pour recevoir et analyser un chunk audio du client Angular
    """
    if not monitoring_manager.is_running:
        raise HTTPException(status_code=400, detail="Surveillance non démarrée")
    
    temp_path = None
    try:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = tmp.name
        
        # Charger l'audio
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        
        # ✅ AJOUT: Enregistrer l'audio
        # monitoring_manager._save_audio_for_debug(y, sr)
        
        # Analyser
        result = monitoring_manager.analyze_audio_chunk(y, sr)
        
        if result:
            # Créer événement de détection
            detection = monitoring_manager.process_stream_detection(result)
            
            if detection:
                # Envoyer via WebSocket
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    monitoring_manager.event_queue.put({
                        'event': 'detection',
                        'data': detection.dict()
                    }),
                    monitoring_manager.loop
                )
                
                return {
                    'success': True,
                    'detection': detection.dict()
                }
        
        return {
            'success': True,
            'detection': None,
            'message': 'Analysé mais pas de détection'
        }
        
    except Exception as e:
        logger.error(f"Erreur analyze_stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Nettoyer
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/start_stream")
async def start_monitoring_stream(request: StartMonitoringRequest):
    """
    Démarrer une session de surveillance en mode streaming (sans PyAudio)
    """
    try:
        config_path = request.config_path or "config_online.json"
        
        # Charger config et modèle SANS initialiser PyAudio
        monitoring_manager.load_configuration(config_path)
        monitoring_manager.initialize_model()
        monitoring_manager.initialize_index()
        
        # Créer session
        from uuid import uuid4
        import datetime
        
        session_id = str(uuid4())[:8]
        monitoring_manager.current_session = MonitoringSession(
            session_id=session_id,
            start_time=datetime.datetime.now().isoformat(),
            status="running"
        )
        
        monitoring_manager.is_running = True
        monitoring_manager.start_time = time.time()
        monitoring_manager.current_track = None
        monitoring_manager.track_start_time = None
        monitoring_manager.consecutive_detections = 0
        
        logger.info(f"Surveillance streaming démarrée - Session: {session_id}")
        
        return ApiResponse(
            success=True,
            message="Surveillance streaming démarrée",
            data={"session_id": session_id}
        )
        
    except Exception as e:
        logger.error(f"Erreur start_stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/export/{session_id}")
async def export_session(
    session_id: str,
    format: str = Query("csv", regex="^(csv|json)$")
):
    """Exporter une session en CSV ou JSON"""
    session = monitoring_manager.session_storage.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    
    if format == "json":
        return session.dict()
    else:
        return export_session_csv(session)


def export_session_csv(session: MonitoringSession) -> StreamingResponse:
    """Génère un export CSV de la session"""

    # Étape 1 : regrouper par titre pour connaître la durée totale jouée
    durations = {}
    for detection in session.detections:
        durations[detection.track] = durations.get(detection.track, 0) + detection.duration_played

    # Étape 2 : filtrer les détections selon le total cumulé
    data = []
    for detection in session.detections:
        total_duration = durations.get(detection.track, 0)
        if total_duration < 10:
            # Exclure ce titre entièrement
            continue
        
        dt = datetime.datetime.fromisoformat(detection.timestamp)
        data.append({
            'Session ID': session.session_id,
            'Date': dt.strftime('%d/%m/%Y'),
            'Heure': dt.strftime('%H:%M:%S'),
            'Titre': detection.track,
            # 'Durée de diffusion (sec)': round(detection.duration_played, 2),
            'Durée Totale Titre (sec)': round(total_duration, 2),
            'ID Détection': detection.id
        })

    # Étape 3 : création DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(columns=[
            'Session ID', 'Date', 'Heure', 'Titre',
            'Durée Totale Titre (sec)', 'ID Détection'
        ])

    # Étape 4 : générer le CSV avec encodage UTF-8 BOM
    output = StringIO()
    output.write('\ufeff')
    df.to_csv(output, index=False, encoding='utf-8', sep=',')
    output.seek(0)

    # Étape 5 : nom de fichier
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"rapport_session_{session.session_id}_{date_str}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )





# ----------------- WEBSOCKET -----------------
@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            event = await monitoring_manager.event_queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté")
    except Exception as e:
        logger.error(f"WebSocket erreur: {e}")

