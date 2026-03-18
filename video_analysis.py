"""
video_analysis.py
=================
Analisi video calcetto — pipeline completa:
  1. Estrazione frame dal video (OpenCV)
  2. Rilevamento persone per frame (YOLOv8)
  3. Tracking giocatori multi-frame (ByteTrack integrato in Ultralytics)
  4. Classificazione maglia (bianca vs colorata) via HSV
  5. Aggregazione statistiche per giocatore
  6. Generazione voto (1-10) + commento testuale (Qwen2-VL + Ollama)
  7. Output JSON + CSV pronti per importare in app.py

Requisiti:
    pip install opencv-python ultralytics requests numpy
    ollama pull qwen2.5vl:7b
    ollama serve   (in un terminale separato)

Utilizzo:
    python video_analysis.py --video partita.mp4 --fps 1 --model qwen2.5vl:7b
"""

import argparse
import base64
import csv
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests

# ──────────────────────────────────────────────
# CONFIGURAZIONE DEFAULT
# ──────────────────────────────────────────────
DEFAULT_FPS_SAMPLE  = 1          # frame da analizzare al secondo
DEFAULT_YOLO_MODEL  = "yolov8n.pt"  # nano = veloce; yolov8s.pt = più preciso
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:7b"
DEFAULT_OLLAMA_URL   = "http://localhost:11434"
CONF_THRESHOLD       = 0.55      # confidenza minima YOLO
IOU_THRESHOLD        = 0.5       # soglia IoU ByteTrack

# ──────────────────────────────────────────────
# RANGE HSV PER CLASSIFICAZIONE MAGLIA
# ──────────────────────────────────────────────
# Bianca: alta luminosità, bassa saturazione
WHITE_HSV_LOWER = np.array([0,   0,  160], dtype=np.uint8)
WHITE_HSV_UPPER = np.array([180, 60, 255], dtype=np.uint8)

# Colorata: tutto ciò che non è bianco né pelle
# (esclusi toni pelle e bianco — cattura jersey colorati)
COLOR_HSV_RANGES = [
    (np.array([0,   80, 50],  dtype=np.uint8), np.array([20,  255, 255], dtype=np.uint8)),  # rosso/arancio
    (np.array([20,  80, 50],  dtype=np.uint8), np.array([35,  255, 255], dtype=np.uint8)),  # giallo
    (np.array([35,  80, 50],  dtype=np.uint8), np.array([85,  255, 255], dtype=np.uint8)),  # verde
    (np.array([85,  80, 50],  dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8)),  # blu/viola
    (np.array([140, 80, 50],  dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),  # rosso scuro
]

# ──────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def frame_to_base64(frame: np.ndarray, max_side: int = 512) -> str:
    """
    Converte un frame OpenCV in stringa base64 JPEG.
    Ridimensiona a max_side px sul lato lungo per ridurre il payload.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf).decode("utf-8")


def crop_bbox(frame: np.ndarray, bbox) -> np.ndarray:
    """Ritaglia il bounding box da un frame. bbox = (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


# ──────────────────────────────────────────────
# STEP 1 — ESTRAZIONE FRAME
# ──────────────────────────────────────────────
def get_video_info(video_path: str) -> tuple[float, int, float]:
    """Ritorna (video_fps, total_frames, duration_sec)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {video_path}")
    video_fps     = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec  = total_frames / video_fps if video_fps > 0 else 0
    cap.release()
    return video_fps, total_frames, duration_sec


# ──────────────────────────────────────────────
# STEP 2+3 — YOLO + BYTETRACK
# ──────────────────────────────────────────────
def detect_and_track_streaming(video_path: str, yolo_model_name: str,
                               fps_sample: float, action_fps: float = 0.0) -> tuple[dict, list, int]:
    """
    Processa il video in streaming — un frame alla volta, zero accumulo in RAM.
    Ritorna:
        track_data      — dati tracking per giocatore
        action_frames   — lista (frame_idx, frame) per analisi azioni LLM
        total_tracked   — numero frame usati per il tracking
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log("ERRORE: ultralytics non installato. Esegui: pip install ultralytics")
        sys.exit(1)

    video_fps, total_frames, duration_sec = get_video_info(video_path)
    log(f"Video: {video_fps:.1f} fps, {total_frames} frame, {duration_sec/60:.1f} min")

    track_step  = max(1, int(video_fps / fps_sample))
    action_step = max(1, int(video_fps / action_fps)) if action_fps > 0 else 0
    expected_track = total_frames // track_step

    log(f"Tracking: 1/{track_step} frame (≈{fps_sample:.1f}/s) → ~{expected_track} frame")
    if action_step:
        log(f"Azioni LLM: 1/{action_step} frame (≈{action_fps:.3f}/s) → ~{total_frames//action_step} frame")

    log(f"Caricamento modello YOLO: {yolo_model_name}")
    yolo = YOLO(yolo_model_name)

    track_data    = defaultdict(lambda: {"frames": [], "crops": []})
    action_frames = []
    frame_idx     = 0
    tracked_count = 0

    cap = cv2.VideoCapture(video_path)
    log("Avvio tracking in streaming...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame per azioni LLM — salva una copia ridotta
        if action_step and frame_idx % action_step == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            action_frames.append((frame_idx, small))

        # Frame per tracking YOLO
        if frame_idx % track_step == 0:
            tracked_count += 1
            if tracked_count % 500 == 0:
                pct = tracked_count / expected_track * 100
                log(f"  Tracking {tracked_count}/{expected_track} ({pct:.0f}%)")

            results = yolo.track(
                frame,
                persist=True,
                classes=[0],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for box, track_id in zip(boxes.xyxy, boxes.id.int()):
                    tid  = int(track_id)
                    bbox = box.cpu().numpy()
                    crop = crop_bbox(frame, bbox)
                    if crop.size == 0:
                        continue
                    track_data[tid]["frames"].append((frame_idx, bbox.tolist()))
                    if len(track_data[tid]["crops"]) < 10:
                        track_data[tid]["crops"].append(crop)

        frame_idx += 1

    cap.release()
    log(f"Streaming completato — {len(track_data)} ID unici, {len(action_frames)} frame azioni")
    return dict(track_data), action_frames, tracked_count


# ──────────────────────────────────────────────
# STEP 4 — CLASSIFICAZIONE MAGLIA
# ──────────────────────────────────────────────
def classify_jersey(crops: list[np.ndarray]) -> str:
    """
    Classifica il colore della maglia da una lista di crop.
    Ritorna 'bianca', 'colorata' o 'sconosciuta'.

    Strategia:
    - Converte in HSV
    - Analizza la zona torso (50% centrale del crop)
    - Vota a maggioranza su tutti i crop disponibili
    """
    votes = {"bianca": 0, "colorata": 0}

    for crop in crops:
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        # Taglia solo il torso (20%-70% verticale, 20%-80% orizzontale)
        h, w = crop.shape[:2]
        torso = crop[int(h*0.20):int(h*0.70), int(w*0.20):int(w*0.80)]
        if torso.size == 0:
            continue

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # Maschera bianco
        white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
        white_px = cv2.countNonZero(white_mask)

        # Maschera colorata (unione di tutti i range)
        color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_HSV_RANGES:
            color_mask |= cv2.inRange(hsv, lower, upper)
        color_px = cv2.countNonZero(color_mask)

        total_px = torso.shape[0] * torso.shape[1]
        if total_px == 0:
            continue

        white_ratio = white_px / total_px
        color_ratio = color_px / total_px

        if white_ratio > 0.30:
            votes["bianca"] += 1
        elif color_ratio > 0.20:
            votes["colorata"] += 1

    if votes["bianca"] == 0 and votes["colorata"] == 0:
        return "sconosciuta"
    return max(votes, key=votes.get)


# ──────────────────────────────────────────────
# STEP 5 — AGGREGAZIONE STATISTICHE
# ──────────────────────────────────────────────
def aggregate_stats(track_data: dict, total_frames: int, fps_sample: float) -> list[dict]:
    """
    Per ogni track_id calcola metriche calcistiche derivate dal tracking:
    - presenza e continuità (quanto è rimasto in campo)
    - zona prevalente e ampiezza di copertura
    - mobilità e velocità media di spostamento
    - attività offensiva/difensiva (tempo nelle rispettive metà campo)
    - costanza (presenza continua vs apparizioni sporadiche)
    """
    players = []

    for tid, data in track_data.items():
        if len(data["frames"]) < 8:
            continue

        squadra = classify_jersey(data["crops"])
        frame_count = len(data["frames"])
        presenza_sec = frame_count / fps_sample

        # Posizioni nel tempo (ordinate per frame)
        frames_sorted = sorted(data["frames"], key=lambda x: x[0])
        positions = []
        frame_indices = []
        for fidx, bbox in frames_sorted:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            positions.append((cx, cy))
            frame_indices.append(fidx)

        pos_arr = np.array(positions)
        avg_x = float(np.mean(pos_arr[:, 0]))
        avg_y = float(np.mean(pos_arr[:, 1]))

        # Mobilità totale (std posizione)
        mobility = float(np.std(pos_arr[:, 0]) + np.std(pos_arr[:, 1]))

        # Velocità media frame-to-frame (spostamento in pixel/frame)
        if len(pos_arr) > 1:
            deltas = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)
            avg_speed = float(np.mean(deltas))
            max_speed = float(np.max(deltas))
        else:
            avg_speed = 0.0
            max_speed = 0.0

        # Copertura del campo: area del rettangolo minimo che racchiude le posizioni
        if len(pos_arr) > 2:
            x_range = float(np.max(pos_arr[:, 0]) - np.min(pos_arr[:, 0]))
            y_range = float(np.max(pos_arr[:, 1]) - np.min(pos_arr[:, 1]))
            campo_coperto = round(x_range * y_range / 1000, 1)  # in unità relative
        else:
            campo_coperto = 0.0

        # Tempo in zona offensiva vs difensiva
        # y bassa = zona alta del frame = solitamente offensiva (dipende dall'orientamento)
        zona_off  = sum(1 for p in positions if p[1] < np.percentile(pos_arr[:, 1], 33))
        zona_dif  = sum(1 for p in positions if p[1] > np.percentile(pos_arr[:, 1], 66))
        zona_cen  = frame_count - zona_off - zona_dif
        perc_off  = round(zona_off / frame_count * 100)
        perc_dif  = round(zona_dif / frame_count * 100)
        perc_cen  = round(zona_cen / frame_count * 100)

        # Costanza: quanti "blocchi continui" di presenza (meno blocchi = più continuo)
        if len(frame_indices) > 1:
            gaps = [frame_indices[i+1] - frame_indices[i] for i in range(len(frame_indices)-1)]
            n_interruzioni = sum(1 for g in gaps if g > fps_sample * 5)  # gap > 5 secondi
        else:
            n_interruzioni = 0
        costanza = "alta" if n_interruzioni == 0 else "media" if n_interruzioni <= 2 else "bassa"

        # Ruolo inferito dalla posizione media e mobilità
        if mobility > 200 and perc_cen > 40:
            ruolo_inferito = "centrocampista (mezzala)"
        elif perc_off > 45:
            ruolo_inferito = "attaccante"
        elif perc_dif > 45:
            ruolo_inferito = "difensore"
        elif mobility > 150:
            ruolo_inferito = "centrocampista"
        else:
            ruolo_inferito = "centrocampista / jolly"

        # Crop migliore
        best_crop = None
        best_area = 0
        for crop in data["crops"]:
            area = crop.shape[0] * crop.shape[1]
            if area > best_area:
                best_area = area
                best_crop = crop

        players.append({
            "track_id":       tid,
            "squadra":        squadra,
            "frame_count":    frame_count,
            "presenza_sec":   round(presenza_sec, 1),
            "avg_x":          round(avg_x, 1),
            "avg_y":          round(avg_y, 1),
            "mobility":       round(mobility, 1),
            "avg_speed":      round(avg_speed, 2),
            "max_speed":      round(max_speed, 2),
            "campo_coperto":  campo_coperto,
            "perc_off":       perc_off,
            "perc_dif":       perc_dif,
            "perc_cen":       perc_cen,
            "costanza":       costanza,
            "n_interruzioni": n_interruzioni,
            "ruolo_inferito": ruolo_inferito,
            "best_crop":      best_crop,
        })

    players.sort(key=lambda p: p["presenza_sec"], reverse=True)
    log(f"Giocatori validi: {len(players)} (bianca: {sum(1 for p in players if p['squadra']=='bianca')}, colorata: {sum(1 for p in players if p['squadra']=='colorata')})")
    return players


# ──────────────────────────────────────────────
# STEP 6 — VALUTAZIONE CON QWEN2-VL VIA OLLAMA
# ──────────────────────────────────────────────

def unload_model(ollama_url: str, model: str):
    """Forza Ollama a scaricare il modello dalla VRAM dopo ogni inferenza."""
    try:
        requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10
        )
    except Exception:
        pass  # non bloccante


# ──────────────────────────────────────────────
# STEP 5b — ANALISI AZIONI FRAME PER FRAME
# ──────────────────────────────────────────────
ACTION_PROMPT = """Sei un analista di calcetto amatoriale. Guarda questo frame di una partita.

Identifica SOLO azioni tecniche visibili in questo preciso momento:
- PASSAGGIO: un giocatore sta passando la palla a un compagno
- TIRO: un giocatore sta calciando verso la porta
- TACKLE: un giocatore sta contrastando un avversario
- GOL: la palla è in rete o un giocatore sta esultando
- POSSESSO: un giocatore controlla la palla
- CONTRASTO_AEREO: due giocatori saltano per il pallone
- NESSUNA: nessuna azione tecnica visibile (giocatori in movimento libero)

Per ogni azione visibile rispondi in questo formato JSON, nient'altro:
{"azioni": [{"tipo": "TIPO_AZIONE", "zona": "sinistra|centro|destra", "meta_campo": "offensiva|difensiva|centrale", "maglia": "bianca|colorata|incerto"}]}

Se non vedi azioni tecniche: {"azioni": []}"""

def analyze_actions(frames: list[tuple[int, np.ndarray]],
                    track_data: dict,
                    ollama_url: str, model: str,
                    n_frames: int = 30,
                    use_vision: bool = True) -> dict:
    """
    Analizza N frame campionati dall'intero video per rilevare azioni tecniche.
    Associa ogni azione al giocatore più vicino alla zona dell'azione.
    Ritorna:
        action_stats = {
            track_id: {
                "passaggi": int, "tiri": int, "tackle": int,
                "gol": int, "possesso": int, "contrasti_aerei": int
            }
        }
    """
    if not use_vision:
        log("Analisi azioni saltata (--no-vision attivo)")
        return {}

    # Campiona N frame distribuiti uniformemente
    step = max(1, len(frames) // n_frames)
    sampled = frames[::step][:n_frames]
    log(f"Analisi azioni su {len(sampled)} frame con {model}...")

    # Mappa frame_idx -> {track_id: (cx, cy)} per associazione spaziale
    frame_positions = {}
    for tid, data in track_data.items():
        for fidx, bbox in data["frames"]:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if fidx not in frame_positions:
                frame_positions[fidx] = {}
            frame_positions[fidx][tid] = (cx, cy)

    # Contatori azioni per track_id
    action_stats = defaultdict(lambda: {
        "passaggi": 0, "tiri": 0, "tackle": 0,
        "gol": 0, "possesso": 0, "contrasti_aerei": 0
    })

    ACTION_MAP = {
        "PASSAGGIO":        "passaggi",
        "TIRO":             "tiri",
        "TACKLE":           "tackle",
        "GOL":              "gol",
        "POSSESSO":         "possesso",
        "CONTRASTO_AEREO":  "contrasti_aerei",
    }

    # Dimensioni frame per normalizzare zona -> coordinate approssimative
    sample_frame = sampled[0][1]
    frame_h, frame_w = sample_frame.shape[:2]

    for i, (fidx, frame) in enumerate(sampled):
        if i % 10 == 0:
            log(f"  Azioni frame {i+1}/{len(sampled)}...")

        image_b64 = frame_to_base64(frame, max_side=768)

        payload = {
            "model": model,
            "prompt": ACTION_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300, "num_ctx": 2048}
        }

        try:
            resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
            if resp.status_code != 200:
                time.sleep(8)
                continue
            raw = resp.json().get("response", "").strip()

            # Pulizia JSON (il modello a volte aggiunge testo prima/dopo)
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                time.sleep(4)
                continue
            data_json = json.loads(raw[start:end])
            azioni = data_json.get("azioni", [])

        except (json.JSONDecodeError, Exception):
            time.sleep(4)
            continue

        # Associa ogni azione al giocatore più vicino alla zona
        players_in_frame = frame_positions.get(fidx, {})
        if not players_in_frame:
            # Usa frame adiacenti se questo non ha tracking
            for delta in [-1, 1, -2, 2]:
                players_in_frame = frame_positions.get(fidx + delta, {})
                if players_in_frame:
                    break

        for azione in azioni:
            tipo = azione.get("tipo", "").upper()
            if tipo not in ACTION_MAP or tipo == "NESSUNA":
                continue

            zona       = azione.get("zona", "centro")
            meta_campo = azione.get("meta_campo", "centrale")
            maglia     = azione.get("maglia", "incerto")

            # Stima coordinate della zona dell'azione
            zone_x = {"sinistra": frame_w * 0.2, "centro": frame_w * 0.5, "destra": frame_w * 0.8}.get(zona, frame_w * 0.5)
            zone_y = {"offensiva": frame_h * 0.2, "centrale": frame_h * 0.5, "difensiva": frame_h * 0.8}.get(meta_campo, frame_h * 0.5)

            # Trova il giocatore più vicino alla zona, con filtro maglia
            best_tid  = None
            best_dist = float("inf")

            for tid, (cx, cy) in players_in_frame.items():
                dist = ((cx - zone_x)**2 + (cy - zone_y)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid

            if best_tid is not None and best_dist < frame_w * 0.4:
                action_stats[best_tid][ACTION_MAP[tipo]] += 1

        unload_model(ollama_url, model)
        time.sleep(3)

    # Riepilogo
    total_actions = sum(sum(v.values()) for v in action_stats.values())
    log(f"Azioni rilevate totali: {total_actions} su {len(sampled)} frame")
    return dict(action_stats)

def build_prompt(player: dict, player_index: int, total_players: int) -> str:
    """
    Costruisce il prompt con metriche calcistiche complete.
    """
    presenza_min = int(player["presenza_sec"] // 60)
    presenza_sec_r = int(player["presenza_sec"] % 60)
    presenza_fmt = f"{presenza_min}m {presenza_sec_r}s" if presenza_min > 0 else f"{presenza_sec_r}s"

    mobilita_desc = (
        "molto alta — giocatore in costante movimento"  if player["mobility"] > 250 else
        "alta — buona copertura del campo"              if player["mobility"] > 150 else
        "media — movimenti selettivi"                   if player["mobility"] > 80  else
        "bassa — giocatore statico"
    )

    velocita_desc = (
        "veloce — scatti frequenti"    if player["avg_speed"] > 15 else
        "nella media"                  if player["avg_speed"] > 7  else
        "lento — ritmo compassato"
    )

    copertura_desc = (
        "ampia — copre gran parte del campo"  if player["campo_coperto"] > 50 else
        "media"                               if player["campo_coperto"] > 20 else
        "ristretta — agisce in zona limitata"
    )

    # Sezione azioni tecniche (se disponibili)
    actions = player.get("actions", {})
    if actions and any(v > 0 for v in actions.values()):
        azioni_str = "\n".join([
            f"  Passaggi:         {actions.get('passaggi', 0)}",
            f"  Tiri in porta:    {actions.get('tiri', 0)}",
            f"  Tackle difensivi: {actions.get('tackle', 0)}",
            f"  Gol/esultanze:    {actions.get('gol', 0)}",
            f"  Possesso palla:   {actions.get('possesso', 0)}",
            f"  Contrasti aerei:  {actions.get('contrasti_aerei', 0)}",
        ])
        totale_azioni = sum(actions.values())
        azioni_section = f"""
Azioni tecniche rilevate (totale: {totale_azioni}):
{azioni_str}"""
    else:
        azioni_section = "\nAzioni tecniche: nessuna rilevata nei frame analizzati"

    return f"""Sei un commentatore esperto di calcetto amatoriale. Analizza la performance del giocatore {player_index} di {total_players} basandoti ESCLUSIVAMENTE sui dati rilevati dal video.

═══ DATI OGGETTIVI ═══
Squadra:          {player["squadra"]}
Ruolo inferito:   {player["ruolo_inferito"]}
Tempo in campo:   {presenza_fmt}
Costanza:         {player["costanza"]} ({player["n_interruzioni"]} interruzioni)

Distribuzione per zona:
  Offensiva:  {player["perc_off"]}% del tempo
  Centrale:   {player["perc_cen"]}% del tempo
  Difensiva:  {player["perc_dif"]}% del tempo

Mobilità:         {mobilita_desc} (indice: {player["mobility"]:.0f})
Velocità media:   {velocita_desc} ({player["avg_speed"]:.1f} px/frame)
Velocità massima: {player["max_speed"]:.1f} px/frame
Copertura campo:  {copertura_desc} (indice: {player["campo_coperto"]:.0f})
{azioni_section}
═══════════════════════

Scrivi in italiano una valutazione calcistica realistica. Considera:
- Le azioni tecniche (passaggi, tiri, tackle) sono il fattore PIÙ IMPORTANTE
- Un giocatore con molti passaggi e tackle merita voto alto anche se poco mobile
- Zero azioni tecniche con alta mobilità = 5-6 (tanto movimento, poco contributo)
- Gol o tiri = bonus significativo sul voto
- La costanza premia chi è sempre in campo senza interruzioni

Formato risposta (rispetta ESATTAMENTE, nessun testo prima):
VOTO: <numero intero da 1 a 10>
COMMENTO: <2-3 frasi in italiano che citano PRIMA le azioni tecniche poi mobilità e zona>"""


def evaluate_player_with_llm(player: dict, player_index: int, total_players: int,
                               ollama_url: str, model: str,
                               max_retries: int = 3, pause_sec: float = 15.0,
                               use_vision: bool = True) -> tuple[float, str]:
    """
    Invia crop + prompt a Qwen2-VL via Ollama.
    - Immagine ridimensionata a 512px per ridurre il carico in RAM
    - Retry automatico con backoff esponenziale su errore 500
    - Pausa tra chiamate per lasciare tempo a Ollama di liberare la memoria
    Ritorna (voto, commento).
    """
    if player["best_crop"] is None:
        return 5.0, "Nessuna immagine disponibile per la valutazione."

    prompt = build_prompt(player, player_index, total_players)
    if use_vision and player["best_crop"] is not None:
        image_b64 = frame_to_base64(player["best_crop"], max_side=512)
    else:
        image_b64 = None

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 200,
            "num_ctx": 2048
        }
    }
    if image_b64:
        payload["images"] = [image_b64]

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=120
            )

            # 500 = Ollama sotto pressione — aspetta e riprova
            if resp.status_code == 500:
                wait = pause_sec * attempt  # backoff: 8s, 16s, 24s
                log(f"    500 Server Error (tentativo {attempt}/{max_retries}) — attendo {wait:.0f}s...")
                time.sleep(wait)
                last_error = f"500 dopo {max_retries} tentativi"
                continue

            resp.raise_for_status()
            text = resp.json().get("response", "").strip()

            # Parsing risposta
            voto = 5.0
            commento = text

            for line in text.split("\n"):
                line = line.strip()
                if line.upper().startswith("VOTO:"):
                    try:
                        voto = float(line.split(":", 1)[1].strip().split()[0])
                        voto = max(1.0, min(10.0, voto))
                    except (ValueError, IndexError):
                        pass
                elif line.upper().startswith("COMMENTO:"):
                    commento = line.split(":", 1)[1].strip()

            # Scarica il modello dalla VRAM e aspetta che la memoria si liberi
            unload_model(ollama_url, model)
            time.sleep(pause_sec)
            return voto, commento

        except requests.exceptions.ConnectionError:
            return 5.0, "Errore: Ollama non raggiungibile. Avvia con `ollama serve`."
        except requests.exceptions.Timeout:
            wait = pause_sec * attempt
            log(f"    Timeout (tentativo {attempt}/{max_retries}) — attendo {wait:.0f}s...")
            time.sleep(wait)
            last_error = "timeout"
        except Exception as e:
            return 5.0, f"Errore LLM: {e}"

    return 5.0, f"Valutazione fallita dopo {max_retries} tentativi ({last_error})."


# ──────────────────────────────────────────────
# STEP 7 — OUTPUT JSON + CSV
# ──────────────────────────────────────────────
def save_results(players: list[dict], output_dir: str, video_name: str):
    """
    Salva i risultati in:
    - video_analysis_results.json  (completo)
    - video_analysis_votes.csv     (importabile in app.py)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepara dati serializzabili (rimuovi crop numpy)
    clean = []
    for p in players:
        clean.append({
            "track_id":    p["track_id"],
            "squadra":     p["squadra"],
            "presenza_sec": p["presenza_sec"],
            "zona":        "offensiva" if p["avg_y"] < 0.4 else "difensiva" if p["avg_y"] > 0.6 else "centrale",
            "mobilita":    "alta" if p["mobility"] > 150 else "media" if p["mobility"] > 80 else "bassa",
            "voto":        p.get("voto", 5.0),
            "commento":    p.get("commento", ""),
        })

    # JSON
    json_path = os.path.join(output_dir, f"{video_name}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"video": video_name, "players": clean}, f, ensure_ascii=False, indent=2)
    log(f"JSON salvato: {json_path}")

    # CSV (compatibile con votes_comments.csv dell'app principale)
    csv_path = os.path.join(output_dir, f"{video_name}_votes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "matchId", "matchDate", "playerId", "playerName", "vote", "comment"
        ])
        writer.writeheader()
        for p in clean:
            writer.writerow({
                "matchId":    f"video_{video_name}",
                "matchDate":  time.strftime("%d/%m/%Y %H:%M"),
                "playerId":   f"track_{p['track_id']}",
                "playerName": f"Giocatore {p['track_id']} ({p['squadra']})",
                "vote":       p["voto"],
                "comment":    p["commento"],
            })
    log(f"CSV salvato: {csv_path}")

    return json_path, csv_path



# ──────────────────────────────────────────────
# FILTRO PER NUMERO GIOCATORI ATTESO
# ──────────────────────────────────────────────
def filter_by_player_count(players: list[dict], expected: int) -> list[dict]:
    """
    Filtra i giocatori tenendo solo i `expected` track più stabili.

    Strategia:
    1. Divide i candidati per squadra (bianca / colorata)
    2. Prova a prendere expected//2 per squadra (es. 5+5 per una 10v10)
    3. Se una squadra ha meno giocatori del previsto, compensa dall'altra
    4. In caso di pareggio nella presenza, preferisce track con più frame

    Questo elimina i falsi positivi (arbitro, raccattapalle, spettatori
    entrati nell'inquadratura) che tendono ad avere presenza molto bassa.
    """
    if expected <= 0:
        return players

    half = expected // 2
    odd  = expected % 2  # se dispari (es. 11), una squadra ha un giocatore in più

    bianche   = [p for p in players if p["squadra"] == "bianca"]
    colorate  = [p for p in players if p["squadra"] == "colorata"]
    # Già ordinati per presenza_sec desc, ma ri-ordiniamo per sicurezza
    bianche.sort(key=lambda p: (p["presenza_sec"], p["frame_count"]), reverse=True)
    colorate.sort(key=lambda p: (p["presenza_sec"], p["frame_count"]), reverse=True)

    selected_b = bianche[:half + odd]
    selected_c = colorate[:half]

    # Se una squadra è corta, prende il surplus dall'altra
    deficit_b = (half + odd) - len(selected_b)
    deficit_c = half - len(selected_c)

    if deficit_b > 0:
        extra = colorate[half: half + deficit_b]
        selected_c = colorate[:half + deficit_b]
        selected_b = bianche[:half + odd]
    if deficit_c > 0:
        selected_b = bianche[:half + odd + deficit_c]
        selected_c = colorate[:half]

    result = selected_b + selected_c
    result.sort(key=lambda p: p["presenza_sec"], reverse=True)

    scartati = len(players) - len(result)
    log(f"Filtro giocatori: attesi={expected}, selezionati={len(result)} "
        f"(bianca={len(selected_b)}, colorata={len(selected_c)}), scartati={scartati}")

    return result

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analisi video calcetto con AI")
    parser.add_argument("--video",   required=True,               help="Percorso del video (.mp4/.mov)")
    parser.add_argument("--fps",     type=float, default=DEFAULT_FPS_SAMPLE,
                        help="Frame al secondo per il tracking YOLO (default: 1). Aumenta per video lunghi (es. 10)")
    parser.add_argument("--action-fps", type=float, default=0.033,
                        help="Frame al secondo per analisi azioni LLM (default: 0.033 = 1 frame ogni 30s)")
    parser.add_argument("--yolo",    default=DEFAULT_YOLO_MODEL,  help="Modello YOLO (default: yolov8n.pt)")
    parser.add_argument("--model",   default=DEFAULT_OLLAMA_MODEL, help="Modello Ollama (default: qwen2.5vl:7b)")
    parser.add_argument("--ollama",  default=DEFAULT_OLLAMA_URL,  help="URL Ollama (default: http://localhost:11434)")
    parser.add_argument("--output",  default="output",            help="Cartella output (default: ./output)")
    parser.add_argument("--no-llm",  action="store_true",         help="Salta la valutazione LLM (solo tracking)")
    parser.add_argument("--max-players", type=int, default=20,    help="Max giocatori da valutare con LLM (default: 20)")
    parser.add_argument("--no-vision", action="store_true",
                        help="Usa solo dati statistici senza inviare immagini (più leggero, compatibile con llama3.2)")
    parser.add_argument("--players", type=int, default=None,
                        help="Numero TOTALE di giocatori in campo (es. 10 per 5v5). Filtra i falsi positivi.")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        log(f"ERRORE: file video non trovato: {args.video}")
        sys.exit(1)

    video_name = Path(args.video).stem
    start_time = time.time()

    # ── STEP 1+2+3: Tracking in streaming (nessun accumulo RAM)
    action_fps_val = args.action_fps if not args.no_vision else 0.0
    track_data, action_frames_list, total_frames = detect_and_track_streaming(
        args.video, args.yolo,
        fps_sample=args.fps,
        action_fps=action_fps_val
    )

    # ── STEP 4+5: Classificazione + aggregazione
    players = aggregate_stats(track_data, total_frames, args.fps)

    if not players:
        log("Nessun giocatore rilevato. Verifica il video e le soglie di confidenza.")
        sys.exit(0)

    # ── FILTRO per numero giocatori atteso
    if args.players:
        log(f"Numero giocatori atteso: {args.players} — applico filtro falsi positivi")
        players = filter_by_player_count(players, args.players)
        if not players:
            log("ERRORE: nessun giocatore rimasto dopo il filtro.")
            sys.exit(0)

    # Stampa riepilogo tracking
    print("\n" + "="*55)
    print(f"{'ID':>4}  {'Squadra':12} {'Presenza':>9}  {'Zona':12} {'Mobilità':>9}")
    print("-"*55)
    for p in players:
        zona = "offensiva" if p["avg_y"] < 0.4 else "difensiva" if p["avg_y"] > 0.6 else "centrale"
        mob  = "alta" if p["mobility"] > 150 else "media" if p["mobility"] > 80 else "bassa"
        print(f"{p['track_id']:>4}  {p['squadra']:12} {p['presenza_sec']:>7.0f}s  {zona:12} {mob:>9}")
    print("="*55 + "\n")

    # ── STEP 5b: Analisi azioni frame per frame
    if not args.no_llm and not args.no_vision:
        action_stats = analyze_actions(
            action_frames_list, track_data,
            args.ollama, args.model,
            n_frames=len(action_frames_list),
            use_vision=True
        )
        # Inietta le azioni nei dati dei giocatori
        for p in players:
            p["actions"] = action_stats.get(p["track_id"], {})
        # Stampa riepilogo azioni
        log("Azioni per giocatore:")
        for p in players:
            a = p.get("actions", {})
            if any(v > 0 for v in a.values()):
                log(f"  Track {p['track_id']}: pass={a.get('passaggi',0)} tiri={a.get('tiri',0)} tackle={a.get('tackle',0)} gol={a.get('gol',0)}")
    else:
        for p in players:
            p["actions"] = {}

    # ── STEP 6: Valutazione LLM
    if not args.no_llm:
        log(f"Avvio valutazione con {args.model} (max {args.max_players} giocatori)...")
        candidates = [p for p in players if p["squadra"] != "sconosciuta"][:args.max_players]

        for i, player in enumerate(candidates, 1):
            log(f"  Valuto giocatore {i}/{len(candidates)} (track_id={player['track_id']}, squadra={player['squadra']})")
            voto, commento = evaluate_player_with_llm(
                player, i, len(candidates), args.ollama, args.model,
                use_vision=not args.no_vision
            )
            player["voto"]    = voto
            player["commento"] = commento
            log(f"    → Voto: {voto} | {commento[:60]}...")

        # Giocatori senza LLM (sconosciuta o oltre il limite)
        for p in players:
            if "voto" not in p:
                p["voto"]    = 5.0
                p["commento"] = "Valutazione non disponibile (maglia non classificata o limite raggiunto)."
    else:
        log("Salto valutazione LLM (--no-llm attivo)")
        for p in players:
            p["voto"]    = 5.0
            p["commento"] = "Valutazione LLM disabilitata."

    # ── STEP 7: Salva output
    json_path, csv_path = save_results(players, args.output, video_name)

    elapsed = time.time() - start_time
    log(f"\nCompletato in {elapsed:.0f}s")
    log(f"Risultati: {json_path}")
    log(f"CSV per app.py: {csv_path}")

    # Riepilogo finale con voti
    print("\n" + "="*65)
    print(f"{'ID':>4}  {'Squadra':12} {'Presenza':>9}  {'Voto':>5}  Commento")
    print("-"*65)
    for p in players:
        commento_short = p.get("commento","")[:30] + "..."
        voto_display = f"{p.get('voto', '-'):.1f}"
        print(f"{p['track_id']:>4}  {p['squadra']:12} {p['presenza_sec']:>7.0f}s  {voto_display:>5}  {commento_short}")
    print("="*65)


if __name__ == "__main__":
    main()
