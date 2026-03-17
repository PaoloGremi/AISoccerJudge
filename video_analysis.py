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
    ollama pull qwen2-vl
    ollama serve   (in un terminale separato)

Utilizzo:
    python video_analysis.py --video partita.mp4 --fps 1 --model qwen2-vl
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
DEFAULT_OLLAMA_MODEL = "qwen2-vl"
DEFAULT_OLLAMA_URL   = "http://localhost:11434"
CONF_THRESHOLD       = 0.4       # confidenza minima YOLO
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


def frame_to_base64(frame: np.ndarray) -> str:
    """Converte un frame OpenCV in stringa base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
def extract_frames(video_path: str, fps_sample: float) -> list[tuple[int, np.ndarray]]:
    """
    Estrae frame dal video a cadenza fps_sample.
    Ritorna lista di (frame_number, frame).
    """
    log(f"Apertura video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire il video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0

    log(f"Video: {video_fps:.1f} fps, {total_frames} frame totali, {duration_sec:.0f}s")

    step = max(1, int(video_fps / fps_sample))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append((frame_idx, frame.copy()))
        frame_idx += 1

    cap.release()
    log(f"Estratti {len(frames)} frame (1 ogni {step} frame originali)")
    return frames


# ──────────────────────────────────────────────
# STEP 2+3 — YOLO + BYTETRACK
# ──────────────────────────────────────────────
def detect_and_track(frames: list[tuple[int, np.ndarray]], yolo_model_name: str) -> dict:
    """
    Rileva persone con YOLOv8 e traccia con ByteTrack.
    Ritorna:
        track_data = {
            track_id: {
                "frames": [(frame_idx, bbox), ...],
                "crops":  [np.ndarray, ...]    # crop per classificazione maglia
            }
        }
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log("ERRORE: ultralytics non installato. Esegui: pip install ultralytics")
        sys.exit(1)

    log(f"Caricamento modello YOLO: {yolo_model_name}")
    model = YOLO(yolo_model_name)

    track_data = defaultdict(lambda: {"frames": [], "crops": []})

    log(f"Rilevamento + tracking su {len(frames)} frame...")
    for i, (frame_idx, frame) in enumerate(frames):
        if i % 20 == 0:
            log(f"  Frame {i+1}/{len(frames)}")

        results = model.track(
            frame,
            persist=True,           # mantieni stato ByteTrack tra frame
            classes=[0],            # classe 0 = persona
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            continue

        boxes = results[0].boxes
        for box, track_id in zip(boxes.xyxy, boxes.id.int()):
            tid = int(track_id)
            bbox = box.cpu().numpy()
            crop = crop_bbox(frame, bbox)

            if crop.size == 0:
                continue

            track_data[tid]["frames"].append((frame_idx, bbox.tolist()))
            # Salva al massimo 10 crop per giocatore (per non saturare la RAM)
            if len(track_data[tid]["crops"]) < 10:
                track_data[tid]["crops"].append(crop)

    log(f"Tracciati {len(track_data)} ID unici")
    return dict(track_data)


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
    Per ogni track_id calcola:
    - squadra (bianca/colorata)
    - frame_count, presenza_sec (tempo in campo)
    - posizione media in campo (normalizzata 0-1)
    - mobilità (deviazione standard posizione)
    - rappresentante crop per l'analisi LLM
    """
    players = []

    for tid, data in track_data.items():
        if len(data["frames"]) < 3:  # ignora tracce troppo brevi (rumore)
            continue

        squadra = classify_jersey(data["crops"])
        frame_count = len(data["frames"])
        presenza_sec = frame_count / fps_sample

        # Posizione media normalizzata (centro del bounding box)
        positions = []
        for _, bbox in data["frames"]:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            positions.append((cx, cy))

        pos_arr = np.array(positions)
        avg_x = float(np.mean(pos_arr[:, 0]))
        avg_y = float(np.mean(pos_arr[:, 1]))
        mobility = float(np.std(pos_arr[:, 0]) + np.std(pos_arr[:, 1]))

        # Scegli il crop migliore (il più grande) come rappresentante
        best_crop = None
        best_area = 0
        for crop in data["crops"]:
            area = crop.shape[0] * crop.shape[1]
            if area > best_area:
                best_area = area
                best_crop = crop

        players.append({
            "track_id":    tid,
            "squadra":     squadra,
            "frame_count": frame_count,
            "presenza_sec": round(presenza_sec, 1),
            "avg_x":       round(avg_x, 1),
            "avg_y":       round(avg_y, 1),
            "mobility":    round(mobility, 1),
            "best_crop":   best_crop,
        })

    # Ordina per tempo di presenza decrescente
    players.sort(key=lambda p: p["presenza_sec"], reverse=True)
    log(f"Giocatori validi: {len(players)} (bianca: {sum(1 for p in players if p['squadra']=='bianca')}, colorata: {sum(1 for p in players if p['squadra']=='colorata')})")
    return players


# ──────────────────────────────────────────────
# STEP 6 — VALUTAZIONE CON QWEN2-VL VIA OLLAMA
# ──────────────────────────────────────────────
def build_prompt(player: dict, player_index: int, total_players: int) -> str:
    """
    Costruisce il prompt per Qwen2-VL.
    """
    zona = "offensiva" if player["avg_y"] < 0.4 else "difensiva" if player["avg_y"] > 0.6 else "centrale"
    mobilita_desc = "alta" if player["mobility"] > 150 else "media" if player["mobility"] > 80 else "bassa"

    return f"""Sei un talent scout di calcetto amatoriale. Stai analizzando il giocatore {player_index} di {total_players}.

Dati oggettivi rilevati automaticamente dal video:
- Squadra: {player["squadra"]}
- Tempo in campo: {player["presenza_sec"]} secondi
- Zona prevalente: {zona}
- Mobilità (movimento): {mobilita_desc}

Nell'immagine allegata vedi un frame del giocatore durante la partita.

Basandoti sull'immagine e sui dati forniti, scrivi in italiano:
1. VOTO: un numero da 1 a 10 (solo il numero, sulla prima riga)
2. COMMENTO: 2-3 frasi di descrizione della performance, citando posizione in campo, dinamismo e qualsiasi dettaglio visibile nell'immagine

Formato risposta (rispetta esattamente):
VOTO: <numero>
COMMENTO: <testo>"""


def evaluate_player_with_llm(player: dict, player_index: int, total_players: int,
                               ollama_url: str, model: str) -> tuple[float, str]:
    """
    Invia crop + prompt a Qwen2-VL via Ollama.
    Ritorna (voto, commento).
    """
    if player["best_crop"] is None:
        return 5.0, "Nessuna immagine disponibile per la valutazione."

    prompt = build_prompt(player, player_index, total_players)
    image_b64 = frame_to_base64(player["best_crop"])

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.3,  # bassa temperatura = risposte più consistenti
            "num_predict": 200
        }
    }

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=120
        )
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
                    voto = max(1.0, min(10.0, voto))  # clamp 1-10
                except (ValueError, IndexError):
                    pass
            elif line.upper().startswith("COMMENTO:"):
                commento = line.split(":", 1)[1].strip()

        return voto, commento

    except requests.exceptions.ConnectionError:
        return 5.0, "Errore: Ollama non raggiungibile. Avvia con `ollama serve`."
    except requests.exceptions.Timeout:
        return 5.0, "Errore: timeout Ollama. Prova un modello più leggero."
    except Exception as e:
        return 5.0, f"Errore LLM: {e}"


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
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analisi video calcetto con AI")
    parser.add_argument("--video",   required=True,               help="Percorso del video (.mp4/.mov)")
    parser.add_argument("--fps",     type=float, default=DEFAULT_FPS_SAMPLE,   help="Frame al secondo da analizzare (default: 1)")
    parser.add_argument("--yolo",    default=DEFAULT_YOLO_MODEL,  help="Modello YOLO (default: yolov8n.pt)")
    parser.add_argument("--model",   default=DEFAULT_OLLAMA_MODEL, help="Modello Ollama (default: qwen2-vl)")
    parser.add_argument("--ollama",  default=DEFAULT_OLLAMA_URL,  help="URL Ollama (default: http://localhost:11434)")
    parser.add_argument("--output",  default="output",            help="Cartella output (default: ./output)")
    parser.add_argument("--no-llm",  action="store_true",         help="Salta la valutazione LLM (solo tracking)")
    parser.add_argument("--max-players", type=int, default=20,    help="Max giocatori da valutare con LLM (default: 20)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        log(f"ERRORE: file video non trovato: {args.video}")
        sys.exit(1)

    video_name = Path(args.video).stem
    start_time = time.time()

    # ── STEP 1: Estrazione frame
    frames = extract_frames(args.video, args.fps)

    # ── STEP 2+3: YOLO + ByteTrack
    track_data = detect_and_track(frames, args.yolo)

    # ── STEP 4+5: Classificazione + aggregazione
    total_frames = len(frames)
    players = aggregate_stats(track_data, total_frames, args.fps)

    if not players:
        log("Nessun giocatore rilevato. Verifica il video e le soglie di confidenza.")
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

    # ── STEP 6: Valutazione LLM
    if not args.no_llm:
        log(f"Avvio valutazione con {args.model} (max {args.max_players} giocatori)...")
        candidates = [p for p in players if p["squadra"] != "sconosciuta"][:args.max_players]

        for i, player in enumerate(candidates, 1):
            log(f"  Valuto giocatore {i}/{len(candidates)} (track_id={player['track_id']}, squadra={player['squadra']})")
            voto, commento = evaluate_player_with_llm(
                player, i, len(candidates), args.ollama, args.model
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
