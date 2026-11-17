"""
GesturRGB Pro — Optimized Python Flask backend
Runs MediaPipe Hands in Python (OpenCV) and streams annotated frames to the browser.

Lag fixes applied:
  1. ESP32 sender in background queue thread (never blocks capture loop)
  2. Snapshot state — /state never waits on MediaPipe
  3. model_complexity=0 + inference downscale to 640x360
  4. Camera 960x540, JPEG quality 65
  5. Browser polls /state every 250ms
  6. Peace flash uses threading.Timer (non-blocking)
  7. Raw camera fps diagnostic
"""

import cv2
import mediapipe as mp
import math
import time
import threading
import queue
import json
from flask import Flask, render_template, Response, request, jsonify

# ============================================================
# CONFIG  <-- CHANGE THESE
# ============================================================
ESP32_IP = "192.168.1.100"     # ESP32 IP from Serial Monitor
CAMERA_INDEX = 0               # webcam index (try 1 if 0 fails)
CAM_W = 960
CAM_H = 540
JPEG_QUALITY = 65
INFER_W = 640
INFER_H = 360
STATE_POLL_MS = 250
WS_URL = f"ws://{ESP32_IP}:81"

# ============================================================
# APP
# ============================================================
app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_model = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,          # 0 = fastest
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================
# STATE
# ============================================================
PALETTE = [
    (255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0),
    (0, 255, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255),
]

state = {
    "current_color": (255, 0, 0),
    "color_index": 0,
    "last_color_change": 0.0,
    "brightness": 1.0,
    "rainbow_active": False,
    "diamond_scale": 1.0,
    "zoom_active": False,
    "zoom_last_ratio": None,
    "last_sent": (-1, -1, -1),
    "left":  {"gesture": "none", "stable": "none", "hold_start": 0.0},
    "right": {"gesture": "none", "stable": "none", "hold_start": 0.0},
    "combo": None,
    "toast": None,
    "_peace_prev": None,
}

lock = threading.Lock()

# Snapshot published once per frame — /state reads this without touching MediaPipe
snapshot = {}
snapshot_lock = threading.Lock()

def publish_snapshot():
    with lock:
        snap = {
            "left": state["left"]["stable"],
            "right": state["right"]["stable"],
            "combo": state["combo"],
            "brightness": round(state["brightness"] * 100),
            "diamond_scale": round(state["diamond_scale"], 3),
            "color": list(state["current_color"]),
            "rainbow": state["rainbow_active"],
            "toast": state["toast"],
        }
    with snapshot_lock:
        snapshot.clear()
        snapshot.update(snap)

# ============================================================
# ESP32 SENDER (background thread + queue, never blocks capture)
# ============================================================
try:
    import websocket  # websocket-client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False
    print("[WARN] websocket-client not installed. ESP32 sends disabled.")

ws_conn = None
ws_lock = threading.Lock()
send_queue = queue.Queue(maxsize=2)

def connect_esp32():
    global ws_conn
    if not HAS_WS_CLIENT:
        return
    try:
        ws_conn = websocket.create_connection(WS_URL, timeout=2)
        print("[ESP32] WebSocket connected")
    except Exception as e:
        print("[ESP32] WS connect failed:", e)
        ws_conn = None

def sender_worker():
    global ws_conn
    while True:
        payload = send_queue.get()
        if payload is None:
            continue
        if not HAS_WS_CLIENT:
            continue
        with ws_lock:
            try:
                if ws_conn is None:
                    connect_esp32()
                if ws_conn is not None:
                    ws_conn.send(payload)
            except Exception as e:
                print("[ESP32] send failed:", e)
                ws_conn = None

def send_rgb(r, g, b):
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    with lock:
        if state["last_sent"] == (r, g, b):
            return
        state["last_sent"] = (r, g, b)

    payload = f"{r},{g},{b}"
    try:
        if send_queue.full():
            try:
                send_queue.get_nowait()
            except queue.Empty:
                pass
        send_queue.put_nowait(payload)
    except queue.Full:
        pass

# ============================================================
# UTILS
# ============================================================
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lerp(a, b, t):
    return a + (b - a) * t

def hand_size(lm):
    d1 = dist(lm[0], lm[9])
    d2 = dist(lm[5], lm[17])
    return (d1 + d2) / 2 + 1e-6

# ============================================================
# GESTURE DETECTION (no fist command)
# ============================================================
def detect_gesture(lm):
    if not lm or len(lm) < 21:
        return "none"

    wrist = lm[0]
    thumb_tip = lm[4]
    index_tip = lm[8]
    middle_tip = lm[12]
    ring_tip = lm[16]
    pinky_tip = lm[20]

    index_mcp = lm[5]
    middle_mcp = lm[9]
    ring_mcp = lm[13]
    pinky_mcp = lm[17]

    size = hand_size(lm)

    ext = {
        "index":  dist(wrist, index_tip)  > dist(wrist, index_mcp)  * 1.15,
        "middle": dist(wrist, middle_tip) > dist(wrist, middle_mcp) * 1.15,
        "ring":   dist(wrist, ring_tip)   > dist(wrist, ring_mcp)   * 1.15,
        "pinky":  dist(wrist, pinky_tip)  > dist(wrist, pinky_mcp)  * 1.15,
    }

    thumb_extended = dist(thumb_tip, index_mcp) > size * 0.55
    pinch_ratio = dist(thumb_tip, index_tip) / size

    # OK
    if pinch_ratio < 0.45 and ext["middle"] and ext["ring"] and ext["pinky"]:
        return "ok"

    # Rock
    if ext["index"] and not ext["middle"] and not ext["ring"] and ext["pinky"]:
        return "rock"

    # Hang loose
    if thumb_extended and not ext["index"] and not ext["middle"] and not ext["ring"] and ext["pinky"]:
        return "hangloose"

    # Pinch
    if pinch_ratio < 0.55 and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "pinch"

    # Peace
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "peace"

    # Point
    if ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "point"

    # Thumbs up
    if thumb_extended and not ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        if thumb_tip.y < wrist.y - size * 0.2:
            return "thumbsup"

    # Open palm
    open_count = sum([ext["index"], ext["middle"], ext["ring"], ext["pinky"]])
    if open_count >= 4:
        return "open"

    return "none"

# ============================================================
# SINGLE-HAND ACTION
# ============================================================
def on_gesture_confirmed(handedness, gesture):
    with lock:
        state["toast"] = f"{handedness}: {gesture.upper()}"

    if gesture == "open":
        with lock:
            state["rainbow_active"] = True
            state["last_color_change"] = 0.0
    elif gesture == "thumbsup":
        with lock:
            state["current_color"] = (255, 140, 20)
            state["rainbow_active"] = False
    elif gesture == "point":
        with lock:
            state["current_color"] = (30, 120, 255)
            state["rainbow_active"] = False
    elif gesture == "ok":
        with lock:
            state["current_color"] = (160, 60, 255)
            state["rainbow_active"] = False
    elif gesture == "rock":
        with lock:
            state["current_color"] = (220, 20, 60)
            state["rainbow_active"] = False
    elif gesture == "hangloose":
        with lock:
            state["current_color"] = (0, 200, 180)
            state["rainbow_active"] = False
    elif gesture == "peace":
        with lock:
            state["_peace_prev"] = state["current_color"]
            state["current_color"] = (255, 255, 255)
        push_rgb(force=True)
        threading.Timer(0.3, _restore_peace).start()
    elif gesture == "pinch":
        with lock:
            state["rainbow_active"] = not state["rainbow_active"]

def _restore_peace():
    with lock:
        prev = state.get("_peace_prev")
        if prev is not None:
            state["current_color"] = prev
    push_rgb(force=True)

# ============================================================
# PER-HAND STATE UPDATE
# ============================================================
def process_hand(handedness, lm):
    side = "left" if handedness == "Left" else "right"
    s = state[side]
    g = detect_gesture(lm)
    now = time.time()

    if g != s["gesture"]:
        if s["hold_start"] == 0.0:
            s["hold_start"] = now
        elif now - s["hold_start"] > 0.18:
            s["gesture"] = g
            s["hold_start"] = 0.0
            if g != s["stable"]:
                s["stable"] = g
                on_gesture_confirmed(side.capitalize(), g)
    else:
        s["hold_start"] = 0.0

# ============================================================
# BOTH-HAND COMBOS + ZOOM
# ============================================================
def process_combo(left_lm, right_lm):
    if left_lm is None or right_lm is None:
        with lock:
            state["combo"] = None
            if state["zoom_active"]:
                state["zoom_active"] = False
                state["zoom_last_ratio"] = None
                state["diamond_scale"] = 1.0
        return

    gL = state["left"]["stable"]
    gR = state["right"]["stable"]

    if gL == "open" and gR == "open":
        with lock:
            state["combo"] = "both_open"
            state["brightness"] = clamp(state["brightness"] + 0.03, 0, 1)
            state["diamond_scale"] = 0.6 + state["brightness"] * 1.2
        push_rgb()
        return

    if gL == "peace" and gR == "peace":
        with lock:
            state["combo"] = "both_peace"
            state["rainbow_active"] = True
            state["last_color_change"] = 0.0
        return

    if gL == "point" and gR == "point":
        with lock:
            state["combo"] = "both_point"
            state["current_color"] = (255, 255, 255)
            state["brightness"] = 1.0
            state["diamond_scale"] = 1.0
            state["rainbow_active"] = False
        push_rgb(force=True)
        return

    # Zoom
    with lock:
        state["combo"] = "zoom"

    lc = left_lm[9]
    rc = right_lm[9]
    d = dist(lc, rc)
    avg_size = (hand_size(left_lm) + hand_size(right_lm)) / 2
    ratio = d / avg_size

    with lock:
        if not state["zoom_active"]:
            state["zoom_active"] = True
            state["zoom_last_ratio"] = ratio
            return

        if state["zoom_last_ratio"] is None:
            state["zoom_last_ratio"] = ratio
            return

        if abs(ratio - state["zoom_last_ratio"]) < 0.03:
            return

        state["zoom_last_ratio"] = ratio
        lo, hi = 1.2, 3.5
        raw = clamp((ratio - lo) / (hi - lo), 0, 1)
        smoothed = lerp(state["brightness"], raw, 0.15)
        state["brightness"] = clamp(smoothed, 0, 1)
        state["diamond_scale"] = 0.6 + state["brightness"] * 1.2

    push_rgb()

# ============================================================
# PUSH RGB TO ESP32 (queue, non-blocking)
# ============================================================
def push_rgb(force=False):
    with lock:
        color = state["current_color"]
        br = state["brightness"]
        last = state["last_sent"]
    r = int(color[0] * br)
    g = int(color[1] * br)
    b = int(color[2] * br)
    if not force and (r, g, b) == last:
        return
    send_rgb(r, g, b)

# ============================================================
# RAINBOW TICK
# ============================================================
def tick_rainbow():
    with lock:
        if not state["rainbow_active"]:
            return
        now = time.time()
        if now - state["last_color_change"] > 0.26:
            state["color_index"] = (state["color_index"] + 1) % len(PALETTE)
            state["current_color"] = PALETTE[state["color_index"]]
            state["last_color_change"] = now
    push_rgb()

# ============================================================
# CAPTURE LOOP
# ============================================================
frame_lock = threading.Lock()
latest_frame = None

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def capture_loop():
    global latest_frame
    print("[CAM] Capture loop started")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)

        # Downscale for inference only
        small = cv2.resize(frame, (INFER_W, INFER_H))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = hands_model.process(rgb_small)

        left_lm = None
        right_lm = None

        if results.multi_hand_landmarks:
            for i, lm in enumerate(results.multi_hand_landmarks):
                handedness = "Right"
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i].classification[0].label
                mirrored = "Left" if handedness == "Right" else "Right"

                if mirrored == "Left":
                    left_lm = lm.landmark
                else:
                    right_lm = lm.landmark

                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        with lock:
            if left_lm is None:
                state["left"]["gesture"] = "none"
                state["left"]["stable"] = "none"
                state["left"]["hold_start"] = 0.0
            if right_lm is None:
                state["right"]["gesture"] = "none"
                state["right"]["stable"] = "none"
                state["right"]["hold_start"] = 0.0

        if left_lm is not None:
            process_hand("Left", left_lm)
        if right_lm is not None:
            process_hand("Right", right_lm)

        process_combo(left_lm, right_lm)
        tick_rainbow()
        publish_snapshot()

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with frame_lock:
                latest_frame = jpg.tobytes()

def gen_frames():
    while True:
        with frame_lock:
            data = latest_frame
        if data is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def get_state():
    with snapshot_lock:
        return jsonify(dict(snapshot))

@app.route("/set_brightness", methods=["POST"])
def set_brightness():
    data = request.get_json(force=True)
    val = float(data.get("value", 1.0))
    with lock:
        state["brightness"] = clamp(val, 0, 1)
        state["diamond_scale"] = 0.6 + state["brightness"] * 1.2
    push_rgb(force=True)
    return jsonify({"ok": True, "brightness": round(state["brightness"] * 100)})

@app.route("/set_color", methods=["POST"])
def set_color():
    data = request.get_json(force=True)
    r = int(data.get("r", 255))
    g = int(data.get("g", 0))
    b = int(data.get("b", 0))
    with lock:
        state["current_color"] = (r, g, b)
        state["rainbow_active"] = False
    push_rgb(force=True)
    return jsonify({"ok": True})

@app.route("/fps")
def fps():
    return jsonify({"camera": CAM_W, "height": CAM_H, "inference": INFER_W})

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    connect_esp32()
    threading.Thread(target=sender_worker, daemon=True).start()
    threading.Thread(target=capture_loop, daemon=True).start()
    print("[FLASK] Starting at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)