# app.py
import os, cv2, time, json, threading, queue
from pathlib import Path
from datetime import datetime
import numpy as np
from flask import Flask, Response, render_template

# ===== TensorFlow / Keras =====
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.applications import MobileNetV2

# ===== 경로/환경 =====
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CAPTURE_DIR = STATIC_DIR / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ 가중치 파일 경로 수정 필요
WEIGHTS_PATH = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\new_violence_detection_model.weights.h5"

# 파라미터
CAM_INDEX   = 0
IMG_SIZE    = (64, 64)
SEQ_LEN     = 16
THRESHOLD   = 0.7
MIRROR_VIEW = True

app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(BASE_DIR / "templates"))

# ===== 이벤트 큐 (SSE) =====
event_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)

# ===== 전역 카메라/프레임 공유 =====
camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # 필요 시 주석/해제해서 테스트

latest_frame = None
frame_lock = threading.Lock()
running = True

def capture_loop():
    """카메라를 읽는 유일한 스레드: 최신 프레임을 공유 변수에 보관"""
    global latest_frame
    # warm-up
    for _ in range(5):
        camera.read(); time.sleep(0.02)

    while running:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.02); continue
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        with frame_lock:
            latest_frame = frame

def mjpeg_generator():
    """공유 프레임을 JPEG로 인코딩해 스트리밍"""
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.02); continue
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.01); continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/events")
def events():
    def gen():
        yield "retry: 3000\n\n"
        while True:
            item = event_q.get()  # blocking
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    return Response(gen(), mimetype="text/event-stream")

# ===== TF 유틸/모델 =====
def set_tf_growth():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

def build_feature_extractor(img_size=(64,64)):
    # input_shape=(H,W,3)
    return MobileNetV2(weights="imagenet", include_top=False, pooling=None,
                       input_shape=(img_size[1], img_size[0], 3))

def build_mobilstm(seq_len=16):
    return Sequential([
        TimeDistributed(Flatten(), input_shape=(seq_len,2,2,1280), name="td0"),
        Dropout(0.5), TimeDistributed(Flatten(), name="td1"),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.5), Dense(256, activation="relu"),
        Dropout(0.5), Dense(128, activation="relu"),
        Dropout(0.5), Dense(64, activation="relu"),
        Dropout(0.5), Dense(32, activation="relu"),
        Dropout(0.5), Dense(2, activation="softmax"),
    ])

def detector_loop():
    """공유 프레임을 읽어 feature→BiLSTM 분류→SSE 이벤트 발송"""
    set_tf_growth()
    fe = build_feature_extractor(IMG_SIZE)
    model = build_mobilstm(SEQ_LEN)
    model.load_weights(WEIGHTS_PATH)

    frame_buf = []

    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.02); continue

        # 전처리 → feature (1,2,2,1280)
        resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb, 0) / 255.0
        feat = fe.predict(inp, verbose=0)
        feat = np.squeeze(feat, axis=0)  # (2,2,1280)

        frame_buf.append(feat)
        if len(frame_buf) > SEQ_LEN:
            frame_buf.pop(0)
        if len(frame_buf) < SEQ_LEN:
            continue

        seq = np.expand_dims(np.array(frame_buf, dtype=np.float32), 0)
        prob = model.predict(seq, verbose=0)[0]  # [safe, violence]
        is_violence = int(np.argmax(prob)) == 1
        confidence  = float(prob[1] if is_violence else prob[0])

        cap_url = ""
        if is_violence and confidence >= THRESHOLD:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"cap_{ts}.jpg"
            out_path = CAPTURE_DIR / filename
            cv2.imwrite(str(out_path), frame)
            cap_url = f"/static/captures/{filename}"

        evt = {
            "ts": int(time.time()*1000),
            "label": "violence" if is_violence else "safe",
            "confidence": confidence,
            "capture_url": cap_url,
            "summary": ""  # LLM 요약 붙일 거면 여기에
        }
        try:
            event_q.put_nowait(evt)
        except queue.Full:
            try: event_q.get_nowait()
            except queue.Empty: pass
            event_q.put_nowait(evt)

# ===== 스레드 시작 =====
threading.Thread(target=capture_loop,  daemon=True).start()
threading.Thread(target=detector_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
