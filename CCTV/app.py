from flask import Flask, render_template, Response, render_template_string
import os, cv2, time, json, threading, queue
from pathlib import Path
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import base64, mimetypes, io
from PIL import Image

# 환경 변수
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ===== TensorFlow / Keras =====
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.applications import MobileNetV2

# ===== 경로/환경 =====
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CAPTURE_DIR = STATIC_DIR / "captures"           # 캡처 저장 폴더
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ 가중치 파일 경로를 네 환경에 맞게 수정
WEIGHTS_PATH = r"C:\coding\fortuneMain\CCTV\model\new_violence_detection_model.weights.h5"

# 카메라/모델 파라미터
CAM_INDEX   = 0
IMG_SIZE    = (64, 64)
SEQ_LEN     = 16
THRESHOLD   = 0.7
MIRROR_VIEW = True   # 좌우반전 원하면 True

app = Flask(__name__, static_folder=str(STATIC_DIR))

# ===== 탐지 이벤트 큐 (SSE로 전달) =====
event_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)

# ===== 비디오 스트림 (MJPEG) =====
camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def mjpeg_generator():
    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.03)
            continue
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/video")
def video():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/history")
def history():
    records = []
    return render_template("history.html", records=records)

# ===== SSE 이벤트 스트림 =====
@app.route("/events")
def events():
    def gen():
        # 재연결 간격(밀리초)
        yield "retry: 3000\n\n"
        while True:
            item = event_q.get()          # block until event
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    return Response(gen(), mimetype="text/event-stream")

# ===== TensorFlow 유틸/모델 =====
def set_tf_growth():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

def build_feature_extractor(img_size=(64,64)):
    # input_shape=(H, W, 3)
    fe = MobileNetV2(weights="imagenet", include_top=False, pooling=None,
                     input_shape=(img_size[1], img_size[0], 3))
    return fe

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

# LLM
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)

def encode_image_to_base64(image_path, max_size=(1024, 1024)):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image"):
        raise ValueError(f"유효하지 않은 이미지 파일: {image_path}")
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        buffer = io.BytesIO()
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

def summarize_one_image(image_path):
    prompt = (
        "이 CCTV 장면은 폭력 사건의 일부입니다.\n"
        "등장인물의 공격적/방어적 행동과 물리적 충돌의 방향을 관찰하고,\n"
        "한 문장으로 행동 중심으로 요약하세요. 감정적 표현은 배제하세요."
    )
    base64_image = encode_image_to_base64(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
        }
    ]
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("❌ 요약 오류:", e)
        return "(요약 실패)"

# ===== 백그라운드 탐지 루프 (웹캠→feature→BiLSTM) =====
def detector_loop():
    set_tf_growth()
    fe = build_feature_extractor(IMG_SIZE)
    model = build_mobilstm(SEQ_LEN)
    model.load_weights(WEIGHTS_PATH)

    frame_buf = []

    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.03)
            continue
        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

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
        prob = model.predict(seq, verbose=0)[0]   # [safe_prob, violence_prob]
        is_violence = int(np.argmax(prob)) == 1
        confidence  = float(prob[1] if is_violence else prob[0])

        cap_url = ""
        summary_text = ""
        # 폭력 + 임계값 초과 시 캡처 저장
        if is_violence and confidence >= THRESHOLD:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"cap_{ts}.jpg"
            out_path = CAPTURE_DIR / filename
            cv2.imwrite(str(out_path), frame)
            cap_url = f"/static/captures/{filename}"

            # LLM 요약 호출
            summary_text = summarize_one_image(str(out_path))

        # 이벤트 푸시
        evt = {
            "ts": int(time.time()*1000),
            "label": "violence" if is_violence else "safe",
            "confidence": confidence,
            "capture_url": cap_url,
            "summary": summary_text
        }
        try:
            event_q.put_nowait(evt)
        except queue.Full:
            # 큐가 가득 찼으면 가장 오래된 것 버리고 새 것 삽입
            try: event_q.get_nowait()
            except queue.Empty: pass
            event_q.put_nowait(evt)

# 서버 기동 시 탐지 스레드 시작
if __name__ == "__main__":
    # 스레드 시작
    threading.Thread(target=detector_loop, daemon=True).start()
    # 재로더 끄고 실행
    app.run(debug=True, threaded=True, use_reloader=False)
