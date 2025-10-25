import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    TimeDistributed, Dropout, Bidirectional, LSTM,
    Dense, Flatten
)
from tensorflow.keras.applications import MobileNetV2


# === 1️⃣ 경로 설정 ===
VIDEO_PATH = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\V_341.mp4"   # 분석할 영상
OUTPUT_PATH = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\output.mp4"     # 결과 영상 저장 경로
CAPTURE_DIR = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\violence_captures"  # 폭력 캡처 저장 폴더
os.makedirs(CAPTURE_DIR, exist_ok=True)


# === 2️⃣ Feature extractor (MobileNetV2) ===
feature_extractor = MobileNetV2(
    weights='imagenet',
    include_top=False,     # fully connected 제거
    pooling=None,          # (2,2,1280) 그대로 유지
    input_shape=(64, 64, 3)
)
print("✅ Feature extractor 준비 완료:", feature_extractor.output_shape)

# === 3️⃣ Violence Detection Model ===
model = Sequential([
    TimeDistributed(Flatten(), input_shape=(16, 2, 2, 1280), name="time_distributed"),
    Dropout(0.5, name='dropout'),
    TimeDistributed(Flatten(), name='time_distributed_1'),
    Bidirectional(LSTM(32, return_sequences=False), name='bidirectional'),
    Dropout(0.5, name='dropout_1'),
    Dense(256, activation='relu', name='dense'),
    Dropout(0.5, name='dropout_2'),
    Dense(128, activation='relu', name='dense_1'),
    Dropout(0.5, name='dropout_3'),
    Dense(64, activation='relu', name='dense_2'),
    Dropout(0.5, name='dropout_4'),
    Dense(32, activation='relu', name='dense_3'),
    Dropout(0.5, name='dropout_5'),
    Dense(2, activation='softmax', name='dense_4')
])

# === 4️⃣ 가중치 로드 ===
model.load_weights(r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\new_violence_detection_model.weights.h5")
print("✅ Violence Detection 모델 가중치 로드 완료!")

# === 5️⃣ 분석 파라미터 ===
SEQUENCE_LENGTH = 16
IMG_SIZE = (64, 64)
frame_buffer = []
capture_count = 0
MAX_CAPTURES = 8

# === 6️⃣ 비디오 로드 ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ 비디오 파일을 열 수 없습니다.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("🎥 폭력 감지 중... (Q로 종료)")

# === 7️⃣ 프레임 분석 루프 ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize + Normalize
    resized = cv2.resize(frame, IMG_SIZE)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    resized = np.expand_dims(resized, axis=0) / 255.0

    # === MobileNetV2 Feature 추출 ===
    features = feature_extractor.predict(resized, verbose=0)  # (1, 2, 2, 1280)
    features = np.squeeze(features, axis=0)                   # (2, 2, 1280)
    frame_buffer.append(features)

    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer.pop(0)

    label_text = "Analyzing..."
    color = (255, 255, 0)

    # === 16프레임 단위로 폭력 판별 ===
    if len(frame_buffer) == SEQUENCE_LENGTH:
        sequence = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)  # (1,16,2,2,1280)
        print("🧩 입력 시퀀스 shape:", sequence.shape)

        preds = model.predict(sequence, verbose=0)
        pred_label = np.argmax(preds)
        confidence = preds[0][pred_label]

        # === 폭력 감지 ===
        if pred_label == 1 and confidence > 0.7:
            label_text = f" VIOLENCE DETECTED ({confidence*100:.1f}%)"
            color = (0, 0, 255)

            if capture_count < MAX_CAPTURES:
                capture_path = os.path.join(CAPTURE_DIR, f"capture_{capture_count+1}.jpg")
                cv2.imwrite(capture_path, frame)
                capture_count += 1
                print(f"🚨 폭력 감지! 캡처 저장됨: {capture_path}")
        else:
            label_text = f" SAFE ({confidence*100:.1f}%)"
            color = (0, 255, 0)

    # === 결과 표시 및 저장 ===
    cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    out.write(frame)
    cv2.imshow("Violence Detection Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === 8️⃣ 종료 ===
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ 분석 완료!")
print(f"🎞️ 결과 영상 저장: {OUTPUT_PATH}")
print(f"🖼️ 캡처 이미지 저장 폴더: {CAPTURE_DIR}")