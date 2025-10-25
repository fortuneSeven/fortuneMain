# wepdetect.py — Webcam Violence Detection with sane defaults
import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.applications import MobileNetV2


# ---------------- Utils ---------------- #
def set_tf_memory_growth():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

def build_feature_extractor(img_size=(64, 64)):
    fe = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling=None,
        input_shape=(img_size[1], img_size[0], 3),
    )
    print("✅ Feature extractor:", fe.output_shape)
    return fe

def build_mobilstm(seq_len=16):
    return Sequential([
        TimeDistributed(Flatten(), input_shape=(seq_len, 2, 2, 1280), name="time_distributed"),
        Dropout(0.5, name="dropout"),
        TimeDistributed(Flatten(), name="time_distributed_1"),
        Bidirectional(LSTM(32, return_sequences=False), name="bidirectional"),
        Dropout(0.5, name="dropout_1"),
        Dense(256, activation="relu", name="dense"),
        Dropout(0.5, name="dropout_2"),
        Dense(128, activation="relu", name="dense_1"),
        Dropout(0.5, name="dropout_3"),
        Dense(64, activation="relu", name="dense_2"),
        Dropout(0.5, name="dropout_4"),
        Dense(32, activation="relu", name="dense_3"),
        Dropout(0.5, name="dropout_5"),
        Dense(2, activation="softmax", name="dense_4"),
    ])

def safe_imwrite(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"이미지 저장 실패: {path}")


# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Webcam Violence Detection (MoBiLSTM) — defaults enabled")

    # ✅ 기본값으로 자동 실행되게 수정 (required 제거 + default 추가)
    ap.add_argument(
        "--weights",
        type=str,
        default=r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\new_violence_detection_model.weights.h5",
        help="Keras .h5 가중치 경로 (기본값 내장, CLI로 덮어쓰기 가능)"
    )
    ap.add_argument(
        "--captures",
        type=str,
        default=r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\violence_captures",
        help="캡처 저장 폴더"
    )
    ap.add_argument("--out", type=str, default="", help="결과 영상 mp4 경로 (빈 값이면 저장 안 함)")
    ap.add_argument("--camera-index", type=int, default=0, help="웹캠 인덱스")
    ap.add_argument("--img-size", type=int, nargs=2, default=[64, 64], metavar=("W", "H"))
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-captures", type=int, default=8)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    weight_path = Path(args.weights).expanduser().resolve()
    cap_dir = Path(args.captures).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else None

    if not weight_path.exists():
        print(f"❌ 가중치 파일을 찾을 수 없습니다:\n   {weight_path}\n"
              f"→ 코드 상단 기본값을 수정하거나 --weights 로 직접 경로를 넣어주세요.")
        sys.exit(1)

    set_tf_memory_growth()
    fe = build_feature_extractor(tuple(args.img_size))
    model = build_mobilstm(args.seq_len)

    print(f"🔍 가중치 로드: {weight_path}")
    model.load_weights(str(weight_path))
    print("✅ 모델 가중치 로드 완료!")

    # Windows에서는 CAP_DSHOW가 호환이 좋은 편
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"❌ 웹캠({args.camera_index})을 열 수 없습니다. 다른 인덱스나 장치 점유 여부를 확인하세요.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"⚠️ 결과 파일을 열 수 없습니다: {out_path} (저장 생략)")
                writer = None

    print("🎥 웹캠 폭력 감지 시작 (종료: Q)")
    print(f"   해상도: {width}x{height}, FPS: {fps:.1f}")
    print(f"   캡처 폴더: {cap_dir}")
    if writer:
        print(f"   출력 파일: {out_path}")

    SEQ = args.seq_len
    IMG_W, IMG_H = args.img_size
    frame_buffer = []
    capture_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ 프레임을 읽을 수 없습니다.")
                break

            # 전처리
            resized = cv2.resize(frame, (IMG_W, IMG_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb, axis=0) / 255.0

            # Feature 추출 -> (1,2,2,1280)
            features = fe.predict(inp, verbose=0)
            features = np.squeeze(features, axis=0)  # (2,2,1280)
            frame_buffer.append(features)
            if len(frame_buffer) > SEQ:
                frame_buffer.pop(0)

            label_text = "Analyzing..."
            color = (0, 255, 255)

            if len(frame_buffer) == SEQ:
                seq = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)  # (1,SEQ,2,2,1280)
                preds = model.predict(seq, verbose=0)
                pred_label = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][pred_label])

                if pred_label == 1 and confidence >= args.threshold:
                    label_text = f"VIOLENCE DETECTED ({confidence*100:.1f}%)"
                    color = (0, 0, 255)
                    if capture_count < args.max_captures:
                        cap_path = cap_dir / f"capture_{capture_count+1}.jpg"
                        safe_imwrite(cap_path, frame)
                        capture_count += 1
                        print(f"🚨 캡처 저장: {cap_path}")
                else:
                    label_text = f"SAFE ({confidence*100:.1f}%)"
                    color = (0, 200, 0)

            # 오버레이 + 저장/표시
            cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            if writer:
                writer.write(frame)
            cv2.imshow("Violence Detection - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ 종료")


if __name__ == "__main__":
    # ⚠️ TensorFlow는 Python 3.13에서 공식 휠이 부족할 수 있습니다.
    #    문제가 생기면 Python 3.11 venv에서 실행하세요.
    main()
