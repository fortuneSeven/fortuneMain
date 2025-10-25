import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional, LSTM, Dense, Flatten
from tensorflow.keras.applications import MobileNetV2

# === 1️⃣ 경로 설정 ===
VIDEO_PATH = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\V_341.mp4"
OUTPUT_PATH = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\output.mp4"
CAPTURE_DIR = r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\violence_captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

model = Sequential([
    TimeDistributed(Flatten(), input_shape=(16, 2, 2, 1280), name="time_distributed"),
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
    Dense(2, activation="softmax", name="dense_4")
])