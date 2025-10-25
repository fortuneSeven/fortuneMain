from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dropout, Bidirectional, LSTM, Dense, Flatten

# === Violence Detection MoBiLSTM Model ===
model = Sequential([
    # (1) CNN feature sequence input
    TimeDistributed(
        Flatten(),  # 입력은 (16, 2, 2, 1280)
        input_shape=(16, 2, 2, 1280),
        name="time_distributed"
    ),
    Dropout(0.5, name="dropout"),

    # (2) Flatten frame features → sequence
    TimeDistributed(Flatten(), name="time_distributed_1"),

    # (3) BiLSTM sequence encoder
    Bidirectional(LSTM(32, return_sequences=False), name="bidirectional"),
    Dropout(0.5, name="dropout_1"),

    # (4) Fully-connected classifier
    Dense(256, activation='relu', name="dense"),
    Dropout(0.5, name="dropout_2"),
    Dense(128, activation='relu', name="dense_1"),
    Dropout(0.5, name="dropout_3"),
    Dense(64, activation='relu', name="dense_2"),
    Dropout(0.5, name="dropout_4"),
    Dense(32, activation='relu', name="dense_3"),
    Dropout(0.5, name="dropout_5"),
    Dense(2, activation='softmax', name="dense_4")
])

# === Load weights ===
model.load_weights(r"C:\coding\fortuneseven\fortuneMain-1\CCTV\model\new_violence_detection_model.weights.h5")

print("✅ Violence Detection Model 가중치 로드 성공!")
model.summary()