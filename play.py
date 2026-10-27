"""
Passo 3 (Libras) - Reconhecimento de sinais de Libras em tempo real.
Uso: python play.py
Coloque a mao dentro do quadrado verde; o modelo mostra o sinal que reconhece.
q = sair
"""

import json
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
NONE_CLASS = "nada"
CONF_MIN = 0.60  # abaixo disso o modelo "nao tem certeza" e mostramos "?"

model = tf.keras.models.load_model("gesture_model.keras")

with open("class_names.json") as f:
    class_names = json.load(f)


def predict(roi):
    img = cv2.cvtColor(
        cv2.resize(roi, (IMG_SIZE, IMG_SIZE)),
        cv2.COLOR_BGR2RGB
    )

    probs = model(
        img[None].astype("float32"),
        training=False
    ).numpy()[0]

    return (
        class_names[int(np.argmax(probs))],
        float(np.max(probs)),
        probs
    )


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam.")

recent = deque(maxlen=7)  # suaviza: voto da maioria nos ultimos frames

while True:
    ok, frame = cap.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2

    roi = frame[y1:y1 + side, x1:x1 + side]

    label, conf, probs = predict(roi)
    recent.append(label if conf >= CONF_MIN else "?")
    stable = Counter(recent).most_common(1)[0][0]

    if stable == NONE_CLASS:
        big, color = "-", (180, 180, 180)
        message = "Nenhum sinal"
    elif stable == "?":
        big, color = "?", (0, 165, 255)
        message = "Nao tenho certeza"
    else:
        big, color = stable, (0, 255, 0)
        message = f"Sinal: {stable}"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + side, y1 + side),
        color,
        3
    )

    cv2.putText(
        frame,
        f"Modelo ve: {label} ({conf:.0%})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # barras de probabilidade (mostram a "incerteza" do modelo)
    for i, c in enumerate(class_names):
        bw = int(probs[i] * 150)

        cv2.rectangle(
            frame,
            (10, 50 + i * 22),
            (10 + bw, 64 + i * 22),
            (0, 200, 255),
            -1
        )

        cv2.putText(
            frame,
            c,
            (170, 63 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # letra reconhecida em destaque (canto direito)
    cv2.putText(
        frame,
        big,
        (w - 140, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        4.0,
        color,
        8
    )

    cv2.putText(
        frame,
        message,
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3
    )

    cv2.imshow("Libras - reconhecimento de sinais", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()