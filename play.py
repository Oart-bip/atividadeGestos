
"""
Passo 3 - Jogo Pedra, Papel e Tesoura contra o computador.
Uso: python play.py
ESPACO = nova rodada (contagem 3-2-1, depois o modelo "le" sua mao)
q = sair
"""

import json
import random
import time
from collections import Counter, deque

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

model = tf.keras.models.load_model("gesture_model.keras")
class_names = json.load(open("class_names.json"))

BEATS = {
    "pedra": "tesoura",
    "papel": "pedra",
    "tesoura": "papel"
}  # chave vence valor

MOVES = ["pedra", "papel", "tesoura"]


def predict(roi):
    img = cv2.cvtColor(
        cv2.resize(roi, (IMG_SIZE, IMG_SIZE)),
        cv2.COLOR_BGR2RGB
    )

    probs = model.predict(
        img[None].astype("float32"),
        verbose=0
    )[0]

    return (
        class_names[int(np.argmax(probs))],
        float(np.max(probs)),
        probs
    )


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Nao consegui abrir a webcam.")

recent = deque(maxlen=7)  # suaviza: voto da maioria nos ultimos frames

score = {
    "voce": 0,
    "pc": 0,
    "empates": 0
}

state, t0, message, pc_move = (
    "idle",
    0,
    "Aperte ESPACO para jogar",
    None
)

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
    recent.append(label)
    stable = Counter(recent).most_common(1)[0][0]

    if state == "count":
        left = 3 - int(time.time() - t0)

        if left <= 0:
            you = stable

            if you not in MOVES:
                message = "Nao vi sua mao! Tente de novo"
            else:
                pc_move = random.choice(MOVES)

                if you == pc_move:
                    score["empates"] += 1
                    message = f"Empate! PC: {pc_move}"

                elif BEATS[you] == pc_move:
                    score["voce"] += 1
                    message = f"Voce ganhou! ({you} x {pc_move})"

                else:
                    score["pc"] += 1
                    message = f"PC ganhou! ({you} x {pc_move})"

            state, t0 = "result", time.time()

        else:
            message = f"{left}..."

    elif state == "result" and time.time() - t0 > 3:
        state = "idle"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + side, y1 + side),
        (0, 255, 0),
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

    # barras de probabilidade (otimo para mostrar a "incerteza" do modelo)
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

    cv2.putText(
        frame,
        message,
        (10, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Voce {score['voce']} x {score['pc']} PC "
        f"(empates: {score['empates']})",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Pedra, Papel e Tesoura com CNN", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32 and state == "idle":
        state, t0 = "count", time.time()

cap.release()
cv2.destroyAllWindows()
