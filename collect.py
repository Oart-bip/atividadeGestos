
"""
Passo 1 (Libras) - Coleta de dados pela webcam.
Uso: python collect.py
IMPORTANTE: clique uma vez na JANELA DO VIDEO para ela receber as teclas
(se o terminal estiver em foco, nada acontece).

Sinais (letras do alfabeto manual de Libras, todas estaticas):
A B C F I L O V W Y   + a classe "nada" (sem mao / mao fora de sinal)

Teclas (aperte UMA vez, nao segure):
  1 2 3 4 5 6 7 8 9 0  ou a letra do proprio sinal (a b c f i l o v w y)
  n = nada
  ESPACO = troca a iluminacao atual (escuro <-> claro)
  q = sair

Ao apertar a tecla de um sinal, ha uma contagem de 3s para voce posicionar a
mao e a gravacao comeca sozinha. Ela grava uma "rajada" de BURST_SIZE fotos e
pausa sozinha: troque a iluminacao (ESPACO), aperte a tecla de novo e repita.
Assim as duas iluminacoes ficam INTERCALADAS dentro de cada sinal (o train.py
usa os ultimos 20% das fotos de cada classe como validacao, entao as duas
iluminacoes precisam aparecer ate o fim da coleta de cada sinal).
Aperte a mesma tecla durante a gravacao para pausar antes.

Varie tambem: angulo, distancia e a mao usada (direita e esquerda).
"""

import os
import time
import cv2

SIGNS = ["A", "B", "C", "F", "I", "L", "O", "V", "W", "Y"]
NONE_CLASS = "nada"
CLASSES = SIGNS + [NONE_CLASS]

# Teclas 1..9 e 0 (10 sinais) ou a letra do sinal; "n" = nada.
KEYS = {}

for i, s in enumerate(SIGNS):
    KEYS[ord(str((i + 1) % 10))] = s
    KEYS[ord(s.lower())] = s

KEYS[ord("n")] = NONE_CLASS

KEY_LABEL = {s: f"{(i + 1) % 10}/{s.lower()}" for i, s in enumerate(SIGNS)}
KEY_LABEL[NONE_CLASS] = "n"

LIGHTS = ["escuro", "claro"]

DATA_DIR = "data"
IMG_SIZE = 224
SAVE_EVERY_N_FRAMES = 3
MAX_PER_CLASS = 300   # 150 por iluminacao
BURST_SIZE = 50       # fotos por rajada antes de pausar e trocar a luz
COUNTDOWN = 3.0
DEBOUNCE = 0.5        # ignora repeticao de tecla (segurar) por 0.5s

for c in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, c), exist_ok=True)


def scan():
    """Conta as fotos ja existentes por classe e por iluminacao."""
    result = {c: {"escuro": 0, "claro": 0, "outra": 0} for c in CLASSES}

    for c in CLASSES:
        for f in os.listdir(os.path.join(DATA_DIR, c)):
            if not f.endswith(".jpg"):
                continue

            if f.endswith("_escuro.jpg"):
                result[c]["escuro"] += 1
            elif f.endswith("_claro.jpg"):
                result[c]["claro"] += 1
            else:
                result[c]["outra"] += 1

    return result


counts = scan()


def total(cls):
    return sum(counts[cls].values())


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit(
        "Nao consegui abrir a webcam. Feche outros apps que usam a camera "
        "(Zoom, Meet...) ou tente cv2.VideoCapture(1)."
    )

recording = None  # classe sendo gravada
countdown_start = None
last_key_time = 0
frame_count = 0
burst_saved = 0
light_idx = 0
notice, notice_until = "", 0
WIN = "Coleta Libras"

print(
    "Janela aberta. Clique nela e aperte 1-9/0 ou a letra do sinal "
    "('n' = nada). ESPACO troca a iluminacao. 'q' sai."
)

while True:
    ok, frame = cap.read()

    if not ok:
        print("Falha ao ler a webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    side = min(h, w) // 2 + 60
    x1, y1 = w // 2 - side // 2, h // 2 - side // 2
    roi = frame[y1:y1 + side, x1:x1 + side]
    light = LIGHTS[light_idx]

    status, color = "PARADO - aperte a tecla de um sinal", (0, 255, 0)

    if recording and countdown_start is not None:
        left = COUNTDOWN - (time.time() - countdown_start)

        if left > 0:
            status, color = (
                f"{recording}: prepare a mao... {int(left) + 1}",
                (0, 255, 255)
            )
        else:
            countdown_start = None

    if recording and countdown_start is None:
        status, color = (
            f"GRAVANDO: {recording} ({burst_saved}/{BURST_SIZE})",
            (0, 0, 255)
        )
        frame_count += 1

        if frame_count % SAVE_EVERY_N_FRAMES == 0:
            if total(recording) >= MAX_PER_CLASS:
                print(f"{recording}: completo ({MAX_PER_CLASS} fotos).")
                notice = f"{recording}: completo!"
                notice_until = time.time() + 4
                recording = None
            else:
                img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(
                    os.path.join(
                        DATA_DIR,
                        recording,
                        f"{int(time.time() * 1000)}_{light}.jpg"
                    ),
                    img
                )
                counts[recording][light] += 1
                burst_saved += 1

                if burst_saved >= BURST_SIZE:
                    print(
                        f"{recording}: rajada de {BURST_SIZE} fotos "
                        f"({light}) concluida. Troque a iluminacao (ESPACO)."
                    )
                    notice = "Rajada pronta: troque a luz (ESPACO)"
                    notice_until = time.time() + 5
                    recording = None

    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + side, y1 + side),
        color,
        3
    )

    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"ILUMINACAO: {light.upper()} (ESPACO troca)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255) if light == "claro" else (255, 200, 0),
        2
    )

    for i, c in enumerate(CLASSES):
        full = total(c) >= MAX_PER_CLASS

        cv2.putText(
            frame,
            f"[{KEY_LABEL[c]}] {c}: {total(c)}/{MAX_PER_CLASS} "
            f"(esc {counts[c]['escuro']} / claro {counts[c]['claro']})",
            (10, 90 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if full else (255, 255, 255),
            1
        )

    if time.time() < notice_until:
        cv2.putText(
            frame,
            notice,
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    cv2.putText(
        frame,
        "Varie: angulo, distancia, mao direita/esquerda",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.imshow(WIN, frame)
    key = cv2.waitKey(10) & 0xFF

    if key == 255:
        continue

    if key == ord("q"):
        break

    if key == 32:  # ESPACO: troca a iluminacao atual
        if time.time() - last_key_time > DEBOUNCE:
            last_key_time = time.time()
            light_idx = 1 - light_idx
            print(f"Iluminacao atual: {LIGHTS[light_idx]}")

    elif key in KEYS and time.time() - last_key_time > DEBOUNCE:
        last_key_time = time.time()
        cls = KEYS[key]

        if recording == cls:
            recording = None
            print(f"Pausado: {cls}")
        elif total(cls) >= MAX_PER_CLASS:
            print(f"{cls}: ja esta completo ({MAX_PER_CLASS} fotos).")
            notice = f"{cls}: ja completo"
            notice_until = time.time() + 3
        else:
            recording, countdown_start = cls, time.time()
            frame_count, burst_saved = 0, 0
            print(f"Preparando: {cls} (iluminacao {light})")

    elif key not in KEYS:
        print(f"(tecla nao reconhecida: codigo {key})")

cap.release()
cv2.destroyAllWindows()

print("\nResumo da coleta:")

for c in CLASSES:
    print(
        f" {c:>5}: {total(c):>3} fotos "
        f"(escuro {counts[c]['escuro']}, claro {counts[c]['claro']})"
    )

totals = [total(c) for c in CLASSES]

if max(totals) - min(totals) > 0.2 * MAX_PER_CLASS:
    print(
        "\nAtencao: as classes estao desbalanceadas. Complete as que "
        "tem menos fotos antes de treinar."
    )

for c in CLASSES:
    if abs(counts[c]["escuro"] - counts[c]["claro"]) > 0.2 * MAX_PER_CLASS:
        print(f"Atencao: {c} esta com iluminacoes desiguais.")
