import cv2
import mediapipe as mp
import threading
import time
import numpy as np

# Inicializa o módulo de mãos do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Desenhos dos pontos e linhas
mp_draw = mp.solutions.drawing_utils

# Inicializa a câmera
url = 'http://192.168.0.4:4747/video'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Ajusta a resolução desejada
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Índices das pontas dos dedos: polegar, indicador, médio, anelar, mindinho
finger_tips = [4, 8, 12, 16, 20]

# Variáveis globais
frame = None
ret = False
lock = threading.Lock()
running = True
mode = 'count'  # Modo inicial
last_switch_time = 0
draw_points = []

def capture_frames():
    global frame, ret, running
    while running:
        grabbed, img = cap.read()
        if not grabbed:
            continue
        with lock:
            frame = img
            ret = grabbed

# Função para identificar gesto de "joinha"
def is_thumb_up(fingers):
    return fingers == [1, 0, 0, 0, 0]

# Inicia a thread de captura
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    with lock:
        if frame is None:
            continue
        img = frame.copy()
        got_frame = ret

    if not got_frame:
        continue

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    total_fingers = 0  # total de dedos levantados (para exibir na barra lateral)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            handness = result.multi_handedness[idx].classification[0].label

            fingers_up = []

            # Dedos (exceto polegar)
            for tip_id in finger_tips[1:]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Polegar
            if handness == 'Right':
                fingers_up.insert(0, 1 if landmarks[4].x < landmarks[3].x else 0)
            else:
                fingers_up.insert(0, 1 if landmarks[4].x > landmarks[3].x else 0)

            # Alternância de modo com "joinha"
            current_time = time.time()
            if is_thumb_up(fingers_up) and current_time - last_switch_time > 1:
                mode = 'draw' if mode == 'count' else 'count'
                print(f"[INFO] Alternando para modo: {mode}")
                last_switch_time = current_time
                draw_points = []

            # Ação conforme o modo
            if mode == 'count':
                total = sum(fingers_up)
                total_fingers += total
                cv2.putText(img, f"Mao {idx+1} ({handness}): {total} dedos", (10, 50 + idx * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            elif mode == 'draw':
                # Desenho com dedo indicador
                h, w, _ = img.shape
                x, y = int(landmarks[8].x * w), int(landmarks[8].y * h)
                draw_points.append((x, y))

    # Desenha os pontos se estiver no modo draw
    if mode == 'draw':
        for i in range(1, len(draw_points)):
            cv2.line(img, draw_points[i - 1], draw_points[i], (255, 0, 0), 4)

    # --- BARRA LATERAL ---
    sidebar_width = 300
    h, w, _ = img.shape
    sidebar = 255 * np.ones((h, sidebar_width, 3), dtype=np.uint8)

    # Título
    cv2.putText(sidebar, "INFO", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # Modo atual
    cv2.putText(sidebar, "Modo:", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    if mode == 'draw':
        cv2.putText(sidebar, "DESENHO", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    else:
        cv2.putText(sidebar, "CONTAGEM", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)

    # Joinha p/ alternar
    cv2.putText(sidebar, "Joinha para", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.putText(sidebar, "alternar modo", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

    # Total de dedos
    if mode == 'count':
        cv2.putText(sidebar, "Dedos detectados:", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(sidebar, f"{total_fingers}", (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 3)

    # Junta imagem e barra lateral
    combined = np.hstack((img, sidebar))
    cv2.imshow("Webcam com Info", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Finaliza
thread.join()
cap.release()
cv2.destroyAllWindows()
