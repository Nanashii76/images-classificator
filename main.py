import cv2
import mediapipe as mp
import threading

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
# Caso seja uma câmera externa, utilize url, caso seja webcam, troque `cap = cv2.VideoCapture(url)` para `cap = cv2.VideoCapture(0)`
url = 'http://192.168.0.4:4747/video'
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Índices das pontas dos dedos: polegar, indicador, médio, anelar, mindinho
finger_tips = [4, 8, 12, 16, 20]

# Variável global para armazenar o frame capturado
frame = None
ret = False
lock = threading.Lock()
running = True

def capture_frames():
    global frame, ret, running
    while running:
        grabbed, img = cap.read()
        if not grabbed:
            continue
        with lock:
            frame = img
            ret = grabbed

# Inicia a thread de captura
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    # Copia o frame com lock para evitar conflito
    with lock:
        if frame is None:
            continue
        img = frame.copy()
        got_frame = ret

    if not got_frame:
        continue

    # Processa o frame
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Lógica polegar depende da mão (direita ou esquerda)
            handness = result.multi_handedness[idx].classification[0].label  # 'Left' ou 'Right'

            fingers_up = []

            # Dedos (exceto polegar)
            for tip_id in finger_tips[1:]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # Polegar
            if handness == 'Right':
                if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
                    fingers_up.insert(0, 1)
                else:
                    fingers_up.insert(0, 0)
            else:  # mão esquerda
                if landmarks[finger_tips[0]].x > landmarks[finger_tips[0] - 1].x:
                    fingers_up.insert(0, 1)
                else:
                    fingers_up.insert(0, 0)

            total = sum(fingers_up)

            # Escreve na tela com posição diferente para cada mão
            cv2.putText(img, f"Mao {idx+1} ({handness}): {total} dedos", (10, 50 + idx * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

# Finaliza
thread.join()
cap.release()
cv2.destroyAllWindows()
