# Contador de Dedos com MediaPipe e OpenCV

Este projeto utiliza MediaPipe e OpenCV para detectar mãos em tempo real através da webcam ou celular (via IP) e contar quantos dedos estão levantados em cada mão.

## Demonstração

<img src="./img/ezgif-4e44264dd77714.gif" alt="demo" width="600" />

## Funcionalidades

- Detecção de até duas mãos simultaneamente
- Reconhecimento individual dos dedos levantados
- Compatível com webcam e com apps de câmera IP (ex: DroidCam, IP Webcam)
- Interface em tempo real com contagem para cada mão (esquerda/direita)

## Tecnologias

- OpenCV
- MediaPipe Hands
- Python threading para captura assíncrona da câmera

## Requisitos

- Python 3.7+
- Bibliotecas:
    > pip install opencv-python mediapipe

## Como Usar

1. Clone o Projeto:
``` bash
git clone <link-repositório>
cd <nome-pasta>
```

2. Configure a entrada de vídeo

No código, troque a origem da câmera conforme necessário:

- Para webcam:
``` python
cap = cv2.VideoCapture(0)
```

- Para câmera IP
``` python
url = 'http://SEU_IP:PORTA/video'
cap = cv2.VideoCapture(url)
```

3. Execute o script
``` bash
python main.py
```

4. Ecerre apertando `Q`

- A contagem de dedos funciona para mãos esquerda e direita, adaptando a lógica do polegar com base na mão detectada.

- O código usa threading para evitar travamentos causados pela leitura contínua da câmera IP.

## Melhorias futuas

- [ ] Reconhecimento de gestos personalizados (paz, joinha, etc.)
- [ ] Comandos acionados por gestos
- [ ] Desenho na tela com o dedo
- [ ] Detecção de distâncias e movimentos entre os dedos