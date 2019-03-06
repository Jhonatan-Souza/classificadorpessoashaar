import cv2
import numpy as np

# Classificador de pessoas com haarcascade
classificadordepessoas = cv2.CascadeClassifier('Haarcascades\haarcascade_fullbody.xml')


# mostrar o video
mostrarvideo = cv2.VideoCapture('curitiba1.mp4')

while mostrarvideo.isOpened():

    # ler cada frame do video e faz um redimensionamento, uso isso para acelerar a classificação, quanto maior a imagem
    # mais pixels para classificar, e diminui a resolução do video pela metade
    ret, frame = mostrarvideo.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # passa o frame para o classificador
    pessoas = classificadordepessoas.detectMultiScale(cinza, 1.2, 4) #quanto menor esses valores é mais sensivel a classificação
    #mas acontece um maior numero de falsos positivos.

    # faz o bounding box
    for (x, y, w, h) in pessoas:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.imshow('Pessoas', frame)

    if cv2.waitKey(1) == 27:  # tecla esc para sair
        break

mostrarvideo.release()
cv2.destroyAllWindows()
