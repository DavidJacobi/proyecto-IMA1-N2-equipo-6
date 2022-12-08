#librerias
import cv2
import os
import mediapipe as mp

#importamos carpetas
direccion = 'C:/Users/david/Desktop/nuevo/datos'
lista = os.listdir(direccion)
print('nombres: ', lista)

#modelo
modelo= cv2.face.LBPHFaceRecognizer_create()

#leer modelo
modelo.read('modelo.xml')

#declaracion del detector
detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

#video captura
captura = cv2.VideoCapture(0)

#parametros de deteccion
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:

    while True:
        #video captura
        ret, frame = captura.read()

        #frame espejo
        frame = cv2.flip(frame, 1)

        #color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #resultado deteccion de rostro
        resultado = rostros.process(rgb)

        #filtro
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame,rostro)

                #ancho y alto de la ventana
                al, an, dat = frame.shape

                #extraer xinicial y yinicial
                xi = rostro.location_data.relative_bounding_box.xmin
                yi = rostro.location_data.relative_bounding_box.ymin

                #extraer el ancho y el alto
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height

                #conversion a pixel
                xi = int(xi * an)
                yi = int(yi * al)
                ancho = int(ancho * an)
                alto = int(alto * al)

                #xfinal yfinal
                xf = xi + ancho
                yf = yi + alto

                #extraccion de pixeles
                cara = frame[yi:yf, xi:xf]

                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)
                cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)

                #prediccion
                prediccion = modelo.predict(cara)

                #visualizar resultados
                if prediccion[0] == 0:
                    cv2.putText(frame, '{}'.format(lista[0]), (xi,yi - 5), 1, 1.3, (0,255,0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf, yf), (0,255,0), 2)
                elif prediccion[0] == 1:
                    cv2.putText(frame, '{}'.format(lista[1]), (xi,yi - 5), 1, 1.3, (255,0,0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf, yf), (255,0,0), 2)



        #fotogramas
        cv2.imshow("reconocimiento facial y de tapabocas", frame)
        #leyendo tecla
        t = cv2.waitKey(1)
        if t == 27:
            break

captura.release()
cv2.destroyAllWindows()