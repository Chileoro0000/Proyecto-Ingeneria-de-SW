import cv2
import os
import numpy as np
import webbrowser
import time
import pyautogui

def GestoReconocido(emotion):
	#Gestones de gestos
	if emotion == 'Beso': 
		pyautogui.press('tab')
	if emotion == 'InflarMejillas': 
		pyautogui.press('down')
	if emotion == 'SacarLengua':
		pyautogui.press('enter') 
	if emotion == 'Sonrisa':
		webbrowser.open("http://www.youtube.com", new=2, autoraise=True)
		

method = 'LBPH'
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')

dataPath = 'C:/Users/gabil/Desktop/PROYECTO VISION ARTIFICIAL/Proyecto-Ingeneria-de-SW/DATA' #Cambia a la ruta donde hayas almacenado Data
GestoPaths = os.listdir(dataPath)
print('GestoPaths=',GestoPaths)

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def Main():
	while True:

		ret,frame = cap.read()
		if ret == False: break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()

		nFrame = cv2.hconcat([frame])

		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
			result = emotion_recognizer.predict(rostro)

			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

			if method == 'LBPH':
				if result[1] < 60:
					cv2.putText(frame, "Reconociendo..." ,(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
					Gesto = GestoReconocido(GestoPaths[result[0]])
					time.sleep(1)
					Main()
				else:
					cv2.putText(frame,'Humano',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
					cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
					nFrame = cv2.hconcat([frame])

		cv2.imshow('Reconocedor de gestos',nFrame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	

Main()

		
cap.release()
cv2.destroyAllWindows()