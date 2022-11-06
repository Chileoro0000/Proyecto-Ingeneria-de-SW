import cv2
import os
import numpy as np
import time

def obtenerModelo(method,facesData,labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face_LBPHFaceRecognizer.create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando ( "+method+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+method+".xml")

dataPath = 'C:/Users/gabil/Desktop/PROYECTO VISION ARTIFICIAL/DATA' 
ListaDeGestos = os.listdir(dataPath)
print('Lista de personas: ', ListaDeGestos)

labels = []
facesData = []
label = 0

for nameDir in ListaDeGestos:
	CarpetaGestos = dataPath + '/' + nameDir

	for fileName in os.listdir(CarpetaGestos):
		labels.append(label)
		facesData.append(cv2.imread(CarpetaGestos+'/'+fileName,0))
	label = label + 1

obtenerModelo('EigenFaces',facesData,labels)
obtenerModelo('FisherFaces',facesData,labels)
obtenerModelo('LBPH',facesData,labels)