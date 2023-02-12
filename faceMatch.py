import face_recognition
import numpy as np
import cv2
import copy
from PIL import Image
from PIL import ImageDraw
#feats:
#right_eyebrow
#left_eyebrow
#top_lip
#bottom_lip
#nose_tip
#chin
#nose_bridge



def getWebcam():
    webcam = cv2.VideoCapture(0)
    #Frame coordinates go frame[y][x]
    while True:
        ret, frame = webcam.read()
        lowFiFrame = cv2.resize(copy.deepcopy(frame), (0,0), fy=.25, fx=.25)
        locations = face_recognition.face_locations(lowFiFrame)
        feats = face_recognition.face_landmarks(lowFiFrame)

        tagFaces(frame,locations)
        featureSwap(frame, feats, "left_eye", "right_eye")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()


#Draws red box around faces it sees
def tagFaces(frame, locations):
    for spot in locations:
        tL, bR = getCorners(spot, 4)
        cv2.rectangle(frame, tL, bR, (0, 0, 255), 3)

def featureSwap(frame, feats, feat1, feat2):
    if len(feats) == 0:
        return False
    eyeList = []
    for person in feats:
        eyeList.append(person[feat1])
        eyeList.append(person[feat2])
    eyeRectList = []
    for eye in eyeList:
        eyeRectList.append(maxAndMin(eye))
    for i in range(len(eyeList)//2):
        eye_bounding_box(frame, eyeRectList[i], eyeRectList[i+1])

def maxAndMin(featCoords):
    adj = 5
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    return [min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj]

def eye_bounding_box(frame, eye1, eye2):
    # Coords in the eyes go minx, miny, maxx, maxy
    try:
        for x in range((eye1[2]-eye1[0]) * 4):
            for y in range((eye1[3]-eye1[1]) * 4):
                frame[eye2[1]*4 + y][eye2[0]*4 + x] = 0
                frame[eye1[1]*4 + y][eye1[0]*4 + x] = 0

    except:
        pass
    
def getCorners(fL,n):
    tL = (fL[3]*n, fL[0]*n)
    bR = (fL[1]*n, fL[2]*n)
    return tL, bR

if __name__ == "__main__":
    getWebcam()
