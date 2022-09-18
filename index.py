import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# myOrjResList=os.listdir("OrjResim")
# imageOrj=cv2.imread("OrjResim/"+f'{myOrjResList[0]}')
# imgElon=face_recognition.load_image_file("OrjResim/"+f'{myOrjResList[0]}')
# imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
# encodeElon=face_recognition.face_encodings(imgElon)[0]
# cv2.imshow("Test",imageOrj)
# cv2.waitKey(0)

path="resimler"
images=[]
classNames=[]
myList=os.listdir(path)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList=[]

    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

        # result = face_recognition.compare_faces([encodeElon], encode)
        #print(result)

    return(encodeList)
def detectedNameAndTime(name):
    with open("detectedNames.csv","r+") as f:
        myDetectedNameAndDateList=f.readlines()
        nameList=[]
        for entry in myDetectedNameAndDateList:
            entryList=entry.split(',')
            nameList.append(entryList[0])
        if not name in nameList:
            now=datetime.now()
            dateString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')

encodeListKnown=findEncodings(images)

camera=cv2.VideoCapture(0)

while True:
    success,img=camera.read()
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgSmall)
    encodesCurFrame=face_recognition.face_encodings(imgSmall,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 0, 0),cv2.FILLED)
            cv2.putText(img,name,(x1,y2-4),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
            print(name)
            detectedNameAndTime(name)





    cv2.imshow("video",img)
    cv2.waitKey(1)




#test







# imgElon=face_recognition.load_image_file("resimler/Elon_Musk1.jpg")
# imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
#
# imgTest=face_recognition.load_image_file("resimler/elon-musk2.jpg")
# imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#
# imgTest2=face_recognition.load_image_file("resimler/Elon_Musk3.jpg")
# imgTest2=cv2.cvtColor(imgTest2,cv2.COLOR_BGR2RGB)
# encodeElonTest2=face_recognition.face_encodings(imgTest2)[0]
#
#
# imgTest3=face_recognition.load_image_file("resimler/elon_musk4.jpg")
# imgTest3=cv2.cvtColor(imgTest3,cv2.COLOR_BGR2RGB)
# encodeElonTest3=face_recognition.face_encodings(imgTest3)[0]
#
#
#
#
# faceLoc=face_recognition.face_locations(imgElon)[0]
# encodeElon=face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)
#
# faceLocTest=face_recognition.face_locations(imgTest)[0]
# encodeElonTest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)
#
#
#
# result=face_recognition.compare_faces([encodeElon],encodeElonTest)
# result2=face_recognition.compare_faces([encodeElon],encodeElonTest2)
# result3=face_recognition.compare_faces([encodeElon],encodeElonTest3)
#
# print(result)
# print(result2)
# print(result3)
#
# cv2.imshow("Elon",imgElon)
# cv2.imshow("Elon_Test",imgTest)
#
# cv2.waitKey(0)
#

