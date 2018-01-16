import os 
import cv2
import time

def read_img(path,*suffix):
    s=os.listdir(path)
    resultArray=[]
    for i in s:
        if endwith(i,suffix):
            document=os.path.join(path,i)
            img=cv2.imread(document)
            resultArray.append(img)
            
    return resultArray

def endwith(s,*endstring):
    resultArray=map(s.endswith,endstring)
    if True in resultArray:
        return True
    else:
        return False
    
    
def readPicSaveFace(sourcePath,objectPath,*suffix):
    count=1
    resultArray=read_img(sourcePath,*suffix)
    face_cascade=cv2.CascadeClassifier(r'F:\opencv-3.4.0\data\haarcascades\haarcascade_frontalface_default.xml')
    for i in resultArray:
        gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            f=cv2.resize(gray[y:(y+h),x:(x+w)],(200,200))
            cv2.imwrite(os.path.join(objectPath,'1_%s.jpg'%count),f)
            count+=1
if __name__=='__main__':
    readPicSaveFace(r'F:\photo',r'F:\objectphoto','.jpg','.JPG','png','PNG')
    print('finish')