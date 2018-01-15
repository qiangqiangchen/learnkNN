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
    resultArray=read_img(sourcePath,*suffix)
    
    