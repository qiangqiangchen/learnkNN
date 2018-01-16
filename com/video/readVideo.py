import numpy as np
import cv2
import time,math

"""
图像旋转操作函数

"""
def rotate(img,angle):
    height,width=img.shape[:2]
    
    if angle%180==0:
        scale=1
        rang=(width,height)
    elif angle%90==0:
        scale=float(max(height,width)/min(height,width))
        rang=(height,width)
    else:
        pass
        scale=math.sqrt(pow(height,2)+pow(width,2))/min(height, width) 
        
        
    M=cv2.getRotationMatrix2D((width/2,height/2),angle,scale) 
    res=cv2.warpAffine(img,M,rang)
    return res

index=1
cap=cv2.VideoCapture(r'F:\test.mp4')
while(cap.isOpened()):
    ret,frame=cap.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('fram',gray)
    time.sleep(0.1)
    cv2.imwrite("F:\\photo\\1_%d.png"%index,gray)
    index+=1
    k=cv2.waitKey(1)
    if k==ord('q') or k==27:
        break
    elif k==ord('s'):
        cv2.imwrite("test%d.png"%index,gray)
        index+=1
cap.release()
cv2.destroyAllWindows()
print('finish')
