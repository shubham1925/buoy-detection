import cv2 as cv
import numpy as np

#https://www.life2coding.com/crop-image-using-mouse-click-movement-python/
#https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
vertices = []
image = cv.imread('frame2.jpg') 

path_y = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Yellow'
path_o = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Orange'
path_g = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Green'
    
def mouse_crop(event, x, y, flag, params):  
    if event==cv.EVENT_LBUTTONDOWN:
        vertices.append((x,y))       
        if len(vertices) == 10:
            blank = np.zeros_like(image)
            vertices_arr = np.array(vertices)
            #To draw the contours, cv.drawContours function is used. It can 
            #also be used to draw any shape provided you have its boundary points
            cv.drawContours(blank,[vertices_arr],-1,(255,255,255),-1)
            cv.imshow("after contour", blank)
            blank = cv.bitwise_not(blank)
            cv.imshow("after not", blank)
            only_boi = cv.add(image,blank)   
            cv.imshow("after add", only_boi)
            x,y,c = np.where(only_boi != 255)            
            cropped = only_boi[np.min(x):np.max(x),np.min(y):np.max(y)]
            cv.imshow("buoy", only_boi)
#            cropped = cv.resize(cropped, (100,100), interpolation = cv.INTER_AREA)
#            cv.imwrite(path_y+"\yellow.jpg",only_boi)
#            cv.imwrite(path_o+"\orange.jpg",only_boi)
            cv.imwrite(path_g+"\green.jpg",cropped)
            cv.imshow("cropped",cropped)            
            cv.imshow("name",image)  
        if len(vertices) >=2 :
            cv.line(image,vertices[-1],vertices[-2],(255,0,0),1)
            cv.imshow("name",image)


cv.imshow("name",image)
cv.setMouseCallback("name",mouse_crop)
cv.waitKey(0)