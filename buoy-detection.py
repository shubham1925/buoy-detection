import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import time
#from roipoly import roipoly
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

hist_size = [256]
hist_range = [0,256]

vid = cv.VideoCapture("detectbuoy.avi")


def GaussianEquation(sigma, x, mean):
    equation = (1/(sigma*math.sqrt(2*math.pi)))*np.exp(-0.5*(x-mean)**2/sigma**2)
    return equation
    

path = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\frame2.jpg'
image = cv.imread(path)
#cv.imshow("image", image)
color = ("b","g","r")

for j,c in enumerate(color):
    if c == "r":
        histogram_red = cv.calcHist([image], [j], None, hist_size, hist_range)
    if c == "g":
        histogram_green = cv.calcHist([image], [j], None, hist_size, hist_range)
    if c == "b":
        histogram_blue = cv.calcHist([image], [j], None, hist_size, hist_range)
        
plt.subplot(3,1,1)        
plt.plot(histogram_red, color = "r")
plt.subplot(3,1,2)
plt.plot(histogram_green, color = "g")
plt.subplot(3,1,3)
plt.plot(histogram_blue, color = "b")




#average histogram code --->assuming we have images
for i in images:
    img = cv.imread("path"+"image")
    b,g,r = cv.split(img)
    color = ("b","g","r")

    for j,c in enumerate(color):
        if c == "r":
            histogram_red = cv.calcHist([img], [j], None, hist_size, hist_range)
        if c == "g":
            histogram_green = cv.calcHist([img], [j], None, hist_size, hist_range)
        if c == "b":
            histogram_blue = cv.calcHist([img], [j], None, hist_size, hist_range)
        
    
#
#code for extracting frames from video
# =============================================================================
# i = 0;
# while(vid.isOpened()):
#     ret, frame = vid.read()
#     if ret == False:
#         break
#     cv.imwrite("frame"+str(i)+".jpg", frame)
#     i = i + 1
#     
# vid.release()
# cv.destroyAllWindows()
# =============================================================================
#    
#
