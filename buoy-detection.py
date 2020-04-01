import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import time
import os
import cv2 as cv

#from roipoly import roipoly
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

hist_size = [255]
hist_range = [0,254]
dataset = []
path_g = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Green'
vid = cv.VideoCapture("detectbuoy.avi")
for i in os.listdir(path_g):
    dataset.append(i)

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

#define the total number of training imgs for each chan
green_data = 29
orange_data = 117
yellow = 140

"""
mean_b = []
mean_g = []
mean_r = []

std_dev_b = []
std_dev_g = []
std_dev_r = []

# =============================================================================
#trial_str = path_g + "\green10.jpg"
#img = cv.imread(trial_str)
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#thresh = cv.threshold(gray, 240,255,cv.THRESH_BINARY_INV)[1]
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
#morphed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
#cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
#cv.drawContours(img, cnts, -1, [0,0,255], 2)
#cnt = sorted(cnts, key=cv.contourArea)[-1]
#x,y,w,h = cv.boundingRect(cnt)
#dst = img[y:y+h, x:x+w]
# 
#cv.imshow("trial",img)
#cv.waitKey(0)
# =============================================================================




for i in os.listdir(path_g):
    dataset.append(i)
    
histogram_r = np.zeros((255,1))
histogram_g = np.zeros((255,1))
histogram_b = np.zeros((255,1))


for i in range(0,green_data):
#    img = i
    string_path = path_g+"\green"+str(i)+".jpg"
    img = cv.imread(string_path) 
    
#    image_q = cv.imread(string_path)
#    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#    threshold = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)[1]
#    threshold_arr = np.array(threshold)
##    cv.imshow("thresh", threshold_arr)
#    
##    cv.waitKey(0)
##    print(type(gray))
#    
#    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
#    morphed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)
#    cnts = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
#    cnt = sorted(cnts, key=cv.contourArea)[-1]
#    x,y,w,h = cv.boundingRect(cnt)
#    dst = img[y:y+h, x:x+w]
#    img = dst
    color = ("b","g","r")
    (mean, stds) = cv.meanStdDev(img)
    for j,c in enumerate(color):
        if c == "b":
            temp_b = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
            histogram_b = np.column_stack((histogram_b, temp_b))
            mean_b.append(mean[0])
            std_dev_b.append(stds[0])
        if c == "g":
            temp_g = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
            histogram_g = np.column_stack((histogram_g, temp_g))
            mean_g.append(mean[1])
            std_dev_g.append(stds[1])
        if c == "r":
            temp_r = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
            histogram_r = np.column_stack((histogram_r, temp_r))
            mean_r.append(mean[2])
            std_dev_r.append(stds[2])

print(histogram_b.shape[1])
#sum across the columns, divide by total number of observations
histogram_avg_b = np.sum(histogram_b, axis=1) / (green_data)
histogram_avg_g = np.sum(histogram_g, axis=1) / (green_data)
histogram_avg_r = np.sum(histogram_r, axis=1) / (green_data)
#plt.subplot(3,1,1)
#plt.plot(histogram_avg_b, color = "b")
#plt.subplot(3,1,2)
#plt.plot(histogram_avg_g, color = "g")
#plt.subplot(3,1,3)
#plt.plot(histogram_avg_r, color = "r")


avg_mean_b = sum(mean_b)/len(mean_b)
avg_mean_g = sum(mean_g)/len(mean_g)
avg_mean_r = sum(mean_r)/len(mean_r)

avg_std_dev_b = sum(std_dev_b)/len(std_dev_b)
avg_std_dev_g = sum(std_dev_g)/len(std_dev_g)
avg_std_dev_r = sum(std_dev_r)/len(std_dev_r)

gaussian_b = GaussianEquation(avg_std_dev_b, list(range(0,255)), avg_mean_b)
gaussian_g = GaussianEquation(avg_std_dev_g, list(range(0,255)), avg_mean_g)
gaussian_r = GaussianEquation(avg_std_dev_r, list(range(0,255)), avg_mean_r)

plt.plot(gaussian_b, "b", gaussian_g, "g", gaussian_r, "r")
"""

def GaussianEquation(sigma, x, mean):
    equation = (1/(sigma*math.sqrt(2*math.pi)))*np.exp(-0.5*(x-mean)**2/sigma**2)
    return equation

def AverageHistogram():
    mean_b = []
    mean_g = []
    mean_r = []
    
    std_dev_b = []
    std_dev_g = []
    std_dev_r = []
    histogram_r = np.zeros((255,1))
    histogram_g = np.zeros((255,1))
    histogram_b = np.zeros((255,1))
    for i in range(0,green_data):
        string_path = path_g+"\green"+str(i)+".jpg"
        img = cv.imread(string_path) 
        color = ("b","g","r")
        (mean, stds) = cv.meanStdDev(img)
        for j,c in enumerate(color):
            if c == "b":
                temp_b = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_b = np.column_stack((histogram_b, temp_b))
                mean_b.append(mean[0])
                std_dev_b.append(stds[0])
            if c == "g":
                temp_g = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_g = np.column_stack((histogram_g, temp_g))
                mean_g.append(mean[1])
                std_dev_g.append(stds[1])
            if c == "r":
                temp_r = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_r = np.column_stack((histogram_r, temp_r))
                mean_r.append(mean[2])
                std_dev_r.append(stds[2])
                
    #Uncomment to plot histograms
    #plt.subplot(3,1,1)
    #plt.plot(histogram_avg_b, color = "b")
    #plt.subplot(3,1,2)
    #plt.plot(histogram_avg_g, color = "g")
    #plt.subplot(3,1,3)
    #plt.plot(histogram_avg_r, color = "r")
    return mean_r, mean_g, mean_b, std_dev_r, std_dev_g, std_dev_b

def EM():
    K = 3
    
    datapoint = []
    
    for i in range(0, green_data):
        string_path = path_g+"\green"+str(i)+".jpg"
        img = cv.imread(string_path)
#        print(string_path)
        #select green channel
        green_chan_img = img[:,:,1]
        height, width = green_chan_img.shape
        for h in range(0, height):
            for w in range(0, width):
                #collect all the datapoints (x) in algorithm
                datapoint.append(green_chan_img[h][w])
    
    #intital estimates
    mean_b_init = 160
    mean_g_init = 200
    mean_r_init = 160
    std_dev_b_init = 50
    std_dev_g_init = 30
    std_dev_r_init = 60
    
    iterations = 0
    
    while(iterations <= 3):        
#        print("iteration: "+str(iterations))
        responsibility_1 = []
        responsibility_2 = []
        responsibility_3 = []
        
        probability_dist_1 = []
        probability_dist_2 = []
        probability_dist_3 = []
        
        pi_k_1 = []
        pi_k_2 = []
        pi_k_3 = []            
        
        #perform e and m for each datapoint
        for i in datapoint:
#            print("in for")
            #calculate probabilty at that pixel
            probability_1 = GaussianEquation(std_dev_b_init, i, mean_b_init)
            probability_2 = GaussianEquation(std_dev_g_init, i, mean_g_init)
            probability_3 = GaussianEquation(std_dev_r_init, i, mean_r_init)
            
            probability_dist_1.append(probability_1)
            probability_dist_2.append(probability_2)
            probability_dist_3.append(probability_3)
            
            temp_pi_1 = probability_1/(probability_1 + probability_2 + probability_3)
            temp_pi_2 = probability_2/(probability_1 + probability_2 + probability_3)
            temp_pi_3 = probability_3/(probability_1 + probability_2 + probability_3)
            
            pi_k_1.append(temp_pi_1)
            pi_k_2.append(temp_pi_2)
            pi_k_3.append(temp_pi_3)
#            print(i)
            
        #formula for calculating new mean from pdf    
        mean_b_init = np.sum(np.array(pi_k_1)*np.array(datapoint))/np.sum(np.array(pi_k_1))
        mean_g_init = np.sum(np.array(pi_k_2)*np.array(datapoint))/np.sum(np.array(pi_k_2))
        mean_r_init = np.sum(np.array(pi_k_3)*np.array(datapoint))/np.sum(np.array(pi_k_3))
        
        std_dev_b_init = (np.sum(np.array(pi_k_1) * ((np.array(datapoint)) 
        - mean_b_init) ** (2)) / np.sum(np.array(pi_k_1))) ** (1 / 2)
        std_dev_g_init = (np.sum(np.array(pi_k_2) * ((np.array(datapoint)) 
        - mean_g_init) ** (2)) / np.sum(np.array(pi_k_2))) ** (1 / 2)
        std_dev_r_init = (np.sum(np.array(pi_k_1) * ((np.array(datapoint)) 
        - mean_r_init) ** (2)) / np.sum(np.array(pi_k_3))) ** (1 / 2)
        
        print("bgr :"+str(mean_b_init) + " " + str(mean_g_init) + " " + str(mean_r_init))
        
        iterations = iterations + 1
        print(iterations)
        
    return mean_b_init, mean_g_init, mean_r_init, std_dev_b_init, std_dev_g_init, std_dev_r_init  

if __name__ == "__main__":
    mean_r, mean_g, mean_b, std_dev_r, std_dev_g, std_dev_b = AverageHistogram()
    avg_mean_b = sum(mean_b)/len(mean_b)
    avg_mean_g = sum(mean_g)/len(mean_g)
    avg_mean_r = sum(mean_r)/len(mean_r)
    
    avg_std_dev_b = sum(std_dev_b)/len(std_dev_b)
    avg_std_dev_g = sum(std_dev_g)/len(std_dev_g)
    avg_std_dev_r = sum(std_dev_r)/len(std_dev_r)
    
    gaussian_b = GaussianEquation(avg_std_dev_b, list(range(0,255)), avg_mean_b)
    gaussian_g = GaussianEquation(avg_std_dev_g, list(range(0,255)), avg_mean_g)
    gaussian_r = GaussianEquation(avg_std_dev_r, list(range(0,255)), avg_mean_r)
    
    plt.plot(gaussian_b, "b", gaussian_g, "g", gaussian_r, "r")
    
    mean_b_init, mean_g_init, mean_r_init, std_dev_b_init, std_dev_g_init, std_dev_r_init = EM()
    
    
    
#p 
#
#path_y = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Yellow\yellow.jpg'
#path_o = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Orange\orange.jpg'
#image_o = cv.imread(path_o)
#path_g = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Green\green.jpg' 
#image_y = cv.imread(path_y)
#image_g = cv.imread(path_g)
#image_main = cv.imread("frame2.jpg")
##cv.imshow("image", image)
#color = ("b","g","r")
#
##for j,c in enumerate(color):
##    if c == "r":
##        histogram_red = cv.calcHist([image_y], [j], None, hist_size, hist_range)
##    if c == "g":
##        histogram_green = cv.calcHist([image_y], [j], None, hist_size, hist_range)
##    if c == "b":
##        histogram_blue = cv.calcHist([image_y], [j], None, hist_size, hist_range)
##        
##for j,c in enumerate(color):
##    if c == "r":
##        histogram_red = cv.calcHist([image_o], [j], None, hist_size, hist_range)
##    if c == "g":
##        histogram_green = cv.calcHist([image_o], [j], None, hist_size, hist_range)
##    if c == "b":
##        histogram_blue = cv.calcHist([image_o], [j], None, hist_size, hist_range)
#        
#for j,c in enumerate(color):
#    if c == "r":
#        histogram_red = cv.calcHist([image_g], [j], None, hist_size, hist_range, accumulate = 1)
#        
#    if c == "g":
#        histogram_green = cv.calcHist([image_g], [j], None, hist_size, hist_range)
#    if c == "b":
#        histogram_blue = cv.calcHist([image_g], [j], None, hist_size, hist_range)
#
#final_histo = np.column_stack((histogram_red, histogram_green))
#final_histo_avg = np.sum(final_histo, axis = 1)/(final_histo.shape[1] - 1)
#
#mean_y, std_dev_y = cv.meanStdDev(image_main)
#
#x = list(range(0,255))
#
#g_eqn_y_g = GaussianEquation(std_dev_y[1], x, mean_y[1])
#
#print(np.sum(g_eqn_y_g))
#
#plt.subplot(2,1,1)
#plt.plot(g_eqn_y_g, color = "g")
#plt.subplot(2,1,2)
#plt.plot(histogram_green, color = "g")
#
#while True:
#    ret, frame = vid.read()
#    green_frame = frame[:,:,1]
#    width, height = green_frame.shape
#    img_thresholded = np.zeros(green_frame.shape, dtype = np.uint8)
#    
#    for w in range(0,width):
#        for h in range(0,height):
#            pixel_value =green_frame[w][h]
##            print("pixel val: "+str(pixel_value))
#            if g_eqn_y_g[pixel_value-1] > 0.005 and g_eqn_y_g[pixel_value-1] < 0.015:
#                img_thresholded[w][h] = 255
#            else:
#                img_thresholded[w][h] = 0
#    cv.imshow("thresh", img_thresholded)
#                
#                
#                
#
#
#
#
##        
##plt.subplot(3,1,1)        
##plt.plot(histogram_red, color = "r")
##plt.subplot(3,1,2)
##plt.plot(histogram_green, color = "g")
##plt.subplot(3,1,3)
##plt.plot(histogram_blue, color = "b")
#
#
#
#
##average histogram code --->assuming we have images
#for i in images:
#    img = cv.imread("path"+"image")
#    b,g,r = cv.split(img)
#    color = ("b","g","r")
#
#    for j,c in enumerate(color):
#        if c == "r":
#            histogram_red = cv.calcHist([img], [j], None, hist_size, hist_range)
#        if c == "g":
#            histogram_green = cv.calcHist([img], [j], None, hist_size, hist_range)
#        if c == "b":
#            histogram_blue = cv.calcHist([img], [j], None, hist_size, hist_range)
#        
    
#

#    
#
