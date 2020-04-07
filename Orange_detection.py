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
hist_range = [0,256]
dataset = []
path_g = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Green_new'
path_o = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Orange'
path_y = r'C:\Users\shubh\Desktop\PMRO\SEM2\Perception\P3\Training Set\Yellow_new'

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
green_data = 28
orange_data = 117
yellow = 140

"""
mean_b = []
mean_g = []
mean_r = []
std_dev_b = []
std_dev_g = []
std_dev_r = []
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

    # Iterate for no of green buoy images
    for i in range(0,green_data):
        string_path = path_o+"\orange"+str(i)+".jpg"
        img = cv.imread(string_path) 
        color = ("b","g","r")

        # New Mean Calculation
        mask= np.zeros((img.shape[0],img.shape[0],3), np.uint8)
        coordinates = np.indices((img.shape[0], img.shape[0]))
        coordinates = coordinates.reshape(2, -1)
        x,y=coordinates[0],coordinates[1]
        indices=np.where((x-img.shape[0]/2)**2+(y-img.shape[0]/2)**2 < (img.shape[0]/2)**2)
        xnew,ynew=x[indices[0]],y[indices[0]]
        mask[xnew,ynew]=img[xnew,ynew]
        pixels=img[xnew,ynew]
        mean=np.sum(pixels, axis=0) / len(pixels)
        stds=[np.std(pixels[:,0]),np.std(pixels[:,1]),np.std(pixels[:,2])]

        
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

    histogram_avg_b = np.sum(histogram_b, axis=1) / (green_data)
    histogram_avg_g = np.sum(histogram_g, axis=1) / (green_data)
    histogram_avg_r = np.sum(histogram_r, axis=1) / (green_data)

    #Uncomment to plot histograms
    plt.subplot(3,1,1)
    plt.plot(histogram_avg_b, color = "b")
    plt.subplot(3,1,2)
    plt.plot(histogram_avg_g, color = "g")
    plt.subplot(3,1,3)
    plt.plot(histogram_avg_r, color = "r")
    plt.show()
    
    return mean_r, mean_g, mean_b, std_dev_r, std_dev_g, std_dev_b





def EM():
    K = 5
    
    datapoint = []
    
    for i in range(0, green_data):
        string_path = path_o+"\orange"+str(i)+".jpg"
        img = cv.imread(string_path)
#        img = cv.resize(img, (10,10))
#        print(string_path)
        #select green channel
        green_chan_img = img[:,:,2]
        height, width = green_chan_img.shape
        
        coordinates = np.indices((green_chan_img.shape[0], green_chan_img.shape[0]))
        coordinates = coordinates.reshape(2, -1)
        x,y=coordinates[0],coordinates[1]
        indices=np.where((x-green_chan_img.shape[0]/2)**2+(y-green_chan_img.shape[0]/2)**2 < (green_chan_img.shape[0]/2)**2)
        xnew,ynew=x[indices[0]],y[indices[0]]

        # Prepared Dataset
        datapoint=green_chan_img[xnew,ynew]
        
    #intital estimates
    mean_b_init = 130
    mean_g_init = 210
    mean_r_init = 130
    std_dev_b_init = 20
    std_dev_g_init = 35
    std_dev_r_init = 42
    
    iterations = 0
    
    while(iterations <= 500):        
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
        for i in range(len(datapoint)):
            #calculate probabilty at that pixel
            probability_1 = GaussianEquation(std_dev_b_init, datapoint[i], mean_b_init)
            probability_2 = GaussianEquation(std_dev_g_init, datapoint[i], mean_g_init)
            probability_3 = GaussianEquation(std_dev_r_init, datapoint[i], mean_r_init)
            
            #gaussian of 3 channels
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
        
        #calculating SD from mean and data points
        std_dev_b_init = (np.sum(np.array(pi_k_1) * ((np.array(datapoint)) 
        - mean_b_init) ** (2)) / np.sum(np.array(pi_k_1))) ** (1 / 2)
        std_dev_g_init = (np.sum(np.array(pi_k_2) * ((np.array(datapoint)) 
        - mean_g_init) ** (2)) / np.sum(np.array(pi_k_2))) ** (1 / 2)
        std_dev_r_init = (np.sum(np.array(pi_k_1) * ((np.array(datapoint)) 
        - mean_r_init) ** (2)) / np.sum(np.array(pi_k_3))) ** (1 / 2)      
        
        
#        print("bgr :"+str(mean_b_init) + " " + str(mean_g_init) + " " + str(mean_r_init))
        
        iterations = iterations + 1
        print(iterations)
    return mean_b_init, mean_g_init, mean_r_init, std_dev_b_init, std_dev_g_init, std_dev_r_init  

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
out = cv.VideoWriter('green_1D.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


if __name__ == "__main__":
    mean_r, mean_g, mean_b, std_dev_r, std_dev_g, std_dev_b = AverageHistogram()
    avg_mean_b = sum(mean_b)/len(mean_b)
    avg_mean_g = sum(mean_g)/len(mean_g)
    avg_mean_r = sum(mean_r)/len(mean_r)
    
    avg_std_dev_b = sum(std_dev_b)/len(std_dev_b)
    avg_std_dev_g = sum(std_dev_g)/len(std_dev_g)
    avg_std_dev_r = sum(std_dev_r)/len(std_dev_r)
    
    gaussian_b = GaussianEquation(avg_std_dev_b, list(range(0,256)), avg_mean_b)
    gaussian_g = GaussianEquation(avg_std_dev_g, list(range(0,256)), avg_mean_g)
    gaussian_r = GaussianEquation(avg_std_dev_r, list(range(0,256)), avg_mean_r)
    
#    plt.plot(gaussian_b, "b", gaussian_g, "g", gaussian_r, "r")
    
    mean_b_init, mean_g_init, mean_r_init, std_dev_b_init, std_dev_g_init, std_dev_r_init = EM()
    
    
    greenboi_r = GaussianEquation(std_dev_r_init, list(range(0,256)), mean_r_init)
    greenboi_g = GaussianEquation(std_dev_g_init, list(range(0,256)), mean_g_init)
    greenboi_b = GaussianEquation(std_dev_b_init, list(range(0,256)), mean_b_init)
    
    plt.plot(greenboi_r, "r", greenboi_g, "g", greenboi_b, "b")
    while True:
      ret,frame = vid.read()
      if frame is not None:
        frame_g=frame[:,:,2]
        frame_r=frame[:,:,2]
        if ret == True:
            frame_updated=np.zeros(frame_g.shape, dtype = np.uint8)
#            print("here " + str(frame_updated.shape))
            coordinates = np.indices((frame_g.shape[0], frame_g.shape[1]))
            coordinates = coordinates.reshape(2, -1)
            x,y=coordinates[0],coordinates[1]
            pixel_val=frame_g[x,y]
            
            indices1=np.where((greenboi_r[pixel_val]<0.02) & (greenboi_g[pixel_val]>0.01) & (greenboi_b[pixel_val]<0.02) )
            indices2=np.where((greenboi_r[pixel_val]>0.02) & (greenboi_g[pixel_val]<0.01) & (greenboi_b[pixel_val]>0.02) )
            x1,y1=x[indices1[0]],y[indices1[0]]
            x2,y2=x[indices2[0]],y[indices2[0]]
            frame_updated[x1,y1]=255
            frame_updated[x2,y2]=0
            
            kernel_square = np.ones((10,10),np.uint8)
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
            np.array([[0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 0]], dtype=np.uint8)
            blur = cv.blur(frame_updated,(10,10)) 
            ret_thresh, thresholded = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)
            cv.imshow("r1",thresholded)
            edges = cv.Canny(thresholded, 200, 300)
            dilated = cv.dilate(thresholded, kernel_ellipse, iterations = 1)
#            dilated_updated = cv.cvtColor(dilated, cv.COLOR_BGR2GRAY)
            contours, _ = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#            cont_img = cv.drawContours(frame, contours, -1, (0,0,255), 5)
#
            # Draw circle to fit the contours enclosing specified area
            for c in contours:
#                print("area: "+str(cv.contourArea(c)))
                if cv.contourArea(c) > 40:
                    print("inside1")
                    (x,y),r = cv.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    r = int(r)
                    print(r)
                    if r > 8 and r < 35 and y<350 and y>150 :
                        print("inside")
                        cv.circle(frame,center,r,(36,171,255),2)
#            cv.line(frame, (0,440), (600, 440), (255,0,0), 2)
            cv.imshow("threshold", frame)
            #cv.waitKey(0)
            out.write(frame)
            k = cv.waitKey(15) & 0xff
            if k == 27:
                break

        else:
            break
        
        
    vid.release()
    cv.destroyAllWindows()