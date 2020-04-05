# https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
# https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
# https://github.com/DFoly/Gaussian-Mixture-Modelling/blob/master/gaussian-mixture-model.ipynb
# https://cmsc426.github.io/colorseg/#colorclassification

from matplotlib import pyplot as plt
import numpy as np
import math
import time
import os
import sys
try:
    # print("Ye")
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv
from scipy.stats import multivariate_normal as mvn
from imutils import contours

path_orange = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Orange'
vid = cv.VideoCapture("detectbuoy.avi")
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
out = cv.VideoWriter('model_orange_3D.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
def mean_orange():    
    orange_data = 117
    datapoints_orange = []
    for i in range(0, orange_data):
        string_path = path_orange +"/orange"+str(i)+".jpg"
        img = cv.imread(string_path)
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img = np.reshape(img, (height * width, channels))
        for pixels in range(img.shape[0]):
            datapoints_orange.append(img[pixels, :])
    return datapoints_orange

def look_at_histogram():
    orange_data = 117
    hist_size = [255]
    hist_range = [0,256]
    histogram_r = np.zeros((255,1))
    histogram_g = np.zeros((255,1))
    histogram_b = np.zeros((255,1))
    mean_b = []
    mean_g = []
    mean_r = []
    
    std_dev_b = []
    std_dev_g = []
    std_dev_r = []

    for i in range(0, orange_data):
        string_path = path_orange + "/orange"+str(i)+".jpg"
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

    #sum across the columns, divide by total number of observations
    histogram_avg_b = np.sum(histogram_b, axis=1) / (yellow_data)
    histogram_avg_g = np.sum(histogram_g, axis=1) / (yellow_data)
    histogram_avg_r = np.sum(histogram_r, axis=1) / (yellow_data) 

    # Uncomment to plot histograms
    plt.subplot(3,1,1)
    plt.plot(histogram_avg_b, color = "b")
    plt.subplot(3,1,2)
    plt.plot(histogram_avg_g, color = "g")
    plt.subplot(3,1,3)
    plt.plot(histogram_avg_r, color = "r")
    plt.show()

def learn_with_em(xtrain, K, iters):
    n_points, dimen = xtrain.shape
    mean = np.float64(xtrain[np.random.choice(n_points, K, False), :])
    # # print(mean)  
    covar = [np.random.randint(1,255) * np.eye(dimen)] * K# [150*np.eye(d)] * K
    # print(covar)
    for i in range(K):
        covar[i]=np.multiply(covar[i],np.random.rand(dimen,dimen))
    

    max_bound = 0.0001
    pi_k = [1./K] * K
    prob_cluster__given_x = np.zeros((n_points, K))

    log_likelihoods_array = []
    # print(covar)
    while len(log_likelihoods_array) < iters:
        # Expectation Step
        print(len(log_likelihoods_array))
        for k in range(K):
            tmp = pi_k[k] * mvn.pdf(xtrain, mean[k], covar[k], allow_singular=True)
            prob_cluster__given_x[:,k]=tmp.reshape((n_points,))

        log_likelihood = np.sum(np.log(np.sum(prob_cluster__given_x, axis = 1)))

        print ("{0} -> {1}".format(len(log_likelihoods_array),log_likelihood))

        log_likelihoods_array.append(log_likelihood)
        prob_cluster__given_x = (prob_cluster__given_x.T / np.sum(prob_cluster__given_x, axis = 1)).T
         
        N_ks = np.sum(prob_cluster__given_x, axis = 0)
        
        # Maximization Step
        for k in range(K):
            temp = math.fsum(prob_cluster__given_x[:,k])
            mean[k] = 1. / N_ks[k] * np.sum(prob_cluster__given_x[:, k] * xtrain.T, axis = 1).T
            diff_x_mean = xtrain - mean[k]
            covar[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(diff_x_mean.T,  prob_cluster__given_x[:, k]), diff_x_mean))
            pi_k[k] = 1. / n_points * N_ks[k]
        if len(log_likelihoods_array) < 2 : continue
        if np.abs(log_likelihood - log_likelihoods_array[-2]) < max_bound or len(log_likelihoods_array) > 1000: break

    plt.plot(log_likelihoods_array)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
    return mean, covar, pi_k

def orange_buoy_visual(trained_mean, trained_covar, train_pi_k, K):
    print("Frame Reading started..")
    while (vid.isOpened()):
        ret,frame = vid.read()
        if frame is not None:
            frame_orig = frame
            if ret == True:            
                height, width, channels = frame.shape[0], frame.shape[1], frame.shape[2]
                frame = np.reshape(frame, (height * width, channels))
                prob_of_orange_buoy = np.zeros((height * width, K))
                orange_likelihood = np.zeros((height * width, K))

                for k in range(0, K):
                    # Look at probabilities from here.
                    prob_of_orange_buoy[:, k] = train_pi_k[k] * mvn.pdf(frame, trained_mean[k], trained_covar[k])
                    orange_likelihood = prob_of_orange_buoy.sum(1)

                orange_prob = np.reshape(orange_likelihood, (height, width))
                orange_prob[np.where(orange_prob == np.max(orange_prob))] = 255
                mask_image =np.zeros((height, width, channels), np.uint8)
                mask_image[:,:,0] = orange_prob
                mask_image[:,:,1] = orange_prob
                mask_image[:,:,2] = orange_prob
                # blur = cv.GaussianBlur(mask_image, (3, 3), 5)
                # kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
                # np.array([[0, 0, 1, 0, 0],
                #        [0, 1, 1, 1, 0],
                #        [1, 1, 1, 1, 1],
                #        [0, 1, 1, 1, 0],
                #        [0, 0, 1, 0, 0]], dtype=np.uint8)
                kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
                np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
                
                dilated = cv.dilate(mask_image, kernel_ellipse, iterations = 1)
                edges = cv.Canny(dilated, 50, 255)
                # cv.imshow("orig", frame_orig)
                
                contours_image, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                # cont_img = cv.drawContours(frame_orig, contours_image, -1, (0,0,255), 5)
                cv.imshow("Mask", edges)

                (contour_sorted, bounds) = contours.sort_contours(contours_image)
                hull = cv.convexHull(contour_sorted[0])
                (x, y), radius = cv.minEnclosingCircle(hull)
                # print(radius, x, y)
                if radius > 2.6 and ((x > 120 and y > 320) or (x > 350 and y > 200)):
                    cv.circle(frame_orig, (int(x), int(y) - 10), int(radius + 10), (7,150,250), 4)
                cv.imshow("Final", frame_orig)
                out.write(frame_orig)
                k = cv.waitKey(15) & 0xff
                if k == 27:
                    break

        else:
            break
    vid.release()
    out.release()
# Uncomment to see the Average Histogram
# look_at_histogram()

# # Uncomment this
# K = 5
# mean_green_pts = mean_green()
# trained_mean, trained_covar, train_pi_k = learn_with_em(np.array(mean_green_pts), 5, 500)

K = 9
# mean_orange_pts = mean_orange()
# trained_mean, trained_covar, train_pi_k = learn_with_em(np.array(mean_orange_pts), K, 500)
# np.save('mean.npy', trained_mean)
# np.save('covar.npy', trained_covar)
# np.save('weights.npy', train_pi_k)

trained_mean = np.load('mean.npy') 
trained_covar = np.load('covar.npy')
train_pi_k = np.load('weights.npy')


# orangeboi_r = mvn.pdf(list(range(0,256)), trained_mean[0][0], trained_covar[0][2, 2])
# orangeboi_g = mvn.pdf(list(range(0,256)), trained_mean[0][1], trained_covar[0][1, 1])
# orangeboi_b = mvn.pdf(list(range(0,256)), trained_mean[0][2], trained_covar[0][0, 0])

# plt.plot(orangeboi_r, "r", orangeboi_g, "g", orangeboi_b, "b")
# plt.title('Gaussian Curve for only first mean value')
# plt.xlabel('x (0-256)')
# plt.ylabel('Probabilites')
# plt.show()
print("EM finished..")

orange_buoy_visual(trained_mean, trained_covar, train_pi_k, K)
cv.destroyAllWindows()