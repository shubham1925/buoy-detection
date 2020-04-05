# https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95

from matplotlib import pyplot as plt
import numpy as np
import math
import time
import os
import sys
#from roipoly import roipoly
try:
    # print("Ye")
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv
from scipy.stats import multivariate_normal as mvn

path_green = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Green'
path_orange = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Orange'
path_yellow = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Yellow'

def GaussianEquation(sigma, x, mean):
    equation = (1/(sigma*math.sqrt(2*math.pi)))*np.exp(-0.5*(x-mean)**2/sigma**2)
    return equation

def gauss_wala_eqn(data, mean, covar):
        det_cov = np.linalg.det(covar)
        cov_inv = np.linalg.inv(covar)
        diff = np.matrix(data-mean)
        N = (2.0 * np.pi) ** (-len(data[1]) / 2.0) * (1.0 / (np.linalg.det(covar) ** 0.5)) *\
            np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv,diff),axis=1))
        
        return N
        
        

def mean_green():    
    green_data = 29
    # orange_data = 117
    # yellow = 140
    datapoints_green = []
    # mean_pts_green = []
    for i in range(0, green_data):
        string_path = path_green+"/green"+str(i)+".jpg"
        img = cv.imread(string_path)

        # blue_chan_img = img[:,:,0]
        # green_chan_img = img[:,:,1]
        # red_chan_img = img[:,:,2]
        # height, width = img.shape[0], img.shape[1]
        # for h in range(0, height):
        #     for w in range(0, width):
        #         datapoint_blue_for_green.append(blue_chan_img[h][w])
        #         datapoint_green_for_green.append(green_chan_img[h][w])
        #         datapoint_red_for_green.append(red_chan_img[h][w])
                # mean_pts_green.append([np.mean(blue_chan_img[h][w]), np.mean(green_chan_img[h][w]), np.mean(red_chan_img[h][w])])
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img = np.reshape(img, (height * width, channels))
        for pixels in range(img.shape[0]):
            # Get all channel points
            datapoints_green.append(img[pixels, :])
    return datapoints_green

def mean_orange():    
    orange_data = 29
    # orange_data = 117
    # yellow = 140
    datapoints_orange = []
    # mean_pts_green = []
    for i in range(0, orange_data):
        string_path = path_orange +"/orange"+str(i)+".jpg"
        img = cv.imread(string_path)

        # blue_chan_img = img[:,:,0]
        # green_chan_img = img[:,:,1]
        # red_chan_img = img[:,:,2]
        # height, width = img.shape[0], img.shape[1]
        # for h in range(0, height):
        #     for w in range(0, width):
        #         datapoint_blue_for_green.append(blue_chan_img[h][w])
        #         datapoint_green_for_green.append(green_chan_img[h][w])
        #         datapoint_red_for_green.append(red_chan_img[h][w])
                # mean_pts_green.append([np.mean(blue_chan_img[h][w]), np.mean(green_chan_img[h][w]), np.mean(red_chan_img[h][w])])
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img = np.reshape(img, (height * width, channels))
        for pixels in range(img.shape[0]):
            # Get all channel points
            datapoints_orange.append(img[pixels, :])
    return datapoints_orange

def look_at_histogram_son():
    green_data = 29
    yellow_data = 140
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

    for i in range(0,yellow_data):
        string_path = path_yellow + "/yellow"+str(i)+".jpg"
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
    
    # print(covar)
    max_bound = 0.0001
    # gamma = np.zeros((n_points, clusters))

    pi_k = [1./K] * K
    prob_cluster__given_x = np.zeros((n_points, K))

    log_likelihoods_array = []
    # print(covar)
    while len(log_likelihoods_array) < iters:
        # Expectation
        print(len(log_likelihoods_array))
        for k in range(K):
            tmp = pi_k[k] * mvn.pdf(xtrain, mean[k], covar[k], allow_singular=True)
            prob_cluster__given_x[:,k]=tmp.reshape((n_points,))

        log_likelihood = np.sum(np.log(np.sum(prob_cluster__given_x, axis = 1)))

        # print ("{0} -> {1}".format(len(log_likelihoods_array),log_likelihood))
        # if log_likelihood>-334735: break

        log_likelihoods_array.append(log_likelihood)
        # print(prob_cluster__given_x)
        prob_cluster__given_x = (prob_cluster__given_x.T / np.sum(prob_cluster__given_x, axis = 1)).T
        # prob_cluster__given_x = np.divide(prob_cluster__given_x.T, np.tile(prob_cluster__given_x,(K,1)).T)
         
        N_ks = np.sum(prob_cluster__given_x, axis = 0)
        
        # Maximization
        for k in range(K):
            temp = math.fsum(prob_cluster__given_x[:,k])
            # print(temp)
            mean[k] = 1. / N_ks[k] * np.sum(prob_cluster__given_x[:, k] * xtrain.T, axis = 1).T
            # print(mean[k])
            diff_x_mean = xtrain - mean[k]
            covar[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(diff_x_mean.T,  prob_cluster__given_x[:, k]), diff_x_mean))
            pi_k[k] = 1. / n_points * N_ks[k]
        # print(log_likelihoods_array[-2])  
        if len(log_likelihoods_array) < 2 : continue
        if np.abs(log_likelihood - log_likelihoods_array[-2]) < max_bound or len(log_likelihoods_array) > 10000: break

    plt.plot(log_likelihoods_array)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
    return mean, covar, pi_k

# Yellow Only, change inside function.
# look_at_histogram_son()

mean_green_pts = mean_green()
mean_orange_pts = mean_orange()

# mean_green_pt = [np.mean(blue_channel_pts), np.mean(green_channel_pts), np.mean(red_channel_pts)]
# x_train = np.array([blue_channel_pts, green_channel_pts, red_channel_pts]).T
# trained_mean, trained_covar, train_pi_k = learn_with_em(np.array(mean_green_pts), 5, 500)
# print(trained_mean[0][0], trained_covar,trained_covar[0][0, 1])

# greenboi_r = mvn.pdf(list(range(0,256)), trained_mean[0][0], trained_covar[0][0, 0])
# greenboi_g = mvn.pdf(list(range(0,256)), trained_mean[0][1], trained_covar[0][1, 1])
# greenboi_b = mvn.pdf(list(range(0,256)), trained_mean[0][2], trained_covar[0][2, 2])

# plt.plot(greenboi_r, "r", greenboi_g, "g", greenboi_b, "b")
# plt.title('Gaussian Curve for only first mean value')
# plt.xlabel('x (0-256)')
# plt.ylabel('Probabilites')
# plt.show()

trained_mean, trained_covar, train_pi_k = learn_with_em(np.array(mean_orange_pts), 9, 500)
print(trained_mean[0][0], trained_covar,trained_covar[0][0, 1])

orangeboi_r = mvn.pdf(list(range(0,256)), trained_mean[0][0], trained_covar[0][0, 0])
orangeboi_g = mvn.pdf(list(range(0,256)), trained_mean[0][1], trained_covar[0][1, 1])
orangeboi_b = mvn.pdf(list(range(0,256)), trained_mean[0][2], trained_covar[0][2, 2])

plt.plot(orangeboi_r, "r", orangeboi_g, "g", orangeboi_b, "b")
plt.title('Gaussian Curve for only first mean value')
plt.xlabel('x (0-256)')
plt.ylabel('Probabilites')
plt.show()

# x_train = np.array([blue_channel_pts, green_channel_pts, red_channel_pts]).T

# print(x_train.shape)


# string_path = path_green  + "/green0" + ".jpg"
# img = cv.imread(string_path)
# mean_std = cv.meanStdDev(img)
# img_mean_blue = mean_std[0].reshape(3,)[0] # Blue
# img_std_blue = mean_std[1].reshape(3,)[0] # Blue
# blue_chan_img = img[:,:,0].reshape(img.shape[0]*img.shape[1],)

# img_mean_green = mean_std[0].reshape(3,)[1] # Green
# img_std_green = mean_std[1].reshape(3,)[1] # Green
# green_chan_img = img[:,:,1].reshape(img.shape[0]*img.shape[1],)

# img_mean_red = mean_std[0].reshape(3,)[2] # Red
# img_std_red = mean_std[1].reshape(3,)[2] # Red
# red_chan_img = img[:,:,2].reshape(img.shape[0]*img.shape[1],)
# # print(green_chan_img.reshape(676,))
# y_b = gauss_wala_eqn(green_chan_img, img_mean_green, img_std_green)
# y_g = gauss_wala_eqn(blue_chan_img, img_mean_blue, img_std_blue)
# y_r = gauss_wala_eqn(red_chan_img, img_mean_red, img_std_red)
# # print(y)
# # plt.plot(green_chan_img, y)
# plt.subplot(3,1,1)
# plt.plot(blue_chan_img, y_b, color = "b")
# plt.subplot(3,1,2)
# plt.plot(green_chan_img, y_g, color = "g")
# plt.subplot(3,1,3)
# plt.plot(red_chan_img, y_r, color = "r")
# plt.show()

# bg_img_path = "/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Frames/frame0.jpg"
# bg_img = cv.imread(bg_img_path)
# # https://github.com/yashv28/Color-Segmentation-using-GMM/blob/master/em.py


key = cv.waitKey(3000)#pauses for 3 seconds before fetching next image
if key == 27:
    cv.destroyAllWindows()