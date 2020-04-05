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
#from roipoly import roipoly
try:
    # print("Ye")
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv
from scipy.stats import multivariate_normal as mvn
from imutils import contours

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
        if np.abs(log_likelihood - log_likelihoods_array[-2]) < max_bound or len(log_likelihoods_array) > 1000: break

    # plt.plot(log_likelihoods_array)
    # plt.title('Log Likelihood vs iteration plot')
    # plt.xlabel('Iterations')
    # plt.ylabel('log likelihood')
    # plt.show()
    return mean, covar, pi_k

# Yellow Only, change inside function.
# look_at_histogram_son()

def gote_dekho(trained_mean, trained_covar, train_pi_k, K):
    vid = cv.VideoCapture("detectbuoy.avi")
    print("Frame Reading started..")
    while (vid.isOpened()):
        ret,frame = vid.read()
        if frame is not None:
            # frame_g=frame[:,:,2]
            frame_orig = frame
            if ret == True:            
                # K = 9
                # frame = cv.GaussianBlur(frame, (3,3), 5)
                height, width, channels = frame.shape[0], frame.shape[1], frame.shape[2]
                frame = np.reshape(frame, (height * width, channels))
                prob_of_single_gote = np.zeros((height * width, K))
                gote_likelihood = np.zeros((height * width, K))

                for k in range(0, K):
                    prob_of_single_gote[:, k] = train_pi_k[k] * mvn.pdf(frame, trained_mean[k], trained_covar[k])
                    gote_likelihood = prob_of_single_gote.sum(1)

                gote_prob = np.reshape(gote_likelihood, (height, width))
                # print("!")
                # print(np.min(gote_prob))
                # # For possible gote area, make image dark
                # print(np.max(gote_prob))
                gote_prob[np.where(gote_prob == np.max(gote_prob))] = 255
                mask_image =np.zeros((height, width, channels), np.uint8)
                mask_image[:,:,0] = gote_prob
                mask_image[:,:,1] = gote_prob
                mask_image[:,:,2] = gote_prob
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
                print(radius, x, y)
                if radius > 2.6 and ((x > 120 and y > 320) or (x > 350 and y > 200)):
                    cv.circle(frame_orig, (int(x), int(y) - 10), int(radius + 10), (0,165,255), 4)
                cv.imshow("Final", frame_orig)
                k = cv.waitKey(15) & 0xff
                if k == 27:
                    break

        else:
            break
    vid.release()


# greenboi_r = mvn.pdf(list(range(0,256)), trained_mean[0][0], trained_covar[0][0, 0])
# greenboi_g = mvn.pdf(list(range(0,256)), trained_mean[0][1], trained_covar[0][1, 1])
# greenboi_b = mvn.pdf(list(range(0,256)), trained_mean[0][2], trained_covar[0][2, 2])

# plt.plot(greenboi_r, "r", greenboi_g, "g", greenboi_b, "b")
# plt.title('Gaussian Curve for only first mean value')
# plt.xlabel('x (0-256)')
# plt.ylabel('Probabilites')
# plt.show()

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

gote_dekho(trained_mean, trained_covar, train_pi_k, K)
            # for c in contours:
#                print("area: "+str(cv.contourArea(c)))
#                 if cv.contourArea(c) > 40:
#                     print("inside1")
#                     (x,y),r = cv.minEnclosingCircle(c)
#                     center = (int(x),int(y))
#                     r = int(r)
#                     print(r)
#                     if r > 10 and r < 15.5 and y < 440:
#                         print("inside")
#                         cv.circle(frame,center,r,(0,255,0),2)
# #            cv.line(frame, (0,440), (600, 440), (255,0,0), 2)
            # cv.imshow("threshold", frame)

            # coordinates = np.indices((frame_dummy.shape[0], frame_dummy.shape[1]))
            # coordinates = coordinates.reshape(2, -1)
            # x,y=coordinates[0],coordinates[1]
            # pixel_val=frame_dummy[x,y]
            # print("!")
            # print(coordinates.shape)
            # print(pixel_val.shape)
    
#             indices1=np.where((orangeboi_r[pixel_val]<0.5) & (orangeboi_g[pixel_val]>0.021) & (orangeboi_b[pixel_val]<0.5) )
#             indices2=np.where((orangeboi_r[pixel_val]>0.5) & (orangeboi_g[pixel_val]<0.021) & (orangeboi_b[pixel_val]>0.5) )
#             x1,y1=x[indices1[0]],y[indices1[0]]
#             x2,y2=x[indices2[0]],y[indices2[0]]
#             frame_updated[x1,y1]=255
#             frame_updated[x2,y2]=0

#             # kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#             # np.array([[0, 0, 1, 0, 0],
#             #        [1, 1, 1, 1, 1],
#             #        [1, 1, 1, 1, 1],
#             #        [1, 1, 1, 1, 1],
#             #        [0, 0, 1, 0, 0]], dtype=np.uint8)
#             blur = cv.GaussianBlur(frame_updated,(15,15), 0) 
#             ret_thresh, thresholded = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
#             cv.imshow("r1",thresholded)
#             edges = cv.Canny(thresholded, 50, 255)
#             # dilated = cv.dilate(thresholded, kernel_ellipse, iterations = 1)
# #           # dilated_updated = cv.cvtColor(dilated, cv.COLOR_BGR2GRAY)
#             contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#             # cont_img = cv.drawContours(frame, contours, -1, (0,0,255), 5)
#             # cv.imshow("threshold", cont_img)
#             # Draw circle to fit the contours enclosing specified area
#             for c in contours:
# #                print("area: "+str(cv.contourArea(c)))
#                 if cv.contourArea(c) > 40:
#                     print("inside1")
#                     (x,y),r = cv.minEnclosingCircle(c)
#                     center = (int(x),int(y))
#                     r = int(r)
#                     print(r)
#                     if r > 8 and r < 35 and y<350 and y>150 :
#                         print("inside")
#                         cv.circle(frame,center,r,(0,255,0),2)
# #            cv.line(frame, (0,440), (600, 440), (255,0,0), 2)
#             cv.imshow("threshold", frame)
# #            # cv.waitKey(0)

# images = []
# while (vid.isOpened()):
#     success, frame = vid.read()
#     if success == False:
#         break    

#     test_image = frame
#     K = 6
#     nx = test_image.shape[0]
#     ny = test_image.shape[1]
#     img = test_image
#     ch = img.shape[2]
#     img = np.reshape(img, (nx*ny,ch))
    
#     weights = train_pi_k # np.load('weights_o.npy')
#     # parameters = parameters # np.load('parameters_o.npy')
#     prob = np.zeros((nx*ny,K))
#     likelihood = np.zeros((nx*ny,K))
    
#     for cluster in range(K):
#        prob[:,cluster] = weights[cluster]*mvn.pdf(img, trained_mean[cluster], trained_covar[cluster])
       
#        likelihood = prob.sum(1)
       
    
#     probabilities = np.reshape(likelihood,(nx,ny))
    
#     probabilities[probabilities>np.max(probabilities)/3.0] = 255
    
    
    
#     output = np.zeros_like(frame)
#     output[:,:,0] = probabilities
#     output[:,:,1] = probabilities
#     output[:,:,2] = probabilities
#     blur = cv.GaussianBlur(output,(3,3),5)
    
#     # edged = cv.Canny(blur,50,255 )
    
#     # cnts,h = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     # (cnts_sorted, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")
#     cv.imshow("out", output)
#     # hull = cv.convexHull(cnts_sorted[0])
#     # (x,y),radius = cv.minEnclosingCircle(hull)
    
#     # if radius > 7:
#     #     cv.circle(test_image,(int(x),int(y)-1),int(radius+1),(0,165,255),4)

#     #     cv.imshow("Final output",test_image)
#     #     images.append(test_image)
#     # else:
#     #     cv.imshow("Final output",test_image)
#     #     images.append(test_image)
    
#     # cv2.waitKey(5)

        

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


cv.destroyAllWindows()