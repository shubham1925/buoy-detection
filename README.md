### ENPM673 Project 3 -
Buoy Detection | GMM - EM

## Authors
Prasheel Renkuntla- 116925570
Raj Prakash Shinde- 116852104
Shubham Sonawane- 116808996
 
## Description
This project deals with the detection of underwater buoys of different colors using EM algorithm

## Dependencies
* Ubuntu 16
* Python 3.7
* OpenCV 4.2
* Numpy
* matplotlib
* sys
* math


## Run
To run the code for detecting all buoys using 3D gaussian
```
python3.7 model_all_colors_3D.py
```
Enter the choice of code (histogram, training or detection of buoys) when prompted

To run the code for 1D detection of yellow buoy
```
python3.7 yellow_detection.py
```
To run the code for 1D detection of green buoy
```
python3.7 green_detection.py
```
To run the code for 1D detection of orange buoy
```
python3.7 orange_detection.py
```

 
## Reference
* https://cmsc426.github.io/colorseg/#colorclassification
* https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
* https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f

