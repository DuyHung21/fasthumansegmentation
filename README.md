# Human Segmentation

This is a graduation project, which aims to build understanding about deep learning and convolutional neural network. The project contains
  - Two pretrained models for performing human segmentation on CPU devices.
  - An background replacement application using two pretrained models.

### Tech

This project uses a number of open source projects to work properly:

* [Python](https://www.python.org/)
* [Pytorch](https://pytorch.org/)
* [Opencv](https://opencv.org/)
* [Numpy](https://www.numpy.org/)

### Installation

This project requires [Python](https://www.python.org/) 3.5+ to run.
Install python packages
```sh
$ cd code
$ pip install -r requirements.txt
```
Grant execution permission for running files
```s
$ cd code
$ chmod +x run_model_custom.sh
$ chmod +x run_model_esp.sh
```

### Run the code
Run Network 1 for human silhouette extraction
```sh
./run_model_esp.sh
```
Run Network 1 for background replacement
```sh
# Put your background images into the backgrounds folder
./run_model_esp.sh backgrounds/wonders.png 
```
Run Network 2 for human silhouette extraction
```sh
./run_model_custom.sh
```
Run Network 2 for background replacement
```sh
# Put your background images into the backgrounds folder
./run_model_custom.sh backgrounds/wonders.png 
```

