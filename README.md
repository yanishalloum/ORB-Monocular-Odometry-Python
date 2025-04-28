# ORB MONOCULAR ODOMETRY PYTHON

**Author:** [Yanis Halloum](https://github.com/yanishalloum)


# 1. Prerequisites
Tested in **Ubuntu 20.04**

## C++13 Compiler

## Pangolin
The library [Pangolin](https://github.com/stevenlovegrove/Pangolin) is used for visualization and user interface. 
I use an older version for compatibility reasons.
```
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin 
git checkout 86eb4975fc4fc8b5d92148c2e370045ae9bf9f5d
mkdir build 
cd build 
cmake .. -D CMAKE_BUILD_TYPE=Release 
make -j 3 
sudo make install
cd ..
python setup.py install
```
Pangolin uses an old FFmpeg API that causes the build to fail. Deactivate the installation of FFmpeg.

## OpenCV
[OpenCV](http://opencv.org) is used. Dowload and install instructions can be found at: http://opencv.org. **Tested with OpenCV 3.0+**.

## Eigen3
Requirements for g2o. Download and install instructions can be found at: http://eigen.tuxfamily.org. 

## g2o 
```
sudo apt update
sudo apt install python3-venv
python3 -m venv env
source env/bin/activate
pip install -U g2o-python
```

# Dataset

Dowmload the dataset [EuRoc](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and place it in your repo.



