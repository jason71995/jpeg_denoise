# Remove JPEG noise via CNN on Keras and Tensorflow 2.0

## Introduction

Simple way to remove JPEG noise via CNN. 

#### Results on JPEG Quality=5 (1-100)
| JPEG | Removed | Real |
| :---: | :---: | :---: |
| ![alt text](https://i.imgur.com/rBzIfat.jpg "JPEG") |![alt text](https://i.imgur.com/ktUIaUF.png "Removed") |![alt text](https://i.imgur.com/jeJNxWV.png "Real") |
| ![alt text](https://i.imgur.com/3uNI8s1.jpg "JPEG") |![alt text](https://i.imgur.com/Ikej6IS.png "Removed") |![alt text](https://i.imgur.com/8WDixsQ.png "Real") |

## Data

- Download training data from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Get trained model from [Google Drive](https://drive.google.com/open?id=1-SK5iPDlq9_mYaG2KebWsZwKggiAQUNl)

## Environment
```
python==3.6
tensorflow==2.0
```

## How to use

#### Train
```
python train.py
```
#### Predict
```
python predict.py --image test.jpg --model weights.h5
```