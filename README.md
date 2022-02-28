# Automated Classification of General Movements in Infants Using Two-stream Spatiotemporal Fusion Network
---

## Overview
 <img src=img/overview.png><br>
 **Fig. 1.** Overview of the proposed GMs classification method<br>

## folder structure
```
.
├── config         
├── data
|   ├── fold
|   |   ├── test
|   |   |   ├── fold1.csv
|   |   |   └── ...
|   |   └── train        
|   ├── FMs
|   |   ├──sub1
|   |   |   ├── video.mp4                
|   |   |   ├── org
|   |   |   |   ├── 000000.png
|   |   |   |   └── ...
|   |   |   ├── removed
|   |   |   |   ├── 000000.png
|   |   |   |   └── ...
|   |   |   ├── 224x224
|   |   |   |   ├── 000000.png
|   |   |   |   └── ...
|   |   |   └── 224x224_flow
|   |   |       ├── 000005.png
|   |   |       └── ...
|   |   └ ...
|   ├── PR
|   └── WMs
├── model
|   ├── openpose.pth
|   └── u2net.pth   
├── result      
└── src
```
## Getting Started
### Install Requriements
Create a python 3.8 environment, e.g.:<br>
```
python3.8 -m venv venv
. venv/bin/activate
```
Install pytorch by following [the official tutorial](https://pytorch.org/get-started/locally/).<br>
Install other requirements with pip.<br>
```
pip install -r requirements.txt
```
### Download the models
- [U2Net](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view)
- [OpenPose](https://drive.google.com/file/d/1EULkcH_hhSU28qVc1jSJpCh2hGOrzpjK/view)
### Data preparation (including preprocessing networks)
```
# convert video to images
python src/data/preprocess/mov2img.py
# body extractor (remove background)
python src/data/preprocess/extractor.py
# pose adjuster
python src/data/preprocess/extractor.py
# optical flow
python src/data/preprocess/optical_flow.py
# split data for cross validation
python src/data/preprocess/split.py
```
### Train and validate the model
```
# train and eval model
python src/train.py
```
