# Automated Classification of General Movements in Infants Using Two-stream Spatiotemporal Fusion Network
---
by Yuki Hashimoto, Akira Furui<br>
<br>
This is the official repo for our paper general movements assessment using two-stream spatiotemporal fusion network published in MICCAI 2022.<br>

## Overview
<img src=img/overview.png><br>
**Fig. 1.** Overview of the proposed GMs classification method. The abbreviations GAP and CONV $1 \times 1$ denote global average pooling and pointwise convolution, respectively.<br>
<br>
**Abstract**: The assessment of general movements (GMs) in infants is a useful tool in the early diagnosis of neurodevelopmental disorders. However, its evaluation in clinical practice relies on visual inspection by experts, and an automated solution is eagerly awaited. Recently, video-based GMs classification has attracted attention, but this approach would be strongly affected by irrelevant information, such as background clutter in the video. Furthermore, for reliability, it is necessary to properly extract the spatiotemporal features of infants during GMs. In this study, we propose an automated GMs classification method, which consists of preprocessing networks that remove unnecessary background information from GMs videos and adjust the infant's body position, and a subsequent motion classification network based on a two-stream structure. The proposed method can efficiently extract the essential spatiotemporal features for GMs classification while preventing overfitting to irrelevant information for different recording environments. We validated the proposed method using videos obtained from 100 infants. The experimental results demonstrate that the proposed method outperforms several baseline models and the existing methods.<br>
<br>

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