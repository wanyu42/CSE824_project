# CSE824_project

DATASET can be NTU_Fi-HAR/UT_HAR_data, MODEL can be MLP/LeNet/LSTM.
The dataset can be downloaded through (https://drive.google.com/drive/folders/1dAupSo_UoN5JKQ7_iV1r5DjPcBRz5ZIQ?usp=sharing)
## Natural Training
```
python run_dev.py --dataset DATASET --model MODEL 
```
## Adversarial Training
```
python run_adv.py --dataset DATASET --model MODEL 
```
## Attacking Evaluation
### On natural trained model
```
python attack.py --dataset DATASET --model MODEL 
```
### On robust trained model
```
python attack.py --dataset DATASET --model MODEL 
```
## Adversarial Detection
### On natural trained model
```
python detect.py --dataset DATASET --model MODEL
```
### On robust trained model
```
python detect_adv.py --dataset DATASET --model MODEL
```
## Temporal Shuffling
```
python shuffling.py --dataset DATASET --model MODEL
```
