# CytoBrain

This is a repository for [Algonauts2023 competition](http://algonauts.csail.mit.edu).  
Developed by [@Cytosine](https://github.com/Catosine)

## Dataset
You may access the dataset from [here](https://naturalscenesdataset.org)

## Quick Start
```Bash
    # To start a training
    python task_train.py --data ./data --subject subj01
```

## ToDos
### 0. Preparation
- [x] [Dataset](./src/dataset.py)
- [x] [Training Script](./task_train_torch.py)
- [x] [Feature Extraction Script](./feature_extract.py)
- [x] [Inference Script](./task_inference_torch.py)

### 1. End2End Model 
- Pretrained ResNet50 + 1 FCN

### 2. CNN Extractor & Regression
- Use ResNet50 as Extractor
- Rigid/Lasso Regression
- Elastic Net (L1+L2)

## Experiment Results
### Preatrained Feature Modelling
| Subject | Feature           | Model   | Dev Median Pearons's R (Left) | Dev Median Pearons's R (Right) | Figures                                           | Note     |  
| ------  | ----------------- | ------- | ----------------------------- | ------------------------------ | ------------------------------------------------- | -------- |  
| Subj01  | resnet50-avgpool  | LinearR | 0.243                         | 0.245                          | [fig](./img/linear_regression_baseline/histogram_pearson_20230308144134.png) | baseline/random crop 256 |
| Subj01  | resnet50-avgpool  | RidgeR  | 0.250                         | 0.249                          | [fig](./img/ridge_baseline/histogram_pearson_20230308144134.png) | baseline/random crop 256 |
| Subj01  | resnet50-avgpool  | RidgeR  | 0.358                         | 0.361                          | [fig](./img/ridge_baseline/histogram_pearson_20230308144134.png) | alpha=1e5/random crop 256 |
| Subj01  | resnet50-avgpool | RidgeR  | 0.375                         | 0.374                          | [fig](./img/ridge_resnet50_full_img_alpha_1e4/histogram_pearson_20230309171031.png) | alpha=1e4/full img |
| Subj01  | resnet50-avgpool | RidgeR  | 0.376                         | 0.376                          | [fig](./img/ridge_resnet50_full_img_alpha_2e4/histogram_pearson_20230309171821.png) | alpha=2e4/full img |
| Subj01  | resnet152-avgpool | RidgeR  | 0.355                         | 0.357                          | [fig](./img/ridge_resnet152_alpha_1e5/histogram_pearson_20230309141541.png) | alpha=1e5/random crop 256 |


### Best Figure
![](./img/ridge_resnet50_full_img_alpha_2e4/histogram_pearson_20230309171821.png)
![](./img/ridge_resnet50_full_img_alpha_2e4/box_pearson_20230309171829.png)