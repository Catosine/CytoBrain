# CytoBrain

This is a **personal** repository for [Algonauts2023 competition](http://algonauts.csail.mit.edu).  
Developed by [@Cytosine](https://github.com/Catosine)

## Quick Start
```Bash
    # To start a training
    python task_train.py --data ./data --subject subj01
```

## ToDos
### 0. Preparation
- [x] Dataset
- [x] Training Script
- [ ] Feature Extraction Script
- [x] Inference Script

### 1. End2End Model 
- ResNet50 + 3 of FCN

### 2. CNN Extractor & Regression
- Use ResNet50 as Extractor
- Rigid/Lasso Regression
- Elastic Net (L1+L2)

## Results
| Subject | Model | Epoch | Batch | Train Loss | Train Median Pearson Coef. | Dev Loss | Dev Median Pearson Coef. | Test Median Pearson Coef. | Note |
| - | - | - | - | - | - | - | - | - | - |
| Subj01  | End2end ResNet50 | NA | NA | NA | NA | NA | NA | NA | NA |