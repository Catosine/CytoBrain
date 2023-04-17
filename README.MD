# CytoBrain

This is a repository for [Algonauts2023 competition](http://algonauts.csail.mit.edu).  
Developed by [@Cytosine](https://github.com/Catosine)

## Dataset
You may access the dataset from [here](https://naturalscenesdataset.org)

## Quick Start
### Feature Extraction
```Bash
# Extract train set features of subj01 via pretrained Resnet50
python feature_extract.py --data ~/algonauts2023/data --subject subj01 --train \ 
        --save_path ~/algonauts2023/data/subj01/training_split/test_features \
        --pretrained_weights ~/backbone/resnet50-imagenet1k-v2.pth \ 
        --layers layer3 avgpool

# Also extract test set features of subj01 via pretrained Resnet50
python feature_extract.py --data ~/algonauts2023/data --subject subj01 \ 
        --save_path ~/algonauts2023/data/subj01/test_split/test_features \
        --pretrained_weights ~/backbone/resnet50-imagenet1k-v2.pth \ 
        --layers layer3 avgpool

# Or, extract train set features of last 4 layer of image captioning model
python hf_feature_extract.py --data ~/algonauts2023/data --subject subj01 --train \
        --save_path ~/algonauts2023/data/subj01/training_split/training_features \
        --pretrained_weights nlpconnect/vit-gpt2-image-captioning \
        --feature_type decoder

# And same to test set
python hf_feature_extract.py --data ~/algonauts2023/data --subject subj01 \
        --save_path ~/algonauts2023/data/subj01/training_split/training_features \
        --pretrained_weights nlpconnect/vit-gpt2-image-captioning \
        --feature_type decoder
```

### Feature Decomposition Via PCA
TBU

### Modelling
Please see [example notebook](example.ipynb)

### Submission
Please see [submission notebook](submission.ipynb)

## Results
### Submission
| Method | Test Median Pearson's R | Note                  |  
| ------ | ----------------------- | --------------------- |  
| RidgeR | 38.406                  | res50(layer3+avgpool) |
| RidgeR | 49.793                  | [ViT-GPT2 Image Captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)(Last4Layer) |  

### Best Submission
![fig](./img/Submission_Report_Summary.svg)

### Experiments: Preatrained Feature Modelling
#### Resnet50
| Subject | Feature          | Model   | Dev Median Pearson's R (Left) | Dev Median Pearson's R (Right) | Note                     |  
| ------- | ---------------- | ------- | ----------------------------- | ------------------------------ | ------------------------ |  
| Subj01  | avgpool          | LinearR | 0.243                         | 0.245                          | baseline/random crop 256 |  
| Subj01  | avgpool          | RidgeR  | 0.376                         | 0.376                          | alpha=2e4                |  
| Subj01  | layer3           | RidgeR  | 0.391                         | 0.392                          | alpha=1e3/avgpool        |  
| Subj01  | layer2           | RidgeR  | 0.328                         | 0.316                          | alpha=1e2/avgpool        |  
| Subj01  | layer1           | RidgeR  | 0.288                         | 0.282                          | alpha=1e1/avgpool        |  
| Subj01  | layer3+avgpool   | RidgeR  | 0.398                         | 0.394                          | alpha=1e4                |  
| Subj01  | layer123+avgpool | RidgeR  | 0.391                         | 0.390                          | alpha=1e3                |  q

#### [ViT-GPT2 Image Captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
| Subject | Feature                       | Model   | Dev Median Pearson's R (Left) | Dev Median Pearson's R (Right) | Note      |  
| ------  | ----------------------------- | ------- | ----------------------------- | ------------------------------ | --------- |  
| Subj01  | encoder-pca-512               | RidgeR  | 0.394                         | 0.393                          | alpha=1   |
| Subj01  | encoder-pca-2048              | RidgeR  | 0.350                         | 0.345                          | alpha=1e4 |
| Subj01  | encoder-avg-768               | RidgeR  | 0.378                         | 0.379                          | alpha=1e4 |
| Subj01  | encoder-cls                   | RidgeR  | 0.370                         | 0.370                          | alpha=1e4 |
| Subj01  | encoder-last4-pca-512         | RidgeR  | 0.445                         | 0.440                          | alpha=2e3 |
| Subj01  | decoder-pca-512               | RidgeR  | 0.394                         | 0.386                          | alpha=5e3 |
| Subj01  | decoder-pca-2048              | RidgeR  | 0.351                         | 0.345                          | alpha=2e4 |
| Subj01  | decoder-avg-768               | RidgeR  | 0.378                         | 0.379                          | alpha=1e4 |
| Subj01  | decoder-cls                   | RidgeR  | 0.371                         | 0.370                          | alpha=1e4 |
| Subj01  | decoder-last4-pca-512         | RidgeR  | 0.446                         | 0.442                          | alpha=2e3 |
| Subj01  | decoder-last1234-pca-512      | RidgeR  | 0.454                         | 0.449                          | alpha=5e5 |
| Subj01  | encoder+decoder_pca_512       | RidgeR  | 0.405                         | 0.399                          | alpha=1e6 |
| Subj01  | encoder+decoder_last4_pca_512 | RidgeR  | 0.447                         | 0.443                          | alpha=1e4 |
