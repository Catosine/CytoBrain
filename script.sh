#!/bin/bash

python feature_extract.py --subject subj01 --pretrained_weight ./backbone.nosync/resnet50-imagenet1k-v2.pth --layers layer3 avgpool