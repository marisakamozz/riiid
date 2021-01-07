#!/bin/bash
mkdir input
mkdir input/riiid-test-answer-prediction
mkdir data
mkdir data/cvfiles
mkdir data/saint
mkdir logs
mkdir models
kaggle competitions download -c riiid-test-answer-prediction
unzip -d input/riiid-test-answer-prediction riiid-test-answer-prediction.zip