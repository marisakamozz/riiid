# Riiid! Answer Correctness Prediction - SAINT+ Solution

This repository is our solution of the "Riiid! Answer Correctness Prediction" competition at Kaggle.

You can use [this notebook](https://www.kaggle.com/marisakamozz/riiid-saint-solution) to reproduce our submission.

## 1. Instructions

### 1-1. Create Virtual Environment

Execute the following command to create the virtual environment.

```
conda env create --file environment.yaml
```

### 1-2. Preprocess

Execute the following command to download the data and perform preprocessing.

```
./makedir.sh
cd src
python makecv.py
python saintdata.py
```

### 1-3. Train

You can train your own model by running this command.
Please note that this solution has a problem with early stopping not working well.
You have to stop training at the best point for yourself.

```
python sainttrain.py
```

### 1-4. Test

If you train your model, a checkpoint file is created in the `/models/` directory.
Copy the checkpoint file to the `/src/` directory and name it `saint.ckpt`.

Then, you can test your model by running this command.

```
python sainttest.py
```

## 2. Model Summary

* [SAINT+](https://arxiv.org/abs/2010.12042) based model
* used train.csv only.
* used questions only.

We didn't use lectures, because lectures didn't improve validation score.

### 2-1. Input

Encoder input:

* positional embedding
* question_id embedding
* lag embedding (= timestamp - last timestamp)
* user_count: number of questions the user has solved

lag feature is categorized by 100 quantiles.
user_count is categorized by [proficiency test](https://www.kaggle.com/shuntarotanaka/riiid-the-first-30-questions-are-proficiency-test).

Decoder input:

* positional embedding
* answered_correctly
* prior_question_elapsed_time
* prior_question_had_explanation

prior_question_elapsed_time and prior_question_had_explanation are shifted -1.
prior_question_elapsed_time is categorized by 100 quantiles.

### 2-2. Model Architecture

Embedding Layer + Tranformer + Feed Forward Layer (with residual connection)

Weights of embedding layers are sampled from a normal distribution with a standard deviation of 0.01.

Transformer architecture:

* sequence length: 150
* number of dimension: 128
* number of encoder layers: 3
* number of decoder layers: 3 (self-attention) + 3 (encoder-attention)
* number of heads: 8

### 2-3. Training

* batch size: 256
* optimizer: Adam
* training time is almost 20 hours on 1 K80 GPU machine.
* early stopping by 1% unseen user's auc score. (but it didn't work well.)

## 3. Acknowledgments

I would like to thank the organizers for hosting the wonderful competition, the participants who shared a lot of knowledge and fought together, and my best partner, Tanaka Shuntaro.
