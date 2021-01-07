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

## 2. Acknowledgments

I would like to thank the organizers for hosting the wonderful competition, the participants who shared a lot of knowledge and fought together, and my best partner, Tanaka Shuntaro.
