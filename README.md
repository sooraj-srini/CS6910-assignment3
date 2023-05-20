# Introduction

This is an implementation of the Encoder-Decoder architecture used to solve a transliteration problem: Converting English words to Hindi words. 
For example, the word 'ghar' would be transliterated to 'घर'.

There are two different implementations to solve the same problem:
- one without using attention and instead using a RNN based architecture
- one using attention for the same RNN based architecture

# Running the code

The barebones code is in `train.py`. Inside, you will find comments on the architecture and the hyperparameters used.
The code can be run using the following command:

```
python train.py
```
The appropriate arguments can be added using the `--help` argument.

The actual results of running the code are in `a3.ipynb`. This is where you would find the sweeps.

# Results

The predictions of both the implementations are in the `predictions` folder. There, you would find the predictions with and without attention. 

