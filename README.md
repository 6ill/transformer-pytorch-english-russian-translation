# Transformer PyTorch English-Russian Translation

This repository contains a PyTorch implementation of an English to Russian translation model using transformers. The implementation is based on the [pytorch-transformer](https://github.com/hkproj/pytorch-transformer/) repository. In addition to the original implementation, this project includes label smoothing and a learning rate scheduler in the training process.


## Installation
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Setup 
Create folders named "runs", "vocabs", and "weights".

## Usage
To train the model, use the following command:
```bash
python train.py
```

## Configuration
The training configuration can be adjusted in the `config.py` file.

## Acknowledgements
This project is based on the [pytorch-transformer](https://github.com/hkproj/pytorch-transformer/) repository.

