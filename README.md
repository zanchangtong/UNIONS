# UNIONS
This repository is the official implementation of [Unlikelihood Tuning on Negative Samples Amazingly Improves Zero-Shot Translation](https://arxiv.org/abs/2309.16599). 
## Requirements

To install requirements:

```setup
conda create -n unions python=3.8
conda activate unions
bash setup.sh 
```

## Training

To train the models as described in our paper, please follow the commands below, using the opus100(v1.0) experiment as an example:

Stage1: 
```train
cd exps/opus100
# Requires setting up the Path first
bash opus100_baseline.sh 
```
Stage2: 
```train
# Requires setting up the Path first
bash opus100_unions.sh 
```

## Evaluation
To evaluate the model using the opus100(v1.0) test set, run the following commands:

```eval
# Requires setting up the Path first
bash evaluate.sh 
```

## Reference

```
@article{zan2023unions,
  title={Unlikelihood Tuning on Negative Samples Amazingly Improves Zero-Shot Translation},
  author={Zan, Changtong and Ding, Liang and Shen, Li and Lei, Yibin and Zhan, Yibing and Liu, Weifeng and Tao, Dacheng},
  journal={arXiv preprint},
  year={2023}
}
```

