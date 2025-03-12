# Knowledge Distillation for Semantic Segmentation
A Label Space Unification Approach

This is a complimentary repository to our paper: [Knowledge Distillation for Semantic Segmentation A Label Space Unification Approach](https://arxiv.org/abs/2502.19177).

![example_output](figures/example.gif)
![filtered_trajectory](figures/filter_points.gif "Removing keypoints on dynamic objects leads to a better trajectory estimation.")


## Installation
### Requirements
- python >= 3.10
- pytorch >= 2.3
- accelerate >= 1.2
- transformers >= 4.45.0 

```
$ pip install -r requirements.txt
```


Huggingface accelerate is a wrapper used mainly for multi-gpu and half-precision training.
You can adjust the settings prior to training with (recommended for faster training) or just skip it:
```
$ accelerate config
```


## Download Weights

| Model | Taxonomy   | IoU  |
|-------|------------|------|
| [M2FB](https://huggingface.co/antopost/data-priors/resolve/main/23_best.pth?download=true)  | GOOSE      | 64.4 |
| [M2FL](https://huggingface.co/antopost/data-priors/resolve/main/64_best.pth?download=true)  | GOOSE      | 67.9 |
| [M2FB](https://huggingface.co/antopost/data-priors/resolve/main/24_best.pth?download=true)  | Cityscapes | 75.5 |
| [M2FL](https://huggingface.co/antopost/data-priors/resolve/main/12_best.pth?download=true)  | Cityscapes | 78.3 |
| [M2FL](https://huggingface.co/antopost/data-priors/resolve/main/40.pth?download=true)  | Mapillary  | 52.7 |


## Training steps


### 1. Pretrain on source taxonomy

Train a standard Mask2Former on a source dataset.
```
$ accelerate launch train.py --config config.yaml --exper_name <experiment_name>
```
Track progress in Tensorboard:
```
$ tensorboard --logdir experiments/<experiment_name>/logs
```

### 2. Ontology mapping

Before you can generate pseudo-labels with priors, you need to define an ontology mapping between target and extra datasets.
You can find examples in `datasets/<dataset_name>/lists/master_labels_<source_dataset>.csv`

### 3. Generate pseudo-labels
Generate pseudo-labels using dataset priors.
Use config_inference.yaml to set your labeling parameters.
```
$ accelerate launch inference.py
```

### 4. Train on the compound dataset
Now that you have generated labels with priors
```
$ accelerate launch inference.py
```

## Inference

Inference is done using the same script as pseudo-label generation.
Adjust the config_inference.yaml to save or display images with or w/o priors.
```
$ accelerate launch inference.py
```


## TODOs

- [x] Publish weights
- [ ] Simpler config files (too many legacy parameters)
- [ ] Code cleanup, remove legacy code
