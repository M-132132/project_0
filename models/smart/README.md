## This part is to add [SMART](https://github.com/rainmaker22/SMART) model in the TrajAttr pipeline

0. Create environment & preperation

```bash
conda create -n TrajAttr_smart python=3.9
conda activate TrajAttr_smart
```

1. Install ScenarioNet

2. Install TrajAttr as in the main README

3. Install SMART requirement from [SMART](https://github.com/rainmaker22/SMART) or try install from here:
```bash
pip install -r requirements.txt
```

You can verify the installation of TrajAttr via running the training script:

```bash
python train_pl.py method=SMART
```

The model will be trained on several sample data.

## Training

### 1. Data Preparation

TrajAttr takes data from [ScenarioNet](https://github.com/metadriverse/scenarionet) as input. Process the data with
ScenarioNet in advance.

### 2. Configuration

SMART model has its own configuration file in `TrajAttr/config/method/SMART.yaml`.


### 2. Train
```python train.py```

The default training setups are the same as SMART-7M and when train a SMART model in TrajAttr, this should be uncommentted in train.py:

```bash
# accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
```
The latest TrajAttr-SMART version ckpt has minADE=0.827，minFDE=3.300 for reference.
