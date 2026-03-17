# Spiking Traffic Sign Recognition

This repository contains convolutional and spiking convolutional models for traffic sign recognition on the GTSRB dataset. It includes training, checkpoint evaluation, and inference energy estimation for CNN, SCNN, and SCNN-PD variants.

## Features

- CNN baseline for GTSRB classification
- Spiking CNN (`SCNN`) with configurable timesteps
- Spiking CNN with population decoding (`SCNN-PD`)
- Checkpoint evaluation on the official GTSRB test split
- Energy estimation for trained checkpoints

## Repository Structure

```text
.
├── checkpoints/
│   ├── CNN/
│   ├── SCNN/
│   └── SCNN-PD/
├── splits/
│   ├── Train.csv
│   ├── Test.csv
│   ├── train_split.csv
│   └── val_split.csv
├── data.py
├── energy_estimation.py
├── eval.py
├── model.py
├── train.py
└── utils.py
```

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB):

- Dataset homepage: https://benchmark.ini.rub.de/gtsrb_news.html
- Kaggle mirror used in the code layout:
  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The scripts expect the dataset root at:

```text
../../Data/gtsrb-german-traffic-sign
```

Inside that directory, the expected files are:

```text
gtsrb-german-traffic-sign/
├── Train/
└── Test/
```

The repository uses the split metadata stored in:

```text
splits/
├── Test.csv
├── train_split.csv
└── val_split.csv
```

`train.py` reads `splits/train_split.csv` and `splits/val_split.csv`. `eval.py` and `energy_estimation.py` read `splits/Test.csv`. The CSV `Path` column is resolved relative to the dataset root above.

The dataset loader in [data.py](data.py) reads image paths and crops each image to the ROI specified in the CSV metadata.

## Training

The training entry point is [train.py](train.py).

Run:

```bash
python3 train.py
```

The script is currently configured through constants inside `main()`, including:

- seeds
- timesteps
- `run_snn`
- `run_cnn`
- `use_population_code`
- batch size
- patience

Generated checkpoints follow these naming schemes:

- CNN: `cnn_seed=<seed>.pth`
- SCNN: `scnn_t=<timesteps>_seed=<seed>.pth`
- SCNN-PD: `scnn_pd_t=<timesteps>_seed=<seed>.pth`

## Evaluation

Use [eval.py](eval.py) to compute the test accuracy of a checkpoint.

Examples:

```bash
python3 eval.py --checkpoint checkpoints/CNN/cnn_seed=0.pth
python3 eval.py --checkpoint checkpoints/SCNN/scnn_t=1_seed=0.pth
python3 eval.py --checkpoint checkpoints/SCNN-PD/scnn_pd_t=1_seed=0.pth
```

Optional overrides:

```bash
python3 eval.py \
  --checkpoint checkpoints/SCNN/scnn_t=10_seed=0.pth \
  --model-type scnn \
  --num-steps 10 \
  --device cuda
```

## Energy Estimation

Use [energy_estimation.py](energy_estimation.py) to estimate inference energy from a checkpoint.

Examples:

```bash
python3 energy_estimation.py --checkpoint checkpoints/CNN/cnn_seed=0.pth
python3 energy_estimation.py --checkpoint checkpoints/SCNN/scnn_t=1_seed=0.pth
python3 energy_estimation.py --checkpoint checkpoints/SCNN-PD/scnn_pd_t=1_seed=0.pth
```

Example with explicit processing unit:

```bash
python3 energy_estimation.py \
  --checkpoint checkpoints/SCNN/scnn_t=10_seed=0.pth \
  --processing-unit loihi
```

Supported processing units are defined in [energy_estimation.py](energy_estimation.py).

## Notes

- `eval.py` and `energy_estimation.py` automatically infer the model type from the checkpoint name.
- Legacy CNN checkpoints with old layer names are handled automatically.
- Training currently uses only train and validation splits. Test evaluation is intentionally separated into `eval.py`.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
