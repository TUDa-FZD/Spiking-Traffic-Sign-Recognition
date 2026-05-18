# Spiking Traffic Sign Recognition

This repository contains convolutional and directly trained spiking convolutional neural networks for traffic sign recognition on the German Traffic Sign Recognition Benchmark (GTSRB). The project includes training, evaluation, and inference energy estimation for CNN, SCNN, and SCNN with Population Decoding (SCNN-PD) variants.

The implementation accompanies the following paper:

> **Energy-Efficient Traffic Sign Recognition Using Directly Trained Spiking Neural Networks and Population Decoding**  
> Jonas V. Schulte, Steven Peters  
> *Frontiers in Neuroscience, 2026*

📄 Paper: https://doi.org/10.3389/fnins.2026.1771436

---

## Features

- CNN baseline for GTSRB classification
- Directly trained Spiking CNN (`SCNN`)
- Spiking CNN with Population Decoding (`SCNN-PD`)
- Configurable number of simulation timesteps
- Checkpoint evaluation on the official GTSRB test split
- Inference energy estimation for CNNs and SNNs

---

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

---

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

- Dataset homepage:  
  https://benchmark.ini.rub.de/gtsrb_news.html

- Kaggle mirror used for the repository structure:  
  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

The scripts expect the dataset root at:

```text
../../Data/gtsrb-german-traffic-sign
```

Expected structure:

```text
gtsrb-german-traffic-sign/
├── Train/
└── Test/
```

The repository uses split metadata stored in:

```text
splits/
├── Test.csv
├── train_split.csv
└── val_split.csv
```

- `train.py` uses:
  - `splits/train_split.csv`
  - `splits/val_split.csv`

- `eval.py` and `energy_estimation.py` use:
  - `splits/Test.csv`

The dataset loader in [data.py](data.py) reads image paths and crops each image to the region of interest (ROI) specified in the CSV metadata.

---

## Training

The training entry point is:

```bash
python3 train.py
```

The script is currently configured through constants inside `main()`, including:

- random seeds
- number of timesteps
- `run_snn`
- `run_cnn`
- `use_population_code`
- batch size
- early stopping patience

Generated checkpoints follow these naming schemes:

- CNN:
  ```text
  cnn_seed=<seed>.pth
  ```

- SCNN:
  ```text
  scnn_t=<timesteps>_seed=<seed>.pth
  ```

- SCNN-PD:
  ```text
  scnn_pd_t=<timesteps>_seed=<seed>.pth
  ```

---

## Evaluation

Use `eval.py` to compute test accuracy.

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

---

## Energy Estimation

Use `energy_estimation.py` to estimate inference energy consumption.

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

---

## Notes

- `eval.py` and `energy_estimation.py` automatically infer the model type from the checkpoint filename.
- Legacy CNN checkpoints with older layer names are supported automatically.
- Training currently uses only train and validation splits.
- Test evaluation is intentionally separated into `eval.py`.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{schulte2026energy,
  title={Energy-Efficient Traffic Sign Recognition Using Directly Trained Spiking Neural Networks and Population Decoding},
  author={Schulte, Jonas V. and Peters, Steven},
  journal={Frontiers in Neuroscience},
  volume={20},
  year={2026},
  doi={10.3389/fnins.2026.1771436},
  url={https://doi.org/10.3389/fnins.2026.1771436}
}
```

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
