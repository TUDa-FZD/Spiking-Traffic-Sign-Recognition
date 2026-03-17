import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import snntorch.functional as SF
import torch
from sklearn.model_selection import train_test_split
from snntorch import utils
from torch.utils.data import DataLoader


def measure_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> float:
    """Compute classification accuracy for a spiking model."""
    model.eval()
    device = torch.device(device)
    running_length = 0
    running_accuracy = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            spk_rec, _ = model(data)
            spike_count = spk_rec.sum(dim=0)
            predictions = spike_count.argmax(dim=1)

            running_length += len(targets)
            running_accuracy += int((predictions == targets).sum().item())

    return running_accuracy / max(1, running_length)


def predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device | str,
) -> int:
    """Run a single-image prediction with a spiking model."""
    model.eval()
    device = torch.device(device)

    with torch.no_grad():
        batch = image.unsqueeze(0).to(device)
        spk_rec, _ = model(batch)
        spike_count = spk_rec.sum(dim=0)
        prediction = spike_count.argmax(dim=1)
        return int(prediction.item())


def validate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device | str,
    population_code: bool = False,
    num_classes: int = 10,
    spiking: bool = True,
) -> Tuple[float, float]:
    """Evaluate a model and return average loss and accuracy."""
    model.eval()
    device = torch.device(device)
    running_loss = 0.0
    correct = 0.0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            if spiking:
                utils.reset(model)
                spk_rec, _ = model(data)
                loss = loss_fn(spk_rec, targets)
                if population_code:
                    correct += SF.accuracy_rate(
                        spk_rec,
                        targets,
                        population_code=True,
                        num_classes=num_classes,
                    ) * spk_rec.size(1)
                else:
                    correct += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            else:
                outputs = model(data)
                loss = loss_fn(outputs, targets)
                predictions = outputs.argmax(dim=1)
                correct += float((predictions == targets).sum().item())

            running_loss += float(loss.item())
            total += targets.size(0)

    average_loss = running_loss / max(1, len(test_loader))
    accuracy = correct / max(1, total)
    return average_loss, accuracy


def create_stratified_split(
    csv_path: str | Path,
    val_fraction: float = 0.1,
    split_seed: int = 12345,
    save: bool = True,
    train_out: str | Path = "splits/train_split.csv",
    val_out: str | Path = "splits/val_split.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible stratified train/validation split."""
    data_frame = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        data_frame,
        test_size=val_fraction,
        stratify=data_frame["ClassId"],
        random_state=split_seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Total samples: {len(data_frame)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    train_distribution = train_df["ClassId"].value_counts(normalize=True)
    val_distribution = val_df["ClassId"].value_counts(normalize=True)
    print("\nDistribution difference (should be very small):")
    print((train_distribution - val_distribution).abs().max())

    if save:
        train_df.to_csv(train_out, index=False)
        val_df.to_csv(val_out, index=False)
        print(f"\nSaved: {train_out}, {val_out}")

    return train_df, val_df


def summarize_cnn_ckpt_acc(
    ckpt_dir: str | Path = ".",
    pattern: str = "cnn_seed=*.pth",
    seeds: Optional[Sequence[int]] = None,
    ddof: int = 1,
    as_percent: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize CNN checkpoints from old and new naming schemes."""
    regex = re.compile(r"cnn_seed=(\d+)(?:_test_acc=(\d+))?\.pth$")
    seed_set = set(seeds) if seeds is not None else None
    rows = []

    for checkpoint_path in Path(ckpt_dir).glob(pattern):
        match = regex.match(checkpoint_path.name)
        if match is None:
            continue

        seed = int(match.group(1))
        acc_int = match.group(2)
        if seed_set is not None and seed not in seed_set:
            continue

        rows.append(
            {
                "seed": seed,
                "acc": None if acc_int is None else int(acc_int) / 100.0,
                "file": checkpoint_path.name,
            }
        )

    return _summarize_checkpoint_rows(rows, ddof=ddof, as_percent=as_percent)


def summarize_ckpt_acc_by_t(
    ckpt_dir: str | Path = ".",
    pattern: str = "scnn*_t=*_seed=*.pth",
    seeds: Optional[Sequence[int]] = None,
    mode: Optional[str] = None,
    ddof: int = 1,
    as_percent: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize SCNN checkpoints by timestep and mode.

    Supported formats:
    - ``scnn_t=10_seed=0.pth``
    - ``scnn_pd_t=10_seed=0.pth``
    - old formats with ``_test_acc=...``
    """
    regex = re.compile(
        r"scnn(_pd)?_t=(\d+)_seed=(\d+)(?:_test_acc=(\d+))?\.pth$"
    )
    seed_set = set(seeds) if seeds is not None else None
    rows = []

    for checkpoint_path in Path(ckpt_dir).glob(pattern):
        match = regex.match(checkpoint_path.name)
        if match is None:
            continue

        mode_tag, timestep, seed, acc_int = match.groups()
        seed = int(seed)
        if seed_set is not None and seed not in seed_set:
            continue

        resolved_mode = "pd" if mode_tag == "_pd" else "standard"
        if mode is not None and resolved_mode != mode:
            continue

        rows.append(
            {
                "t": int(timestep),
                "seed": seed,
                "mode": resolved_mode,
                "acc": None if acc_int is None else int(acc_int) / 100.0,
                "file": checkpoint_path.name,
            }
        )

    if not rows:
        summary_columns = ["t", "mode", "n", "acc_mean", "acc_std", "acc_min", "acc_max"]
        raw_columns = ["t", "seed", "mode", "acc", "file"]
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=raw_columns),
        )

    raw = pd.DataFrame(rows).sort_values(["t", "seed"]).reset_index(drop=True)
    raw_with_acc = raw.dropna(subset=["acc"])

    if raw_with_acc.empty:
        summary = (
            raw.groupby(["t", "mode"], as_index=False)
            .agg(n=("seed", "size"))
            .sort_values(["mode", "t"])
            .reset_index(drop=True)
        )
        for column in ["acc_mean", "acc_std", "acc_min", "acc_max"]:
            summary[column] = pd.NA
        return summary, raw

    summary = (
        raw_with_acc.groupby(["t", "mode"], as_index=False)
        .agg(
            n=("acc", "size"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", lambda values: values.std(ddof=ddof if len(values) > 1 else 0)),
            acc_min=("acc", "min"),
            acc_max=("acc", "max"),
        )
        .sort_values(["mode", "t"])
        .reset_index(drop=True)
    )

    if as_percent:
        raw = raw.copy()
        summary = summary.copy()
        for column in ["acc"]:
            raw[column] = raw[column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else pd.NA
            )
        for column in ["acc_mean", "acc_std", "acc_min", "acc_max"]:
            summary[column] = summary[column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else pd.NA
            )

    return summary, raw


def _summarize_checkpoint_rows(
    rows: list[dict],
    ddof: int,
    as_percent: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize checkpoint rows with optional accuracy metadata."""
    if not rows:
        summary_columns = ["n", "acc_mean", "acc_std", "acc_min", "acc_max"]
        raw_columns = ["seed", "acc", "file"]
        return (
            pd.DataFrame(columns=summary_columns),
            pd.DataFrame(columns=raw_columns),
        )

    raw = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    raw_with_acc = raw.dropna(subset=["acc"])

    if raw_with_acc.empty:
        summary = pd.DataFrame([{"n": len(raw), "acc_mean": pd.NA, "acc_std": pd.NA, "acc_min": pd.NA, "acc_max": pd.NA}])
        return summary, raw

    summary = pd.DataFrame(
        [
            {
                "n": len(raw_with_acc),
                "acc_mean": raw_with_acc["acc"].mean(),
                "acc_std": raw_with_acc["acc"].std(
                    ddof=ddof if len(raw_with_acc) > 1 else 0
                ),
                "acc_min": raw_with_acc["acc"].min(),
                "acc_max": raw_with_acc["acc"].max(),
            }
        ]
    )

    if as_percent:
        raw = raw.copy()
        summary = summary.copy()
        raw["acc"] = raw["acc"].map(
            lambda value: f"{value:.2f}%" if pd.notna(value) else pd.NA
        )
        for column in ["acc_mean", "acc_std", "acc_min", "acc_max"]:
            summary[column] = summary[column].map(
                lambda value: f"{value:.2f}%" if pd.notna(value) else pd.NA
            )

    return summary, raw
