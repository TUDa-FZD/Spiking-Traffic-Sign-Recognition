import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import snntorch.functional as SF
import torch
from snntorch import utils
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(_: int) -> None:
    """Assign a reproducible random seed to each dataloader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _resolve_population_code(
    model: torch.nn.Module,
    population_code: Optional[bool],
) -> bool:
    """Validate population coding against the model configuration."""
    model_population_code = bool(getattr(model, "population_code", False))
    if population_code is None:
        return model_population_code
    if population_code != model_population_code:
        raise ValueError(
            "population_code does not match the model configuration: "
            f"{population_code=} vs {model_population_code=}."
        )
    return population_code


def _compute_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    spiking: bool,
    population_code: bool,
    num_classes: int,
) -> float:
    """Return the number of correct predictions in one batch."""
    if spiking:
        if population_code:
            return float(
                SF.accuracy_rate(
                    outputs,
                    targets,
                    population_code=True,
                    num_classes=num_classes,
                )
                * outputs.size(1)
            )
        return float(SF.accuracy_rate(outputs, targets) * outputs.size(1))

    predictions = outputs.argmax(dim=1)
    return float((predictions == targets).sum().item())


def _forward_pass(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_fn,
    spiking: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return model outputs with loss."""
    if spiking:
        utils.reset(model)
        outputs, _ = model(data)
        loss = loss_fn(outputs, targets)
        return outputs, loss

    outputs = model(data)
    loss = loss_fn(outputs, targets)
    return outputs, loss


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    step_size: int,
    gamma: float,
    loss_fn,
    device: Union[str, torch.device],
    patience: int = 10,
    max_epochs: int = 200,
    spiking: bool = True,
    population_code: Optional[bool] = None,
    num_classes: int = 43,
    ckpt_path: str = "best_model.pth",
    monitor: str = "val_acc",
    verbose: bool = True,
) -> Dict[str, Union[float, int, List[float], str]]:
    """Train a model with early stopping based on validation performance."""
    device = torch.device(device)
    population_code = _resolve_population_code(model, population_code)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=gamma,
        patience=step_size,
        min_lr=1e-6,
    )

    train_loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    val_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    best_epoch = -1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if monitor not in {"val_acc", "val_loss"}:
        raise ValueError("monitor must be either 'val_acc' or 'val_loss'.")

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        running_total = 0

        loop = (
            tqdm(train_loader, leave=False, desc=f"Epoch {epoch:03d}")
            if verbose
            else train_loader
        )
        for data, targets in loop:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, loss = _forward_pass(
                model,
                data,
                targets,
                loss_fn=loss_fn,
                spiking=spiking,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            running_correct += _compute_accuracy(
                outputs,
                targets,
                spiking=spiking,
                population_code=population_code,
                num_classes=num_classes,
            )
            running_total += targets.size(0)

            if verbose and hasattr(loop, "set_postfix"):
                loop.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * running_correct / max(1, running_total)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            loss_fn=loss_fn,
            device=device,
            spiking=spiking,
            population_code=population_code,
            num_classes=num_classes,
        )
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        improved = False
        if monitor == "val_acc" and val_acc > best_val_acc:
            improved = True
        if monitor == "val_loss" and val_loss < best_val_loss:
            improved = True

        if improved:
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)

            if verbose:
                tqdm.write(
                    f"=> Saved checkpoint at epoch {epoch}: "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
                )
        else:
            epochs_without_improvement += 1

        scheduler.step(val_loss)

        if verbose:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f}% | val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.2f}% | "
                f"no_improve={epochs_without_improvement}/{patience}",
                end="\r",
                flush=True,
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print()
                print(f"=> Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "val_loss": val_loss_hist,
        "val_acc": val_acc_hist,
        "last_epoch": epoch,
        "ckpt_path": ckpt_path,
        "monitor": monitor,
    }


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    loss_fn,
    device: Union[str, torch.device],
    spiking: bool = True,
    population_code: Optional[bool] = None,
    num_classes: int = 43,
) -> Tuple[float, float]:
    """Return average loss and accuracy in percent."""
    device = torch.device(device)
    population_code = _resolve_population_code(model, population_code)
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs, loss = _forward_pass(
                model,
                data,
                targets,
                loss_fn=loss_fn,
                spiking=spiking,
            )
            total_loss += float(loss.item())
            total_correct += _compute_accuracy(
                outputs,
                targets,
                spiking=spiking,
                population_code=population_code,
                num_classes=num_classes,
            )
            total_samples += targets.size(0)

    average_loss = total_loss / max(1, len(data_loader))
    accuracy = 100.0 * total_correct / max(1, total_samples)
    return average_loss, accuracy


def _build_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Create a reproducible dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
        drop_last=False,
    )


def _make_snn_checkpoint_name(num_steps: int, seed: int, population_code: bool) -> str:
    """Create a checkpoint name for SNN runs."""
    prefix = "scnn_pd" if population_code else "scnn"
    return f"{prefix}_t={num_steps}_seed={seed}.pth"


def _make_data_transforms(image_size: int):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def main() -> None:
    from data import GTSRBDataset
    from model import CNN, SCNN

    seeds = [0, 1, 2, 3, 4]
    timesteps = [1]
    batch_size = 128
    num_workers = 8
    run_snn = True
    run_cnn = False
    use_population_code = True
    population_size = 10
    image_size = 32
    patience = 10
    device = "cuda"
    data_path = Path("../../Data/gtsrb-german-traffic-sign")
    splits_path = Path("splits")

    train_transform, eval_transform = _make_data_transforms(image_size)
    gtsrb_train = GTSRBDataset(
        data_path,
        csv_file=splits_path / "train_split.csv",
        transform=train_transform,
    )
    gtsrb_val = GTSRBDataset(
        data_path,
        csv_file=splits_path / "val_split.csv",
        transform=eval_transform,
    )

    print(f"Number of train images:\t{len(gtsrb_train)}")
    print(f"Number of val images:\t{len(gtsrb_val)}")

    if use_population_code:
        loss_fn = SF.ce_count_loss(population_code=True, num_classes=43)
    else:
        loss_fn = SF.ce_count_loss()

    if run_snn:
        for num_steps in timesteps:
            for seed in seeds:
                print(f"\n=== Training SNN with t={num_steps}, seed={seed} ===")
                seed_everything(seed)

                train_loader = _build_dataloader(
                    gtsrb_train,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    seed=seed,
                )
                val_loader = _build_dataloader(
                    gtsrb_val,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    seed=seed + 1,
                )

                if use_population_code:
                    model = SCNN(
                        num_steps=num_steps,
                        beta=0.9,
                        population_code=True,
                        population_size=population_size,
                        num_classes=43,
                    ).to(device)
                else:
                    model = SCNN(
                        num_steps=num_steps,
                        beta=0.9,
                        num_classes=43,
                    ).to(device)

                tmp_ckpt_path = f"tmp_scnn_t={num_steps}_seed={seed}.pth"
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    lr=1e-3,
                    step_size=5,
                    gamma=0.5,
                    loss_fn=loss_fn,
                    device=device,
                    patience=patience,
                    max_epochs=200,
                    spiking=True,
                    population_code=use_population_code,
                    num_classes=43,
                    ckpt_path=tmp_ckpt_path,
                    monitor="val_loss",
                )

                final_ckpt_path = _make_snn_checkpoint_name(
                    num_steps,
                    seed,
                    use_population_code,
                )
                os.replace(tmp_ckpt_path, final_ckpt_path)

                print(
                    f"t={num_steps}, seed={seed} done | "
                    f"best_epoch={int(history['best_epoch'])} | "
                    f"val_loss={float(history['best_val_loss']):.4f} "
                    f"val_acc={float(history['best_val_acc']):.2f}% | "
                    f"ckpt={final_ckpt_path}"
                )

    if run_cnn:
        print("\n=== CNN baseline over multiple seeds ===")
        cnn_loss_fn = torch.nn.CrossEntropyLoss()

        for seed in seeds:
            print(f"\n=== Training CNN with seed={seed} ===")
            seed_everything(seed)

            train_loader = _build_dataloader(
                gtsrb_train,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                seed=seed,
            )
            val_loader = _build_dataloader(
                gtsrb_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                seed=seed + 1,
            )

            model = CNN(num_classes=43).to(device)
            tmp_ckpt_path = f"tmp_cnn_seed={seed}.pth"
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=1e-3,
                step_size=5,
                gamma=0.5,
                loss_fn=cnn_loss_fn,
                device=device,
                patience=patience,
                max_epochs=200,
                spiking=False,
                population_code=False,
                num_classes=43,
                ckpt_path=tmp_ckpt_path,
                monitor="val_loss",
            )

            final_ckpt_path = f"cnn_seed={seed}.pth"
            os.replace(tmp_ckpt_path, final_ckpt_path)

            print(
                f"cnn seed={seed} done | best_epoch={int(history['best_epoch'])} | "
                f"val_loss={float(history['best_val_loss']):.4f} "
                f"val_acc={float(history['best_val_acc']):.2f}% | "
                f"ckpt={final_ckpt_path}"
            )


if __name__ == "__main__":
    main()
