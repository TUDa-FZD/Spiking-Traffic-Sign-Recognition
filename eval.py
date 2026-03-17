import argparse
import re
from pathlib import Path

import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms

from data import GTSRBDataset
from model import CNN, SCNN
from utils import validate


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the GTSRB test split.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../../Data/gtsrb-german-traffic-sign"),
        help="Path to the GTSRB dataset root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for evaluation.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=43,
        help="Number of output classes.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population size for population-decoding checkpoints.",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "cnn", "scnn", "scnn_pd"],
        default="auto",
        help="Override automatic model type detection from the filename.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override the inferred number of timesteps for SCNN checkpoints.",
    )
    return parser.parse_args()


def infer_checkpoint_config(
    checkpoint_path: Path,
    model_type: str,
    num_steps: int | None,
) -> tuple[str, int | None]:
    """Infer model type and timestep count from the checkpoint filename."""
    if model_type != "auto":
        return model_type, num_steps

    checkpoint_name = checkpoint_path.name
    if re.fullmatch(r"cnn_seed=\d+\.pth", checkpoint_name):
        return "cnn", None

    scnn_match = re.fullmatch(r"scnn_t=(\d+)_seed=\d+\.pth", checkpoint_name)
    if scnn_match:
        return "scnn", int(scnn_match.group(1))

    scnn_pd_match = re.fullmatch(r"scnn_pd_t=(\d+)_seed=\d+\.pth", checkpoint_name)
    if scnn_pd_match:
        return "scnn_pd", int(scnn_pd_match.group(1))

    raise ValueError(
        "Could not infer model type from checkpoint name. "
        "Use --model-type and optionally --num-steps."
    )


def build_model(
    model_type: str,
    num_classes: int,
    population_size: int,
    num_steps: int | None,
    device: torch.device,
) -> tuple[torch.nn.Module, bool, bool]:
    """Instantiate the correct model for the given checkpoint."""
    if model_type == "cnn":
        model = CNN(num_classes=num_classes).to(device)
        return model, False, False

    if num_steps is None:
        raise ValueError("num_steps is required for SCNN checkpoints.")

    if model_type == "scnn":
        model = SCNN(
            num_steps=num_steps,
            beta=0.9,
            population_code=False,
            num_classes=num_classes,
        ).to(device)
        return model, True, False

    if model_type == "scnn_pd":
        model = SCNN(
            num_steps=num_steps,
            beta=0.9,
            population_code=True,
            population_size=population_size,
            num_classes=num_classes,
        ).to(device)
        return model, True, True

    raise ValueError(f"Unsupported model type: {model_type}")


def adapt_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    """Adapt checkpoint keys when legacy model names differ from current ones."""
    if model_type != "cnn":
        return state_dict

    if "features.0.weight" in state_dict or "classifier.0.weight" in state_dict:
        return state_dict

    key_mapping = {
        "conv1.weight": "features.0.weight",
        "conv1.bias": "features.0.bias",
        "bn1.weight": "features.1.weight",
        "bn1.bias": "features.1.bias",
        "bn1.running_mean": "features.1.running_mean",
        "bn1.running_var": "features.1.running_var",
        "bn1.num_batches_tracked": "features.1.num_batches_tracked",
        "conv2.weight": "features.4.weight",
        "conv2.bias": "features.4.bias",
        "bn2.weight": "features.5.weight",
        "bn2.bias": "features.5.bias",
        "bn2.running_mean": "features.5.running_mean",
        "bn2.running_var": "features.5.running_var",
        "bn2.num_batches_tracked": "features.5.num_batches_tracked",
        "conv3.weight": "features.8.weight",
        "conv3.bias": "features.8.bias",
        "bn3.weight": "features.9.weight",
        "bn3.bias": "features.9.bias",
        "bn3.running_mean": "features.9.running_mean",
        "bn3.running_var": "features.9.running_var",
        "bn3.num_batches_tracked": "features.9.num_batches_tracked",
        "fc1.weight": "classifier.0.weight",
        "fc1.bias": "classifier.0.bias",
        "fc2.weight": "classifier.2.weight",
        "fc2.bias": "classifier.2.bias",
        "fc3.weight": "classifier.4.weight",
        "fc3.bias": "classifier.4.bias",
    }

    adapted_state_dict = OrderedDict()
    for key, value in state_dict.items():
        adapted_state_dict[key_mapping.get(key, key)] = value
    return adapted_state_dict


def main() -> None:
    """Load a checkpoint and report test accuracy on GTSRB."""
    args = parse_args()
    device = torch.device(args.device)

    model_type, inferred_num_steps = infer_checkpoint_config(
        args.checkpoint,
        args.model_type,
        args.num_steps,
    )
    model, spiking, population_code = build_model(
        model_type=model_type,
        num_classes=args.num_classes,
        population_size=args.population_size,
        num_steps=inferred_num_steps,
        device=device,
    )

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = adapt_state_dict_for_model(state_dict, model_type)
    model.load_state_dict(state_dict)

    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    splits_path = Path("splits")
    test_dataset = GTSRBDataset(
        args.data_path,
        csv_file=splits_path / "Test.csv",
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    loss_fn = (
        SF.ce_count_loss(population_code=True, num_classes=args.num_classes)
        if population_code
        else (SF.ce_count_loss() if spiking else torch.nn.CrossEntropyLoss())
    )
    _, accuracy = validate(
        model,
        test_loader,
        loss_fn=loss_fn,
        device=device,
        population_code=population_code,
        num_classes=args.num_classes,
        spiking=spiking,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {model_type}")
    if inferred_num_steps is not None:
        print(f"Timesteps: {inferred_num_steps}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    import snntorch.functional as SF

    main()
