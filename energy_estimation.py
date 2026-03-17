import argparse
import re
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchvision import transforms

from data import GTSRBDataset
from model import CNN, SCNN


class EnergyEstimator:
    # energy values are adopted from keras spiking model energy
    # https://github.com/nengo/keras-spiking/blob/main/keras_spiking/model_energy.py#L402
    PROCESSING_UNIT = {
        # energy for mac oeprations on 45 nm technology
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6757323
        "default_device": {
            "spiking": False,
            "energy_per_synop": 3.7e-12 + 0.9e-12,
            "energy_per_neuron": 3.7e-12 + 0.9e-12,
        },
        # https://ieeexplore.ieee.org/abstract/document/7054508
        "cpu": {
            "spiking": False,
            "energy_per_synop": 8.6e-9,
            "energy_per_neuron": 8.6e-9,
        },
        "gpu": {
            "spiking": False,
            "energy_per_synop": 0.3e-9,
            "energy_per_neuron": 0.3e-9,
        },
        "arm": {
            "spiking": False,
            "energy_per_synop": 0.9e-9,
            "energy_per_neuron": 0.9e-9,
        },
        # energy for acc oeprations on 45 nm technology
        "default_spiking_device": {
            "spiking": True,
            "energy_per_synop": 0.9e-12,
            "energy_per_neuron": 0.9e-12,
        },
        # https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf
        "loihi": {
            "spiking": True,
            "energy_per_synop": 23.6e-12,
            "energy_per_neuron": 81e-12,
        },
        # https://arxiv.org/abs/1903.08941
        "spinnaker": {
            "spiking": True,
            "energy_per_synop": 13.3e-9,
            "energy_per_neuron": 26e-9,
        },
        # energy values of the spinnaker2 prototype
        "spinnaker2": {
            "spiking": True,
            "energy_per_synop": 1.5e-9,
            "energy_per_neuron": 2.19e-9,
        },
    }

    def __init__(
        self,
        model: nn.Module,
        processing_unit: str = None,
        input_data: torch.Tensor = None,
        dataloader: torch.utils.data.DataLoader = None,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("The 'model' must be an instance of torch.nn.Module")

        if input_data is not None and not isinstance(input_data, torch.Tensor):
            raise TypeError("The 'input_data' must be an instance of torch.Tensor")

        if dataloader is not None and not isinstance(
            dataloader, torch.utils.data.DataLoader
        ):
            raise TypeError(
                "The 'dataloader' must be an instance of torch.utils.data.DataLoader"
            )

        # Determine if the model is spiking
        self.is_spiking = any(layer.__class__.__module__.startswith('snn') for layer in model.modules())

        # Select default processing unit if none provided
        if processing_unit is None:
            if self.is_spiking:
                processing_unit = "default_spiking_device"
            else:
                processing_unit = "default_device"

        if processing_unit not in self.PROCESSING_UNIT:
            raise ValueError(
                f"Invalid processing unit '{processing_unit}'. "
                f"Available processing units are: {list(self.PROCESSING_UNIT.keys())}"
            )
        
        if not self.is_spiking and self.PROCESSING_UNIT[processing_unit]["spiking"]:
            raise ValueError(
            f"The processing unit '{processing_unit}' is not suitable for a non-spiking model"
            )

        self.model = model
        self.input_data = input_data
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_unit = processing_unit

    @staticmethod
    def _shape_info(x):
        if isinstance(x, torch.Tensor):
            return x.shape
        elif isinstance(x, (list, tuple)):
            return [EnergyEstimator._shape_info(item) for item in x]
        else:
            return None

    @staticmethod
    def _remove_hooks(hooks):
        for hook in hooks:
            hook.remove()

    def _register_hooks_with_layer_names(self, model):
        layer_info = {}
        hooks = []

        def forward_hook(module, input, output):
            for name, mod in model.named_modules():
                if mod is module:
                    if name and mod is not model:
                        out_shape = self._shape_info(output)
                        in_shape = self._shape_info(input)
                        neurons = 0

                        if isinstance(mod, (snn.Leaky, nn.ReLU)):
                            if isinstance(output, torch.Tensor):
                                neurons = output.numel()

                            elif (
                                isinstance(output, (tuple, list))
                                and len(output) > 0
                                and isinstance(output[0], torch.Tensor)
                            ):
                                neurons = output[0].numel()

                        layer_dict = {
                            "input_shape": in_shape,
                            "output_shape": out_shape,
                            "neurons": neurons,
                        }

                        layer_info[name] = layer_dict
                    break

        for layer in model.modules():
            hooks.append(layer.register_forward_hook(forward_hook))

        return layer_info, hooks

    def _register_hooks_spikes(self, model):
        spikes = {}
        hooks = []

        if not any(isinstance(layer, snn.Leaky) for layer in model.modules()):
            raise ValueError("The model does not contain any spiking neurons")

        def forward_hook(module, _, output):
            for name, mod in model.named_modules():
                if mod is module and isinstance(mod, snn.Leaky):
                    if isinstance(output, torch.Tensor):
                        if name in spikes:
                            spikes[name] += output.sum().item()
                        else:
                            spikes[name] = output.sum().item()
                    elif (
                        isinstance(output, (tuple, list))
                        and len(output) > 0
                        and isinstance(output[0], torch.Tensor)
                    ):
                        if name in spikes:
                            spikes[name] += output[0].sum().item()
                        else:
                            spikes[name] = output[0].sum().item()

        for layer in model.modules():
            if isinstance(layer, snn.Leaky):
                hooks.append(layer.register_forward_hook(forward_hook))
        return spikes, hooks

    def get_spikes_per_layer(self):
        spikes, hooks = self._register_hooks_spikes(self.model)
        num_samples = 1
        if self.dataloader is not None:
            print("Processing data through the model ...")
            num_samples = len(self.dataloader.dataset)
            with torch.no_grad():
                for data in self.dataloader:
                    inputs, _ = data
                    inputs = inputs.to(self.device)
                    _ = self.model(inputs)
        else:
            _ = self.model(self.input_data)
        self._remove_hooks(hooks)
        spikes = {layer: count / num_samples for layer, count in spikes.items()}
        return spikes

    def get_layer_sparsity(self):
        spikes = self.get_spikes_per_layer()
        layer_info = self.get_output_shapes()
        num_neurons = {
            layer: info["neurons"]
            for layer, info in layer_info.items()
            if isinstance(dict(self.model.named_modules())[layer], snn.Leaky)
        }
        layer_sparsity = {
            layer: count / num_neurons[layer]
            for layer, count in spikes.items()
            if layer in num_neurons
        }
        return layer_sparsity

    def get_output_shapes(self):
        layer_info, hooks = self._register_hooks_with_layer_names(self.model)
        _ = self.model(self.input_data)
        self._remove_hooks(hooks)
        return layer_info

    def estimate_energy(self):
        layer_info = self.get_output_shapes()
        energy_table = []
        total_energy = 0
        layer_sparsity = {}

        if self.is_spiking:
            layer_sparsity = self.get_layer_sparsity()

        for name, info in layer_info.items():
            input_shape = info["input_shape"]
            output_shape = info["output_shape"]
            neurons = info.get("neurons", 0)
            params = 0
            connections = 0
            energy = 0

            sparsity = layer_sparsity.get(name, "")
            if isinstance(sparsity, float):
                sparsity = f"{sparsity:.2f}"

            # Find the previous Leaky layer
            prev_leaky_layer_name = None
            for prev_name in reversed(
                list(layer_info.keys())[: list(layer_info.keys()).index(name)]
            ):
                prev_layer = dict(self.model.named_modules())[prev_name]
                if isinstance(prev_layer, snn.Leaky):
                    prev_leaky_layer_name = prev_name
                    break

            layer = dict(self.model.named_modules())[name]
            if isinstance(layer, nn.Conv2d):
                params = layer.weight.numel()
                connections = params * torch.tensor(output_shape[1:]).prod().item()

                if layer_sparsity:
                    if prev_leaky_layer_name in layer_sparsity:
                        energy = connections * self.PROCESSING_UNIT[self.processing_unit]["energy_per_synop"] * layer_sparsity[prev_leaky_layer_name]
                    else:
                        energy = connections * self.PROCESSING_UNIT[self.processing_unit]["energy_per_synop"] * self.model.num_steps
                else:
                    energy = connections * self.PROCESSING_UNIT[self.processing_unit]["energy_per_synop"]

            elif isinstance(layer, nn.Linear):
                params = layer.weight.numel()
                connections = params

                if layer_sparsity:
                    if prev_leaky_layer_name in layer_sparsity:
                        energy = connections * self.PROCESSING_UNIT[self.processing_unit]["energy_per_synop"] * layer_sparsity[prev_leaky_layer_name]
                    else:
                        energy = connections * self.PROCESSING_UNIT[self.processing_unit]["energy_per_synop"] * self.model.num_steps
                else:
                    energy = connections * self.PROCESSING_UNIT[self.processing_unit]["energy_per_synop"]

            # elif isinstance(layer, nn.ReLU):
            #     energy = neurons * self.PROCESSING_UNIT[self.processing_unit]["energy_per_neuron"]

            # elif isinstance(layer, snn.Leaky):
            #     energy = neurons * self.PROCESSING_UNIT[self.processing_unit]["energy_per_neuron"] * self.model.num_steps

            total_energy += energy
            energy_table.append(
                [name, input_shape, output_shape, connections, neurons, energy, sparsity]
            )

        headers = [
            "Layer",
            "Input Shape",
            "Output Shape",
            "Connections",
            "Neurons",
            "Energy (J)",
            "Sparsity",
        ]
        table = tabulate(energy_table, headers=headers, floatfmt=".3e", tablefmt="github")
        return table, total_energy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for checkpoint-based energy estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate inference energy for a checkpoint.",
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
        help="Batch size used for spike statistics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for loading the model.",
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
        help="Population size for SCNN-PD checkpoints.",
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
    parser.add_argument(
        "--processing-unit",
        choices=list(EnergyEstimator.PROCESSING_UNIT.keys()),
        default=None,
        help="Processing unit used for the energy estimate.",
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
) -> torch.nn.Module:
    """Instantiate the correct model for the given checkpoint."""
    if model_type == "cnn":
        return CNN(num_classes=num_classes).to(device)

    if num_steps is None:
        raise ValueError("num_steps is required for SCNN checkpoints.")

    if model_type == "scnn":
        return SCNN(
            num_steps=num_steps,
            beta=0.9,
            population_code=False,
            num_classes=num_classes,
        ).to(device)

    if model_type == "scnn_pd":
        return SCNN(
            num_steps=num_steps,
            beta=0.9,
            population_code=True,
            population_size=population_size,
            num_classes=num_classes,
        ).to(device)

    raise ValueError(f"Unsupported model type: {model_type}")


def adapt_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model_type: str,
) -> dict[str, torch.Tensor]:
    """Adapt legacy CNN checkpoint keys to the current module layout."""
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
    """Load a checkpoint and estimate its inference energy."""
    args = parse_args()
    device = torch.device(args.device)

    model_type, inferred_num_steps = infer_checkpoint_config(
        args.checkpoint,
        args.model_type,
        args.num_steps,
    )
    model = build_model(
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
    sample_input = torch.randn(1, 3, 32, 32, device=device)

    estimator = EnergyEstimator(
        model=model,
        processing_unit=args.processing_unit,
        input_data=sample_input,
        dataloader=test_loader,
    )
    energy_table, total_energy = estimator.estimate_energy()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {model_type}")
    if inferred_num_steps is not None:
        print(f"Timesteps: {inferred_num_steps}")
    print(f"Processing unit: {estimator.processing_unit}")
    print(energy_table)
    print(f"Total energy per inference: {total_energy:.2e} J")


if __name__ == "__main__":
    main()
