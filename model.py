import torch
import torch.nn as nn
import snntorch as snn


class _BaseSCNN(nn.Module):
    """Shared architecture for spiking CNN variants."""

    def __init__(
        self,
        num_steps: int = 10,
        beta: float = 0.9,
        population_code: bool = False,
        population_size: int | None = None,
        num_classes: int = 43,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.population_code = population_code
        self.num_classes = num_classes

        if population_code:
            if population_size is None:
                raise ValueError(
                    "population_size must be set when population_code=True."
                )
            self.population_size = population_size
            self.output_neurons = num_classes * population_size
        else:
            self.population_size = None
            self.output_neurons = num_classes

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif2 = snn.Leaky(beta=beta)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lif3 = snn.Leaky(beta=beta)

        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.lif4 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(1024, 512)
        self.lif5 = snn.Leaky(beta=beta)

        self.fc3 = nn.Linear(512, self.output_neurons)
        self.lif6 = snn.Leaky(beta=beta)

    def _init_memories(self) -> list:
        return [
            self.lif1.init_leaky(),
            self.lif2.init_leaky(),
            self.lif3.init_leaky(),
            self.lif4.init_leaky(),
            self.lif5.init_leaky(),
            self.lif6.init_leaky(),
        ]


class SCNN(_BaseSCNN):
    """Spiking CNN used for training and evaluation."""

    def __init__(
        self,
        num_steps: int = 10,
        beta: float = 0.9,
        population_code: bool = False,
        population_size: int | None = None,
        num_classes: int = 43,
    ) -> None:
        super().__init__(
            num_steps=num_steps,
            beta=beta,
            population_code=population_code,
            population_size=population_size,
            num_classes=num_classes,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem1, mem2, mem3, mem4, mem5, mem6 = self._init_memories()
        spike_recordings = []
        membrane_recordings = []

        for _ in range(self.num_steps):
            cur1 = self.mp1(self.bn1(self.conv1(x)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.mp2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.mp3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc1(spk3.flatten(1))
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            spike_recordings.append(spk6)
            membrane_recordings.append(mem6)

        return torch.stack(spike_recordings, dim=0), torch.stack(
            membrane_recordings,
            dim=0,
        )


class SCNN_eval(_BaseSCNN):
    """SCNN variant that returns spike traces from all layers."""

    def __init__(
        self,
        num_steps: int = 10,
        beta: float = 0.9,
        population_code: bool = False,
        population_size: int | None = None,
        num_classes: int = 43,
    ) -> None:
        super().__init__(
            num_steps=num_steps,
            beta=beta,
            population_code=population_code,
            population_size=population_size,
            num_classes=num_classes,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mem1, mem2, mem3, mem4, mem5, mem6 = self._init_memories()
        layer_recordings = [[] for _ in range(6)]

        for _ in range(self.num_steps):
            cur1 = self.mp1(self.bn1(self.conv1(x)))
            spk1, mem1 = self.lif1(cur1, mem1)
            layer_recordings[0].append(spk1)

            cur2 = self.mp2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)
            layer_recordings[1].append(spk2)

            cur3 = self.mp3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(cur3, mem3)
            layer_recordings[2].append(spk3)

            cur4 = self.fc1(spk3.flatten(1))
            spk4, mem4 = self.lif4(cur4, mem4)
            layer_recordings[3].append(spk4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            layer_recordings[4].append(spk5)

            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)
            layer_recordings[5].append(spk6)

        return tuple(torch.stack(recording, dim=0) for recording in layer_recordings)


class SCNN_BNTT(_BaseSCNN):
    """SCNN variant with time-dependent batch normalization."""

    def __init__(
        self,
        num_steps: int = 10,
        beta: float = 0.9,
        population_code: bool = False,
        population_size: int | None = None,
        num_classes: int = 43,
    ) -> None:
        super().__init__(
            num_steps=num_steps,
            beta=beta,
            population_code=population_code,
            population_size=population_size,
            num_classes=num_classes,
        )
        self.bn1 = snn.BatchNormTT2d(input_features=32, time_steps=num_steps)
        self.bn2 = snn.BatchNormTT2d(input_features=64, time_steps=num_steps)
        self.bn3 = snn.BatchNormTT2d(input_features=128, time_steps=num_steps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem1, mem2, mem3, mem4, mem5, mem6 = self._init_memories()
        spike_recordings = []
        membrane_recordings = []

        for _ in range(self.num_steps):
            cur1 = self.mp1(self.bn1(self.conv1(x)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.mp2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.mp3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc1(spk3.flatten(1))
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            cur6 = self.fc3(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)

            spike_recordings.append(spk6)
            membrane_recordings.append(mem6)

        return torch.stack(spike_recordings, dim=0), torch.stack(
            membrane_recordings,
            dim=0,
        )


class CNN(nn.Module):
    """Conventional CNN baseline with the same feature extractor depth."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
