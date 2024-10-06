import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram


class LogMelSpec(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.melspec = MelSpectrogram(*args, **kwargs)

    def __call__(self, data: Tensor):
        return torch.log(self.melspec(data).clamp_(min=1e-9, max=1e9))
