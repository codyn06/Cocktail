"""GPU models for live separation (task C3). Needs `uv sync --group gpu`.

- VoiceSeparator: MossFormer2_SS_16K (ClearerVoice-Studio, Apache-2.0), 2 voices at 16 kHz.
  Chosen in A6: 714 MiB peak, RTF ~0.13 on the RTX 5070 Ti Laptop (docs/STATUS.md).
- SoundLabeler: AST fine-tuned on AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593, BSD-3),
  527 classes, mapped to coarse labels.
"""

from __future__ import annotations

import numpy as np
import torch

SR = 16000


class VoiceSeparator:
    n_out = 2

    def __init__(self, device: str = "cuda"):
        from clearvoice import ClearVoice

        self.device = device
        self.net = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"]).models[0].model
        self.net.to(device).eval()

    @torch.inference_mode()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (n,) float32 mono at 16 kHz -> (2, n) voice estimates (arbitrary scale)."""
        out = self.net(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(self.device))
        return np.stack([o.squeeze().float().cpu().numpy()[: len(x)] for o in out])


# AudioSet class name -> coarse label. First match in this order wins.
COARSE = [
    ("alarm", ("smoke detector", "fire alarm", "alarm", "buzzer")),
    ("siren", ("siren", "emergency vehicle", "police car", "ambulance", "fire engine")),
    ("speech", ("speech", "conversation", "narration", "male speech", "female speech", "child speech")),
    ("music", ("music", "musical instrument", "orchestra", "string section", "violin", "guitar",
               "piano", "drum", "singing")),
]


def coarse_label(audioset_name: str) -> str:
    n = audioset_name.lower()
    if "car alarm" in n or "alarm clock" in n or "babble" in n:
        return "other"
    for coarse, keys in COARSE:
        if any(k in n for k in keys):
            return coarse
    return "other"


class SoundLabeler:
    def __init__(self, device: str = "cuda", model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device = device
        self.fe = ASTFeatureExtractor.from_pretrained(model_id)
        self.model = ASTForAudioClassification.from_pretrained(model_id).to(device).eval()
        self.names = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        groups: dict[str, list[int]] = {}
        for i, n in enumerate(self.names):
            groups.setdefault(coarse_label(n), []).append(i)
        self.groups = {k: np.array(v) for k, v in groups.items()}

    @torch.inference_mode()
    def probs(self, xs: list[np.ndarray]) -> np.ndarray:
        """Batch of 16 kHz mono clips -> (batch, 527) sigmoid class probabilities."""
        f = self.fe(xs, sampling_rate=SR, return_tensors="pt")["input_values"].to(self.device)
        return torch.sigmoid(self.model(f).logits).float().cpu().numpy()

    def coarse(self, p: np.ndarray) -> dict[str, float]:
        """Class probabilities -> coarse label scores (max over member classes)."""
        return {k: float(p[idx].max()) for k, idx in self.groups.items()}
