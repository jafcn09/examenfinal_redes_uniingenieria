from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .augment import spec_augment
from .config import CFG, CLASSES


def conv1d_block(cin, cout, k, s, p):
    return nn.Sequential(nn.Conv1d(cin, cout, k, s, p, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.MaxPool1d(2))


class CNN1D(nn.Module):
    def __init__(self, n_cls: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(conv1d_block(1, 16, 64, 16, 24), conv1d_block(16, 32, 3, 1, 1),
                                      conv1d_block(32, 64, 3, 1, 1), conv1d_block(64, 64, 3, 1, 1), conv1d_block(64, 64, 3, 1, 1))
        self.cam_layer = self.features[1]
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(64, 100), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(100, n_cls))
        self.specaug = False

    def embed(self, x):
        return self.pool(self.features(x.unsqueeze(1))).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


class STFTFrontend(nn.Module):
    def __init__(self, n_fft: int = 256, hop: int = 8):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x, specaug: bool = False):
        S = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True).abs()
        S = torch.log1p(S).unsqueeze(1)
        S = (S - S.mean(dim=(2, 3), keepdim=True)) / (S.std(dim=(2, 3), keepdim=True) + 1e-6)
        return spec_augment(S, 16, 16) if specaug else S


def conv2d_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class CNN2D(nn.Module):
    def __init__(self, n_cls: int = len(CLASSES)):
        super().__init__()
        self.frontend = STFTFrontend()
        self.features = nn.Sequential(conv2d_block(1, 32), conv2d_block(32, 64), conv2d_block(64, 128), conv2d_block(128, 128))
        self.cam_layer = self.features[1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_cls))
        self.specaug = False

    def embed(self, x):
        return self.pool(self.features(self.frontend(x, self.training and self.specaug))).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


class ASTClassifier(nn.Module):
    def __init__(self, cfg: CFG, n_cls: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        from transformers import ASTForAudioClassification, ASTConfig
        self.frames = cfg.ast_frames
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.fs, n_fft=512, win_length=300, hop_length=24, n_mels=128, f_min=20.0, power=2.0)
        if pretrained:
            self.ast = ASTForAudioClassification.from_pretrained(cfg.ast_name, num_labels=n_cls, ignore_mismatched_sizes=True, attn_implementation="eager")
        else:
            c = ASTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, num_labels=n_cls, attn_implementation="eager")
            self.ast = ASTForAudioClassification(c)
        self._interpolate_pos_embed(self.frames)
        self.specaug = False

    def _interpolate_pos_embed(self, frames: int):
        emb, c = self.ast.audio_spectrogram_transformer.embeddings, self.ast.config
        f_p = (c.num_mel_bins - c.patch_size) // c.frequency_stride + 1
        t_p = (c.max_length - c.patch_size) // c.time_stride + 1
        new_t = (frames - c.patch_size) // c.time_stride + 1
        pe = emb.position_embeddings.data
        tok, grid = pe[:, :2], pe[:, 2:].reshape(1, f_p, t_p, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(f_p, new_t), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).reshape(1, f_p * new_t, -1)
        emb.position_embeddings = nn.Parameter(torch.cat([tok, grid], 1))
        c.max_length = frames
        self.grid = (f_p, new_t)

    def frontend(self, x):
        m = torch.log(self.mel(x) + 1e-6)
        m = m[:, :, :self.frames] if m.shape[-1] >= self.frames else F.pad(m, (0, self.frames - m.shape[-1]))
        m = (m - m.mean(dim=(1, 2), keepdim=True)) / (2 * m.std(dim=(1, 2), keepdim=True) + 1e-6)
        if self.training and self.specaug:
            m = spec_augment(m.unsqueeze(1), 24, 16).squeeze(1)
        return m.transpose(1, 2)

    def forward(self, x, output_attentions: bool = False):
        out = self.ast(input_values=self.frontend(x), output_attentions=output_attentions)
        return (out.logits, out.attentions) if output_attentions else out.logits

    def embed(self, x):
        h = self.ast.audio_spectrogram_transformer(input_values=self.frontend(x)).last_hidden_state
        return (h[:, 0] + h[:, 1]) / 2


def build_ast(cfg: CFG) -> ASTClassifier:
    try:
        return ASTClassifier(cfg, pretrained=True)
    except Exception as e:
        if not (cfg.smoke or os.environ.get("CWRU_ALLOW_RANDOM_AST") == "1"):
            raise
        print(f"AST preentrenado no disponible ({type(e).__name__}): se usa AST reducido aleatorio solo para validar el flujo")
        return ASTClassifier(cfg, pretrained=False)


def build_model(name: str, cfg: CFG) -> nn.Module:
    return {"CNN1D": lambda: CNN1D(), "CNN2D": lambda: CNN2D(), "AST": lambda: build_ast(cfg)}[name]()


def model_spec(cfg: CFG) -> dict:
    return {"CNN1D": (cfg.epochs_cnn, cfg.lr_cnn), "CNN2D": (cfg.epochs_cnn, cfg.lr_cnn), "AST": (cfg.epochs_ast, cfg.lr_ast)}
