"""Vendored LP-MusicCaps captioning model (SeungHeon Doh et al., ISMIR 2023).

Adapted from ``github.com/seungheondoh/lp-music-caps`` — files
``lpmc/music_captioning/model/bart.py`` and ``.../modules.py`` (MIT-tagged
on the Hugging Face model repo; the GitHub README mentions CC-BY-NC for
the *dataset*, which is NOT redistributed here).

Why vendored: the upstream repo is a 2024-era research tree pinned to
``transformers==4.26.1`` and python 3.10 (its ``pip install -e .`` cannot
install on Python 3.14), but the actual inference path — a small mel-CNN
audio encoder feeding ``facebook/bart-base`` — runs fine on modern
torch/transformers.  Only these two files are needed, so they are ported
here instead of asking users to install a broken package.

Changes vs upstream: type hints, docstrings, checkpoint loading via
``weights_only=True``, and BART calls kept compatible with transformers
v5 (plain ``BartForConditionalGeneration``).  The audio front-end is a
Whisper-style mel projection (16 kHz, 128 mels, 10 s windows) followed by
a strided Conv1d stack; the text side is bart-base beam decoding.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# hard-coded audio hyperparameters (upstream values)
SAMPLE_RATE = 16000
N_FFT = 1024
N_MELS = 128
HOP_LENGTH = int(0.01 * SAMPLE_RATE)
DURATION = 10
N_SAMPLES = int(DURATION * SAMPLE_RATE)
N_FRAMES = N_SAMPLES // HOP_LENGTH + 1


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0):
    """Sinusoidal positional embedding (Whisper-style)."""
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2))
    scaled_time = (torch.arange(length)[:, np.newaxis]
                   * inv_timescales[np.newaxis, :])
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MelEncoder(nn.Module):
    """STFT power spectrogram → mel scale → dB (torchaudio transforms)."""

    def __init__(self, sample_rate: int = 16000, f_min: float = 0.0,
                 f_max: float = 8000.0, n_fft: int = 1024,
                 win_length: int = 1024,
                 hop_length: int = int(0.01 * 16000), n_mels: int = 128,
                 power=None, pad: int = 0, normalized: bool = False,
                 center: bool = True, pad_mode: str = "reflect") -> None:
        super().__init__()
        self.window = torch.hann_window(win_length)
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            power=power)
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: Tensor) -> Tensor:
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec


class AudioEncoder(nn.Module):
    """Mel spectrogram → strided convs → (batch, n_ctx, text_dim) tokens."""

    def __init__(self, n_mels: int, n_ctx: int, audio_dim: int,
                 text_dim: int, num_of_stride_conv: int) -> None:
        super().__init__()
        self.mel_encoder = MelEncoder(n_mels=n_mels)
        self.conv1 = nn.Conv1d(n_mels, audio_dim, kernel_size=3, padding=1)
        self.conv_stack = nn.ModuleList([])
        for _ in range(num_of_stride_conv):
            self.conv_stack.append(
                nn.Conv1d(audio_dim, audio_dim, kernel_size=3, stride=2,
                          padding=1))
        self.register_buffer("positional_embedding",
                             sinusoids(n_ctx, text_dim))

    def forward(self, x: Tensor) -> Tensor:
        """*(batch, waveform)* mono samples → *(batch, n_ctx, text_dim)*."""
        x = self.mel_encoder(x)          # (batch, n_mels, n_ctx)
        x = F.gelu(self.conv1(x))
        for conv in self.conv_stack:
            x = F.gelu(conv(x))
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding).to(x.dtype)
        return x


class BartCaptionModel(nn.Module):
    """Mel-CNN audio encoder + bart-base caption decoder.

    ``generate`` returns one caption per (10 s) audio sample row.
    ``forward_encoder`` also exposes the pre-decoder audio embeddings —
    the app mean-pools those into a fixed 768-d embedding vector.
    """

    def __init__(self, n_mels: int = 128, num_of_conv: int = 6,
                 sr: int = 16000, duration: int = 10, max_length: int = 128,
                 label_smoothing: float = 0.1,
                 bart_type: str = "facebook/bart-base",
                 audio_dim: int = 768) -> None:
        super().__init__()
        from transformers import BartConfig, BartForConditionalGeneration
        from transformers import BartTokenizer

        bart_config = BartConfig.from_pretrained(bart_type)
        self.tokenizer = BartTokenizer.from_pretrained(bart_type)
        self.bart = BartForConditionalGeneration(bart_config)

        self.n_sample = sr * duration
        self.hop_length = int(0.01 * sr)
        self.n_frames = int(self.n_sample // self.hop_length)
        self.num_of_stride_conv = num_of_conv - 1
        self.n_ctx = int(self.n_frames // 2 ** self.num_of_stride_conv) + 1
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels,
            n_ctx=self.n_ctx,
            audio_dim=audio_dim,
            text_dim=self.bart.config.hidden_size,
            num_of_stride_conv=self.num_of_stride_conv,
        )
        self.max_length = max_length
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                            ignore_index=-100)

    @property
    def device(self):
        return list(self.parameters())[0].device

    @staticmethod
    def shift_tokens_right(input_ids: Tensor, pad_token_id: int,
                           decoder_start_token_id: int) -> Tensor:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError(
                "self.model.config.pad_token_id has to be defined.")
        shifted_input_ids.masked_fill_(shifted_input_ids == -100,
                                       pad_token_id)
        return shifted_input_ids

    def forward_encoder(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        audio_embs = self.audio_encoder(audio)
        encoder_outputs = self.bart.model.encoder(
            input_ids=None, inputs_embeds=audio_embs,
            return_dict=True)["last_hidden_state"]
        return encoder_outputs, audio_embs

    def forward(self, audio: Tensor, text: list[str]) -> Tensor:
        encoder_outputs, _ = self.forward_encoder(audio)
        text = self.tokenizer(text, padding="longest", truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        attention_mask = text["attention_mask"].to(self.device)
        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100)
        decoder_input_ids = self.shift_tokens_right(
            decoder_targets, self.bart.config.pad_token_id,
            self.bart.config.decoder_start_token_id)
        decoder_outputs = self.bart(
            input_ids=None, attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=attention_mask, inputs_embeds=None,
            labels=None, encoder_outputs=(encoder_outputs,),
            return_dict=True)
        lm_logits = decoder_outputs["logits"]
        return self.loss_fct(
            lm_logits.view(-1, self.tokenizer.vocab_size),
            decoder_targets.view(-1))

    @torch.no_grad()
    def generate(self, samples: Tensor, use_nucleus_sampling: bool = False,
                 num_beams: int = 5, max_length: int = 128,
                 min_length: int = 2, top_p: float = 0.9,
                 repetition_penalty: float = 1.0) -> list[str]:
        """Beam-decode one caption per sample row (*(batch, waveform)*)."""
        audio_embs = self.audio_encoder(samples)
        encoder_outputs = self.bart.model.encoder(
            input_ids=None, attention_mask=None, head_mask=None,
            inputs_embeds=audio_embs, output_attentions=None,
            output_hidden_states=None, return_dict=True)
        input_ids = torch.zeros(
            (encoder_outputs["last_hidden_state"].size(0), 1)).long().to(
            self.device)
        input_ids[:, 0] = self.bart.config.decoder_start_token_id
        decoder_attention_mask = torch.ones(
            (encoder_outputs["last_hidden_state"].size(0), 1)).long().to(
            self.device)
        if use_nucleus_sampling:
            outputs = self.bart.generate(
                input_ids=None, attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs, max_length=max_length,
                min_length=min_length, do_sample=True, top_p=top_p,
                num_return_sequences=1, repetition_penalty=1.1)
        else:
            outputs = self.bart.generate(
                input_ids=None, attention_mask=None,
                decoder_input_ids=input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs, head_mask=None,
                decoder_head_mask=None, inputs_embeds=None,
                decoder_inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                max_length=max_length, min_length=min_length,
                num_beams=num_beams, repetition_penalty=repetition_penalty)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def load_lpmc_checkpoint(model: BartCaptionModel, path, variant="state_dict"):
    """Load an official LP-MusicCaps checkpoint into *model*.

    The released ``pretrain.pth`` / ``supervised.pth`` / ``transfer.pth``
    files store ``{"state_dict": <model state dict>}``.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model
