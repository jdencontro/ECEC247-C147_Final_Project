# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)


class CNNRNNCTCModule(pl.LightningModule):
    """CNN + RNN model trained with CTC loss.

    Expects spectrogram inputs of shape (T, N, bands=2, electrode_channels=16, freq).

    Architecture (all components are provided in the starter codebase):
      SpectrogramNorm -> MultiBandRotationInvariantMLP -> TDSConvEncoder (CNN) -> RNN -> CTC
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        mlp_in_features: int,
        mlp_features: Sequence[int],
        mlp_pooling: Literal["mean", "max"],
        mlp_offsets: Sequence[int],
        tds_block_channels: Sequence[int],
        tds_kernel_width: int,
        cnn_dropout: float,
        rnn_input_size: int | None,
        rnn_type: Literal["lstm", "gru", "rnn"],
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        rnn_bidirectional: bool,
        rnn_output_dropout: float,
        use_packed_sequence: bool,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if mlp_in_features <= 0:
            raise ValueError("mlp_in_features must be positive")
        if len(mlp_features) <= 0:
            raise ValueError("mlp_features must be non-empty")
        if len(tds_block_channels) <= 0:
            raise ValueError("tds_block_channels must be non-empty")
        if tds_kernel_width <= 0:
            raise ValueError("tds_kernel_width must be positive")

        spec_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS  # 32
        num_features = self.NUM_BANDS * mlp_features[-1]
        self.mlp_in_features = mlp_in_features

        # Normalize each electrode channel independently over (N, freq, time).
        self.spec_norm = SpectrogramNorm(channels=spec_channels)

        # Rotation-invariant MLP feature extraction per band.
        # Per band, the MLP input is a flattened (electrodes=16, freq) tensor:
        # in_features = 16 * freq_bins = 16 * (n_fft // 2 + 1) = 528 for n_fft=64.
        self.band_mlp = MultiBandRotationInvariantMLP(
            in_features=mlp_in_features,
            mlp_features=mlp_features,
            pooling=mlp_pooling,
            offsets=mlp_offsets,
            num_bands=self.NUM_BANDS,
        )

        # CNN over time (TDSConvEncoder).
        self.cnn = TDSConvEncoder(
            num_features=num_features,
            block_channels=tds_block_channels,
            kernel_width=tds_kernel_width,
        )
        self.cnn_dropout = nn.Dropout(p=cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        # Optional projection before the RNN to reduce compute.
        rnn_input_size = num_features if rnn_input_size is None else rnn_input_size
        if rnn_input_size <= 0:
            raise ValueError("rnn_input_size must be positive (or null)")
        if rnn_input_size != num_features:
            self.rnn_input_proj: nn.Module = nn.Linear(num_features, rnn_input_size)
        else:
            self.rnn_input_proj = nn.Identity()
        self.rnn_input_norm = nn.LayerNorm(rnn_input_size)

        self.use_packed_sequence = use_packed_sequence

        # RNN backend on per-timestep CNN features.
        rnn_dropout = rnn_dropout if rnn_num_layers > 1 else 0.0
        if rnn_type == "lstm":
            self.rnn: nn.Module = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=rnn_input_size,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                nonlinearity="tanh",
                dropout=rnn_dropout,
                bidirectional=rnn_bidirectional,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        rnn_out_features = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.rnn_output_dropout = (
            nn.Dropout(p=rnn_output_dropout) if rnn_output_dropout > 0 else nn.Identity()
        )

        num_classes = charset().num_classes
        # A direct classifier on CNN features (baseline-style head). This keeps a
        # strong non-recurrent path so the RNN can act as a refinement rather than
        # being the only route to logits.
        self.cnn_classifier = nn.Linear(num_features, num_classes)
        self.rnn_classifier = nn.Linear(rnn_out_features, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (T, N, bands=2, C=16, freq)
        # input_lengths: (N,)
        x = self.spec_norm(inputs)

        # Validate expected per-band flattened feature size (16 * freq).
        _, _, _, C, freq = x.shape
        expected_in_features = C * freq
        if expected_in_features != self.mlp_in_features:
            raise ValueError(
                "mlp_in_features mismatch: "
                f"got {self.mlp_in_features}, but inputs imply {expected_in_features} "
                f"(electrode_channels={C}, freq_bins={freq}). "
                "Update `module.mlp_in_features` to match your spectrogram settings."
            )

        # (T, N, B, C, freq) -> (T, N, B, mlp_features[-1])
        x = self.band_mlp(x)

        # (T, N, B, F) -> (T, N, B*F)
        x = x.flatten(start_dim=2)

        # CNN over time: (T_in, N, F) -> (T_out, N, F)
        x = self.cnn(x)
        x = self.cnn_dropout(x)
        cnn_features = x

        # Account for the CNN shrinking time length (valid convolutions).
        T_diff = inputs.shape[0] - x.shape[0]
        if input_lengths is None:
            emission_lengths = torch.full(
                (x.shape[1],), x.shape[0], dtype=torch.int32, device=x.device
            )
        else:
            emission_lengths = input_lengths - T_diff

        # Project + normalize features before the RNN.
        x = self.rnn_input_norm(self.rnn_input_proj(x))

        # RNN. Use packed sequences for correctness when lengths vary (esp. bidirectional).
        if self.use_packed_sequence and input_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths=emission_lengths.detach().cpu(), enforce_sorted=False
            )
            packed_out = self.rnn(packed)[0]
            x = nn.utils.rnn.pad_packed_sequence(packed_out)[0]
        else:
            x = self.rnn(x)[0]

        x = self.rnn_output_dropout(x)  # (T, N, rnn_out_features)
        logits = self.cnn_classifier(cnn_features) + self.rnn_classifier(x)  # (T, N, C)
        emissions = self.log_softmax(logits)  # (T, N, num_classes)
        return emissions, emission_lengths

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions, emission_lengths = self.forward(inputs, input_lengths=input_lengths)

        # NOTE: As of torch 2.3, CTC loss is not implemented on MPS. We compute
        # it on CPU (while keeping the rest of the model on MPS) so Mac users
        # can still train/evaluate.
        if emissions.device.type == "mps":
            loss = self.ctc_loss(
                log_probs=emissions.cpu(),  # (T, N, num_classes)
                targets=targets.transpose(0, 1).cpu(),  # (T, N) -> (N, T)
                input_lengths=emission_lengths.cpu(),  # (N,)
                target_lengths=target_lengths.cpu(),  # (N,)
            ).to(emissions.device)
        else:
            loss = self.ctc_loss(
                log_probs=emissions,  # (T, N, num_classes)
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
                input_lengths=emission_lengths,  # (N,)
                target_lengths=target_lengths,  # (N,)
            )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
