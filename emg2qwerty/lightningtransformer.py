# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
import torch.nn.functional as F
from torchmetrics import MetricCollection
import math

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class PositionalEncoding(nn.Module):
    r"""Modified from https://github.com/pytorch/examples/tree/main/word_language_model"""
    
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        
   
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        pe = torch.zeros(int(len(x)), int(self.d_model))
        position = torch.arange(0, int(len(x)), dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).to(x.device)
        x = x + pe[:x.size(0), :]
        return x


class TransformerModel(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        mlp_in_features: int,
        mlp_features: Sequence[int],
        mlp_pooling: Literal["mean", "max"],
        mlp_offsets: Sequence[int],
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        # ntoken:int, 
        d_model:int, 
        nhead:int, 
        nlayers:int,
        dropout: float, has_mask: bool
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
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

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)

        
        self.mask = None
        self.has_mask = has_mask

        self.d_model = d_model
        self.dropout = dropout
 


        self.flatten = nn.Flatten(start_dim=2)
        self.input_emb = nn.Linear(num_features, d_model, bias = False)
        self.pos_encoder = PositionalEncoding(d_model = self.d_model)

       
        self.layer = nn.TransformerEncoderLayer(batch_first=False, d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=nlayers)
        
        self.dropout = (
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        )

        
        num_classes = charset().num_classes
        self.trans_classifier = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        
        self.decoder = instantiate(decoder)
        
        # self.init_weights()
        



        
        

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))
    
    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.bias)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16, freq)
        # input_lengths: (N,)
        #x = torch.tensor(inputs).to(torch.int64)
        
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

        if self.has_mask:
            if self.mask is None or self.mask.size(0) != len(inputs):
                mask = self._generate_square_subsequent_mask(len(inputs))
                self.mask = mask

        x = self.flatten(x)
        x= self.input_emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        output = self.encoder(x, mask=self.mask)
        output = self.dropout(output)
        output = self.trans_classifier(output)
        return F.log_softmax(output, dim=-1)




    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

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
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
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

    

   
        
        
        
    
     
      
      
       
