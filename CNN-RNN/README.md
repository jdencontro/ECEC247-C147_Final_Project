# CNN + RNN (CTC) Model

This folder contains a CNN+RNN (LSTM/GRU/RNN) architecture implemented on top
of the provided `emg2qwerty` training pipeline.

## Files

- `CNN-RNN/cnn_rnn_lightning.py`: `CNNRNNCTCModule` (CNN frontend + RNN backend + CTC).
- `config/model/cnn_rnn_ctc.yaml`: Hydra config to train this model.

## Architecture

This implementation intentionally reuses the *provided* building blocks:

`SpectrogramNorm -> MultiBandRotationInvariantMLP -> TDSConvEncoder (CNN) -> RNN -> CTC`

The rotation-invariant MLP pairs well with the included band-rotation augmentation.

## Run

From the repo root:

```bash
conda run -n emg2qwerty python -m emg2qwerty.train model=cnn_rnn_ctc
```

You can override hyperparameters via Hydra, e.g.:

```bash
conda run -n emg2qwerty python -m emg2qwerty.train \\
  model=cnn_rnn_ctc module.rnn_type=gru module.rnn_hidden_size=384
```

For better decoding (usually lower CER), try the provided beam-search decoder:

```bash
conda run -n emg2qwerty python -m emg2qwerty.train \\
  model=cnn_rnn_ctc decoder=ctc_beam
```

## Apple Silicon (MPS) Note

As of `torch==2.3.0`, CTC loss is not implemented on MPS. When training on a
Mac GPU (MPS), this project computes CTC loss on CPU (with gradients flowing
back to the model), so training will be slower than CUDA.

## Notes

- `module.mlp_in_features` must match `electrode_channels * freq_bins`.
  With `n_fft=64`, `freq_bins = 33`, so `16 * 33 = 528` (default).
