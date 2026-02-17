# iclr_simulator: Multi-Resolution RF Signal Simulator

## Overview

This repository provides a configurable simulator for generating synthetic multi-resolution time–frequency datasets.
It is designed for:
- Passive RF signal interception and radar-like scenarios.
- Controlled generation of modulated waveforms and structured interferences.
- Construction of multi-resolution STFT-based datasets for machine learning (e.g., object detection on spectrograms).

The simulator builds *scenarios* composed of several emitters and (optionally) interferences, generates 1D time-domain
signals, computes multiple STFTs at different resolutions, and stores both the spectrograms and the associated
annotations (e.g., waveform type, time–frequency support, class indices).

Example scripts in `examples/` illustrate two typical use cases:
- RF-like signals with a variety of modulation types and interferences.
- Synthetic FSK codes with multiple classes.


## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url> iclr_simulator
   cd iclr_simulator
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux / macOS
   # .venv\Scripts\activate         # Windows (PowerShell or CMD)
   ```

3. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

The simulator is implemented in pure Python (NumPy / SciPy), with optional dependencies for downstream processing
(e.g., PyTorch) if you use the generated datasets for deep learning.


## Core API

The main entry point for users is the module:

```python
from iclr_simulator.simulator.multi_res_generation_with_seg import (
    generate_emitters_scenarios,
    generate_and_store_spectrum_multi,
)
```

The typical workflow is:

1. Define the global acquisition parameters and the STFT configurations.
2. Specify the statistical model of the waveforms and interferences via `signal_defs`.
3. Generate a list of *scenarios* using `generate_emitters_scenarios(...)`.
4. Convert scenarios into multi-resolution spectrograms and store them with `generate_and_store_spectrum_multi(...)`.


### 1. Acquisition and STFT configuration

A scenario is defined over a fixed acquisition time, derived from the sampling frequency `F_E`:

```python
F_E = 4e9  # sampling frequency (Hz)
acquisition_time = 2048 * 64 / F_E

# Minimal duration of a pulse (used as a lower bound for waveform durations)
min_time = 128 / F_E

# Nyquist bandwidth
B = F_E / 2

# Multi-resolution STFTs
stft_cfgs = [
    {"nperseg": 512,  "nfft": 512,  "noverlap": 0, "fs": F_E},
    {"nperseg": 256,  "nfft": 256,  "noverlap": 0, "fs": F_E},
    {"nperseg": 128,  "nfft": 128,  "noverlap": 0, "fs": F_E},
    {"nperseg": 1024, "nfft": 1024, "noverlap": 0, "fs": F_E},
    {"nperseg": 2048, "nfft": 2048, "noverlap": 0, "fs": F_E},
]
```

Each dictionary in `stft_cfgs` controls one resolution of the STFT:
- `nperseg`: window length (number of samples).
- `nfft`: FFT size.
- `noverlap`: overlap between consecutive windows.
- `fs`: sampling frequency (should be consistent with `F_E`).

The simulator will generate, for each scenario, one spectrogram per configuration in `stft_cfgs`.


### 2. Signal model: `signal_defs`

The parameter `signal_defs` defines which *waveforms* and *interferences* can appear in a scenario and with which
statistical properties. It is a nested dictionary with two main sections:

```python
signal_defs = {
    "waveforms": {
        "LFM_short": {
            "duration_min": min_time,
            "duration_max": 1e-6,
            "bandwidth_min": 0.005 * B,
            "bandwidth_max": 0.75 * B,
            "p": 1,
        },
        "FSK": {
            "duration_min": min_time,
            "duration_max": acquisition_time,
            "bandwidth_min": 0.005 * B,
            "bandwidth_max": 0.75 * B,
            "p": 1,
        },
        "none": {
            "duration_min": 0.0,
            "duration_max": 0.0,
            "bandwidth_min": 0.0,
            "bandwidth_max": 0.0,
            "p": 0,
        },
    },
    "interferences": {
        "OFDM": {
            "duration_min": acquisition_time,
            "duration_max": acquisition_time,
            "bandwidth_min": 0.005 * B,
            "bandwidth_max": B / 2,
            "p": 1,
        },
        "none": {
            "duration_min": 0.0,
            "duration_max": 0.0,
            "bandwidth_min": 0.0,
            "bandwidth_max": 0.0,
            "p": 0,
        },
    },
}
```

For each waveform or interference type, you specify:
- `duration_min`, `duration_max` (in seconds): range of possible pulse durations.
- `bandwidth_min`, `bandwidth_max` (in Hz): range of possible occupied bandwidths.
- `p`: activation probability (or relative weight) for sampling this type.

Setting `p = 0` effectively disables a given code or interference type while keeping its configuration in the file
for future experiments.


### 3. Scenario generation: `generate_emitters_scenarios`

A *scenario* contains a random number of emitters and interferences, each instantiated according to the distributions
defined in `signal_defs`. A typical call is:

```python
scenarios = generate_emitters_scenarios(
    nb_scenarios=100000,
    snr_range_db=(-20, 20),
    signal_defs=signal_defs,
    stft_cfgs=stft_cfgs,
    max_nb_emitters=7,
    max_nb_interferences=3,
    inr_range_db=(-20, 20),
    seed=444,
)
```

Main parameters:
- `nb_scenarios`: number of independent scenarios to generate.
- `snr_range_db`: range (min, max) of SNR values for the emitters (in dB).
- `signal_defs`: waveform and interference configuration (see above).
- `stft_cfgs`: list of STFT configurations (used for consistency checks and metadata).
- `max_nb_emitters`: maximum number of emitters per scenario.
- `max_nb_interferences`: maximum number of interferences per scenario.
- `inr_range_db`: range of interference-to-noise ratios (in dB).
- `seed`: random seed for reproducibility.

The returned `scenarios` object is a structured list (or similar container) describing, for each scenario:
- The emitters and their waveforms.
- Their time–frequency parameters (e.g., carrier frequency, bandwidth, duration, SNR).
- The interferences and their parameters.


### 4. Dataset export: `generate_and_store_spectrum_multi`

Once scenarios are generated, they are converted into multi-resolution spectrograms and stored on disk:

```python
base_path = "./rf_dataset"
acquisition_time = 2048 * 64 / F_E

generate_and_store_spectrum_multi(
    scenarios=scenarios,
    base_path=base_path,
    split_train_test=True,
    acquisition_time=acquisition_time,
    stft_cfgs=stft_cfgs,
    seed=444,
)
```

Key parameters:
- `scenarios`: list of scenarios produced by `generate_emitters_scenarios`.
- `base_path`: root directory where the dataset will be stored.
- `split_train_test`: if `True`, automatically split into training / test sets.
- `acquisition_time`: acquisition duration (must match the time definition used in the scenarios).
- `stft_cfgs`: list of STFT configurations (must be identical to the one used at scenario generation).
- `class_index_to_name` (optional): dictionary mapping integer labels to waveform names, useful for classification.

The function will:
- Generate time-domain signals for each scenario.
- Compute one spectrogram per STFT configuration.
- Save the spectrograms and the corresponding annotations (e.g., bounding boxes, class labels) to `base_path`,
  in a format compatible with downstream ML pipelines.


## Example 1: RF-like multi-resolution dataset

A typical RF dataset can be created by combining a rich set of modulations and interference types:

```python
from iclr_simulator.simulator.multi_res_generation_with_seg import (
    generate_emitters_scenarios,
    generate_and_store_spectrum_multi,
)

F_E = 4e9
acquisition_time = 2048 * 64 / F_E
min_time = 128 / F_E
B = F_E / 2

signal_defs = {
    "waveforms": {
        "no_mod":            { "duration_min": min_time, "duration_max": 1e-6,
                               "bandwidth_min": 0.0,       "bandwidth_max": 0.0,       "p": 1 },
        "LFM_short":         { "duration_min": min_time, "duration_max": 1e-6,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.75*B,    "p": 1 },
        "random_biphasique": { "duration_min": min_time, "duration_max": 1e-6,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },
        "QPSK":              { "duration_min": min_time, "duration_max": 1e-6,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },
        "QAM16":             { "duration_min": min_time, "duration_max": 1e-6,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },
        "QAM64":             { "duration_min": min_time, "duration_max": 1e-6,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.05*B,    "p": 1 },
        "FSK":               { "duration_min": min_time, "duration_max": acquisition_time,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.75*B,    "p": 1 },
        "LFM_long":          { "duration_min": min_time, "duration_max": acquisition_time,
                               "bandwidth_min": 0.005*B,   "bandwidth_max": 0.9*B,     "p": 1 },
    },
    "interferences": {
        "OFDM": { "duration_min": acquisition_time, "duration_max": acquisition_time,
                  "bandwidth_min": 0.005*B, "bandwidth_max": B/2, "p": 1 },
        "FHSS": { "duration_min": acquisition_time, "duration_max": acquisition_time,
                  "bandwidth_min": 0.005*B, "bandwidth_max": B/2, "p": 1 },
        "DSSS": { "duration_min": acquisition_time, "duration_max": acquisition_time,
                  "bandwidth_min": 0.005*B, "bandwidth_max": B/2, "p": 1 },
        "none": { "duration_min": 0.0, "duration_max": 0.0,
                  "bandwidth_min": 0.0, "bandwidth_max": 0.0, "p": 0 },
    },
}

stft_cfgs = [
    {"nperseg": 512,  "nfft": 512,  "noverlap": 0, "fs": F_E},
    {"nperseg": 256,  "nfft": 256,  "noverlap": 0, "fs": F_E},
    {"nperseg": 128,  "nfft": 128,  "noverlap": 0, "fs": F_E},
    {"nperseg": 1024, "nfft": 1024, "noverlap": 0, "fs": F_E},
    {"nperseg": 2048, "nfft": 2048, "noverlap": 0, "fs": F_E},
]

scenarios = generate_emitters_scenarios(
    nb_scenarios=100000,
    snr_range_db=(-20, 20),
    signal_defs=signal_defs,
    stft_cfgs=stft_cfgs,
    max_nb_emitters=7,
    max_nb_interferences=3,
    inr_range_db=(-20, 20),
    seed=444,
)

generate_and_store_spectrum_multi(
    scenarios=scenarios,
    base_path="./rf_dataset_v2",
    split_train_test=True,
    acquisition_time=acquisition_time,
    stft_cfgs=stft_cfgs,
    seed=444,
)
```


## Example 2: Synthetic FSK code dataset

The same simulator can be used to generate theoretically clean FSK codes, each associated with a class, and observed
through multiple STFT resolutions:

```python
from iclr_simulator.simulator.multi_res_generation_with_seg import (
    generate_emitters_scenarios,
    generate_and_store_spectrum_multi,
)

F_E = 4e9
acquisition_time = 2048 * 64 / F_E
min_time = 128 / F_E
B = F_E / 2

signal_defs = {
    "waveforms": {
        "FSK_CODE1": {
            "duration_min": min_time * 2,
            "duration_max": (2048 / F_E) * 2,
            "bandwidth_min": 0.01 * B,
            "bandwidth_max": 0.01 * B,
            "p": 1,
        },
        "FSK_CODE2": {
            "duration_min": min_time * 2,
            "duration_max": (2048 / F_E) * 2,
            "bandwidth_min": 0.01 * B,
            "bandwidth_max": 0.01 * B,
            "p": 1,
        },
        "FSK_CODE3": {
            "duration_min": min_time * 2,
            "duration_max": (2048 / F_E) * 2,
            "bandwidth_min": 0.03 * B,
            "bandwidth_max": 0.03 * B,
            "p": 1,
        },
        "FSK_CODE4": {
            "duration_min": min_time * 2,
            "duration_max": (2048 / F_E) * 2,
            "bandwidth_min": 0.03 * B,
            "bandwidth_max": 0.03 * B,
            "p": 1,
        },
    },
}

stft_cfgs = [
    {"nperseg": 128,  "nfft": 128,  "noverlap": 0, "fs": F_E},
    {"nperseg": 2048, "nfft": 2048, "noverlap": 0, "fs": F_E},
]

CLASS_INDEX_TO_NAME = {
    0: "FSK_CODE1",
    1: "FSK_CODE2",
    2: "FSK_CODE3",
    3: "FSK_CODE4",
}

scenarios = generate_emitters_scenarios(
    nb_scenarios=10000,
    snr_range_db=(-10, 10),
    signal_defs=signal_defs,
    stft_cfgs=stft_cfgs,
    max_nb_emitters=3,
    max_nb_interferences=0,
    inr_range_db=(-20, 20),
)

generate_and_store_spectrum_multi(
    scenarios=scenarios,
    base_path="./fsk_codes_dataset",
    split_train_test=True,
    acquisition_time=acquisition_time,
    stft_cfgs=stft_cfgs,
    class_index_to_name=CLASS_INDEX_TO_NAME,
)
```

This configuration shows how to:
- Use a subset of waveform types dedicated to FSK codes.
- Fix bandwidths and duration ranges to control the code structure.
- Attach explicit class indices to waveform names for supervised learning.


## Reproducibility and customization

- Set `seed` in both `generate_emitters_scenarios` and `generate_and_store_spectrum_multi` for reproducible datasets.
- Enable or disable waveform / interference types by adjusting their probability `p`.
- Change `stft_cfgs` to control the time–frequency resolutions available in the final dataset.
- Modify SNR / INR ranges to match the difficulty level required by your downstream task.


## License and citation

- Please refer to the `LICENSE` file for licensing information.
- If you use this simulator in academic work, consider citing the associated publication.
