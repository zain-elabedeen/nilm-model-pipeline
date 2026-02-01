# NILM Research

PyTorch research repository for Non-Intrusive Load Monitoring (NILM) based on the SIDED paper, with AMDA data augmentation.

## Overview

This repository trains disaggregation models on the SIDED (Synthetic Industrial Dataset for Energy Disaggregation) and exports ONNX models for production deployment in the Rust digital twins system.

### Appliance Types

| Appliance | Description | Typical Range |
|-----------|-------------|---------------|
| EVSE | Electric Vehicle Charging | 0 to 22 kW |
| PV | Photovoltaic Generation | -50 to 0 kW |
| CS | Cooling Systems | 0 to 30 kW |
| CHP | Combined Heat and Power | -20 to 5 kW |
| BA | Base Appliances | 0 to 10 kW |

### Model Architectures

- **LSTM**: 3-layer bidirectional LSTM (128 hidden units)
- **TCN**: 8-layer Temporal Convolutional Network (128 channels)
- **ATCN**: TCN with multi-head self-attention

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nilm-research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
# Train with default configuration (TCN + AMDA)
python scripts/train.py

# Train specific model
python scripts/train.py model=lstm

# Train without AMDA augmentation
python scripts/train.py experiment=baseline

# AMDA scaling factor sweep
python scripts/train.py experiment=amda_sweep --multirun
```

### Evaluation

```bash
python scripts/evaluate.py checkpoints/best.ckpt --model-type tcn
```

### ONNX Export

```bash
# Export to ONNX for Rust deployment
python scripts/export_onnx.py checkpoints/best.ckpt \
    --model-type tcn \
    --output-dir exports \
    --copy-to-rust ../digital-twins-project/models
```

## Project Structure

```
nilm-research/
├── configs/                 # Hydra configuration
│   ├── config.yaml         # Root config
│   ├── model/              # Model configs
│   ├── data/               # Dataset configs
│   └── experiment/         # Experiment configs
├── src/nilm_research/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   ├── evaluation/         # Metrics
│   └── export/             # ONNX export
├── scripts/                # CLI scripts
├── notebooks/              # Exploration notebooks
└── tests/                  # Unit tests
```

## AMDA Augmentation

AMDA (Appliance Magnitude-aware Data Augmentation) scales appliances inversely to their contribution:

```
p_i = P_total_i / P_total    # Relative contribution
S_i = s * (1 - p_i)          # Scaling factor
x̃_i = S_i * x_i             # Scaled signal
```

The optimal scaling factor is `s = 2.5` (from the SIDED paper).

## Rust Integration

The exported ONNX models are designed to work with `tract-onnx` in the Rust production system:

- **Input**: `[1, 60]` float32 (60-minute window at 1-minute resolution)
- **Output**: `[1, 5]` float32 (EVSE, PV, CS, CHP, BA)
- **Opset**: 17

The export includes normalisation metadata (`*_config.json`) that aligns with the Rust `NormalisationParams` structure.

## Testing

```bash
pytest tests/ -v
```

## Experiment Tracking

Training logs to Weights & Biases by default. Set `logging.use_wandb=false` to disable.

```bash
# Offline mode
python scripts/train.py logging.wandb_offline=true
```
