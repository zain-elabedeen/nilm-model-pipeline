# NILM Research

Trains disaggregation models for identifying what's consuming and generating power from a single meter reading.

## Where this fits

**Measure** → Model (this) → Orchestrate → Trade

This repository trains PyTorch models on the [SIDED dataset](https://github.com/ChristianInterno/SIDED) and exports ONNX models for deployment in the [Energy Disaggregator](https://github.com/underscoreHQ/underscore-energy-disaggregator-edge) edge daemon.

## What it produces

ONNX models that disaggregate aggregate power into 5 categories:

| Category | Description | Typical Range |
|----------|-------------|---------------|
| **EVSE** | EV charging | 0 to 22 kW |
| **PV** | Solar generation | -50 to 0 kW |
| **CS** | Cooling systems | 0 to 30 kW |
| **CHP** | Combined heat and power | -20 to 5 kW |
| **BA** | Base appliances | 0 to 10 kW |

## Model architectures

- **TCN** (default): 8-layer Temporal Convolutional Network
- **LSTM**: 3-layer bidirectional LSTM
- **ATCN**: TCN with multi-head self-attention

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Training

```bash
# Default (TCN + AMDA augmentation)
python scripts/train.py

# Specific model
python scripts/train.py model=lstm

# Without AMDA
python scripts/train.py experiment=baseline

# AMDA scaling factor sweep
python scripts/train.py experiment=amda_sweep --multirun
```

## Export to ONNX

```bash
python scripts/export_onnx.py checkpoints/best.ckpt \
    --model-type tcn \
    --output-dir exports \
    --copy-to-rust ../underscore-energy-disaggregator-edge/models
```

The export produces:
- ONNX model file compatible with `tract-onnx`
- Normalisation metadata JSON for the Rust runtime

**Model specification:**
- Input: `[1, 60]` float32 (60-minute window)
- Output: `[1, 5]` float32 (power per category)
- Opset: 17

## AMDA augmentation

AMDA (Appliance Magnitude-aware Data Augmentation) improves generalisation by scaling appliances inversely to their power contribution:

```
S_i = s * (1 - p_i)
```

Where `p_i` is the appliance's relative contribution and `s = 2.5` is the optimal scaling factor from the SIDED paper.

## Evaluation

```bash
python scripts/evaluate.py checkpoints/best.ckpt --model-type tcn
```

## Testing

```bash
pytest tests/ -v
```

## Experiment tracking

Logs to Weights & Biases by default. Disable with:

```bash
python scripts/train.py logging.use_wandb=false
```
