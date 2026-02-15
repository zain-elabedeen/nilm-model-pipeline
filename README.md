# Underscore Edge Model Pipeline

Trains disaggregation models for identifying what's consuming and generating power from a single meter reading.

## Where this fits

**Measure** -> Model (this) -> Orchestrate -> Trade

This repository trains PyTorch models and exports ONNX models for deployment in the [Edge Runtime](https://github.com/underscoreHQ/underscore-edge-runtime) daemon. It ships with a built-in loader for the [SIDED dataset](https://huggingface.co/datasets/CInterno/Synthetic_Industrial_Dataset_For_Energy_Disaggregation_SIDED) and supports custom datasets via CSV.

## What it produces

ONNX models that disaggregate aggregate power into per-appliance estimates. The number and names of appliance categories are determined by the dataset loader.

The built-in SIDED loader maps to these categories:

| Category | Description | Typical Range |
|----------|-------------|---------------|
| **BATTERY** | Battery storage | -50 to 50 kW |
| **SOLAR** | Solar generation | -100 to 0 kW |
| **COOLING** | Cooling systems | 0 to 100 kW |
| **GENERATOR** | Diesel/gas generator | -500 to 0 kW |
| **BASE_LOAD** | Factory production machinery | 0 to 200 kW |

Custom datasets define their own appliance categories.

## Model architectures

- **TCN** (default): 8-layer Temporal Convolutional Network
- **LSTM**: 3-layer bidirectional LSTM

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Training

To train on Google Colab without any local setup, use the notebook at `notebooks/colab_training.ipynb`.

If you have a local GPU machine, you can run a local Colab runtime and connect the hosted Colab UI to it:

```bash
docker run --gpus all -p 127.0.0.1:9000:8080 \
  -e GITHUB_TOKEN=ghp_... \
  us-docker.pkg.dev/colab-images/public/runtime
```

Then in Colab, go to **Connect to a local runtime** and enter `http://localhost:9000`. The notebook detects whether it is running on hosted or local Colab and adjusts token retrieval and artefact download accordingly.

```bash
# Default (TCN on SIDED + AMDA augmentation)
python scripts/train.py

# Specific model
python scripts/train.py model=lstm

# Without AMDA
python scripts/train.py experiment=baseline

# AMDA scaling factor sweep
python scripts/train.py experiment=amda_sweep --multirun
```

### Custom CSV datasets

Place one CSV file per site in a directory. Each CSV must have an aggregate power column and one or more appliance columns. Appliance columns are auto-detected from the headers.

```bash
python scripts/train.py data=csv data.data_dir=./my_data
```

See `configs/data/csv.yaml` for all available options.

## Export to ONNX

```bash
python scripts/export_onnx.py checkpoints/best.ckpt \
    --model-type tcn \
    --output-dir exports \
    --copy-to-rust ../underscore-edge-runtime/models
```

The export produces:
- ONNX model file compatible with `tract-onnx`
- Normalisation metadata JSON for the Rust runtime

**Model specification:**
- Input: `[1, window_size]` float32 (default: 60-minute window)
- Output: `[1, num_appliances]` float32 (power per category)
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
uv run --extra dev pytest tests/ -v
```

## Experiment tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai/). Since models are deployed to edge devices rather than cloud endpoints, traditional MLOps platforms (Vertex AI, SageMaker) are not a good fit. W&B provides experiment comparison, loss visualisation, and hyperparameter sweep tracking without the production-serving infrastructure we don't need.

W&B logging is disabled by default. Enable it with:

```bash
python scripts/train.py logging.use_wandb=true
```
