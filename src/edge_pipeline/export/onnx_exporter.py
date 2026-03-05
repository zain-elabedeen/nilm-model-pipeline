"""ONNX export for production deployment in Rust.

Exports trained PyTorch models to ONNX format compatible with tract-onnx.
"""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from edge_pipeline.models.base import NilmModel


class OnnxExporter:
    """
    Export NILM models to ONNX format.

    Ensures compatibility with the Rust production system using tract-onnx.
    Key requirements:
    - Input: [1, 60] float32 (1-minute resolution, 60-minute window)
    - Output: [1, 5] float32 (Battery, Solar, Cooling, Generator, Base Load)
    - Opset version: 17 (tract-onnx compatible)
    """

    OPSET_VERSION = 17
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"

    def __init__(
        self,
        model: NilmModel,
        output_dir: str | Path = "exports",
        model_name: str = "nilm",
    ):
        """
        Initialise exporter.

        Args:
            model: Trained NILM model
            output_dir: Directory for exported files
            model_name: Base name for exported files
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def export(
        self,
        normalisation_metadata: dict | None = None,
        validate: bool = True,
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            normalisation_metadata: Optional normalisation parameters from DataModule
            validate: Whether to validate the exported model

        Returns:
            Path to exported ONNX file
        """
        # Export on CPU to avoid device mismatch when a trained model stays on CUDA.
        # Restore original state afterwards so callers can continue using the model.
        try:
            original_device = next(self.model.parameters()).device
        except StopIteration:
            original_device = torch.device("cpu")
        original_mode_training = self.model.training

        self.model.eval()
        self.model.to("cpu")

        try:
            # Get example input
            example_input = self.model.get_example_input().to("cpu")
            window_size = example_input.shape[1]

            # Export path
            onnx_path = self.output_dir / f"{self.model_name}.onnx"

            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                str(onnx_path),
                input_names=[self.INPUT_NAME],
                output_names=[self.OUTPUT_NAME],
                dynamic_axes=None,  # Fixed batch size of 1
                opset_version=self.OPSET_VERSION,
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )

            # Add metadata to ONNX model
            onnx_model = onnx.load(str(onnx_path))

            # Add model metadata
            metadata = {
                "model_type": self.model.__class__.__name__,
                "window_size": window_size,
                "num_appliances": self.model.num_appliances,
                "appliances": getattr(
                    self.model, "appliance_names",
                    ["BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"],
                ),
                "version": "1.0.0",
            }

            for key, value in metadata.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)

            onnx.save(onnx_model, str(onnx_path))

            # Export normalisation metadata
            if normalisation_metadata is not None:
                metadata_path = self.output_dir / f"{self.model_name}_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(normalisation_metadata, f, indent=2)

            # Validate
            if validate:
                self._validate(onnx_path, example_input)

            return onnx_path
        finally:
            # Restore model state/device
            self.model.to(original_device)
            if original_mode_training:
                self.model.train()
            else:
                self.model.eval()

    def _validate(self, onnx_path: Path, example_input: torch.Tensor) -> None:
        """Validate exported ONNX model."""
        # Check ONNX model validity
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Compare PyTorch and ONNX outputs
        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(example_input).numpy()

        # Run ONNX inference
        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(
            [self.OUTPUT_NAME],
            {self.INPUT_NAME: example_input.numpy()},
        )[0]

        # Check output shapes
        assert pytorch_output.shape == onnx_output.shape, (
            f"Shape mismatch: PyTorch {pytorch_output.shape} vs ONNX {onnx_output.shape}"
        )

        # Check numerical equivalence
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        assert max_diff < 1e-5, f"Output difference too large: {max_diff}"

        print(f"ONNX validation passed:")
        print(f"  Input shape: {example_input.shape}")
        print(f"  Output shape: {onnx_output.shape}")
        print(f"  Max difference: {max_diff:.2e}")

    def export_for_rust(
        self,
        normalisation_metadata: dict | None = None,
    ) -> dict[str, Path]:
        """
        Export model with all files needed for Rust integration.

        Returns:
            Dict mapping file type to path
        """
        paths = {}

        # Export ONNX model
        paths["onnx"] = self.export(normalisation_metadata, validate=True)

        # Export normalisation config for Rust
        if normalisation_metadata is not None:
            rust_config = self._generate_rust_config(normalisation_metadata)
            config_path = self.output_dir / f"{self.model_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(rust_config, f, indent=2)
            paths["config"] = config_path

        return paths

    def _generate_rust_config(self, metadata: dict) -> dict:
        """Generate configuration matching Rust NormalisationParams."""
        config = {
            "window_size": metadata.get("window_size", 60),
            "input_normalisation": None,
            "output_normalisations": {},
        }

        # Input normalisation
        if metadata.get("input"):
            inp = metadata["input"]
            config["input_normalisation"] = {
                "mean": inp["mean"],
                "std_dev": inp["std"],
                "min": inp["min"],
                "max": inp["max"],
            }

        # Output normalisations (per appliance)
        if metadata.get("outputs"):
            for name, params in metadata["outputs"].items():
                config["output_normalisations"][name] = {
                    "mean": params["mean"],
                    "std_dev": params["std"],
                    "min": params["min"],
                    "max": params["max"],
                }

        return config


def export_model(
    checkpoint_path: str | Path,
    model_class: type[NilmModel],
    output_dir: str | Path = "exports",
    model_name: str = "nilm",
) -> Path:
    """
    Convenience function to export a model from checkpoint.

    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint
        model_class: Model class to instantiate
        output_dir: Output directory
        model_name: Output file name

    Returns:
        Path to exported ONNX file
    """
    # Load model from checkpoint
    model = model_class.load_from_checkpoint(checkpoint_path)

    # Export
    exporter = OnnxExporter(model, output_dir, model_name)
    return exporter.export()
