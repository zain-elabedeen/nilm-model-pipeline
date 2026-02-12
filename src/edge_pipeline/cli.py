"""Command-line interface entry points."""

import sys


def train():
    """Entry point for edge-train command."""
    from scripts.train import main

    sys.exit(main())


def evaluate():
    """Entry point for edge-eval command."""
    from scripts.evaluate import main

    main()


def export_onnx():
    """Entry point for edge-export command."""
    from scripts.export_onnx import main

    main()
