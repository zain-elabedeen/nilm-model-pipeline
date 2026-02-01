"""Command-line interface entry points."""

import sys


def train():
    """Entry point for nilm-train command."""
    from scripts.train import main

    sys.exit(main())


def evaluate():
    """Entry point for nilm-eval command."""
    from scripts.evaluate import main

    main()


def export_onnx():
    """Entry point for nilm-export command."""
    from scripts.export_onnx import main

    main()
