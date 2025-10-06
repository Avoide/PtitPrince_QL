"""Integration tests using notebook execution."""

import subprocess

import pytest


class TestNotebookExecution:
    """Test that example notebooks execute without errors."""

    @pytest.mark.slow
    def test_raincloud_plot_notebook(self):
        """Test RainCloud_Plot.ipynb executes successfully."""
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--ExecutePreprocessor.kernel_name=python3",
                "--execute",
                "RainCloud_Plot.ipynb",
                "--output",
                "/tmp/test_raincloud.ipynb",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Notebook execution failed: {result.stderr}"

    @pytest.mark.slow
    def test_tutorial_notebook(self):
        """Test tutorial_python/raincloud_tutorial_python.ipynb executes successfully."""
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--ExecutePreprocessor.kernel_name=python3",
                "--execute",
                "tutorial_python/raincloud_tutorial_python.ipynb",
                "--output",
                "/tmp/test_tutorial.ipynb",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Notebook execution failed: {result.stderr}"
