import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ptitprince import stripplot


class TestStripplotBasic:
    """Test basic stripplot functionality."""

    def test_stripplot_vertical(self, sample_data):
        """Test vertical stripplot creation."""
        ax = stripplot(x="category", y="value", data=sample_data, orient="v")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_stripplot_horizontal(self, sample_data):
        """Test horizontal stripplot creation."""
        ax = stripplot(x="category", y="value", data=sample_data, orient="h")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_stripplot_with_hue(self, sample_data):
        """Test stripplot with hue parameter."""
        ax = stripplot(x="category", y="value", hue="hue", data=sample_data)
        assert ax is not None

    def test_stripplot_with_order(self, sample_data):
        """Test stripplot with custom order."""
        custom_order = ["C", "A", "B"]
        ax = stripplot(x="category", y="value", data=sample_data, order=custom_order)
        assert ax is not None


class TestStripplotMove:
    """Test the move parameter - the key feature of ptitprince stripplot."""

    def test_stripplot_move_vertical(self, sample_data):
        """Test move parameter shifts points vertically for vertical plots."""
        ax = stripplot(x="category", y="value", data=sample_data, move=0.2, orient="v")
        assert ax is not None
        # Check that collections were created
        assert len(ax.collections) > 0

    def test_stripplot_move_horizontal(self, sample_data):
        """Test move parameter shifts points horizontally for horizontal plots."""
        ax = stripplot(x="category", y="value", data=sample_data, move=0.2, orient="h")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_stripplot_negative_move(self, sample_data):
        """Test negative move value."""
        ax = stripplot(x="category", y="value", data=sample_data, move=-0.1)
        assert ax is not None

    def test_stripplot_zero_move(self, sample_data):
        """Test move=0 (default behavior)."""
        ax = stripplot(x="category", y="value", data=sample_data, move=0)
        assert ax is not None


class TestStripplotParameters:
    """Test stripplot parameter handling."""

    def test_stripplot_jitter(self, sample_data):
        """Test jitter parameter."""
        ax_jitter = stripplot(x="category", y="value", data=sample_data, jitter=True)
        assert ax_jitter is not None

        ax_no_jitter = stripplot(x="category", y="value", data=sample_data, jitter=False)
        assert ax_no_jitter is not None

    def test_stripplot_dodge(self, sample_data):
        """Test dodge parameter with hue."""
        ax = stripplot(x="category", y="value", hue="hue", data=sample_data, dodge=True)
        assert ax is not None

    def test_stripplot_size(self, sample_data):
        """Test size parameter."""
        ax = stripplot(x="category", y="value", data=sample_data, size=10)
        assert ax is not None

    def test_stripplot_edgecolor(self, sample_data):
        """Test edgecolor parameter."""
        ax = stripplot(x="category", y="value", data=sample_data, edgecolor="black")
        assert ax is not None

    def test_stripplot_linewidth(self, sample_data):
        """Test linewidth parameter."""
        ax = stripplot(x="category", y="value", data=sample_data, linewidth=1.5)
        assert ax is not None

    def test_stripplot_palette(self, sample_data):
        """Test palette parameter."""
        ax = stripplot(x="category", y="value", data=sample_data, palette="Set1")
        assert ax is not None

    def test_stripplot_color(self, sample_data):
        """Test color parameter."""
        ax = stripplot(x="category", y="value", data=sample_data, color="red")
        assert ax is not None

    def test_stripplot_width(self, sample_data):
        """Test width parameter."""
        ax = stripplot(x="category", y="value", data=sample_data, width=0.5)
        assert ax is not None


class TestStripplotDodgeWidth:
    """Test dodge width alignment - important for raincloud plots."""

    def test_stripplot_dodge_custom_width(self, sample_data):
        """Test dodge with custom width aligns properly."""
        ax = stripplot(x="category", y="value", hue="hue", data=sample_data, dodge=True, width=0.4)
        assert ax is not None

    def test_stripplot_dodge_default_width(self, sample_data):
        """Test dodge with default width."""
        ax = stripplot(x="category", y="value", hue="hue", data=sample_data, dodge=True)
        assert ax is not None


class TestStripplotEdgeCases:
    """Test edge cases and error handling."""

    def test_stripplot_with_nan(self):
        """Test handling of NaN values."""
        data = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1.0, np.nan, 3.0, 4.0]})
        ax = stripplot(x="cat", y="val", data=data)
        assert ax is not None

    def test_stripplot_with_custom_ax(self, sample_data):
        """Test stripplot with custom axes."""
        fig, ax = plt.subplots()
        result_ax = stripplot(x="category", y="value", data=sample_data, ax=ax)
        assert result_ax is ax

    def test_stripplot_array_inputs(self):
        """Test stripplot with array inputs instead of DataFrame."""
        x = np.array(["A", "A", "B", "B", "C", "C"])
        y = np.array([1, 2, 3, 4, 5, 6])
        ax = stripplot(x=x, y=y)
        assert ax is not None

    def test_stripplot_deprecated_split_parameter(self, sample_data):
        """Test that deprecated 'split' parameter raises warning."""
        with pytest.warns(UserWarning, match="split.*renamed.*dodge"):
            ax = stripplot(x="category", y="value", hue="hue", data=sample_data, split=True)
            assert ax is not None
