import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ptitprince import half_violinplot


class TestHalfViolinBasic:
    """Test basic half_violinplot functionality."""

    def test_half_violin_vertical(self, sample_data):
        """Test vertical half violin plot creation."""
        ax = half_violinplot(x="category", y="value", data=sample_data, orient="v")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_half_violin_horizontal(self, sample_data):
        """Test horizontal half violin plot creation."""
        # For horizontal orient, need to swap x and y
        ax = half_violinplot(x="value", y="category", data=sample_data, orient="h")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_half_violin_with_hue(self, sample_data):
        """Test half violin plot with hue parameter."""
        ax = half_violinplot(x="category", y="value", hue="hue", data=sample_data)
        assert ax is not None

    def test_half_violin_with_order(self, sample_data):
        """Test half violin plot with custom order."""
        custom_order = ["C", "A", "B"]
        ax = half_violinplot(x="category", y="value", data=sample_data, order=custom_order)
        assert ax is not None


class TestHalfViolinParameters:
    """Test half_violinplot parameter handling."""

    def test_half_violin_bandwidth(self, sample_data):
        """Test different bandwidth options."""
        for bw in ["scott", "silverman", 0.2]:
            ax = half_violinplot(x="category", y="value", data=sample_data, bw=bw)
            assert ax is not None
            plt.close()

    def test_half_violin_cut(self, sample_data):
        """Test cut parameter."""
        for cut in [0, 1, 2, 3]:
            ax = half_violinplot(x="category", y="value", data=sample_data, cut=cut)
            assert ax is not None
            plt.close()

    def test_half_violin_scale(self, sample_data):
        """Test different scale options."""
        for scale in ["area", "width", "count"]:
            ax = half_violinplot(x="category", y="value", data=sample_data, scale=scale)
            assert ax is not None
            plt.close()

    def test_half_violin_scale_hue(self, sample_data):
        """Test scale_hue parameter."""
        ax = half_violinplot(x="category", y="value", hue="hue", data=sample_data, scale_hue=True)
        assert ax is not None

        ax = half_violinplot(x="category", y="value", hue="hue", data=sample_data, scale_hue=False)
        assert ax is not None

    def test_half_violin_width(self, sample_data):
        """Test width parameter."""
        ax = half_violinplot(x="category", y="value", data=sample_data, width=0.5)
        assert ax is not None

    def test_half_violin_offset(self, sample_data):
        """Test offset parameter."""
        ax = half_violinplot(x="category", y="value", data=sample_data, offset=0.2)
        assert ax is not None

    def test_half_violin_inner_styles(self, sample_data):
        """Test different inner styles."""
        for inner in ["box", "quartile", "stick", "point", None]:
            ax = half_violinplot(x="category", y="value", data=sample_data, inner=inner)
            assert ax is not None
            plt.close()

    def test_half_violin_split(self, sample_data):
        """Test split parameter with hue."""
        ax = half_violinplot(x="category", y="value", hue="hue", data=sample_data, split=True)
        assert ax is not None


class TestHalfViolinEdgeCases:
    """Test edge cases and error handling."""

    def test_half_violin_with_nan(self):
        """Test handling of NaN values."""
        data = pd.DataFrame({"cat": ["A", "A", "B", "B"], "val": [1.0, np.nan, 3.0, 4.0]})
        ax = half_violinplot(x="cat", y="val", data=data)
        assert ax is not None

    def test_half_violin_single_value_per_category(self):
        """Test handling of single value per category."""
        data = pd.DataFrame({"cat": ["A", "B", "C"], "val": [1.0, 2.0, 3.0]})
        ax = half_violinplot(x="cat", y="val", data=data)
        assert ax is not None

    def test_half_violin_identical_values(self):
        """Test handling of identical values in a category."""
        data = pd.DataFrame(
            {"cat": ["A", "A", "A", "B", "B", "B"], "val": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]}
        )
        ax = half_violinplot(x="cat", y="val", data=data)
        assert ax is not None

    def test_half_violin_invalid_inner_style(self, sample_data):
        """Test invalid inner style raises ValueError."""
        with pytest.raises(ValueError, match="Inner style.*not recognized"):
            half_violinplot(x="category", y="value", data=sample_data, inner="invalid")

    def test_half_violin_split_without_enough_hue_levels(self):
        """Test split=True without enough hue levels raises ValueError."""
        data = pd.DataFrame(
            {
                "cat": ["A", "A", "B", "B"],
                "val": [1.0, 2.0, 3.0, 4.0],
                "hue": ["X", "X", "X", "X"],  # Only one hue level
            }
        )
        with pytest.raises(ValueError, match="at least two hue levels"):
            half_violinplot(x="cat", y="val", hue="hue", data=data, split=True)

    def test_half_violin_with_palette(self, sample_data):
        """Test palette parameter."""
        ax = half_violinplot(x="category", y="value", data=sample_data, palette="Set2")
        assert ax is not None

    def test_half_violin_with_color(self, sample_data):
        """Test color parameter."""
        ax = half_violinplot(x="category", y="value", data=sample_data, color="red")
        assert ax is not None
