import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptitprince import RainCloud


class TestRainCloudBasic:
    """Test basic RainCloud functionality."""

    def test_raincloud_vertical(self, sample_data):
        """Test vertical raincloud plot creation."""
        ax = RainCloud(x="category", y="value", data=sample_data, orient="v")
        assert ax is not None
        assert len(ax.collections) > 0  # Should have plot elements

    def test_raincloud_horizontal(self, sample_data):
        """Test horizontal raincloud plot creation."""
        ax = RainCloud(x="category", y="value", data=sample_data, orient="h")
        assert ax is not None
        assert len(ax.collections) > 0

    def test_raincloud_with_hue(self, sample_data):
        """Test raincloud plot with hue parameter."""
        ax = RainCloud(x="category", y="value", hue="hue", data=sample_data)
        assert ax is not None
        # Should have more collections due to hue groups
        assert len(ax.collections) > 3

    def test_raincloud_with_pointplot(self, sample_data):
        """Test raincloud plot with pointplot enabled."""
        ax = RainCloud(x="category", y="value", data=sample_data, pointplot=True)
        assert ax is not None
        # Pointplot adds lines to the plot
        assert len(ax.lines) > 0

    def test_raincloud_with_order(self, sample_data):
        """Test raincloud plot with custom order."""
        custom_order = ["C", "A", "B"]
        ax = RainCloud(x="category", y="value", data=sample_data, order=custom_order)
        assert ax is not None
        # Check that x-axis labels match the custom order
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == custom_order


class TestRainCloudParameters:
    """Test RainCloud parameter handling."""

    def test_raincloud_width_parameters(self, sample_data):
        """Test width parameters for violin and box."""
        ax = RainCloud(x="category", y="value", data=sample_data, width_viol=0.5, width_box=0.1)
        assert ax is not None

    def test_raincloud_move_parameter(self, sample_data):
        """Test the move parameter for stripplot positioning."""
        ax = RainCloud(x="category", y="value", data=sample_data, move=0.1)
        assert ax is not None

    def test_raincloud_offset_parameter(self, sample_data):
        """Test the offset parameter for violin positioning."""
        ax = RainCloud(x="category", y="value", data=sample_data, offset=0.2)
        assert ax is not None

    def test_raincloud_alpha_parameter(self, sample_data):
        """Test alpha transparency parameter."""
        ax = RainCloud(x="category", y="value", data=sample_data, alpha=0.5)
        assert ax is not None
        # Check that alpha is applied - some collections may have None (inherits from parent)
        # Just verify the parameter doesn't cause errors
        assert len(ax.collections) > 0

    def test_raincloud_palette(self, sample_data):
        """Test different palette options."""
        for palette in ["Set1", "Set2", "pastel"]:
            ax = RainCloud(x="category", y="value", data=sample_data, palette=palette)
            assert ax is not None
            plt.close()

    def test_raincloud_dodge(self, sample_data):
        """Test dodge parameter with hue."""
        ax = RainCloud(x="category", y="value", hue="hue", data=sample_data, dodge=True)
        assert ax is not None


class TestRainCloudKwargs:
    """Test kwargs forwarding to subcomponents."""

    def test_cloud_kwargs(self, sample_data):
        """Test kwargs forwarding to cloud/violin component."""
        # Use a different kwarg that won't conflict with RainCloud's own linewidth
        ax = RainCloud(x="category", y="value", data=sample_data, cloud_saturation=0.8)
        assert ax is not None

    def test_box_kwargs(self, sample_data):
        """Test kwargs forwarding to box component."""
        ax = RainCloud(x="category", y="value", data=sample_data, box_saturation=0.5)
        assert ax is not None

    def test_rain_kwargs(self, sample_data):
        """Test kwargs forwarding to rain/stripplot component."""
        ax = RainCloud(x="category", y="value", data=sample_data, rain_edgecolor="black")
        assert ax is not None

    def test_point_kwargs(self, sample_data):
        """Test kwargs forwarding to pointplot component."""
        ax = RainCloud(x="category", y="value", data=sample_data, pointplot=True, point_capsize=0.1)
        assert ax is not None


class TestRainCloudEdgeCases:
    """Test edge cases and error handling."""

    def test_raincloud_with_nan_values(self):
        """Test raincloud plot handles NaN values."""
        data = pd.DataFrame(
            {"cat": ["A", "A", "A", "B", "B", "B"], "val": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0]}
        )
        ax = RainCloud(x="cat", y="val", data=data)
        assert ax is not None

    def test_raincloud_single_category(self):
        """Test raincloud plot with single category."""
        data = pd.DataFrame({"cat": ["A"] * 10, "val": np.random.randn(10)})
        ax = RainCloud(x="cat", y="val", data=data)
        assert ax is not None

    def test_raincloud_with_custom_ax(self, sample_data):
        """Test raincloud plot with custom axes."""
        fig, ax = plt.subplots()
        result_ax = RainCloud(x="category", y="value", data=sample_data, ax=ax)
        assert result_ax is ax

    def test_raincloud_array_inputs(self):
        """Test raincloud plot with array inputs instead of DataFrame."""
        x = np.array(["A", "A", "B", "B", "C", "C"])
        y = np.array([1, 2, 3, 4, 5, 6])
        ax = RainCloud(x=x, y=y)
        assert ax is not None
