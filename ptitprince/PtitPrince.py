from __future__ import division

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from seaborn.categorical import *
from seaborn.categorical import _CategoricalPlotter#, _CategoricalScatterPlotter

__all__ = ["half_violinplot", "stripplot", "RainCloud"]
__version__ = '0.2.7'


class _Half_ViolinPlotter(_CategoricalPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 bw, cut, scale, scale_hue, gridsize,
                 width, inner, split, dodge, orient, linewidth,
                 color, palette, saturation, offset):

        variables = dict(x=x, y=y, hue=hue)

        super().__init__(
            data=data,
            variables=variables,
            order=order,
            orient=orient,
            color=color,
        )
    
        self.map_hue(palette=palette, order=hue_order, saturation=saturation)

        self.bw = bw
        self.cut = cut
        self.scale = scale
        self.scale_hue = scale_hue
        self.gridsize = gridsize
        self.width = width
        self.dodge = dodge
        self.offset = offset

        if inner is not None:
            if not any([inner.startswith("quart"),
                        inner.startswith("box"),
                        inner.startswith("stick"),
                        inner.startswith("point")]):
                err = "Inner style '{}' not recognized".format(inner)
                raise ValueError(err)
        self.inner = inner

        if split and "hue" in self.variables and len(self.var_levels.get('hue', [])) < 2:
            msg = "There must be at least two hue levels to use `split`.'"
            raise ValueError(msg)
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

        self.gray = self._complement_color("auto", color, self._hue_map)
        # Fallback to explicit gray if complement color is not working
        if self.gray is None or self.gray == "none":
            self.gray = "0.3"

    def estimate_densities(self, bw, cut, scale, scale_hue, gridsize):
        """Find the support and density for all of the data."""
        # Initialize data structures to keep track of plotting data
        violin_data = []
        
        # In the modern seaborn structure, the orient might be the actual axis name
        # Let's determine based on which variable is categorical vs numeric
        if "x" in self.variables and "y" in self.variables:
            # Determine orientation: for vertical plots x=categorical, y=numeric
            # For horizontal plots x=numeric, y=categorical
            # Note: modern seaborn sets self.orient to 'y' for horizontal plots, 'x' for vertical
            if self.orient == "y":
                # Horizontal: x=numeric (values), y=categorical (groups)
                value_variable = "x"
                categorical_variable = "y"
            else:
                # Vertical: x=categorical (groups), y=numeric (values)
                value_variable = "y"
                categorical_variable = "x"
        else:
            # Fallback to original logic
            value_variable = "y" if self.orient == "v" else "x"
            categorical_variable = "x" if self.orient == "v" else "y"
        grouping_vars = [categorical_variable]  # e.g., ['x']
        if "hue" in self.variables:
            grouping_vars.append("hue")

        for group_name, group_df in self.plot_data.groupby(grouping_vars):
            # Extract numeric values and remove NaN values
            values = group_df[value_variable]
            kde_data = values.dropna()

            # Handle edge cases for this specific violin
            if kde_data.size == 0:
                support_i = np.array([])
                density_i = np.array([1.])
            elif np.unique(kde_data).size == 1:
                support_i = np.unique(kde_data)
                density_i = np.array([1.])
            else:
                # Fit the KDE for this violin's data
                kde, bw_used = self.fit_kde(kde_data, bw)
                support_i = self.kde_support(kde_data, bw_used, cut, gridsize)
                density_i = kde.evaluate(support_i)
            
            # 4. Store all results for this one violin in a dictionary
            violin_data.append({
                "group_name": group_name,
                "support": support_i,
                "density": density_i,
                "observations": kde_data,
                "max_density": density_i.max() if density_i.size > 1 else 0,
                "count": kde_data.size,
            })

        # 5. Store the complete results list first
        self.violin_data = violin_data

        # 6. Apply scaling
        if scale == "area":
            self.scale_area(scale_hue)

        elif scale == "width":
            self.scale_width

        elif scale == "count":
            self.scale_count(scale_hue)

        else:
            raise ValueError("scale method '{}' not recognized".format(scale))

    def fit_kde(self, x, bw):
        """Estimate a KDE for a vector of data with flexible bandwidth."""
        # Ensure x is a numpy array of floats
        x = np.asarray(x, dtype=float)

        # Allow for the use of old scipy where `bw` is fixed
        try:
            kde = stats.gaussian_kde(x, bw)
        except TypeError:
            kde = stats.gaussian_kde(x)
            if bw != "scott":  # scipy default
                msg = ("Ignoring bandwidth choice, "
                       "please upgrade scipy to use a different bandwidth.")
                warnings.warn(msg, UserWarning)

        # Extract the numeric bandwidth from the KDE object
        bw_used = kde.factor

        # At this point, bw will be a numeric scale factor.
        # To get the actual bandwidth of the kernel, we multiple by the
        # unbiased standard deviation of the data, which we will use
        # elsewhere to compute the range of the support.
        bw_used = bw_used * x.std(ddof=1)

        return kde, bw_used

    def kde_support(self, x, bw, cut, gridsize):
        """Define a grid of support for the violin."""
        support_min = x.min() - bw * cut
        support_max = x.max() + bw * cut
        return np.linspace(support_min, support_max, gridsize)

    def scale_area(self, scale_hue):
        """Scale the densities in self.violin_data to preserve area."""

        # First, find the overall maximum density if we need it
        global_max_density = 1
        if not scale_hue:
            # Get a list of all max_densities and find the true global maximum
            all_max_densities = [d["max_density"] for d in self.violin_data]
            if all_max_densities:
                global_max_density = max(all_max_densities)

        # If scaling by hue, we need to find the max density within each category
        if "hue" in self.variables and scale_hue:
            # Use pandas to quickly find the max density for each x-category
            import pandas as pd
            df = pd.DataFrame(self.violin_data)
            # The group_name is a tuple like ('Sad', 'Friend'), self.orient is 'x'
            df['orient_cat'] = [name[0] if isinstance(name, tuple) else name for name in df['group_name']]
            category_maxes = df.groupby('orient_cat')['max_density'].transform('max')

        # Now, loop through the list of dictionaries and update the density
        for i, violin in enumerate(self.violin_data):
            density = violin["density"]
            if density.size <= 1:
                continue

            if "hue" not in self.variables:
                # Case 1: No hue, scale by the global max
                scaler = max([d["max_density"] for d in self.violin_data])
            elif scale_hue:
                # Case 2: Hue is present, scale within each x-category
                scaler = category_maxes[i]
            else:
                # Case 3: Hue is present, but scale by the global max
                scaler = global_max_density

            if scaler > 0:
                violin["density"] /= scaler

    def scale_width(self, density):
        """Scale each density curve to the same height."""
        if "hue" not in self.variables:
            for d in density:
                d /= d.max()
        else:
            for group in density:
                for d in group:
                    d /= d.max()

    def scale_count(self, scale_hue):
        """Scale each density curve by observation count in self.violin_data."""

        # --- 1. Find the maximum counts needed for scaling ---

        # Get a list of all counts to find the global maximum
        all_counts = [d["count"] for d in self.violin_data]
        global_max_count = max(all_counts) if all_counts else 1

        # If scaling by hue, find the max count within each primary category
        if "hue" in self.variables and scale_hue:
            import pandas as pd
            df = pd.DataFrame(self.violin_data)
            # The group_name can be a tuple like ('Sad', 'Friend') or just 'Sad'
            # We extract the first element to get the primary category
            df['orient_cat'] = [
                name[0] if isinstance(name, tuple) else name for name in df['group_name']
            ]
            # Use pandas `transform` to get the max count for each violin's category
            category_max_counts = df.groupby('orient_cat')['count'].transform('max')

        # --- 2. Loop through violins and apply the scaling ---

        for i, violin in enumerate(self.violin_data):
            density = violin["density"]
            max_density = violin["max_density"]
            count = violin["count"]

            # First, normalize the violin to its own max height of 1
            if max_density > 0:
                normalized_density = density / max_density
            else:
                normalized_density = density

            # Next, determine the scaler based on observation counts
            if "hue" not in self.variables:
                # Case 1: No hue, scale by the global max count
                scaler = count / global_max_count
            elif scale_hue:
                # Case 2: Hue is present, scale within each x-category
                max_count_in_category = category_max_counts[i]
                scaler = count / max_count_in_category if max_count_in_category > 0 else 0
            else:
                # Case 3: Hue is present, but scale by the global max count
                scaler = count / global_max_count

            # Apply the final scaling
            violin["density"] = normalized_density * scaler

    @property
    def scale_width(self):
        """Scale each density curve to the same height in self.violin_data."""
        for violin in self.violin_data:
            density = violin["density"]
            max_density = violin["max_density"]

            # Normalize the density by its own maximum to make the new max 1
            if max_density > 0:
                violin["density"] = density / max_density

    def draw_violins(self, ax, kws):
        """Draw the violins onto `ax`."""
        # Determine correct orientation based on variable types
        # For vertical plots: categorical on x-axis, numeric on y-axis -> use fill_betweenx
        # For horizontal plots: numeric on x-axis, categorical on y-axis -> use fill_between
        if "x" in self.variables and "y" in self.variables:
            if self.orient == "y":
                # Horizontal: x=numeric, y=categorical -> use fill_between
                fill_func = ax.fill_between
            else:
                # Vertical: x=categorical, y=numeric -> use fill_betweenx
                fill_func = ax.fill_betweenx
        else:
            # Fallback to original logic
            fill_func = ax.fill_betweenx if self.orient == "v" else ax.fill_between

        # Set up default drawing properties
        kws.update(dict(edgecolor=self.gray, linewidth=self.linewidth))

        # Calculate width for violin drawing
        if not hasattr(self, 'dwidth'):
            self.dwidth = self.width / 2

        # Single loop through violin_data - handles both hue and no-hue cases
        for violin in self.violin_data:
            support = violin["support"]
            density = violin["density"]
            observations = violin["observations"]
            group_name = violin["group_name"]

            # Handle special case of no observations
            if support.size == 0:
                continue

            # Handle special case of a single observation
            if support.size == 1:
                val = np.ndarray.item(support)
                d = np.ndarray.item(density)
                center = self._get_center_position(group_name)
                if self.split and "hue" in self.variables:
                    d = d / 2
                self.draw_single_observation(ax, center, val, d)
                continue

            # Get center position and color for this violin
            center = self._get_center_position(group_name)
            color = self._get_violin_color(group_name)

            # Draw the violin polygon
            grid = np.ones(len(support)) * center

            if self.split and "hue" in self.variables:
                # Split violins: determine which side based on hue level
                hue_level = group_name[1] if isinstance(group_name, tuple) else None
                hue_levels = self.var_levels.get('hue', [])
                hue_idx = list(hue_levels).index(hue_level) if hue_level in hue_levels else 0

                if hue_idx == 0:  # Left side
                    fill_func(support,
                              -self.offset + grid - density * self.dwidth,
                              -self.offset + grid,
                              facecolor=color, **kws)
                else:  # Right side
                    fill_func(support,
                              -self.offset + grid,
                              -self.offset + grid + density * self.dwidth,
                              facecolor=color, **kws)
            else:
                # Half violin (left side only) - this is the core feature of half violins
                # The violin should extend from the center to the left, with offset applied
                fill_func(support,
                          -self.offset + grid - density * self.dwidth,
                          -self.offset + grid,
                          facecolor=color, **kws)

            # Add legend data for hue plots (but only once per hue level)
            if "hue" in self.variables and isinstance(group_name, tuple):
                hue_level = group_name[1]
                category_level = group_name[0]
                # Add legend only for the first category of each hue level
                # Determine categorical variable - typically x for vertical violins
                if "x" in self.variables and "y" in self.variables:
                    categorical_var = "x"
                else:
                    categorical_var = "x" if self.orient == "v" else "y"
                if category_level == self.var_levels[categorical_var][0]:
                    self.add_legend_data(ax, color, hue_level)

            # Draw the interior representation of the data
            if self.inner is not None:
                self._draw_violin_interior(ax, violin, center)

    def _get_center_position(self, group_name):
        """Get the numeric position on the categorical axis for a violin."""
        # Extract the primary category from group_name
        if "hue" in self.variables and isinstance(group_name, tuple) and len(group_name) >= 2:
            # With hue: group_name is like ('CategoryA', 'HueLevel1')
            primary_cat = group_name[0]
            hue_level = group_name[1]

            # Get base position
            # Determine categorical variable based on orientation
            if "x" in self.variables and "y" in self.variables:
                if self.orient == "y":
                    # Horizontal: y is categorical
                    categorical_var = "y"
                else:
                    # Vertical: x is categorical
                    categorical_var = "x"
            else:
                categorical_var = "x" if self.orient == "v" else "y"
            base_pos = list(self.var_levels[categorical_var]).index(primary_cat)

            # Add hue offset if not splitting
            if not self.split:
                hue_levels = self.var_levels['hue']
                hue_idx = list(hue_levels).index(hue_level)
                # Calculate offset based on hue position
                n_hues = len(hue_levels)
                offset_range = 0.8 * self.width / n_hues
                center_offset = (hue_idx - (n_hues - 1) / 2) * offset_range
                return base_pos + center_offset
            else:
                return base_pos
        else:
            # No hue: group_name is just the category
            if isinstance(group_name, tuple):
                primary_cat = group_name[0]
            else:
                primary_cat = group_name
            # Determine categorical variable based on orientation
            if "x" in self.variables and "y" in self.variables:
                if self.orient == "y":
                    # Horizontal: y is categorical
                    categorical_var = "y"
                else:
                    # Vertical: x is categorical
                    categorical_var = "x"
            else:
                categorical_var = "x" if self.orient == "v" else "y"
            return list(self.var_levels[categorical_var]).index(primary_cat)

    def _get_violin_color(self, group_name):
        """Get the color for a violin based on its group name."""
        if "hue" in self.variables and isinstance(group_name, tuple) and len(group_name) >= 2:
            # With hue: use hue mapping
            hue_level = group_name[1]
            return self._hue_map(hue_level)
        else:
            # No hue: use default color or create a simple palette
            # For modern seaborn, we need to create colors manually
            import seaborn as sns
            if isinstance(group_name, tuple):
                primary_cat = group_name[0]
            else:
                primary_cat = group_name

            # Determine categorical variable based on orientation
            if "x" in self.variables and "y" in self.variables:
                if self.orient == "y":
                    # Horizontal: y is categorical
                    categorical_var = "y"
                else:
                    # Vertical: x is categorical
                    categorical_var = "x"
            else:
                categorical_var = "x" if self.orient == "v" else "y"

            # Get index and create color palette if needed
            cat_idx = list(self.var_levels[categorical_var]).index(primary_cat)
            n_colors = len(self.var_levels[categorical_var])

            # Use a default palette
            if not hasattr(self, '_category_colors'):
                self._category_colors = sns.color_palette("Set2", n_colors)

            return self._category_colors[cat_idx]

    def _draw_violin_interior(self, ax, violin, center):
        """Draw interior elements (box, quartiles, points, or sticks) for a violin."""
        observations = violin["observations"]
        support = violin["support"]
        density = violin["density"]

        # Handle split violins
        split_side = None
        if self.split and "hue" in self.variables:
            group_name = violin["group_name"]
            if isinstance(group_name, tuple):
                hue_level = group_name[1]
                hue_levels = self.var_levels.get('hue', [])
                hue_idx = list(hue_levels).index(hue_level) if hue_level in hue_levels else 0
                split_side = "left" if hue_idx == 0 else "right"

        # Draw interior elements based on inner style
        if self.inner.startswith("box"):
            self.draw_box_lines(ax, observations, support, density, center)
        elif self.inner.startswith("quart"):
            self.draw_quartiles(ax, observations, support, density, center, split_side)
        elif self.inner.startswith("stick"):
            self.draw_stick_lines(ax, observations, support, density, center, split_side)
        elif self.inner.startswith("point"):
            self.draw_points(ax, observations, center)

    def draw_single_observation(self, ax, at_group, at_quant, density):
        """Draw a line to mark a single observation."""
        d_width = density * self.dwidth
        if self.orient == "v":
            ax.plot([at_group - d_width, at_group + d_width],
                    [at_quant, at_quant],
                    color=self.gray,
                    linewidth=self.linewidth)
        else:
            ax.plot([at_quant, at_quant],
                    [at_group - d_width, at_group + d_width],
                    color=self.gray,
                    linewidth=self.linewidth)

    def draw_box_lines(self, ax, data, support, density, center):
        """Draw boxplot information at center of the density."""
        # Compute the boxplot statistics
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        whisker_lim = 1.5 * stats.iqr(data)
        h1 = np.min(data[data >= (q25 - whisker_lim)])
        h2 = np.max(data[data <= (q75 + whisker_lim)])

        # Draw a boxplot using lines and a point
        if self.orient == "v":
            ax.plot([center, center], [h1, h2],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([center, center], [q25, q75],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(center, q50,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))
        else:
            ax.plot([h1, h2], [center, center],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([q25, q75], [center, center],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(q50, center,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))

    def draw_quartiles(self, ax, data, support, density, center, split=None):
        """Draw the quartiles as lines at width of density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])

        self.draw_to_density(ax, center, q25, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)
        self.draw_to_density(ax, center, q50, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 3] * 2)
        self.draw_to_density(ax, center, q75, support, density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)

    def draw_points(self, ax, data, center):
        """Draw individual observations as points at middle of the violin."""
        kws = dict(s=np.square(self.linewidth * 2),
                   color=self.gray,
                   edgecolor=self.gray)

        grid = np.ones(len(data)) * center

        if self.orient == "v":
            ax.scatter(grid, data, **kws)
        else:
            ax.scatter(data, grid, **kws)

    def draw_stick_lines(self, ax, data, support, density,
                         center, split=None):
        """Draw individual observations as sticks at width of density."""
        for val in data:
            self.draw_to_density(ax, center, val, support, density, split,
                                 linewidth=self.linewidth * .5)

    def draw_to_density(self, ax, center, val, support, density, split, **kws):
        """Draw a line orthogonal to the value axis at width of density."""
        idx = np.argmin(np.abs(support - val))
        width = self.dwidth * density[idx] * .99

        kws["color"] = self.gray

        if self.orient == "v":
            if split == "left":
                ax.plot([center - width, center], [val, val], **kws)
            elif split == "right":
                ax.plot([center, center + width], [val, val], **kws)
            else:
                ax.plot([center - width, center + width], [val, val], **kws)
        else:
            if split == "left":
                ax.plot([val, val], [center - width, center], **kws)
            elif split == "right":
                ax.plot([val, val], [center, center + width], **kws)
            else:
                ax.plot([val, val], [center - width, center + width], **kws)

    def plot(self, ax, kws):
        """Make the violin plot."""
        # Estimate densities for all violins first
        self.estimate_densities(self.bw, self.cut, self.scale, self.scale_hue, self.gridsize)

        self.draw_violins(ax, kws)
        if self.orient == "h":
            ax.invert_yaxis()

def stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
              jitter=True, dodge=False, orient=None, color=None, palette=None, move=0,
              size=5, edgecolor="gray", linewidth=0, ax=None, width=.8, **kwargs):
    """
    A wrapper around seaborn's stripplot that adds a `move` parameter
    and preserves specific style defaults.
    """
    # 1. Handle legacy `split` argument if necessary
    if "split" in kwargs:
        dodge = kwargs.pop("split")
        warnings.warn("The `split` parameter has been renamed to `dodge`.", UserWarning)

    # 2. Get the current axes if one isn't provided
    if ax is None:
        ax = plt.gca()

    # 3. Apply old version's smart parameter processing
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == "gray":
        # Convert "gray" string to actual gray color like old version did
        edgecolor = "0.3"  # This matches the gray color used in old version

    # 4. Call the official seaborn stripplot function.
    #    We pass all standard arguments directly to it.
    sns.stripplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                  jitter=jitter, dodge=dodge, orient=orient, color=color, palette=palette,
                  size=size, edgecolor=edgecolor, linewidth=linewidth, ax=ax,
                  **kwargs)

    # 4. Apply the custom `move` functionality if needed
    if move != 0:
        # The stripplot artists are in `ax.collections`.
        # We assume the last one added is the one we want to move.
        if ax.collections:
            points_collection = ax.collections[-1]
            offsets = points_collection.get_offsets()

            # Check orientation to decide which axis to shift
            if orient in ['h', 'y']:
                # Horizontal plot: move the y-positions
                offsets[:, 1] += move
            else:
                # Vertical plot: move the x-positions
                offsets[:, 0] += move
            
            points_collection.set_offsets(offsets)

    # 5. Return the axes object
    return ax


def half_violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
               bw="scott", cut=2, scale="area", scale_hue=True, gridsize=100,
               width=.8, inner="box", split=False, dodge=True, orient=None,
               linewidth=None, color=None, palette=None, saturation=.75,
               ax=None, offset=.15, **kwargs):

    plotter = _Half_ViolinPlotter(x, y, hue, data, order, hue_order,
                             bw, cut, scale, scale_hue, gridsize,
                             width, inner, split, dodge, orient, linewidth,
                             color, palette, saturation, offset)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    return ax


def RainCloud(x = None, y = None, hue = None, data = None,
              order = None, hue_order = None,
              orient = "v", width_viol = .7, width_box = .15,
              palette = "Set2", bw = .2, linewidth = 1, cut = 0.,
              scale = "area", jitter = True, move = 0., offset = None,
              point_size = 3, ax = None, pointplot = False,
              alpha = None, dodge = False, linecolor = 'red', **kwargs):

    '''Draw a Raincloud plot of measure `y` of different categories `x`. Here `x` and `y` different columns of the pandas dataframe `data`.

    A raincloud is made of:

        1) "Cloud", kernel desity estimate, the half of a violinplot.
        2) "Rain", a stripplot below the cloud
        3) "Umberella", a boxplot
        4) "Thunder", a pointplot connecting the mean of the different categories (if `pointplot` is `True`)

    Main inputs:
        x           categorical data. Iterable, np.array, or dataframe column name if 'data' is specified
        y           measure data. Iterable, np.array, or dataframe column name if 'data' is specified
        hue         a second categorical data. Use it to obtain different clouds and rainpoints
        data        input pandas dataframe
        order       list, order of the categorical data
        hue_order   list, order of the hue
        orient      string, vertical if "v" (default), horizontal if "h"
        width_viol  float, width of the cloud
        width_box   float, width of the boxplot
        move        float, adjusts rain position to the x-axis (default value 0.)
        offset      float, adjusts cloud position to the x-axis

    kwargs can be passed to the [cloud (default), boxplot, rain/stripplot, pointplot]
    by preponing [cloud_, box_, rain_ point_] to the argument name.
    '''

    if orient == 'h':  # swap x and y
        x, y = y, x
    if ax is None:
        ax = plt.gca()
        # f, ax = plt.subplots(figsize = figsize) old version had this

    if offset is None:
        offset = max(width_box/1.8, .15) + 0.05
    n_plots = 3
    split = False
    boxcolor = "black"
    boxprops = {'facecolor': 'none', "zorder": 10}
    if hue is not None:
        split = True
        boxcolor = palette
        boxprops = {"zorder": 10}

    kwcloud = dict()
    kwbox   = dict(saturation = 1, whiskerprops = {'linewidth': 2, "zorder": 10})
    kwrain  = dict(zorder = 0, edgecolor = "white")
    kwpoint = dict(capsize = 0., errwidth = 0., zorder = 20)
    for key, value in kwargs.items():
        if "cloud_" in key:
            kwcloud[key.replace("cloud_", "")] = value
        elif "box_" in key:
            kwbox[key.replace("box_", "")] = value
        elif "rain_" in key:
            kwrain[key.replace("rain_", "")] = value
        elif "point_" in key:
            kwpoint[key.replace("point_", "")] = value
        else:
            kwcloud[key] = value

    # Draw cloud/half-violin
    half_violinplot(x = x, y = y, hue = hue, data = data,
                    order = order, hue_order = hue_order,
                    orient = orient, width = width_viol,
                    inner = None, palette = palette, bw = bw,  linewidth = linewidth,
                    cut = cut, scale = scale, split = split, offset = offset, ax = ax, **kwcloud)

    # Draw umberella/boxplot
    sns.boxplot   (x = x, y = y, hue = hue, data = data, orient = orient, width = width_box,
                         order = order, hue_order = hue_order,
                         color = boxcolor, showcaps = True, boxprops = boxprops,
                         palette = palette, dodge = dodge, ax =ax, **kwbox)

    # Set alpha of the two
    if not alpha is None:
        _ = plt.setp(ax.collections + ax.artists, alpha = alpha)

    # Draw rain/stripplot
    ax =  stripplot (x = x, y = y, hue = hue, data = data, orient = orient,
                    order = order, hue_order = hue_order, palette = palette,
                    move = move, size = point_size, jitter = jitter, dodge = dodge,
                    width = width_box, ax = ax, **kwrain)

    # Add pointplot
    if pointplot:
        n_plots = 4
        if not hue is None:
            n_cat = len(np.unique(data[hue]))
            sns.pointplot(x = x, y = y, hue = hue, data = data,
                          orient = orient, order = order, hue_order = hue_order,
                          dodge = width_box * (1 - 1 / n_cat), palette = palette, ax = ax, **kwpoint)
        else:
            sns.pointplot(x = x, y = y, hue = hue, data = data, color = linecolor,
                           orient = orient, order = order, hue_order = hue_order,
                           dodge = width_box/2., ax = ax, **kwpoint)

    # Prune the legend, add legend title
    if not hue is None:
        handles, labels = ax.get_legend_handles_labels()
        _ = plt.legend(handles[0:len(labels)//n_plots], labels[0:len(labels)//n_plots], \
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., \
                       title = str(hue))#, title_fontsize = 25)

    # Adjust the ylim to fit (if needed)
    if orient == "h":
        ylim = list(ax.get_ylim())
        ylim[-1]  -= (width_box + width_viol)/4.
        _ = ax.set_ylim(ylim)
    elif orient == "v":
        xlim = list(ax.get_xlim())
        xlim[-1]  -= (width_box + width_viol)/4.
        _ = ax.set_xlim(xlim)

    return ax
