import warnings

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from vortexclust.io.paths import check_path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError as e:
    raise ImportError(
        "seaborn is required for the demo workflow. "
        "Install via: pip install vortexclust[demo]"
    ) from e

import numpy as np
import pandas as pd

def plot_ssa_grid(data_series, ssa_results, labels,
                  index_format=['%b %d'],
                  figsize=(15, 8),
                  used_signals = 1,
                  titles = None):
    """
    Plot a grid of original vs SSA seasonal component time series.

    Parameters:
    - data_series: list of original time series (e.g., [avg_year, avg_day_over_year])
    - ssa_results: list of SSA seasonal components (same shape as data_series)
    - labels: list of labels (same shape: [ [label1, label2, ...], [label1, label2, ...] ])
    - index_format: datetime x-axis tick format, default is month-day
    - figsize: overall figure size
    - used_signals: number of SSA components to reconstruct signal
    - titles: list of titles for each subplot
    """
    m = len(data_series)
    n = len(labels)

    fig, ax = plt.subplots(m, n, figsize=figsize, squeeze=False)

    for i in range(m):
        if len(index_format) > 1:
            format = index_format[i]
        for j in range(n):
            ax_ij = ax[i][j]

            ts = data_series[i]
            ssa = ssa_results[i][j]
            label = labels[j]

            if used_signals > ssa[:].shape[0]:
                print(f"Number of given signals too high. Reset to maximum number {ssa.shape[0]} of signals in SSA.")
                used_signals = ssa.shape[0]
            if used_signals > 1:
                reconstructed = ssa[:][:used_signals].sum(axis=0)

            seasonal = ssa[:][0]

            if titles:
                title = str(label) + ' ' + titles[i]
            else:
                title = None

            ax_ij.plot(ts.index, ts[label], label='Averaged data')
            ax_ij.plot(ts.index, seasonal, label='Trend', linestyle='--')
            if used_signals > 1: ax_ij.plot(ts.index, reconstructed, label='Reconstructed', linestyle='-.')

            if title: ax_ij.set_title(title)
            ax_ij.xaxis.set_major_formatter(mdates.DateFormatter(format))
            ax_ij.tick_params(axis='x', rotation=45)
            ax_ij.legend()

    plt.tight_layout()
    plt.show()


def plot_timeseries_moments(df, columns, labels,
                            time_column='string', time_format='%m-%y',
                            time_span=-1,
                            positions=None, vertical_line=None,
                            title=None, savefig=None,
                            num_plots=2, figsize=(10, 5)):
    if figsize:
        fig, axes = plt.subplots(num_plots, figsize=figsize)
    else:
        fig, axes = plt.subplots(num_plots, figsize=(10, 2.5*num_plots))

    if num_plots > 1:
        axes[0].set_title(title)
    else:
        axes.set_title(title)

    if len(columns) % 2 == 1:
        warnings.warn(UserWarning(
            'Give an even number of columns. Two columns will be plotted in one plot. Removing last column.'))
        columns = columns[:-1]

    for i in range(len(axes)):
        if num_plots > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.plot(df[columns[2*i]][:time_span], label=labels[2*i])
        ax.plot(df[columns[2*i + 1]][:time_span], label=labels[2*i + 1])
        if isinstance(vertical_line, int):
            ax.axvline(x=vertical_line, color='black', linestyle='--')

        ax.legend(loc='upper right')

        # Add grey vertical lines at positions and the corresponding date
        if positions is not None:
            ax.set_xticks(positions)
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())  # Align with bottom axis
            ax_top.set_xticks(positions)

            if time_column in df.columns:
                labels_dt = df.iloc[positions][time_column].dt.strftime(time_format)
                ax_top.set_xticklabels(labels_dt, ha='center', rotation=0)
            else:
                ax_top.set_xticklabels([str(x) for x in positions], ha='center', rotation=0)

            ax_top.tick_params(axis='x', direction='in', pad=5)
            ax_top.xaxis.set_label_position('top')

    plt.tight_layout()
    if savefig is not None:
        check_path(savefig)
        plt.savefig(savefig)
    plt.show()


def plot_eeof(epc, eeof, expl_var_ratio, savefig=None):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0][0].scatter(np.arange(1, len(expl_var_ratio)+1), expl_var_ratio*100)
    ax[0][0].set_ylabel('Eigenvalues (%)')
    ax[0][0].set_xlabel('Rank')
    ax[0][0].set_title('Eigenvalue Spectrum')

    ax[1][1].plot(epc[:, 0], epc[:, 1])
    ax[1][1].set_xlabel('EPC1')
    ax[1][1].set_ylabel('EPC2')
    ax[1][1].set_title('EPC1 vs. EPC2')

    # ax[0].plot(era5_all['scaled_area'][:2000], label='Data')
    ax[0][1].plot(eeof[0, :400], label='EEOF1')
    ax[0][1].plot(eeof[1, :400], label='EEOF2')
    ax[0][1].set_xlabel('Time lag (day)')
    ax[0][1].set_ylabel('EEOF')
    ax[0][1].set_title("EEOFs 1 and 2")
    ax[0][1].legend(loc='upper left')

    ax[1][0].plot(epc[:2000, 0], label='EPC 1')
    ax[1][0].plot(epc[:2000, 1], label='EPC 2')
    ax[1][0].set_xlabel('Time (day)')
    ax[1][0].set_ylabel('EPC')
    ax[1][0].set_title("EPCs 1 and 2")
    ax[1][0].legend(loc = 'upper left')
    plt.tight_layout()
    if savefig:
        check_path(savefig)
        plt.savefig(savefig)
    plt.show()


def plot_hist_per_class(df, feat_k, y, savefig=None):
    # for idx, feat_k in enumerate(features_kopt):
    var = y
    for feature in feat_k['features']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        unique_classes = sorted(df[var].dropna().unique())
        palette = sns.color_palette('tab10', len(unique_classes))
        color_map = dict(zip(unique_classes, palette))

        # Histogram with percentages
        sns.histplot(data=df,
                     x=feature,
                     hue=var,
                     palette=color_map,
                     hue_order=unique_classes,
                     common_norm=False,
                     multiple='dodge',
                     stat='percent',
                     kde=True,
                     ax=axes[0])
        axes[0].set_title(f"Percentage Histogram of {feature} by {var}")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Percent')

        # Histogram with counts
        sns.histplot(data=df,
                     x=feature,
                     hue=var,
                     palette=color_map,
                     hue_order=unique_classes,
                     common_norm=False,
                     multiple='dodge',
                     stat='count',
                     kde=False,
                     ax=axes[1])
        axes[1].set_title(f"Count Histogram of {feature} by {var}")
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Count')

        group_means = df.groupby(var)[feature].mean()
        # Draw one vertical line at mean per class, with matching color
        for i, label in enumerate(unique_classes):
            mean_val = group_means[label]
            color = color_map[label]
            axes[0].axvline(mean_val, color=color, linewidth=1, linestyle='-.',
                            label=f"Mean of {var} = {label}")
            axes[1].axvline(mean_val, color=color, linewidth=1, linestyle='-.',
                            label=f"Mean of {var} = {label}")

        handles = [Patch(facecolor=color_map[c], label=str(c)) for c in unique_classes]
        handles.append(Line2D([0], [0], linestyle='-.', color='black', label='Class mean'))
        for ax in axes:
            ax.legend(
                handles=handles, title=var,
                loc='upper right',
                bbox_to_anchor=(1, 1),
                bbox_transform=ax.transAxes,
                frameon=True,
            )

        plt.tight_layout()

        if savefig:
            check_path(savefig+f"{feature}")
            plt.savefig(savefig+f"{feature}")

        plt.show()


def compare_cluster(df,
                    compare_col,
                    y_names=['y_ar_latcent', 'y_ar_latcent_scArea', 'y_ar_latcent_filteredArea', 'y_ar_latcent_u'],
                    value_col='string',
                    pred_value='S',
                    gt_value='S',
                    verbose=True):
    r"""
    Compare a cluster classification column with multiple threshold-based ground truth columns.

    :param df: Pandas DataFrame with entire data
    :type df: Pandas DataFrame
    :param compare_col: String or integer of the column that should be compared.
    :type compare_col: str
    :param y_names: List of Strings with all column names that should be compared to ´compare_col´
    :type y_names: List[String]
    :param value_col: String or integer of column that should be counted
    :type value_col: str
    :param pred_value: String or integer of predicted value
    :type pred_value: str or integer
    :param gt_value: String or integer of ground truth value
    :type gt_value: str or integer
    :param verbose: Boolean whether to print information about accuracy, recall, precision and f1 score.
    :type verbose: bool

    :return: None
    """

    if not compare_col in df.columns:
        raise KeyError(f"Compare column {compare_col} not found in dataframe columns.")
    if not value_col in df.columns:
        raise KeyError(f"Value column {value_col} not found in dataframe columns.")

    for y in y_names:
        if y not in df.columns:
            raise KeyError(f"Prediction column {y} not found in dataframe.")

        y_vals = df[y].dropna().unique()
        gt_vals = df[compare_col].dropna().unique()

        if not pred_value in y_vals:
            raise ValueError(f"{pred_value} is not a found value in the given class column {y}."
                             f"Found {y_vals} instead.")
        if not gt_value in gt_vals:
            raise ValueError(f"{gt_value} is not a found value in the given class column {y}."
                             f"Found {gt_vals} instead.")
        print(f'----------------------------------------------')
        print(f'Comparison of Percentages:  {y} (predicted: {pred_value}) vs {compare_col} (ground truth: {gt_value})')
        table = pd.pivot_table(
            data=df[[y, value_col, compare_col]],
            values=value_col,
            index=compare_col,
            columns=y,
            aggfunc='count', margins=True).fillna(0)
        print(np.round(table / df.shape[0] * 100, 2))

        tp = ((df[y] == pred_value) & (df[compare_col] == gt_value)).sum()
        fp = ((df[y] == pred_value) & (df[compare_col] != gt_value)).sum()
        fn = ((df[y] != pred_value) & (df[compare_col] == gt_value)).sum()
        tn = ((df[y] != pred_value) & (df[compare_col] != gt_value)).sum()

        acc = (tn + tp) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        if verbose:
            print(f"Accuracy: {np.round(acc, 2)}\n"
                  f"Precision: {np.round(precision, 2)}\n"
                  f"Recall: {np.round(recall, 2)}\n"
                  f"F1 Score: {np.round(f1, 2)}")