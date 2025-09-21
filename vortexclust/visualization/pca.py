import warnings
from typing import Any, Optional, List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vortexclust.io.paths import check_path


def compute_arrows(pca,
                   score: np.ndarray,
                   n_arrows: int,
                   scales: List[float],
                   n: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Computes and scales the top `n_arrows` vectors for a PCA biplot.

    :param pca: Fitted PCA object containing the principal components.
    :param score: A NumPy array of PCA scores.
    :param n_arrows: The number of top features (arrows) to select.
    :param scales: A list of scaling factors for each principal component.
    :param n: The number of principal components to consider.

    :raises ValueError: If `pca` does not contain 'components_' or
        if `score` has fewer than `n_components` columns.

    :return: A tuple containing:
        - An array of indices corresponding to the selected features.
        - A NumPy array of scaled arrows.
    """
    if not hasattr(pca, "components_"):
        raise ValueError("The provided PCA object must have 'components_' attribute.")
    if score.shape[1] < n:
        raise ValueError(f"Score matrix must have at least {n} columns")

    coeff = pca.components_[:n].T

    # find 'n_arrows' longest arrows
    tops = (coeff ** 2).sum(axis=1).argsort()[-n_arrows:]
    arrows = coeff[tops, :n]

    # Scale arrows
    norm = np.sqrt((arrows ** 2)).sum(axis=0)
    norm[norm == 0] = 1  # avoid division by zero
    arrows /= norm
    arrows *= np.abs(score[:, :n]).max(axis=0)
    scaled_arrows = arrows * np.array(scales[:n])

    return tops, scaled_arrows


def plot_pca(pca: Any, x_reduced: np.ndarray,
             features: Optional[List[str]] = None,
             labels: Optional[Union[List, np.ndarray, pd.Series]] = None,
             **kwargs) -> None:
    r"""
    Plots a 2D or 3D Principal Component Analysis (PCA) biplot, visualizing the dataset
    and the most important principal component vectors.

    :param pca: A fitted PCA object from sklearn.decomposition.PCA.
        Must contain `components_` and `explained_variance_ratio_` attributes.
    :param x_reduced: ndarray of shape (n_samples, n_components). The dataset transformed into principal components.
    :param features: List of feature names correpsonding to the original dataset.
        Used to label principal component vectors. Default is None.
    :param labels: List of cluster labels for coloring data points with its respective cluster. Default is None.
    :param savefig: (str, optional) File path to save the plot. Default is None.
    :param kwargs:
                - plot_type (str, optional): '2D' or '3D' plot. Default is '2D'
                - n_arrows (int, optional): Number of principal component vectors to display. Default is 4.
                - savefig (str, optional): File path to save the plot. Default is None.

    :raises AttributeError: If `pca` is missing required attributes.
    :raises TypeError: If input types are incorrect.
    :raises ValueError: If `plot_type` is not '2D' or '3D'.
    :raises FileNotFoundError: If `savefig` path is invalid.

    :return: None
    """
    plot_type = kwargs.get('plot_type', '2D')
    n_arrows = kwargs.get('n_arrows', 4)
    savefig = kwargs.get('savefig', None)

    if features is None:
        n_features = pca.components_.shape[1]
        features = [f"var{i+1}" for i in range(n_features)]
    else:
        n_features = len(features)

    if n_arrows > n_features:
        n_arrows = n_features

    # Init values for plotting
    try:
        score = x_reduced[:, :]
        pvars = pca.explained_variance_ratio_ * 100

        if score.shape[1] < 2:
            raise ValueError(f"Only {score.shape[1]} columns detected. No plotting available.")
        elif score.shape[1]<3:
            warnings.warn(f"Only {score.shape[1]} columns detected. 3D plot is not possible. Resetting to 2D.", UserWarning)
            xs, ys = score[:, :2].T
            plot_type = '2D'
        else:
            xs, ys, zs = score[:, :3].T

        scales = 1.0 / (score.max(axis=0) - score.min(axis=0))
        if min(3, n_features) < 3:
            scalex, scaley = scales[:n_features]
        else:
            scalex, scaley, scalez = scales[:min(3, n_features)]

        # color by label
        if isinstance(labels, pd.Series):
            if labels.nunique() > 10:
                warnings.warn(f"Currently using 'tab10' to assign colors for each cluster.\n"
                              f"There are {labels.nunique()} unique labels, so the plot might not be displayed correctly.\n"
                              f"Working on this issue.", UserWarning)
            c = labels.to_list()
        else:
            c = None

    except Exception as e:
        raise Exception(f"Error while initialising values for plotting: {e}")

    # Plot data and principal components
    # decision against match-case statement, might not work in older python versions (< 3.10)
    try:
        if plot_type == '2D':
            tops, scaled_arrows = compute_arrows(pca, score, n_arrows, scales, 2)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            if c:
                scatter = ax.scatter(xs * scalex, ys * scaley,
                                     c=c, cmap='tab10', alpha=0.5, zorder=0)
                legend_handles, legend_labels = scatter.legend_elements(prop="colors")
                ax.legend(legend_handles, legend_labels, title="Cluster")

            else:
                ax.scatter(xs * scalex, ys * scaley, alpha=0.5, zorder=0)

            for i, arrow in zip(tops, scaled_arrows):
                ax.quiver(0, 0, *arrow,
                          color='gray',
                          zorder=3,
                          angles='xy', scale_units='xy',
                          scale=1)
                ax.text(*(arrow * 1.15), features[i], ha='center', va='center')

            for i, axis in enumerate('xy'):
                getattr(ax, f'set_{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')

            # make up for plot
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.grid()
            plt.title('2D Biplot')

        elif plot_type == '3D':
            tops, scaled_arrows = compute_arrows(pca, score, n_arrows, scales, 3)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i, arrow in zip(tops, scaled_arrows):
                ax.quiver(0, 0, 0, *arrow,
                          color='gray',
                          zorder=3)
                ax.text(*(arrow * 1.15), features[i], ha='center', va='center')

            # plot points
            if c:
                scatter = ax.scatter3D(xs * scalex, ys * scaley, zs * scalez,
                                       alpha=0.5, c=c, cmap='tab10', zorder=0)
                legend_handles, legend_labels = scatter.legend_elements(prop="colors")
                ax.legend(legend_handles, legend_labels, title="Cluster")
            else:
                ax.scatter3D(xs * scalex, ys * scaley, zs * scalez, alpha=0.5)

            for i, axis in enumerate('xyz'):
                getattr(ax, f'set_{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')

            # make up for plot
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.grid()
            plt.title('3D Biplot')
        else:
            warnings.warn(UserWarning(f"Warning: Unknown plot type '{plot_type}'"))

        # save figure
        if savefig:
            check_path(savefig)
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
        plt.show()

    except Exception as e:
        raise Exception(f"Error while plotting: {e}")
