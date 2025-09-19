from vortexclust.workflows import config

# plotting defaults
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

# logging (reuse config logger + add a file handler)
import logging, os
from datetime import datetime

def set_up_logging():
    logs_dir = config.OUTPUT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"vortexclust_script_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(config.LOG_LEVEL)
    fh.setFormatter(logging.Formatter(config.LOG_FORMAT))
    config.logger.addHandler(fh)
    return config.logger

def main():
    logger = set_up_logging()

    # === Paths from config ===
    demo_d = config.DATA_DIR / "demo_d.csv"
    demo_msw = config.DATA_DIR / "demo_msw.csv"
    output_dir = config.OUTPUT_DIR / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Tunables ===
    to_filter = ["scaled_u"]
    M = 120
    n_components = 4
    show_plots = False
    k_opt = 3

    # --- Optional deps guard (one place) ---
    try:
        import seaborn  as sns
        if hasattr(config, "PLOT_STYLE") and config.PLOT_STYLE:
            # accept either plain seaborn styles ("whitegrid", "darkgrid", "ticks", ...)
            # or full matplotlib style names ("seaborn-v0_8-whitegrid")
            if config.PLOT_STYLE.startswith("seaborn-v"):
                plt.style.use(config.PLOT_STYLE)
            else:
                sns.set_theme(style=config.PLOT_STYLE)
        else:
            sns.set_theme(style="whitegrid")
        import statsmodels
        from pyts.decomposition import SingularSpectrumAnalysis
    except ImportError as e:
        missing = str(e).split("'")[1]
        raise SystemExit(
            f"Missing optional dependency '{missing}'. "
            "Install with: pip install 'vortexclust[demo]'"
        )

    # --- Imports after guard ---
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    # explicit package imports
    from vortexclust.io import read_data, no_white_space, to_date
    from vortexclust.analysis.decomposition import compute_eeof
    from vortexclust.analysis.clustering import gap_statistic
    import vortexclust.visualization as viz
    from vortexclust.workflows.demo import plot_timeseries_moments, plot_eeof, plot_hist_per_class, compare_cluster

    # === Read + prep ===
    df_d = read_data(str(demo_d))
    df_msw = read_data(str(demo_msw))
    no_white_space(df_d); no_white_space(df_msw)
    to_date(df_d, "string", format="mixed")
    to_date(df_msw, "string", format="mixed")

    demo_all = df_d.merge(df_msw, on="string", how="left", suffixes=("_d", "_msw"))

    le = LabelEncoder()
    demo_all["form"] = le.fit_transform(demo_all["form"])
    logger.info(f"Transformed 'form': {dict(zip(le.inverse_transform([0,1]), [0,1]))}")

    demo_all = demo_all.sort_values("string").reset_index(drop=True)

    # DJFM only
    demo = demo_all[demo_all["string"].dt.month.isin([12,1,2,3])].reset_index(drop=True)

    # === Scaling ===
    sc = StandardScaler()
    col = ["area", "ar", "latcent", "u"]
    scaled_col = ["scaled_area", "scaled_ar", "scaled_latcent", "scaled_u"]
    demo[scaled_col] = sc.fit_transform(demo[col])

    # Optional moment plot
    if show_plots:
        plot_timeseries_moments(
            demo, scaled_col, col,
            title="Vortex Geometric Moments",
            time_span=500,
            savefig=str(output_dir / "moments.png")
        ); plt.close()

    # === Seasonality checks (Durbin–Watson + ACF) ===
    from statsmodels.stats.stattools import durbin_watson
    import statsmodels.api as sm
    seasonality_check = []
    for c in scaled_col:
        model_ols = sm.OLS(demo[c], demo[[k for k in scaled_col if k != c]]).fit()
        dw = durbin_watson(model_ols.resid)
        if (dw < 1.5) or (dw > 2.5):
            logger.info(f"Seasonality check for '{c}' recommended. Durbin–Watson: {dw:.2f}")
            seasonality_check.append(c)

    from statsmodels.graphics.tsaplots import plot_acf
    for c in seasonality_check:
        plot_acf(demo[c], lags=500, marker=None)
        plt.axhline(y=0.05, linestyle="--", color="black")
        plt.axhline(y=-0.05, linestyle="--", color="black")
        plt.title(f"Autocorrelation check for '{c}'")
        plt.savefig(output_dir / f"seasonality_check_{c}.png", dpi=config.DPI)
        if show_plots: plt.show()
        plt.close()

    # === Filtering (SSA + EEOF) on 'scaled_u' ===
    ssa = SingularSpectrumAnalysis(window_size=M)
    for c in to_filter:  # c == 'scaled_u'
        ssa_comp = ssa.fit_transform(demo[c].values.reshape(1,-1))
        epc, eeof, expl_var_ratio, eeof_mat, _ = compute_eeof(demo[c], M=400, n_components=n_components)

        if len(to_filter) < 2:
            plot_eeof(epc, eeof, expl_var_ratio, savefig=str(output_dir / "eeof_summary.png")); plt.close()

        # save comparison plot
        plt.figure(figsize=config.FIGSIZE, dpi=config.DPI)
        plt.plot(demo[c][:500], label="original")
        plt.plot(ssa_comp[:n_components].sum(axis=0)[:500], label=f"SSA ({n_components} comps)")
        plt.plot(eeof_mat[399:899, 0], label="EEOF (lead comp)")
        plt.title(f"SSA and EEOF for '{c}'")
        plt.legend()
        plt.savefig(output_dir / f"eeof_ssa_{c}.png", dpi=config.DPI)
        if show_plots: plt.show()
        plt.close()

        # filtered series
        try:
            demo["ssa_"+c] = demo[c] - ssa_comp[:n_components].sum(axis=0)
        except Exception:
            demo["ssa_"+c] = demo[c] - ssa_comp[0, :n_components, :].sum(axis=0)

        demo["eeof_"+c] = np.full_like(demo["ssa_"+c], np.nan, dtype=float)
        demo.loc[:309, "eeof_"+c] = demo.loc[:309, c] - eeof_mat[399:709, 0].T
        demo.loc[310:1000, "eeof_"+c] = demo.loc[310:1000, c] - eeof_mat[399:1090, 309].T
        demo = demo[:1000]

    # === k selection (Gap Statistic) ===
    if k_opt is None:
        k_max = 10
        gap = gap_statistic(
            demo[["scaled_ar", "scaled_latcent", "ssa_scaled_u"]],
            k_max=k_max, n_replicates=10
        )
        xs = np.arange(1, k_max+1)
        plt.errorbar(xs, gap[:,0], yerr=gap[:,1])
        plt.title("Gap Statistic: AR, LatCent, filtered U")
        plt.savefig(output_dir / "gap_ar_latcent_u.png", dpi=config.DPI)
        if show_plots: plt.show()
        plt.close()

        # Tibshirani rule: smallest k with G(k) >= G(k+1) - s(k+1)
        k_opt = 1
        for k in range(1, k_max):  # safe indexing
            if gap[k-1,0] >= gap[k,0] - gap[k,1]:
                k_opt = k
                break
        logger.info(f"Gap statistic selected k_opt={k_opt}")

    # === Train model ===
    model = AgglomerativeClustering(linkage="complete", compute_distances=True, n_clusters=k_opt)
    model.fit(demo[["scaled_ar", "scaled_latcent", "eeof_scaled_u"]])
    demo["y"] = model.labels_.astype(int)

    # === Visualisation ===
    viz.plot_dendrogram(model, truncate_mode="level", p=4, direction="LR",
                        savefig=str(output_dir / "demo_dendrogram.png"))

    logger.info(
        "Averages per class:\n%s",
        demo[["y", "scaled_ar", "scaled_latcent", "eeof_scaled_u"]].groupby(["y"]).mean()
    )
    plot_hist_per_class(
        demo, {"features": ["scaled_ar", "scaled_latcent", "eeof_scaled_u"]}, "y",
        savefig=str(output_dir / "demo_hist_per_class.png")
    )

    # === Compare to form ===
    compare_cluster(demo, compare_col="form", pred_value=1, gt_value=1, y_names=["y"])

    # === Save results ===
    demo.to_csv(output_dir / "demo_results.csv", index=False)
    logger.info("Saved results to %s", output_dir / "demo_results.csv")

if __name__ == "__main__":
    main()
