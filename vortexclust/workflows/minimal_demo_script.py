import argparse
import sys

"""Demo function of vortexclust"""


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="vortexclust-demo",
        description="Run a minimal vortexclust demo pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to demo CSV")
    parser.add_argument("--clusters", type=int, default=3)
    args = parser.parse_args(argv)

    # Optional deps guard (since CLI isn't conditional on extras)
    try:
        import seaborn  # noqa
        import statsmodels  # noqa
    except ImportError as e:
        missing = str(e).split("'")[1]
        print(
            f"Missing optional dependency '{missing}'. "
            "Install with: pip install 'vortexclust[demo]'",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Minimal demo pipeline ---
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    from vortexclust.io import read_data
    from vortexclust.visualization.dendrogram import plot_dendrogram

    # --- STEP 1: read data ---
    df = read_data(args.input) if args.input.endswith(".csv") else pd.read_csv(args.input)
    # --- STEP 2: Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["ar", "latcent"]].dropna())
    # --- STEP 3: detect and filter seasonality - skipped ---
    # --- STEP 4: determine correct number of clusters - skipped ---
    # --- STEP 5: train model ---
    model = AgglomerativeClustering(n_clusters=args.clusters, linkage="complete", compute_distances=True)
    labels = model.fit_predict(X)
    # --- STEP 6: compute thresholds
    df = df.loc[df[["ar", "latcent"]].dropna().index].copy()
    df["cluster"] = labels

    # Example plot (user can redirect display/save in a notebook)
    plot_dendrogram(model, truncate_mode='level', p=4, direction='LR')
    print("Done. First rows:\n", df.head().to_string(index=False))
