# vortexclust: Clustering of Stratospheric Polar Vortex Regimes
`vortexclust` is a Python package for analyzing and clustering climate data, with a focus on the stratospheric polar vortex and sudden stratospheric warmings.

## Table of Contents
- [Installation](#installation)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)
- [FAQ](#faq)
- [Acknowledgments](#acknowledgments)

## Installation
`vortexclust` requires *Python >= 3.11*
### From GitHub
`pip install git+https://github.com/hGl0/vortexclust@main`


#### With optional dependencies: <br>
Adds cartopy/pyproj (needs system GEOS/PROJ) and is only required if `vortexclust.visualization.maps` is used to generate a stereographic map plot<br>
`pip install vortexclust[maps]`

Adds statsmodels, pyts and seaborn, which are only required if the demo script is executed. <br>
`pip install vortexclust[demo]`

### From local source
```bash
git clone https://github.com/hGl0/vortexclust.git
cd vortexclust
pip install -e .
```

### Windows Installation
It is recommended to install `miniconda` or `anaconda`. This handles all Python dependencies without manual compilation.
Example:
```bash
conda create -n vortex python=3.11
conda activate vortex
pip install vortexclust[maps, demo]
```

## Getting started

```python
import vortexclust as vc
from vortexclust.io import read_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# 1. Load and prepare data
df = read_data("data/demo.csv")

# 2. Scale selected columns
scaler = StandardScaler()
df[['scaled_ar', 'scaled_latcent']] = scaler.fit_transform(
    df[['ar', 'latcent']]
)

# 3. Cluster
model = AgglomerativeClustering(n_clusters=3, linkage='complete')
df['cluster'] = model.fit_predict(df[['scaled_ar', 'scaled_latcent']])

# 4. Plot
import vortexclust.visualization as viz
viz.plot_dendrogram(model)
```

For a full pipeline including filtering, seasonality analysis, and statistics, see the sample notebooks or the demo script.

## Usage
You can either build your own pipeline (see example below) or use the included demo scripts and notebooks.  
Typical workflow:
1. Load preprocessed data (CSV).
2. Apply scaling and optionally filter seasonality.
3. Run clustering (e.g., k-means, hierarchical).
4. Evaluate clusters and visualize results.

See `notebooks/` for reproducible examples replicating published studies.

## Tests
Run the test suite with:
```bash
pytest
```
All tests are located in the `tests\` directory. Pytest must be installed separately.

## Contributing <a name="contributing"></a>
Contributions are welcome!
### Code structure
The package follows a functionality-based structure:

- `analysis/` &rarr; core algorithms (clustering, decomposition, evaluation, metrics)  
- `models/` &rarr; wrappers around scikit-learn or custom models  
- `io/` &rarr; input/output (data loading, cleaning, path utilities)  
- `visualization/` &rarr; plotting functions (PCA, dendrograms, radar, violin, stereographic maps)  
- `workflows/` &rarr; demo scripts and integrated workflows

### Adding new algorithms
- **Clustering methods**: add to `analysis/clustering.py` and, if needed, provide a wrapper in `models/`.  
- **Decomposition / filtering**: extend `analysis/decomposition.py`.  
- **Metrics**: place in `analysis/metrics.py`.  
- **Visualization of results**: extend `visualization/`. Keep plotting functions stateless.  
- **High-level workflows**: if you create a new end-to-end workflow, add it to `workflows/`.  

Each function should:
- be pure/stateless (no hidden side-effects),  
- validate inputs explicitly,  
- include a docstring with signature, description, parameters, and return values,  
- have at least one corresponding unit test in `tests/`.

### Guidelines
- Open an issue for major changes before starting implementation.  
- Follow PEP8 and existing code style.  
- Run `pytest` before submitting PRs.  

Contact: Hanna Gloyna (author, hanna.gloyna@gmx.de)

## License <a name="license"></a>
This project is licensed under the [GPL3.0](LICENSE).

## FAQ <a name="faq"></a>
*Q:* How much preprocessing does vortexclust handle automatically?<br>
*A:* It does not preprocess raw NetCDF data. It expects preprocess diagnositic CSVs 
(e.g. ERA5/UA-ICON indices). Scaling, deseasonalization, clustering and visualization are supported
within the package.

*Q:* What do I need to prepare for clustering?<br>
*A:* That depends on your data. In general, clustering algorithms should be applied to scaled 
and deseasonalized data. `vortexclust` provides different tools to filter seasonality. Detect it 
with `statsmodels` durbin watson test or the autocorrelation function. Then the optimal number 
of clusters should be determined, where `vortexclust` implements multiple methods to detect 
the right number of cluster. Other hyperparameters such as choice of model, distance metric or
initial weights might require further tuning.

*Q:* Are there ready-to-use example notebooks or scripts that replicate paper results?<br>
*A:* Yes, the demo script is immediately available, when `vortexclust[demo]` is installed. Some 
python versions. Additionally, 3 ready-to-use notebooks are available in the 'notebooks' directory.

## Acknowledgments <a name="acknowledgments"></a>
I would like to thank my supervisors,  
Dr. Christoph Zülicke (IAP Kühlungsborn), and
Prof. Dr. Anna-Lena Lamprecht (University of Potsdam)  
for their guidance and valuable feedback throughout the development of this project.