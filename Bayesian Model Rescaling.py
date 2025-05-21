import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from tqdm import tqdm
import arviz as az

if __name__ == '__main__':
    ### -------- PART 1: FITTING THE MODEL -------- ###

    # Load and prepare data
    data = pd.read_csv(
        '/Users/robbe_neyns/Documents/research/bee mapping2/Bayesian modelling/Sampled nests csv/sampled_nests_V1.csv',
        sep=",")
    data = data.astype(np.float64)

    # Add quadratic terms
    data['vegetation_squared'] = data['vegetation'] ** 2
    data['impervious_squared'] = data['impervious'] ** 2

    # Variables to rescale
    vars_to_scale = [
        "clay", "sand", "silt", "proximity", "flood", "agriculture",
        "impervious", "vegetation", "vegetation_squared", "impervious_squared", "insolation"
    ]

    # Compute and apply standardization
    scaling_means = {}
    scaling_stds = {}

    for var in vars_to_scale:
        mean = data[var].mean()
        std = data[var].std()
        data[var] = (data[var] - mean) / std
        scaling_means[var] = mean
        scaling_stds[var] = std

    # Save scaling parameters for use in raster prediction
    pd.DataFrame({"mean": scaling_means, "std": scaling_stds}).to_csv("scaling_parameters.csv")

    with pm.Model() as model:
        # Adjusted priors to match scaled space
        beta_insolation = pm.Normal("beta_insolation", mu=0.0838 * scaling_stds["insolation"], sigma=0.026 * scaling_stds["insolation"])
        beta_clay = pm.TruncatedNormal("beta_clay", mu=-0.2 * scaling_stds["clay"], sigma=0.05 * scaling_stds["clay"], upper=0)
        beta_sand = pm.TruncatedNormal("beta_sand", mu=0.1 * scaling_stds["sand"], sigma=0.05 * scaling_stds["sand"], lower=0)
        beta_silt = pm.TruncatedNormal("beta_silt", mu=-0.125 * scaling_stds["silt"], sigma=0.05 * scaling_stds["silt"], upper=0)
        beta_suit = pm.Normal("beta_suit", mu=0, sigma=1)
        beta_flooded = pm.TruncatedNormal("beta_flooded", mu=-5.9 * scaling_stds["flood"], sigma=1.0 * scaling_stds["flood"], upper=0)
        beta_agri = pm.TruncatedNormal("beta_agri", mu=-5.9 * scaling_stds["agriculture"], sigma=1.0 * scaling_stds["agriculture"], upper=0)
        beta_veg = pm.Normal("beta_veg", mu=73.18 * scaling_stds["vegetation"], sigma=30 * scaling_stds["vegetation"])
        beta_veg2 = pm.Normal("beta_veg2", mu=-147.38 * (scaling_stds["vegetation"] ** 2), sigma=30 * (scaling_stds["vegetation"] ** 2))
        beta_building_bright = pm.Normal("beta_building_bright", mu=137.00 * scaling_stds["impervious"], sigma=10 * scaling_stds["impervious"])
        beta_building_bright2 = pm.Normal("beta_building_bright2", mu=-152.23 * (scaling_stds["impervious"] ** 2), sigma=10 * (scaling_stds["impervious"] ** 2))
        intercept = pm.Normal("intercept", mu=0, sigma=1)

        mu = (
            intercept
            + beta_clay * data["clay"]
            + beta_sand * data["sand"]
            + beta_silt * data["silt"]
            + beta_suit * data["proximity"]
            + beta_flooded * data["flood"]
            + beta_agri * data["agriculture"]
            + beta_building_bright * data["impervious"]
            + beta_building_bright2 * data["impervious_squared"]
            + beta_veg * data["vegetation"]
            + beta_veg2 * data["vegetation_squared"]
            + beta_insolation * data["insolation"]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(mu))
        y_obs = pm.Bernoulli("presence", p=p, observed=data["presence"])

        trace = pm.sample(1000, tune=1000, target_accept=0.95, random_seed=42)

        # Plot prior vs. posterior for selected variables
        az.plot_trace(trace, var_names=['intercept', 'beta_insolation', 'beta_veg2'])
        plt.show()

        data["predicted_probability"] = trace.posterior["p"].mean(dim=("chain", "draw")).values
        data.to_csv("/Users/robbe_neyns/Documents/research/bee mapping2/Bayesian modelling/predicted_probabilities.csv")

    ### -------- PART 2: SPATIAL RASTER PREDICTION -------- ###

    print("Starting prediction on raster")
    scaling_df = pd.read_csv("scaling_parameters.csv", index_col=0)

    with rasterio.open('/Users/robbe_neyns/Documents/research/bee mapping2/Bayesian modelling/Merged rasters alligned/merged_input_var_clip.tif') as src:
        layers = {name: src.read(i+1).flatten() for i, name in enumerate([
            "agriculture", "clay", "flood", "insolation", "proximity",
            "sand", "silt", "impervious", "impervious", "vegetation"
        ])}
        meta = src.meta
        transform = src.transform
        height, width = src.height, src.width

    # Compute quadratic terms and rescale all inputs
    layers["vegetation_squared"] = layers["vegetation"] ** 2
    layers["impervious_squared"] = layers["impervious"] ** 2

    for var in layers:
        mean = scaling_df.loc[var, "mean"]
        std = scaling_df.loc[var, "std"]
        layers[var] = (layers[var] - mean) / std

    posterior = trace.posterior
    n_samples = 300
    prob_maps = []

    for i in tqdm(range(n_samples)):
        mu = (
            posterior["intercept"].values[0, i]
            + posterior["beta_insolation"].values[0, i] * layers["insolation"]
            + posterior["beta_clay"].values[0, i] * layers["clay"]
            + posterior["beta_sand"].values[0, i] * layers["sand"]
            + posterior["beta_silt"].values[0, i] * layers["silt"]
            + posterior["beta_suit"].values[0, i] * layers["proximity"]
            + posterior["beta_flooded"].values[0, i] * layers["flood"]
            + posterior["beta_agri"].values[0, i] * layers["agriculture"]
            + posterior["beta_building_bright"].values[0, i] * layers["impervious"]
            + posterior["beta_building_bright2"].values[0, i] * layers["impervious_squared"]
            + posterior["beta_veg"].values[0, i] * layers["vegetation"]
            + posterior["beta_veg2"].values[0, i] * layers["vegetation_squared"]
        )
        p = 1 / (1 + np.exp(-mu))
        prob_maps.append(p)

    prob_stack = np.stack(prob_maps)
    mean_map = prob_stack.mean(axis=0).reshape((height, width))
    std_map = prob_stack.std(axis=0).reshape((height, width))

    meta.update({"dtype": "float32", "count": 1})

    with rasterio.open("/Users/robbe_neyns/Documents/research/bee mapping2/Bayesian modelling/presence_mean_map.tif", "w", **meta) as dst:
        dst.write(mean_map.astype(np.float32), 1)

    with rasterio.open("/Users/robbe_neyns/Documents/research/bee mapping2/Bayesian modelling/presence_std_map.tif", "w", **meta) as dst:
        dst.write(std_map.astype(np.float32), 1)

    print("✅ Saved: presence_mean_map.tif and presence_std_map.tif")
