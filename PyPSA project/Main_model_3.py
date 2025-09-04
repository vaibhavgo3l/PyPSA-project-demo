import pypsa
import pandas as pd
import numpy as np
from pathlib import Path
import os, json

def run_model(ts_data,installed_capacity):

    # -----------------------------
    # Helpers
    # -----------------------------
    ic = installed_capacity.set_index("Fuel")

    def cap(fuel_name):
        """Return installed capacity (MW) for a given row label, or 0 if missing."""
        try:
            return float(ic.loc[fuel_name, "2024 installed capacity (MW)"])
        except KeyError:
            return 0.0

    def vom(fuel_name, default=0.0):
        try:
            v = ic.loc[fuel_name, "vom_adders"]
            return float(v) if pd.notna(v) else default
        except KeyError:
            return default

    # Aggregate wind capacity (one Wind generator using your single wind_cf series)
    wind_cap = cap("Onshore wind") + cap("Offshore wind")
    solar_cap = cap("Solar")
    gas_cap = cap("Gas fired")
    oil_cap = cap("Oil fired")
    biomass_cap = cap("Bioenergy and waste")
    nuclear_cap = cap("Nuclear stations")

    # -----------------------------
    # Build network
    # -----------------------------
    network = pypsa.Network()
    network.set_snapshots(ts_data.index)

    # Bus
    network.add("Bus", "GB")

    # -----------------------------
    # Generators
    # -----------------------------

    # Wind (intermittent)
    if wind_cap > 0:
        network.add("Generator", "Wind",
            bus="GB", p_nom=wind_cap, p_max_pu=1.0, marginal_cost=0.0)
        network.generators_t.p_max_pu["Wind"] = ts_data["wind_cf"].clip(lower=0, upper=1)

    # Solar (intermittent)
    if solar_cap > 0:
        network.add("Generator", "Solar",
            bus="GB", p_nom=solar_cap, p_max_pu=1.0, marginal_cost=0.0)
        network.generators_t.p_max_pu["Solar"] = ts_data["solar_cf"].clip(lower=0, upper=1)

    # Gas (thermal)
    if gas_cap > 0:
        network.add("Generator", "Gas",
            bus="GB", p_nom=gas_cap, p_max_pu=1.0, marginal_cost=0.0)  # set series next
        network.generators_t.marginal_cost["Gas"] = ts_data["mc_gas"]

    # Oil (thermal)
    if oil_cap > 0 and "mc_oil" in ts_data:
        network.add("Generator", "Oil",
            bus="GB", p_nom=oil_cap, p_max_pu=1.0, marginal_cost=0.0)
        network.generators_t.marginal_cost["Oil"] = ts_data["mc_oil"]

    # Biomass & Waste (thermal, CO2-priced as zero already baked into mc_biomass)
    if biomass_cap > 0 and "mc_biomass" in ts_data:
        network.add("Generator", "Biomass",
            bus="GB", p_nom=biomass_cap, p_max_pu=1.0, marginal_cost=0.0)
        network.generators_t.marginal_cost["Biomass"] = ts_data["mc_biomass"]

    # Nuclear (treat as near-zero variable cost; use VOM if you set one)
    if nuclear_cap > 0:
        network.add("Generator", "Nuclear",
            bus="GB", p_nom=nuclear_cap, p_max_pu=1.0, marginal_cost=vom("Nuclear stations", default=0.0))

    # -----------------------------
    # Load (use TSD as demand)
    # -----------------------------
    network.add("Load", "Demand", bus="GB", p_set=0.0)
    network.loads_t.p_set["Demand"] = ts_data["TSD"]

    # -----------------------------
    # Optimize
    # -----------------------------
    network.optimize(solver_name="highs")

    # -----------------------------
    # Results
    # -----------------------------
    print("Objective (total cost):", float(network.objective))
    print("\nGenerator dispatch (first few rows):")
    print(network.generators_t.p.head())

    # If you want totals:
    print("\nTotal generation by tech (MWh):")
    print(network.generators_t.p.sum().round(2))

    return network

if __name__ == "__main__":
    PKL_NAME = "uk_stack_inputs.pkl"

    # Resolve path next to this .py file (works reliably in PyCharm)
    try:
        here = Path(__file__).parent
    except NameError:
        # Fallback if __file__ isn't defined (e.g., interactive run)
        here = Path.cwd()

    pkl_path = here / PKL_NAME

    bundle = pd.read_pickle(pkl_path)
    ts_data = bundle["ts_data"]
    installed_capacity = bundle["installed_capacity"]

    # --- 3) Marginal cost per thermal fuel (€/MWh_el)
    # MC_t = (fuel_price_t / η) + (EF_th / η) * carbon_price_t + VOM
    ic = installed_capacity.set_index("Fuel")
    eta = (ic["Efficiency (%)"] / 100.0)  # convert % → fraction
    EF_th = ic["CO₂ content (tCO₂/MWh_th)"].fillna(0.0)  # biomass already 0; non-thermals NaN→0
    VOM = ic["vom_adders"].fillna(0.0)

    fuel_price_series = {
        "Gas fired": ts_data["gas_price"],
        "Oil fired": ts_data["oil_price_ts"],
        "Bioenergy and waste": ts_data["biomass_price_ts"],
        "Coal fired": ts_data.get("coal_price_ts", pd.Series(index=ts_data.index)),  # only if you add coal row
    }

    # Build MC columns only for fuels present in installed_capacity
    for fuel_label, price_ts in fuel_price_series.items():
        if fuel_label not in ic.index:
            continue
        eta_f = eta.get(fuel_label, np.nan)
        if pd.isna(eta_f) or eta_f <= 0:
            continue  # skip non-thermals / missing efficiency
        ef_f = EF_th.get(fuel_label, 0.0)
        vom_f = VOM.get(fuel_label, 0.0)
        key = {"Gas fired": "gas", "Oil fired": "oil", "Bioenergy and waste": "biomass", "Coal fired": "coal"}[
            fuel_label]
        ts_data[f"mc_{key}"] = (price_ts / eta_f) + (ef_f / eta_f) * ts_data["carbon_price"] + vom_f

    run_model(ts_data, installed_capacity)
