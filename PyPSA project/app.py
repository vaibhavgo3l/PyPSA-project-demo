# app.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Optional dependency used by your backend/model
import pypsa

# If you have Main_model_3.py in the same folder, ensure run_model(...) returns the network.
# Comment this import out if you don't use it.
try:
    from Main_model_3 import run_model
except Exception:
    run_model = None


# -------------------------
# Global page config
# -------------------------
st.set_page_config(page_title="GB Power Market Stack Modelling — Demonstration Tool (PyPSA + Streamlit)",
                   layout="wide")
st.title("GB Power Market Stack Modelling — Demonstration Tool (PyPSA + Streamlit)")


# -------------------------
# Helpers shared across tabs
# -------------------------
def _base_dir() -> Path:
    try:
        return Path(__file__).parent
    except NameError:
        return Path.cwd()


# =========================
# Tab A: Model 3 Generation Stack
# =========================
def page_model3():
    st.subheader("GB Generation Stack: Simple model for 2025 with real carbon and gas prices.")

    if run_model is None:
        st.warning("`Main_model_3.run_model(...)` not found. Make sure Main_model_3.py is in the same folder.")
        return

    def _pickle_path() -> Path:
        return _base_dir() / "uk_stack_inputs_3.pkl"

    @st.cache_data(show_spinner=False)
    def _load_pickle_from_same_dir(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Could not find {path.name} next to the app.")
        return pd.read_pickle(path)

    def _compute_marginal_cost_columns_2(
            ts_data: pd.DataFrame,
            installed_capacity: pd.DataFrame,
            *,
            global_mult: float = 1.0,
            gas_mult: float = 1.0,
            oil_mult: float = 1.0,
            biomass_mult: float = 1.0,
            coal_mult: float = 1.0,
            carbon_mult: float = 1.0,
    ) -> pd.DataFrame:
        """
        Compute marginal cost time series per fuel with optional price multipliers.

        MC_t = (fuel_price_t / η) + (EF_th / η) * carbon_price_t + VOM

        Inputs:
          - ts_data: DataFrame with fuel price columns and optional carbon_price
          - installed_capacity: rows keyed by 'Fuel' with columns:
                'Efficiency (%)', 'CO₂ content (tCO₂/MWh_th)', 'vom_adders'
          - *_mult: scalars in [0, 2], default 1.0. Applied to corresponding price series.
                    'global_mult' multiplies all price series in addition to per-series multipliers.

        Returns:
          - ts_data copy with new columns: mc_gas, mc_oil, mc_biomass, mc_coal (as available)
        """
        ts = ts_data.copy()
        ic = installed_capacity.set_index("Fuel")

        # Parameters from installed capacity
        eta = ic.get("Efficiency (%)", pd.Series(dtype=float)) / 100.0
        EF_th = ic.get("CO₂ content (tCO₂/MWh_th)", pd.Series(dtype=float)).fillna(0.0)
        VOM = ic.get("vom_adders", pd.Series(dtype=float)).fillna(0.0)

        # Map fuels -> ts columns and -> output suffix
        price_cols = {
            "Gas fired": "gas_price",
            "Oil fired": "oil_price_ts",
            "Bioenergy and waste": "biomass_price_ts",
            "Coal fired": "coal_price_ts",
        }
        key_map = {"Gas fired": "gas", "Oil fired": "oil", "Bioenergy and waste": "biomass", "Coal fired": "coal"}

        # Apply multipliers (per series × global)
        per_series_mult = {
            "gas_price": gas_mult,
            "oil_price_ts": oil_mult,
            "biomass_price_ts": biomass_mult,
            "coal_price_ts": coal_mult,
            "carbon_price": carbon_mult,
        }
        for col, m in per_series_mult.items():
            if col in ts.columns:
                ts[col] = ts[col] * (m * global_mult)

        # Compute marginal costs
        for fuel_label, price_col in price_cols.items():
            if price_col not in ts.columns or fuel_label not in ic.index:
                continue

            eff = eta.get(fuel_label, np.nan)
            if pd.isna(eff) or eff <= 0:
                continue

            ef = EF_th.get(fuel_label, 0.0)
            vom = VOM.get(fuel_label, 0.0)
            key = key_map[fuel_label]

            if "carbon_price" in ts.columns:
                ts[f"mc_{key}"] = (ts[price_col] / eff) + (ef / eff) * ts["carbon_price"] + vom
            else:
                ts[f"mc_{key}"] = (ts[price_col] / eff) + vom

        return ts
    def _compute_marginal_cost_columns(ts_data: pd.DataFrame, installed_capacity: pd.DataFrame) -> pd.DataFrame:
        """
        MC_t = (fuel_price_t / η) + (EF_th / η) * carbon_price_t + VOM
        Expects in installed_capacity: Efficiency (%), CO₂ content (tCO₂/MWh_th), vom_adders
        """
        ts = ts_data.copy()
        ic = installed_capacity.set_index("Fuel")

        eta = ic.get("Efficiency (%)", pd.Series(dtype=float)) / 100.0
        EF_th = ic.get("CO₂ content (tCO₂/MWh_th)", pd.Series(dtype=float)).fillna(0.0)
        VOM = ic.get("vom_adders", pd.Series(dtype=float)).fillna(0.0)

        fuel_price_series = {
            "Gas fired": ts.get("gas_price"),
            "Oil fired": ts.get("oil_price_ts"),
            "Bioenergy and waste": ts.get("biomass_price_ts"),
            "Coal fired": ts.get("coal_price_ts"),
        }
        key_map = {"Gas fired": "gas", "Oil fired": "oil", "Bioenergy and waste": "biomass", "Coal fired": "coal"}

        for fuel_label, price_ts in fuel_price_series.items():
            if price_ts is None or fuel_label not in ic.index:
                continue
            eff = eta.get(fuel_label, np.nan)
            if pd.isna(eff) or eff <= 0:
                continue
            ef = EF_th.get(fuel_label, 0.0)
            vom = VOM.get(fuel_label, 0.0)
            key = key_map[fuel_label]
            if "carbon_price" in ts:
                ts[f"mc_{key}"] = (price_ts / eff) + (ef / eff) * ts["carbon_price"] + vom
            else:
                ts[f"mc_{key}"] = (price_ts / eff) + vom

        return ts

    def _extract_results_from_network(network):
        dispatch = network.generators_t.p.copy()  # MW by generator name

        # Load: use 'Demand' if present
        if not network.loads_t.p_set.empty:
            load = (
                network.loads_t.p_set["Demand"]
                if "Demand" in network.loads_t.p_set.columns
                else network.loads_t.p_set.iloc[:, 0]
            ).rename("Load")
        else:
            load = pd.Series(dtype=float, name="Load", index=dispatch.index)

        # MCs: time-varying where available, otherwise static from generators table
        mc_df = pd.DataFrame(index=dispatch.index)
        for gen in dispatch.columns:
            if hasattr(network, "generators_t") and hasattr(network.generators_t, "marginal_cost") \
               and gen in network.generators_t.marginal_cost.columns:
                mc_df[gen] = network.generators_t.marginal_cost[gen]
            else:
                mc_df[gen] = float(network.generators.loc[gen, "marginal_cost"])

        # Total system cost per snapshot (€/h)
        total_cost = (dispatch * mc_df).sum(axis=1).rename("total_cost_eur_per_h")

        # Order generators by average MC (cheapest → priciest)
        order = mc_df.mean(axis=0).sort_values().index.tolist()
        return dispatch, load, mc_df, total_cost, order

    def _plot_stack(dispatch: pd.DataFrame, load: pd.Series, mc_df: pd.DataFrame, order, total_cost: pd.Series):
        fig = go.Figure()
        for gen in order:
            if gen not in dispatch.columns:
                continue
            y = dispatch[gen]
            mc = mc_df[gen] if gen in mc_df.columns else pd.Series(0.0, index=dispatch.index)
            fig.add_trace(
                go.Scatter(
                    name=gen,
                    x=dispatch.index,
                    y=y,
                    mode="lines",
                    stackgroup="one",
                    customdata=np.c_[mc.values],
                    hovertemplate=(
                        "%{fullData.name}: %{y:.2f} MW<br>"
                        "MC: %{customdata[0]:.2f} €/MWh<br>"
                        "%{x}<extra></extra>"
                    ),
                )
            )

        fig.add_trace(
            go.Scatter(
                name="Load",
                x=load.index,
                y=load,
                mode="lines",
                line=dict(width=2),
                customdata=np.c_[total_cost.values],
                hovertemplate="Load: %{y:.2f} MW<br>Total cost: %{customdata[0]:,.0f} €/h<br>%{x}<extra></extra>",
            )
        )

        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Time",
            yaxis_title="Power (MW)",
        )
        return fig

    st.info("Click **Run Model** to load `uk_stack_inputs_3.pkl`, execute the backend, and render the stack plot.")
    run = st.button("Run Model", type="primary", key="run_model3")
    col2, col3, col4, col5, col6 = st.columns(5)

    global_mult = 1

    with col2:
        gas_mult = st.slider("Gas price ×", 0.0, 2.0, 1.0, 0.01)

    with col3:
        oil_mult = st.slider("Oil price ×", 0.0, 2.0, 1.0, 0.01)

    with col4:
        biomass_mult = st.slider("Biomass price ×", 0.0, 2.0, 1.0, 0.01)

    with col5:
        coal_mult = st.slider("Coal price ×", 0.0, 2.0, 1.0, 0.01)

    with col6:
        carbon_mult = st.slider("Carbon price ×", 0.0, 2.0, 1.0, 0.01)

    multipliers = dict(
        global_mult=global_mult,
        gas_mult=gas_mult,
        oil_mult=oil_mult,
        biomass_mult=biomass_mult,
        coal_mult=coal_mult,
        carbon_mult=carbon_mult,
    )

    if run:
        try:
            pkl_path = _pickle_path()
            bundle = _load_pickle_from_same_dir(pkl_path)

            ts_data = bundle["ts_data"]
            installed_capacity = bundle["installed_capacity"]

            # Recreate the marginal-cost time series (frontend step)
            ts_data = _compute_marginal_cost_columns_2(ts_data, installed_capacity, **multipliers)

            # Call backend (ensure run_model returns a Network)
            network = run_model(ts_data, installed_capacity)

            # Extract data for plotting
            dispatch, load, mc_df, total_cost, order = _extract_results_from_network(network)

            st.success("Model run completed.")
            fig = _plot_stack(dispatch, load, mc_df, order, total_cost)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Run failed: {e}")


# =========================
# Tab B: PyPSA Network Dashboard (all inline controls; no sidebar)
# =========================
def page_pypsa():
    st.subheader("PyPSA UK Grid Model Dashboard")

    BASE_DIR = _base_dir()
    DEFAULT_NC = BASE_DIR / "solved_network.nc"
    DEFAULT_HTML = BASE_DIR / "network_map_folium.html"

    # File/path controls inline

    c1, c2 = st.columns([3, 2])
    with c1:
        nc_path_input = str(DEFAULT_NC)
    with c2:
        show_raw = st.checkbox("Show raw dispatch table", value=False, key="pypsa_show_raw")

    nc_path = Path(nc_path_input).expanduser().resolve() if nc_path_input else DEFAULT_NC

    def load_network_from_path(p: Path):
        try:
            return pypsa.Network(str(p))
        except Exception:
            # Fallback: specify engine if needed
            return pypsa.Network(str(p), xarray_open_dataset_kwargs={"engine": "netcdf4"})

    network = None
    if nc_path.exists():
        try:
            network = load_network_from_path(nc_path)
        except Exception as e:
            st.error(f"Error loading network from path:\n{e}")
    else:
        st.warning(f"NetCDF file not found at: {nc_path}")

    if network is not None:
        # Try to get dispatch
        try:
            df = network.generators_t.p
        except Exception as e:
            st.error(f"Could not access generator dispatch from the network: {e}")
            df = None

        if df is not None and not df.empty:
            # Inline controls (no sidebar)
            st.markdown("**Filters & grouping**")
            cA, cB, cC = st.columns([1.2, 1.2, 1.0])

            all_carriers = sorted(network.generators.carrier.unique().tolist())
            all_regions = sorted(network.buses.index.unique().tolist())

            with cA:
                selected_carriers = st.multiselect(
                    "Carriers (types)", all_carriers, default=all_carriers, key="pypsa_carriers"
                )
            with cB:
                selected_regions = st.multiselect(
                    "Regions (buses)", all_regions, default=all_regions, key="pypsa_regions"
                )
            with cC:
                group_by = st.radio(
                    "Group/stack by",
                    ["Carrier (Type)", "Region"],
                    index=0,
                    horizontal=True,
                    key="pypsa_groupby",
                )

            # Time range slider under the filters
            idx_len = len(df.index)
            start_i, end_i = st.slider(
                "Time range (by snapshot index)",
                min_value=0,
                max_value=idx_len - 1,
                value=(0, idx_len - 1),
                key="pypsa_time_range",
            )

            # Derive filtered generator list
            selected_gens = network.generators.index.tolist()
            if selected_regions:
                selected_gens = [g for g in selected_gens if network.generators.at[g, "bus"] in selected_regions]
            if selected_carriers:
                selected_gens = [g for g in selected_gens if network.generators.at[g, "carrier"] in selected_carriers]

            df_filtered = df[selected_gens] if selected_gens else df.iloc[:0, :]

            # Slice time
            if not df_filtered.empty:
                df_filtered = df_filtered.iloc[start_i: end_i + 1]

            # Aggregate
            if not df_filtered.empty:
                if group_by == "Region":
                    gen_to_bus = [network.generators.at[gen, "bus"] for gen in df_filtered.columns]
                    df_aggregated = df_filtered.groupby(gen_to_bus, axis=1).sum()
                    title = "Dispatch stacked by Region"
                else:
                    gen_to_carrier = [network.generators.at[gen, "carrier"] for gen in df_filtered.columns]
                    df_aggregated = df_filtered.groupby(gen_to_carrier, axis=1).sum()
                    title = "Dispatch stacked by Carrier (Type)"
            else:
                df_aggregated, title = df_filtered, "Dispatch"

            # Plot
            if not df_aggregated.empty:
                fig = px.area(
                    df_aggregated,
                    x=df_aggregated.index,
                    y=df_aggregated.columns,
                    title=title,
                    labels={"value": "Power (MW)", "variable": "Group", "index": "Time"},
                )
                fig.update_layout(yaxis_title="Power (MW)", xaxis_title="Time", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data to plot — adjust filters or time range.")

            # Optional raw table
            if show_raw:
                st.dataframe(df_aggregated)

        else:
            st.info("No dispatch data found in the loaded network.")

        # Map (kept inside the tab)
        st.markdown("---")
        st.subheader("Network Map")
        html_path = DEFAULT_HTML
        if html_path.exists():
            components.html(html_path.read_text(encoding="utf-8"), height=700, scrolling=False)
        else:
            st.warning(f"Map file not found next to this script: {html_path.name}")
    else:
        st.info("Provide a valid NetCDF path (.nc) if you want the dispatch plots and map.")


# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["Single-Node GB Stack Model — Simplified Dispatch Simulation", "Multi-Zonal GB Network Model — Transmission-Constrained Dispatch"])
with tab1:
    txt = "This model simulates GB’s generation stack for 2025 using simplified assumptions. Fuel prices, carbon costs, demand, and capacity factors can be adjusted interactively to observe how the supply-demand balance shifts. Use the multipliers to make the base prices more cheap/expensive. Note: only real gas and carbon prices were used; other fuel prices were derived from gas. **Use the crop functionality of the graph to zoom into a specific timeframe.**"
    st.write(txt)
    page_model3()
with (tab2):
    txt ="This model demonstrates PyPSA’s ability to handle regional dispatch complexity, including transmission constraints and varying renewable resource availability. While the dataset is illustrative, the framework is scalable and can integrate real-world market datasets to support competitive stack modelling. **I've also added a map with transmission lines at the bottom to show that it's possible to create spatial graphs for visualisation. Being able to see transmission constraints like this can be very useful.**"
    st.write(txt)
    st.success(
        """
        **Key Capabilities Demonstrated:**
        - Multi-zonal GB grid with **6 buses** *(Scotland, North, Midlands, Wales, SW, SE)*
        - Transmission constraints based on **realistic ESO capacities**
        - Regional generation mixes and **time-varying renewable profiles**
        - **Economic dispatch optimization** using PyPSA's solver
        """
    )
    page_pypsa()
