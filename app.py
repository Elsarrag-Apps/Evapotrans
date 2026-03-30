import math
import io
from datetime import datetime

try:
    import ee
except Exception:
    ee = None

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod
from streamlit_folium import st_folium

st.set_page_config(page_title="Site ET Tool", layout="wide")
st.title("Site Evapotranspiration Tool")
st.caption("Upload an EPW file, draw one site polygon, and optionally add tree polygons.")

# -----------------------------
# Helpers
# -----------------------------
EPW_COLUMNS = [
    "Year", "Month", "Day", "Hour", "Minute", "DataSource",
    "DryBulb", "DewPoint", "RH", "Pressure",
    "ExtraterrestrialHorizontalRadiation",
    "ExtraterrestrialDirectNormalRadiation",
    "HorizontalInfraredRadiation",
    "GlobalHorizontalRadiation",
    "DirectNormalRadiation",
    "DiffuseHorizontalRadiation",
    "GlobalHorizontalIlluminance",
    "DirectNormalIlluminance",
    "DiffuseHorizontalIlluminance",
    "ZenithLuminance",
    "WindDirection", "WindSpeed",
    "TotalSkyCover", "OpaqueSkyCover",
    "Visibility", "CeilingHeight",
    "PresentWeatherObservation", "PresentWeatherCodes",
    "PrecipitableWater", "AerosolOpticalDepth",
    "SnowDepth", "DaysSinceLastSnowfall",
    "Albedo", "LiquidPrecipitationDepth",
    "LiquidPrecipitationQuantity"
]


def read_epw(uploaded_file):
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    if len(raw) < 9:
        raise ValueError("The uploaded EPW file looks incomplete.")

    header = raw[0].split(",")

    def safe_float(parts, idx, default):
        try:
            return float(parts[idx])
        except Exception:
            return default

    meta = {
        "city": header[1].strip() if len(header) > 1 else "",
        "country": header[3].strip() if len(header) > 3 else "",
        "latitude": safe_float(header, 6, 54.5),
        "longitude": safe_float(header, 7, -2.5),
        "timezone": safe_float(header, 8, 0.0),
        "elevation": safe_float(header, 9, 0.0),
    }

    df = pd.read_csv(io.StringIO("\n".join(raw[8:])), header=None)
    if df.shape[1] < 22:
        raise ValueError("The EPW data columns could not be read correctly.")

    df = df.iloc[:, :35].copy()
    df.columns = EPW_COLUMNS
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int) - 1
    for col in ["Year", "Month", "Day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Year", "Month", "Day"]).copy()
    df[["Year", "Month", "Day"]] = df[["Year", "Month", "Day"]].astype(int)

    df["timestamp"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Hour"]),
        errors="coerce"
    )

    numeric_cols = ["DryBulb", "RH", "Pressure", "GlobalHorizontalRadiation", "WindSpeed"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return meta, df


def saturation_vapor_pressure_kpa(t_c):
    return 0.6108 * math.exp((17.27 * t_c) / (t_c + 237.3))


def slope_vapor_pressure_curve_kpa_per_c(t_c):
    es = saturation_vapor_pressure_kpa(t_c)
    return 4098 * es / ((t_c + 237.3) ** 2)


def psychrometric_constant_kpa_per_c(pressure_kpa):
    return 0.000665 * pressure_kpa


def hourly_et0_fao56(row):
    t = float(row["DryBulb"])
    rh = max(0.0, min(100.0, float(row["RH"])))
    p_kpa = float(row["Pressure"]) / 1000.0 if pd.notna(row["Pressure"]) else 101.325
    u2 = max(0.0, float(row["WindSpeed"])) if pd.notna(row["WindSpeed"]) else 0.0
    rs_mj = max(0.0, float(row["GlobalHorizontalRadiation"])) * 0.0036 if pd.notna(row["GlobalHorizontalRadiation"]) else 0.0

    es = saturation_vapor_pressure_kpa(t)
    ea = es * rh / 100.0
    vpd = max(0.0, es - ea)
    delta = slope_vapor_pressure_curve_kpa_per_c(t)
    gamma = psychrometric_constant_kpa_per_c(p_kpa)

    # Simplified hourly net radiation for a first app version.
    albedo = 0.23
    rns = (1 - albedo) * rs_mj
    rnl = 0.0
    rn = max(0.0, rns - rnl)
    g = 0.1 * rn if rs_mj > 0 else 0.5 * rn

    num = 0.408 * delta * (rn - g) + gamma * (37.0 / (t + 273.0)) * u2 * vpd
    den = delta + gamma * (1 + 0.34 * u2)
    et0 = num / den if den > 0 else 0.0
    return max(0.0, et0)


def area_m2(geom):
    geod = Geod(ellps="WGS84")
    area, _ = geod.geometry_area_perimeter(geom)
    return abs(float(area))


def geometry_to_ee(geom):
    return ee.Geometry(mapping(geom))


def init_ee():
    if ee is None:
        return False
    if st.session_state.get("ee_initialized"):
        return True

    project = None
    try:
        project = st.secrets["EE_PROJECT"]
    except Exception:
        project = None

    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        st.session_state["ee_initialized"] = True
        return True
    except Exception:
        return False


def sentinel_ndvi_kc_stats(aoi_geom, tree_geoms):
    if not init_ee():
        raise RuntimeError("Earth Engine is not initialized. Add EE credentials or use the fallback coefficients.")

    aoi_ee = geometry_to_ee(aoi_geom)
    start = "2024-04-01"
    end = "2024-09-30"

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_ee)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    if col.size().getInfo() == 0:
        raise RuntimeError("No Sentinel-2 images were found for the AOI and date window.")

    img = col.median().clip(aoi_ee)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    kc = ndvi.subtract(0.2).divide(0.6).clamp(0.2, 1.05).rename("Kc")

    site_stats = kc.addBands(ndvi).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi_ee,
        scale=10,
        maxPixels=1e10
    ).getInfo() or {}

    results = {
        "site_kc": float(site_stats.get("Kc", 0.5)),
        "site_ndvi": float(site_stats.get("NDVI", 0.3)),
    }

    if tree_geoms:
        tree_union = unary_union(tree_geoms)
        tree_ee = geometry_to_ee(tree_union)
        tree_stats = kc.addBands(ndvi).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=tree_ee,
            scale=10,
            maxPixels=1e10
        ).getInfo() or {}

        rem_geom = aoi_geom.difference(tree_union)
        if not rem_geom.is_empty:
            rem_ee = geometry_to_ee(rem_geom)
            rem_stats = kc.addBands(ndvi).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=rem_ee,
                scale=10,
                maxPixels=1e10
            ).getInfo() or {}
        else:
            rem_stats = {}

        results.update({
            "tree_kc": float(tree_stats.get("Kc", 0.9)),
            "tree_ndvi": float(tree_stats.get("NDVI", 0.6)),
            "rem_kc": float(rem_stats.get("Kc", results["site_kc"])),
            "rem_ndvi": float(rem_stats.get("NDVI", results["site_ndvi"])),
        })
    else:
        results.update({
            "tree_kc": np.nan,
            "tree_ndvi": np.nan,
            "rem_kc": results["site_kc"],
            "rem_ndvi": results["site_ndvi"],
        })

    return results


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Inputs")
    epw_file = st.file_uploader("Upload EPW", type=["epw"])
    clear = st.button("Clear polygons")
    run = st.button("Run model", type="primary")
    st.markdown("**Drawing rule**")
    st.caption("First polygon = site AOI. Any additional polygons = tree zones.")

# -----------------------------
# Session state
# -----------------------------
if "center" not in st.session_state:
    st.session_state.center = [54.5, -2.5]
if "zoom" not in st.session_state:
    st.session_state.zoom = 6

meta = None
df_weather = None
if epw_file is not None:
    try:
        meta, df_weather = read_epw(epw_file)
        st.session_state.center = [meta["latitude"], meta["longitude"]]
        st.session_state.zoom = 15
    except Exception as e:
        st.error(f"Could not read the EPW file: {e}")

# -----------------------------
# Map
# -----------------------------
m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom, control_scale=True)
Draw(
    export=False,
    draw_options={
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "polygon": True,
    },
    edit_options={"edit": True, "remove": True},
).add_to(m)

map_data = st_folium(m, width=1100, height=650, key="site_et_map")

if clear:
    st.session_state["site_et_map"] = None
    st.rerun()

all_drawings = map_data.get("all_drawings", []) if map_data else []
polygons = [shape(feat["geometry"]) for feat in all_drawings if feat.get("geometry", {}).get("type") == "Polygon"]
aoi_geom = polygons[0] if len(polygons) >= 1 else None
tree_geoms = polygons[1:] if len(polygons) >= 2 else []

if aoi_geom is not None and tree_geoms:
    tree_geoms = [g.intersection(aoi_geom) for g in tree_geoms]
    tree_geoms = [g for g in tree_geoms if not g.is_empty]

# -----------------------------
# Geometry summary
# -----------------------------
col1, col2, col3 = st.columns(3)
if aoi_geom is not None:
    aoi_area = area_m2(aoi_geom)
    tree_area = sum(area_m2(g) for g in tree_geoms) if tree_geoms else 0.0
    rem_area = max(0.0, aoi_area - tree_area)
    col1.metric("Site area", f"{aoi_area:,.0f} m2")
    col2.metric("Tree area", f"{tree_area:,.0f} m2")
    col3.metric("Tree cover", f"{(100 * tree_area / aoi_area):.1f}%" if aoi_area > 0 else "0%")
else:
    col1.metric("Site area", "-")
    col2.metric("Tree area", "-")
    col3.metric("Tree cover", "-")

if meta:
    st.info(f"EPW location: {meta['city']}, {meta['country']} ({meta['latitude']:.4f}, {meta['longitude']:.4f})")

# -----------------------------
# Run model
# -----------------------------
if run:
    if epw_file is None:
        st.error("Upload an EPW file first.")
    elif aoi_geom is None:
        st.error("Draw the main site polygon first.")
    else:
        with st.spinner("Calculating hourly ET0 from EPW..."):
            df = df_weather.copy()
            df = df.dropna(subset=["timestamp", "DryBulb", "RH", "GlobalHorizontalRadiation", "WindSpeed"]).copy()
            if df.empty:
                st.error("No valid hourly rows were found in the EPW after parsing.")
                st.stop()
            df["ET0_mm_h"] = df.apply(hourly_et0_fao56, axis=1)

        with st.spinner("Fetching NDVI and Kc from Earth Engine..."):
            try:
                stats = sentinel_ndvi_kc_stats(aoi_geom, tree_geoms)
            except Exception as e:
                st.warning(f"Earth Engine step did not run: {e}")
                stats = {
                    "site_kc": 0.6,
                    "site_ndvi": 0.35,
                    "tree_kc": 0.95 if tree_geoms else np.nan,
                    "tree_ndvi": 0.65 if tree_geoms else np.nan,
                    "rem_kc": 0.45,
                    "rem_ndvi": 0.25,
                }

        df["ET_site_mm_h"] = df["ET0_mm_h"] * stats["site_kc"]
        if tree_geoms:
            df["ET_tree_mm_h"] = df["ET0_mm_h"] * stats["tree_kc"]
            df["ET_rem_mm_h"] = df["ET0_mm_h"] * stats["rem_kc"]

        st.subheader("Spatial coefficients")
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean site NDVI", f"{stats['site_ndvi']:.3f}")
        s2.metric("Mean site Kc", f"{stats['site_kc']:.3f}")
        s3.metric("Mean ET0", f"{df['ET0_mm_h'].mean():.3f} mm/h")

        if tree_geoms:
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Tree NDVI", f"{stats['tree_ndvi']:.3f}")
            t2.metric("Tree Kc", f"{stats['tree_kc']:.3f}")
            t3.metric("Remainder NDVI", f"{stats['rem_ndvi']:.3f}")
            t4.metric("Remainder Kc", f"{stats['rem_kc']:.3f}")

        st.subheader("Hourly results")
        plot_cols = ["ET0_mm_h", "ET_site_mm_h"]
        if tree_geoms:
            plot_cols += ["ET_tree_mm_h", "ET_rem_mm_h"]
        st.line_chart(df.set_index("timestamp")[plot_cols])

        daily = df.set_index("timestamp")[plot_cols].resample("D").sum()
        st.subheader("Daily totals")
        st.line_chart(daily)

        st.subheader("Summary")
        summary = {
            "Total ET0 (mm)": df["ET0_mm_h"].sum(),
            "Total site ET (mm)": df["ET_site_mm_h"].sum(),
        }
        if tree_geoms:
            summary["Total tree ET (mm)"] = df["ET_tree_mm_h"].sum()
            summary["Total remainder ET (mm)"] = df["ET_rem_mm_h"].sum()
        st.dataframe(pd.DataFrame(summary.items(), columns=["Metric", "Value"]).style.format({"Value": "{:.2f}"}), use_container_width=True)

        csv_data = df[["timestamp"] + plot_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_data, file_name="site_et_results.csv", mime="text/csv")
