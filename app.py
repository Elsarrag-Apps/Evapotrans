import math
import io

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

# -----------------------------
# Company logo (optional)
# -----------------------------
# Place your logo file in the repo (e.g., "logo.png") or use a URL
try:
    st.image("logo.png", width=180)
except Exception:
    pass

st.title("Site Evapotranspiration Tool")
st.caption("Upload an EPW file, draw one site polygon, and optionally add additional polygons for trees, grass, water, or hardscape overrides.")
st.markdown("""
### How to use
1. Upload an EPW file
2. Draw the main site polygon first
3. Draw any additional polygons you want to classify
4. In the polygon list below the map, assign each extra polygon to a type
5. Click **Run model**
""")
st.info("The EPW is treated as a single representative year (2024) for hourly, daily, monthly, and annual summaries.")

# -----------------------------
# Helpers
# -----------------------------
ZONE_OPTIONS = ["Trees", "Grass / planting", "Water", "Hardscape"]
ZONE_KC = {
    "Trees": 0.95,
    "Grass / planting": 0.65,
    "Water": 1.05,
    "Hardscape": 0.20,
}

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


def make_representative_timestamp(month_series, day_series, hour_series, fixed_year=2024):
    return pd.to_datetime(
        dict(year=fixed_year, month=month_series, day=day_series, hour=hour_series),
        errors="coerce"
    )


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

    # EPW exports can vary slightly in column count across sources.
    # Keep the available columns, then pad any missing standard fields with NaN.
    n_cols = min(df.shape[1], len(EPW_COLUMNS))
    df = df.iloc[:, :n_cols].copy()
    df.columns = EPW_COLUMNS[:n_cols]
    for missing_col in EPW_COLUMNS[n_cols:]:
        df[missing_col] = np.nan
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int) - 1
    for col in ["Year", "Month", "Day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Year", "Month", "Day"]).copy()
    df[["Year", "Month", "Day"]] = df[["Year", "Month", "Day"]].astype(int)

    df["timestamp"] = make_representative_timestamp(df["Month"], df["Day"], df["Hour"], fixed_year=2024)

    required_cols = ["DryBulb", "RH", "Pressure", "GlobalHorizontalRadiation", "WindSpeed", "Hour", "Year", "Month", "Day"]
    for req in required_cols:
        if req not in df.columns:
            raise ValueError(f"The EPW file is missing the required column: {req}")

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


def mm_over_area_to_m3(et_mm, area_m2_value):
    return (et_mm / 1000.0) * area_m2_value


def et_mm_to_cooling_kwh(et_mm, area_m2_value, latent_heat_mj_per_kg=2.45):
    # Latent cooling equivalent only.
    # 1 mm over 1 m2 = 1 kg water, then convert MJ to kWh.
    mass_kg = et_mm * area_m2_value
    energy_mj = mass_kg * latent_heat_mj_per_kg
    return energy_mj / 3.6


def union_geoms(geoms):
    if not geoms:
        return None
    geom = unary_union(geoms)
    return None if geom.is_empty else geom


def geometry_to_ee(geom):
    return ee.Geometry(mapping(geom))


def init_ee():
    if ee is None:
        return False
    if st.session_state.get("ee_initialized"):
        return True

    # Try service account (Streamlit Cloud)
    try:
        credentials = ee.ServiceAccountCredentials(
            st.secrets["EE_CLIENT_EMAIL"],
            key_data=st.secrets["EE_PRIVATE_KEY"],
        )
        ee.Initialize(credentials=credentials, project=st.secrets["EE_PROJECT"])
        st.session_state["ee_initialized"] = True
        return True
    except Exception as e:
        st.session_state["ee_init_error"] = f"EE service account init failed: {e}"

    # Fallback to default (local auth)
    try:
        ee.Initialize()
        st.session_state["ee_initialized"] = True
        return True
    except Exception as e:
        st.session_state["ee_init_error"] = f"EE local init failed: {e}"
        return False


def sentinel_ndvi_kc_stats(aoi_geom, override_geom=None):
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

    tree_mask = ndvi.gte(0.5).rename("tree_mask")
    grass_mask = ndvi.gte(0.3).And(ndvi.lt(0.5)).rename("grass_mask")
    hard_mask = ndvi.lt(0.3).rename("hard_mask")

    kc_auto = (
        tree_mask.multiply(0.95)
        .add(grass_mask.multiply(0.65))
        .add(hard_mask.multiply(0.20))
        .rename("Kc")
    )

    area_img = ee.Image.pixelArea().rename("px_area")

    def zonal_stats(region_geom):
        combined = ee.Image.cat([
            area_img,
            ndvi,
            kc_auto,
            tree_mask,
            grass_mask,
            hard_mask,
            tree_mask.multiply(area_img).rename("tree_area_m2"),
            grass_mask.multiply(area_img).rename("grass_area_m2"),
            hard_mask.multiply(area_img).rename("hard_area_m2")
        ])
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_geom,
            scale=10,
            maxPixels=1e10
        ).getInfo() or {}
        sums = combined.select(["tree_area_m2", "grass_area_m2", "hard_area_m2"]).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region_geom,
            scale=10,
            maxPixels=1e10
        ).getInfo() or {}
        return stats, sums

    site_stats, site_sums = zonal_stats(aoi_ee)
    tree_area_auto = float(site_sums.get("tree_area_m2", 0.0))
    grass_area_auto = float(site_sums.get("grass_area_m2", 0.0))
    hard_area_auto = float(site_sums.get("hard_area_m2", 0.0))
    total_auto = tree_area_auto + grass_area_auto + hard_area_auto

    results = {
        "site_kc": float(site_stats.get("Kc", 0.5)),
        "site_ndvi": float(site_stats.get("NDVI", 0.3)),
        "tree_area_auto_m2": tree_area_auto,
        "grass_area_auto_m2": grass_area_auto,
        "hard_area_auto_m2": hard_area_auto,
        "tree_frac_auto": (tree_area_auto / total_auto) if total_auto > 0 else 0.0,
        "grass_frac_auto": (grass_area_auto / total_auto) if total_auto > 0 else 0.0,
        "hard_frac_auto": (hard_area_auto / total_auto) if total_auto > 0 else 0.0,
        "tree_kc_auto": 0.95,
        "grass_kc_auto": 0.65,
        "hard_kc_auto": 0.20,
    }

    if override_geom is not None and not override_geom.is_empty:
        rem_geom = aoi_geom.difference(override_geom)
        if not rem_geom.is_empty:
            rem_ee = geometry_to_ee(rem_geom)
            rem_stats, rem_sums = zonal_stats(rem_ee)
            rem_tree_area = float(rem_sums.get("tree_area_m2", 0.0))
            rem_grass_area = float(rem_sums.get("grass_area_m2", 0.0))
            rem_hard_area = float(rem_sums.get("hard_area_m2", 0.0))
            rem_total = rem_tree_area + rem_grass_area + rem_hard_area
            results.update({
                "rem_kc": float(rem_stats.get("Kc", results["site_kc"])),
                "rem_ndvi": float(rem_stats.get("NDVI", results["site_ndvi"])),
                "rem_tree_frac_auto": (rem_tree_area / rem_total) if rem_total > 0 else 0.0,
                "rem_grass_frac_auto": (rem_grass_area / rem_total) if rem_total > 0 else 0.0,
                "rem_hard_frac_auto": (rem_hard_area / rem_total) if rem_total > 0 else 0.0,
            })
        else:
            results.update({
                "rem_kc": results["site_kc"],
                "rem_ndvi": results["site_ndvi"],
                "rem_tree_frac_auto": 0.0,
                "rem_grass_frac_auto": 0.0,
                "rem_hard_frac_auto": 0.0,
            })
    else:
        results.update({
            "rem_kc": results["site_kc"],
            "rem_ndvi": results["site_ndvi"],
            "rem_tree_frac_auto": results["tree_frac_auto"],
            "rem_grass_frac_auto": results["grass_frac_auto"],
            "rem_hard_frac_auto": results["hard_frac_auto"],
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
    st.caption("First polygon = site AOI. Additional polygons can be assigned below the map to Trees, Grass / planting, Water, or Hardscape. You can assign more than one polygon to each type.")

# -----------------------------
# Session state
# -----------------------------
if "center" not in st.session_state:
    st.session_state.center = [54.5, -2.5]
if "zoom" not in st.session_state:
    st.session_state.zoom = 6
if "draw_data" not in st.session_state:
    st.session_state.draw_data = []
if "results" not in st.session_state:
    st.session_state.results = None
if "ee_init_error" not in st.session_state:
    st.session_state.ee_init_error = None
if "polygon_zone_types" not in st.session_state:
    st.session_state.polygon_zone_types = {}
if "map_key_suffix" not in st.session_state:
    st.session_state.map_key_suffix = 0

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

# Add Earth Engine NDVI layer if available
try:
    if init_ee():
        aoi_point = ee.Geometry.Point(st.session_state.center[::-1])
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi_point)
              .filterDate("2024-04-01", "2024-09-30")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
              .median())
        ndvi_img = s2.normalizedDifference(["B8", "B4"])

        vis = {"min": 0, "max": 0.8, "palette": ["white", "yellow", "green"]}
        map_id = ndvi_img.getMapId(vis)

        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr="GEE NDVI",
            name="NDVI",
            overlay=True,
            control=True,
            show=False
        ).add_to(m)
except Exception:
    pass

folium.LayerControl().add_to(m)
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

# Reset drawings properly using session state
if clear:
    st.session_state.draw_data = []
    st.session_state.results = None
    st.session_state.polygon_zone_types = {}
    st.session_state.map_key_suffix += 1
    st.rerun()

map_data = st_folium(m, width=1100, height=650, key=f"site_et_map_{st.session_state.map_key_s

if map_data and map_data.get("all_drawings") is not None:
    st.session_state.draw_data = map_data.get("all_drawings", [])

all_drawings = st.session_state.draw_data
polygons = [shape(feat["geometry"]) for feat in all_drawings if feat.get("geometry", {}).get("type") == "Polygon"]
aoi_geom = polygons[0] if len(polygons) >= 1 else None
extra_polygons = polygons[1:] if len(polygons) >= 2 else []

# Keep zone assignments for extra polygons in session state
for idx in range(len(extra_polygons)):
    st.session_state.polygon_zone_types.setdefault(idx, "Trees")
for idx in list(st.session_state.polygon_zone_types.keys()):
    if idx >= len(extra_polygons):
        del st.session_state.polygon_zone_types[idx]

if extra_polygons:
    with st.expander("Polygon type assignments", expanded=True):
        st.caption("Polygon 1 is always the site boundary. Assign each additional polygon to a zone type below.")
        for idx, geom in enumerate(extra_polygons):
            area_label = f"{area_m2(geom.intersection(aoi_geom)) if aoi_geom is not None else area_m2(geom):,.0f} m2"
            st.session_state.polygon_zone_types[idx] = st.selectbox(
                f"Polygon {idx + 2} type ({area_label})",
                ZONE_OPTIONS,
                index=ZONE_OPTIONS.index(st.session_state.polygon_zone_types.get(idx, "Trees")),
                key=f"zone_type_{idx}",
            )

zone_geoms = {z: [] for z in ZONE_OPTIONS}
if aoi_geom is not None:
    for idx, geom in enumerate(extra_polygons):
        clipped = geom.intersection(aoi_geom)
        if not clipped.is_empty:
            zone_geoms[st.session_state.polygon_zone_types.get(idx, "Trees")].append(clipped)

tree_geoms = zone_geoms["Trees"]
grass_geoms = zone_geoms["Grass / planting"]
water_geoms = zone_geoms["Water"]
hard_geoms = zone_geoms["Hardscape"]
override_geom = union_geoms(tree_geoms + grass_geoms + water_geoms + hard_geoms)

# -----------------------------
# Geometry summary
# -----------------------------
col1, col2, col3 = st.columns(3)
aoi_area = None
tree_area = 0.0
grass_area = 0.0
water_area = 0.0
hard_area = 0.0
rem_area = 0.0
if aoi_geom is not None:
    aoi_area = area_m2(aoi_geom)
    tree_area = sum(area_m2(g) for g in tree_geoms) if tree_geoms else 0.0
    grass_area = sum(area_m2(g) for g in grass_geoms) if grass_geoms else 0.0
    water_area = sum(area_m2(g) for g in water_geoms) if water_geoms else 0.0
    hard_area = sum(area_m2(g) for g in hard_geoms) if hard_geoms else 0.0
    manual_total_area = min(tree_area + grass_area + water_area + hard_area, aoi_area)
    rem_area = max(0.0, aoi_area - manual_total_area)
    col1.metric("Site area", f"{aoi_area:,.0f} m2")
    col2.metric("Manual override area", f"{manual_total_area:,.0f} m2")
    col3.metric("Manual override cover", f"{(100 * manual_total_area / aoi_area):.1f}%" if aoi_area > 0 else "0%")
else:
    col1.metric("Site area", "-")
    col2.metric("Manual override area", "-")
    col3.metric("Manual override cover", "-")

if aoi_geom is not None and (tree_area + grass_area + water_area + hard_area) > 0:
    zc1, zc2, zc3, zc4 = st.columns(4)
    zc1.metric("Trees", f"{tree_area:,.0f} m2")
    zc2.metric("Grass / planting", f"{grass_area:,.0f} m2")
    zc3.metric("Water", f"{water_area:,.0f} m2")
    zc4.metric("Hardscape", f"{hard_area:,.0f} m2")

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
            df["timestamp"] = make_representative_timestamp(df["Month"], df["Day"], df["Hour"], fixed_year=2024)
            df = df.dropna(subset=["timestamp", "DryBulb", "RH", "GlobalHorizontalRadiation", "WindSpeed"]).copy()
            df = df.sort_values("timestamp").reset_index(drop=True)
            if df.empty:
                st.error("No valid hourly rows were found in the EPW after parsing.")
                st.stop()
            df["ET0_mm_h"] = df.apply(hourly_et0_fao56, axis=1)

        ee_warning = None
        with st.spinner("Fetching NDVI and Kc from Earth Engine..."):
            try:
                stats = sentinel_ndvi_kc_stats(aoi_geom, override_geom=override_geom)
            except Exception as e:
                ee_warning = f"Earth Engine step did not run: {e}"
                stats = {
                    "site_kc": 0.6,
                    "site_ndvi": 0.35,
                    
                    "rem_kc": 0.45,
                    "rem_ndvi": 0.25,
                    "tree_area_auto_m2": 0.0,
                    "grass_area_auto_m2": 0.0,
                    "hard_area_auto_m2": aoi_area or 0.0,
                    "tree_frac_auto": 0.0,
                    "grass_frac_auto": 0.0,
                    "hard_frac_auto": 1.0,
                    "tree_kc_auto": 0.95,
                    "grass_kc_auto": 0.65,
                    "hard_kc_auto": 0.20,
                    "rem_tree_frac_auto": 0.0,
                    "rem_grass_frac_auto": 0.0,
                    "rem_hard_frac_auto": 1.0,
                }

        df["ET_site_mm_h"] = df["ET0_mm_h"] * stats["site_kc"]
        df["ET_tree_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Trees"] if tree_area > 0 else np.nan
        df["ET_grass_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Grass / planting"] if grass_area > 0 else np.nan
        df["ET_water_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Water"] if water_area > 0 else np.nan
        df["ET_hard_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Hardscape"] if hard_area > 0 else np.nan
        df["ET_rem_mm_h"] = df["ET0_mm_h"] * stats["rem_kc"]

        df["Site_ET_m3_h"] = df["ET_site_mm_h"].apply(lambda x: mm_over_area_to_m3(x, aoi_area or 0.0))
        df["Site_Cooling_kWh_h"] = df["ET_site_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, aoi_area or 0.0))
        df["Tree_ET_m3_h"] = df["ET_tree_mm_h"].apply(lambda x: mm_over_area_to_m3(x, tree_area)) if tree_area > 0 else np.nan
        df["Tree_Cooling_kWh_h"] = df["ET_tree_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, tree_area)) if tree_area > 0 else np.nan
        df["Grass_ET_m3_h"] = df["ET_grass_mm_h"].apply(lambda x: mm_over_area_to_m3(x, grass_area)) if grass_area > 0 else np.nan
        df["Grass_Cooling_kWh_h"] = df["ET_grass_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, grass_area)) if grass_area > 0 else np.nan
        df["Water_ET_m3_h"] = df["ET_water_mm_h"].apply(lambda x: mm_over_area_to_m3(x, water_area)) if water_area > 0 else np.nan
        df["Water_Cooling_kWh_h"] = df["ET_water_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, water_area)) if water_area > 0 else np.nan
        df["Hard_ET_m3_h"] = df["ET_hard_mm_h"].apply(lambda x: mm_over_area_to_m3(x, hard_area)) if hard_area > 0 else np.nan
        df["Hard_Cooling_kWh_h"] = df["ET_hard_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, hard_area)) if hard_area > 0 else np.nan
        df["Rem_ET_m3_h"] = df["ET_rem_mm_h"].apply(lambda x: mm_over_area_to_m3(x, rem_area))
        df["Rem_Cooling_kWh_h"] = df["ET_rem_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, rem_area))
        total_parts = ["Rem_ET_m3_h"]
        total_cooling_parts = ["Rem_Cooling_kWh_h"]
        for c in ["Tree_ET_m3_h", "Grass_ET_m3_h", "Water_ET_m3_h", "Hard_ET_m3_h"]:
            if c in df.columns:
                total_parts.append(c)
        for c in ["Tree_Cooling_kWh_h", "Grass_Cooling_kWh_h", "Water_Cooling_kWh_h", "Hard_Cooling_kWh_h"]:
            if c in df.columns:
                total_cooling_parts.append(c)
        df["Total_Weighted_ET_m3_h"] = df[total_parts].fillna(0).sum(axis=1)
        df["Total_Weighted_Cooling_kWh_h"] = df[total_cooling_parts].fillna(0).sum(axis=1)

        st.session_state.results = {
            "df": df,
            "stats": stats,
            "aoi_area": aoi_area,
            "tree_area": tree_area,
            "grass_area": grass_area,
            "water_area": water_area,
            "hard_area": hard_area,
            "rem_area": rem_area,
            "override_present": bool(override_geom),
            "ee_warning": ee_warning,
        }

results = st.session_state.results
if results is not None and isinstance(results.get("df"), pd.DataFrame):
    results["df"] = results["df"].sort_values("timestamp").reset_index(drop=True)
if results is not None:
    df = results["df"]
    stats = results["stats"]
    aoi_area = results["aoi_area"]
    tree_area = results["tree_area"]
    grass_area = results["grass_area"]
    water_area = results["water_area"]
    hard_area = results["hard_area"]
    rem_area = results["rem_area"]
    override_present = results["override_present"]
    ee_warning = results["ee_warning"]

    if ee_warning:
        st.warning(ee_warning)
    elif st.session_state.get("ee_init_error"):
        st.info(st.session_state["ee_init_error"])

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Time Series", "Volumes & Cooling", "Download"])

    with tab1:
        st.subheader("Spatial coefficients")
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean site NDVI", f"{stats['site_ndvi']:.3f}")
        s2.metric("Mean site Kc", f"{stats['site_kc']:.3f}")
        s3.metric("Mean ET0", f"{df['ET0_mm_h'].mean():.3f} mm/h")

        st.subheader("Automatic NDVI zoning")
        z1, z2, z3 = st.columns(3)
        z1.metric("Auto tree cover", f"{100 * stats['tree_frac_auto']:.1f}%")
        z2.metric("Auto grass cover", f"{100 * stats['grass_frac_auto']:.1f}%")
        z3.metric("Auto hard cover", f"{100 * stats['hard_frac_auto']:.1f}%")

        za1, za2, za3 = st.columns(3)
        za1.metric("Auto tree area", f"{stats['tree_area_auto_m2']:,.0f} m2")
        za2.metric("Auto grass area", f"{stats['grass_area_auto_m2']:,.0f} m2")
        za3.metric("Auto hard area", f"{stats['hard_area_auto_m2']:,.0f} m2")

        if override_present:
            st.subheader("Manual override zones")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trees", f"{tree_area:,.0f} m2")
            m2.metric("Grass / planting", f"{grass_area:,.0f} m2")
            m3.metric("Water", f"{water_area:,.0f} m2")
            m4.metric("Hardscape", f"{hard_area:,.0f} m2")
            r1, r2, r3 = st.columns(3)
            r1.metric("Remainder auto tree", f"{100 * stats['rem_tree_frac_auto']:.1f}%")
            r2.metric("Remainder auto grass", f"{100 * stats['rem_grass_frac_auto']:.1f}%")
            r3.metric("Remainder auto hard", f"{100 * stats['rem_hard_frac_auto']:.1f}%")

        st.subheader("Key totals")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Annual site ET", f"{df['ET_site_mm_h'].sum():,.1f} mm")
        k2.metric("Annual weighted ET", f"{df['Total_Weighted_ET_m3_h'].sum():,.0f} m3")
        k3.metric("Annual latent cooling equivalent", f"{df['Total_Weighted_Cooling_kWh_h'].sum():,.0f} kWh")
        k4.metric("Peak hourly latent cooling", f"{df['Total_Weighted_Cooling_kWh_h'].max():,.1f} kWh/h")

        summary = {
            "Total ET0 (mm)": df["ET0_mm_h"].sum(),
            "Total site ET depth (mm)": df["ET_site_mm_h"].sum(),
            "Total weighted ET volume (m3)": df["Total_Weighted_ET_m3_h"].sum(),
            "Total weighted latent cooling equivalent (kWh)": df["Total_Weighted_Cooling_kWh_h"].sum(),
            "Site area (m2)": aoi_area or 0.0,
            "Manual tree area (m2)": tree_area,
            "Remainder area (m2)": rem_area,
            "Auto tree area from NDVI (m2)": stats["tree_area_auto_m2"],
            "Auto grass area from NDVI (m2)": stats["grass_area_auto_m2"],
            "Auto hard area from NDVI (m2)": stats["hard_area_auto_m2"],
        }
        summary["Total remainder ET depth (mm)"] = df["ET_rem_mm_h"].sum()
        summary["Remainder ET volume (m3)"] = df["Rem_ET_m3_h"].sum()
        summary["Remainder latent cooling equivalent (kWh)"] = df["Rem_Cooling_kWh_h"].sum()
        if tree_area > 0:
            summary["Tree ET volume (m3)"] = df["Tree_ET_m3_h"].sum()
            summary["Tree latent cooling equivalent (kWh)"] = df["Tree_Cooling_kWh_h"].sum()
        if grass_area > 0:
            summary["Grass ET volume (m3)"] = df["Grass_ET_m3_h"].sum()
            summary["Grass latent cooling equivalent (kWh)"] = df["Grass_Cooling_kWh_h"].sum()
        if water_area > 0:
            summary["Water ET volume (m3)"] = df["Water_ET_m3_h"].sum()
            summary["Water latent cooling equivalent (kWh)"] = df["Water_Cooling_kWh_h"].sum()
        if hard_area > 0:
            summary["Hardscape ET volume (m3)"] = df["Hard_ET_m3_h"].sum()
            summary["Hardscape latent cooling equivalent (kWh)"] = df["Hard_Cooling_kWh_h"].sum()
        st.dataframe(pd.DataFrame(summary.items(), columns=["Metric", "Value"]).style.format({"Value": "{:.2f}"}), use_container_width=True)

    with tab2:
        st.subheader("Hourly ET")
        plot_cols = ["ET0_mm_h", "ET_site_mm_h", "ET_rem_mm_h"]
        if tree_area > 0:
            plot_cols.append("ET_tree_mm_h")
        if grass_area > 0:
            plot_cols.append("ET_grass_mm_h")
        if water_area > 0:
            plot_cols.append("ET_water_mm_h")
        if hard_area > 0:
            plot_cols.append("ET_hard_mm_h")
        st.line_chart(df.set_index("timestamp")[plot_cols])

        daily = df.set_index("timestamp")[plot_cols].resample("D").sum()
        st.subheader("Daily ET totals")
        st.line_chart(daily)

        monthly = df.set_index("timestamp")[plot_cols].resample("ME").sum()
        st.subheader("Monthly ET totals")
        st.line_chart(monthly)

        annual = df.set_index("timestamp")[plot_cols].resample("YE").sum()
        st.subheader("Annual ET totals")
        st.bar_chart(annual)

    with tab3:
        vol_cols = ["Site_ET_m3_h", "Total_Weighted_ET_m3_h", "Site_Cooling_kWh_h", "Total_Weighted_Cooling_kWh_h", "Rem_ET_m3_h", "Rem_Cooling_kWh_h"]
        if tree_area > 0:
            vol_cols += ["Tree_ET_m3_h", "Tree_Cooling_kWh_h"]
        if grass_area > 0:
            vol_cols += ["Grass_ET_m3_h", "Grass_Cooling_kWh_h"]
        if water_area > 0:
            vol_cols += ["Water_ET_m3_h", "Water_Cooling_kWh_h"]
        if hard_area > 0:
            vol_cols += ["Hard_ET_m3_h", "Hard_Cooling_kWh_h"]

        st.subheader("Hourly water volume and latent cooling equivalent")
        st.line_chart(df.set_index("timestamp")[vol_cols])

        daily_vol = df.set_index("timestamp")[vol_cols].resample("D").sum()
        st.subheader("Daily water volume and latent cooling equivalent")
        st.line_chart(daily_vol)

        monthly_vol = df.set_index("timestamp")[vol_cols].resample("ME").sum()
        st.subheader("Monthly water volume and latent cooling equivalent")
        st.line_chart(monthly_vol)

        annual_vol = df.set_index("timestamp")[vol_cols].resample("YE").sum()
        st.subheader("Annual water volume and latent cooling equivalent")
        st.bar_chart(annual_vol)

    with tab4:
        export_cols = [
            "timestamp", "ET0_mm_h", "ET_site_mm_h", "ET_rem_mm_h", "ET_tree_mm_h", "ET_grass_mm_h", "ET_water_mm_h", "ET_hard_mm_h",
            "Site_ET_m3_h", "Tree_ET_m3_h", "Grass_ET_m3_h", "Water_ET_m3_h", "Hard_ET_m3_h", "Rem_ET_m3_h", "Total_Weighted_ET_m3_h",
            "Site_Cooling_kWh_h", "Tree_Cooling_kWh_h", "Grass_Cooling_kWh_h", "Water_Cooling_kWh_h", "Hard_Cooling_kWh_h", "Rem_Cooling_kWh_h", "Total_Weighted_Cooling_kWh_h"
        ]
        export_cols = [c for c in export_cols if c in df.columns]
        csv_data = df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_data, file_name="site_et_results.csv", mime="text/csv")
        st.caption("Weighted ET volume uses polygon area. Cooling is shown as a latent cooling equivalent based on evapotranspiration, not as a direct air-temperature reduction or HVAC load. NDVI zoning is automatic by default, and any additional polygons can be assigned to Trees, Grass / planting, Water, or Hardscape as manual override zones. You can assign more than one polygon to each type.")
