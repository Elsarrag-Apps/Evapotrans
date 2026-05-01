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
import requests
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except Exception:
    ALTAIR_AVAILABLE = False
from folium.plugins import Draw
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Geod
from streamlit_folium import st_folium

st.set_page_config(page_title="Site ET Tool", layout="wide")

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
st.info("The uploaded EPW is treated as a representative weather profile. Its original year is not used for satellite analysis; Month/Day/Hour are mapped onto a fixed display year only for filtering and plotting.")

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


REPRESENTATIVE_WEATHER_YEAR = 2024


def make_representative_timestamp(month_series, day_series, hour_series, fixed_year=REPRESENTATIVE_WEATHER_YEAR):
    return pd.to_datetime(
        dict(year=fixed_year, month=month_series, day=day_series, hour=hour_series),
        errors="coerce"
    )


@st.cache_data(show_spinner=False)
def search_locations(query, limit=5):
    if not query or len(query.strip()) < 3:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
        "countrycodes": "gb"
    }
    headers = {"User-Agent": "site-et-app"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


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
    df["timestamp"] = make_representative_timestamp(df["Month"], df["Day"], df["Hour"], fixed_year=REPRESENTATIVE_WEATHER_YEAR)

    required_cols = ["DryBulb", "RH", "Pressure", "GlobalHorizontalRadiation", "WindSpeed", "Hour", "Year", "Month", "Day"]
    for req in required_cols:
        if req not in df.columns:
            raise ValueError(f"The EPW file is missing the required column: {req}")

    numeric_cols = ["DryBulb", "RH", "Pressure", "GlobalHorizontalRadiation", "WindSpeed", "LiquidPrecipitationDepth"]
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
    mass_kg = et_mm * area_m2_value
    energy_mj = mass_kg * latent_heat_mj_per_kg
    return energy_mj / 3.6


def apply_rain_fed_bucket(et_potential_mm, rain_mm, max_storage_mm=40.0, initial_storage_fraction=0.5):
    """
    Simple hourly rain-fed soil/surface water bucket.

    Potential ET is limited by available stored rainfall. This avoids assuming
    unlimited irrigation for vegetation and wettable ground surfaces.

    Inputs are depths in mm over the relevant surface.
    """
    storage = max_storage_mm * initial_storage_fraction
    actual = []
    storage_trace = []

    for et_pot, rain in zip(et_potential_mm.fillna(0.0), rain_mm.fillna(0.0)):
        rain = max(0.0, float(rain))
        et_pot = max(0.0, float(et_pot))
        storage = min(max_storage_mm, storage + rain)
        et_act = min(et_pot, storage)
        storage = max(0.0, storage - et_act)
        actual.append(et_act)
        storage_trace.append(storage)

    return pd.Series(actual, index=et_potential_mm.index), pd.Series(storage_trace, index=et_potential_mm.index)


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

    try:
        ee.Initialize()
        st.session_state["ee_initialized"] = True
        return True
    except Exception as e:
        st.session_state["ee_init_error"] = f"EE local init failed: {e}"
        return False


def sentinel_ndvi_kc_stats(aoi_geom, override_geom=None, satellite_year=2024):
    if not init_ee():
        raise RuntimeError("Earth Engine is not initialized. Add EE credentials or use the fallback coefficients.")

    aoi_ee = geometry_to_ee(aoi_geom)
    start = f"{satellite_year}-05-01"
    end = f"{satellite_year}-08-31"
    cloud_thresholds = [20, 25, 30, 35, 40]
    col = None
    cloud_threshold_used = None
    image_count_used = 0

    for cloud_limit in cloud_thresholds:
        candidate_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi_ee)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_limit))
        )
        candidate_count = candidate_col.size().getInfo()
        if candidate_count > 0:
            col = candidate_col
            cloud_threshold_used = cloud_limit
            image_count_used = candidate_count
            break

    if col is None:
        raise RuntimeError("No Sentinel-2 images were found for the AOI from May to August, even after relaxing cloud cover to 40%.")

    img = col.median().clip(aoi_ee)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")

    # Classification order matters:
    # 1) Water is detected first using NDWI, because water can have low NDVI and would otherwise be misclassified as hardscape.
    # 2) Trees and grass are then classified from NDVI outside the water mask.
    # 3) Anything remaining is treated as hardscape / low-vegetation surface.
    water_mask = ndwi.gte(0.1).rename("water_mask")
    tree_mask = ndvi.gte(0.5).And(water_mask.Not()).rename("tree_mask")
    grass_mask = ndvi.gte(0.3).And(ndvi.lt(0.5)).And(water_mask.Not()).rename("grass_mask")
    hard_mask = water_mask.Not().And(tree_mask.Not()).And(grass_mask.Not()).rename("hard_mask")

    kc_auto = (
        tree_mask.multiply(0.95)
        .add(grass_mask.multiply(0.65))
        .add(water_mask.multiply(1.05))
        .add(hard_mask.multiply(0.20))
        .rename("Kc")
    )

    area_img = ee.Image.pixelArea().rename("px_area")

    def zonal_stats(region_geom):
        combined = ee.Image.cat([
            area_img,
            ndvi,
            ndwi,
            kc_auto,
            tree_mask,
            grass_mask,
            water_mask,
            hard_mask,
            tree_mask.multiply(area_img).rename("tree_area_m2"),
            grass_mask.multiply(area_img).rename("grass_area_m2"),
            water_mask.multiply(area_img).rename("water_area_m2"),
            hard_mask.multiply(area_img).rename("hard_area_m2")
        ])
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region_geom,
            scale=10,
            maxPixels=1e10
        ).getInfo() or {}
        sums = combined.select(["tree_area_m2", "grass_area_m2", "water_area_m2", "hard_area_m2"]).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region_geom,
            scale=10,
            maxPixels=1e10
        ).getInfo() or {}
        return stats, sums

    site_stats, site_sums = zonal_stats(aoi_ee)
    tree_area_auto = float(site_sums.get("tree_area_m2", 0.0))
    grass_area_auto = float(site_sums.get("grass_area_m2", 0.0))
    water_area_auto = float(site_sums.get("water_area_m2", 0.0))
    hard_area_auto = float(site_sums.get("hard_area_m2", 0.0))
    total_auto = tree_area_auto + grass_area_auto + water_area_auto + hard_area_auto

    results = {
        "site_kc": float(site_stats.get("Kc", 0.5)),
        "site_ndvi": float(site_stats.get("NDVI", 0.3)),
        "tree_area_auto_m2": tree_area_auto,
        "grass_area_auto_m2": grass_area_auto,
        "water_area_auto_m2": water_area_auto,
        "hard_area_auto_m2": hard_area_auto,
        "tree_frac_auto": (tree_area_auto / total_auto) if total_auto > 0 else 0.0,
        "grass_frac_auto": (grass_area_auto / total_auto) if total_auto > 0 else 0.0,
        "water_frac_auto": (water_area_auto / total_auto) if total_auto > 0 else 0.0,
        "hard_frac_auto": (hard_area_auto / total_auto) if total_auto > 0 else 0.0,
        "tree_kc_auto": 0.95,
        "grass_kc_auto": 0.65,
        "water_kc_auto": 1.05,
        "hard_kc_auto": 0.20,
        "satellite_window_start": start,
        "satellite_window_end": end,
        "cloud_threshold_used": cloud_threshold_used,
        "sentinel_image_count": image_count_used,
        "rem_tree_area_auto_m2": 0.0,
        "rem_grass_area_auto_m2": 0.0,
        "rem_water_area_auto_m2": 0.0,
        "rem_hard_area_auto_m2": 0.0,
        "satellite_window_start": start,
        "satellite_window_end": end,
        "cloud_threshold_used": cloud_threshold_used,
        "sentinel_image_count": image_count_used,
    }

    if override_geom is not None and not override_geom.is_empty:
        rem_geom = aoi_geom.difference(override_geom)
        if not rem_geom.is_empty:
            rem_ee = geometry_to_ee(rem_geom)
            rem_stats, rem_sums = zonal_stats(rem_ee)
            rem_tree_area = float(rem_sums.get("tree_area_m2", 0.0))
            rem_grass_area = float(rem_sums.get("grass_area_m2", 0.0))
            rem_water_area = float(rem_sums.get("water_area_m2", 0.0))
            rem_hard_area = float(rem_sums.get("hard_area_m2", 0.0))
            rem_total = rem_tree_area + rem_grass_area + rem_water_area + rem_hard_area
            results.update({
                "rem_kc": float(rem_stats.get("Kc", results["site_kc"])),
                "rem_ndvi": float(rem_stats.get("NDVI", results["site_ndvi"])),
                "rem_tree_frac_auto": (rem_tree_area / rem_total) if rem_total > 0 else 0.0,
                "rem_grass_frac_auto": (rem_grass_area / rem_total) if rem_total > 0 else 0.0,
                "rem_water_frac_auto": (rem_water_area / rem_total) if rem_total > 0 else 0.0,
                "rem_hard_frac_auto": (rem_hard_area / rem_total) if rem_total > 0 else 0.0,
                "rem_tree_area_auto_m2": rem_tree_area,
                "rem_grass_area_auto_m2": rem_grass_area,
                "rem_water_area_auto_m2": rem_water_area,
                "rem_hard_area_auto_m2": rem_hard_area,
            })
        else:
            results.update({
                "rem_kc": results["site_kc"],
                "rem_ndvi": results["site_ndvi"],
                "rem_tree_frac_auto": 0.0,
                "rem_grass_frac_auto": 0.0,
                "rem_water_frac_auto": 0.0,
                "rem_hard_frac_auto": 0.0,
                "rem_tree_area_auto_m2": 0.0,
                "rem_grass_area_auto_m2": 0.0,
                "rem_water_area_auto_m2": 0.0,
                "rem_hard_area_auto_m2": 0.0,
            })
    else:
        results.update({
            "rem_kc": results["site_kc"],
            "rem_ndvi": results["site_ndvi"],
            "rem_tree_frac_auto": results["tree_frac_auto"],
            "rem_grass_frac_auto": results["grass_frac_auto"],
            "rem_water_frac_auto": results["water_frac_auto"],
            "rem_hard_frac_auto": results["hard_frac_auto"],
            "rem_tree_area_auto_m2": results["tree_area_auto_m2"],
            "rem_grass_area_auto_m2": results["grass_area_auto_m2"],
            "rem_water_area_auto_m2": results["water_area_auto_m2"],
            "rem_hard_area_auto_m2": results["hard_area_auto_m2"],
        })

    return results


with st.sidebar:
    st.header("Inputs")
    epw_file = st.file_uploader("Upload EPW", type=["epw"])

    st.markdown("**Satellite classification year**")
    satellite_year = st.number_input("Year used for Sentinel-2 land-cover classification", min_value=2017, max_value=2030, value=2024, step=1)
    st.caption("This year is used only for satellite imagery. The uploaded EPW weather profile is independent of this year.")

    st.markdown("**Rain-fed ET model**")
    use_rain_fed_model = st.checkbox("Limit vegetation/ground ET by rainfall from EPW", value=True)
    max_soil_storage_mm = st.number_input("Maximum rainfall storage (mm)", min_value=5.0, max_value=150.0, value=40.0, step=5.0)
    initial_storage_fraction = st.slider("Initial storage at start of year", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    st.markdown("**Analysis period**")
    analysis_month_options = list(range(1, 13))
    analysis_day_options = list(range(1, 32))

    p1, p2 = st.columns(2)
    with p1:
        start_month = st.selectbox("Start month", analysis_month_options, index=5)
        start_day = st.selectbox("Start day", analysis_day_options, index=0)
    with p2:
        end_month = st.selectbox("End month", analysis_month_options, index=7)
        end_day = st.selectbox("End day", analysis_day_options, index=30)

    try:
        analysis_start_date = pd.to_datetime(f"{REPRESENTATIVE_WEATHER_YEAR}-{start_month:02d}-{start_day:02d}").date()
        analysis_end_date = pd.to_datetime(f"{REPRESENTATIVE_WEATHER_YEAR}-{end_month:02d}-{end_day:02d}").date()
        if analysis_end_date < analysis_start_date:
            st.warning("End date must be after start date within the representative weather profile.")
    except Exception:
        st.warning("Invalid day/month selection. Please choose a valid date.")
        analysis_start_date = pd.to_datetime(f"{REPRESENTATIVE_WEATHER_YEAR}-06-01").date()
        analysis_end_date = pd.to_datetime(f"{REPRESENTATIVE_WEATHER_YEAR}-08-31").date()

    st.markdown("**Location search**")
    search_query = st.text_input("Search address, postcode, or place", placeholder="Start typing at least 3 characters")

    search_results = []
    if search_query and len(search_query.strip()) >= 3:
        try:
            search_results = search_locations(search_query, limit=6)
        except Exception as e:
            st.warning(f"Search failed: {e}")

    selected_result = None
    if search_results:
        labels = [item.get("display_name", "Unknown location") for item in search_results]
        selected_label = st.selectbox("Select a matching location", labels, index=0)
        selected_result = search_results[labels.index(selected_label)]

    if st.button("Go to location"):
        if selected_result is not None:
            lat = float(selected_result["lat"])
            lon = float(selected_result["lon"])
            st.session_state.center = [lat, lon]
            st.session_state.zoom = 18
            st.session_state.map_center_source = "search"
            st.session_state.map_key_suffix += 1
            st.rerun()
        elif search_query:
            st.warning("No matching location found. Try a fuller address or postcode.")

    clear = st.button("Clear polygons")
    run = st.button("Run model", type="primary")
    st.markdown("**Drawing rule**")
    st.caption("First polygon = site AOI. Additional polygons can be assigned below the map to Trees, Grass / planting, Water, or Hardscape. You can assign more than one polygon to each type.")

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
if "map_center_source" not in st.session_state:
    st.session_state.map_center_source = "default"

meta = None
df_weather = None
if epw_file is not None:
    try:
        meta, df_weather = read_epw(epw_file)
        if st.session_state.map_center_source != "search":
            st.session_state.center = [meta["latitude"], meta["longitude"]]
            st.session_state.zoom = 15
            st.session_state.map_center_source = "epw"
    except Exception as e:
        st.error(f"Could not read the EPW file: {e}")

m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom, control_scale=True, tiles=None)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True, show=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite (Esri)",
    control=True,
    show=False
).add_to(m)

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

if clear:
    st.session_state.draw_data = []
    st.session_state.results = None
    st.session_state.polygon_zone_types = {}
    st.session_state.map_key_suffix += 1
    st.rerun()

map_data = st_folium(m, width=1100, height=650, key="site_et_map_" + str(st.session_state.map_key_suffix))
if map_data and map_data.get("all_drawings") is not None:
    st.session_state.draw_data = map_data.get("all_drawings", [])

all_drawings = st.session_state.draw_data
polygons = [shape(feat["geometry"]) for feat in all_drawings if feat.get("geometry", {}).get("type") == "Polygon"]
aoi_geom = polygons[0] if len(polygons) >= 1 else None
extra_polygons = polygons[1:] if len(polygons) >= 2 else []

for idx in range(len(extra_polygons)):
    st.session_state.polygon_zone_types.setdefault(idx, "Trees")
for idx in list(st.session_state.polygon_zone_types.keys()):
    if idx >= len(extra_polygons):
        del st.session_state.polygon_zone_types[idx]

if extra_polygons:
    with st.expander("Polygon type assignments", expanded=True):
        st.caption("Polygon 1 is always the site boundary. Assign each additional polygon to a zone type below.")
        for idx, geom in enumerate(extra_polygons):
            area_val = area_m2(geom.intersection(aoi_geom)) if aoi_geom is not None else area_m2(geom)
            area_label = "{:.0f} m2".format(area_val)
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

col1, col2, col3, col4 = st.columns(4)
aoi_area = None
tree_area = 0.0
grass_area = 0.0
water_area = 0.0
hard_area = 0.0
rem_area = 0.0
manual_green_water_area = 0.0
if aoi_geom is not None:
    aoi_area = area_m2(aoi_geom)
    tree_area = sum(area_m2(g) for g in tree_geoms) if tree_geoms else 0.0
    grass_area = sum(area_m2(g) for g in grass_geoms) if grass_geoms else 0.0
    water_area = sum(area_m2(g) for g in water_geoms) if water_geoms else 0.0
    hard_area = sum(area_m2(g) for g in hard_geoms) if hard_geoms else 0.0
    manual_total_area = min(tree_area + grass_area + water_area + hard_area, aoi_area)
    rem_area = max(0.0, aoi_area - manual_total_area)
    manual_green_water_area = tree_area + grass_area + water_area
    col1.metric("Site area", f"{aoi_area:,.0f} m2")
    col2.metric("Manual override area", f"{manual_total_area:,.0f} m2")
    col3.metric("Remainder area", f"{rem_area:,.0f} m2")
    col4.metric("Manual green + water", f"{manual_green_water_area:,.0f} m2")
else:
    col1.metric("Site area", "-")
    col2.metric("Manual override area", "-")
    col3.metric("Remainder area", "-")
    col4.metric("Manual green + water", "-")

if aoi_geom is not None and (tree_area + grass_area + water_area + hard_area) > 0:
    zc1, zc2, zc3, zc4, zc5 = st.columns(5)
    zc1.metric("Trees", f"{tree_area:,.0f} m2")
    zc2.metric("Grass / planting", f"{grass_area:,.0f} m2")
    zc3.metric("Water", f"{water_area:,.0f} m2")
    zc4.metric("Hardscape", f"{hard_area:,.0f} m2")
    zc5.metric("Remainder", f"{rem_area:,.0f} m2")

if meta:
    st.info(f"EPW location: {meta['city']}, {meta['country']} ({meta['latitude']:.4f}, {meta['longitude']:.4f})")

if run:
    if epw_file is None:
        st.error("Upload an EPW file first.")
    elif aoi_geom is None:
        st.error("Draw the main site polygon first.")
    else:
        with st.spinner("Calculating hourly ET0 from EPW..."):
            df = df_weather.copy()
            df["timestamp"] = make_representative_timestamp(df["Month"], df["Day"], df["Hour"], fixed_year=REPRESENTATIVE_WEATHER_YEAR)
            df = df.dropna(subset=["timestamp", "DryBulb", "RH", "GlobalHorizontalRadiation", "WindSpeed"]).copy()
            if "LiquidPrecipitationDepth" not in df.columns:
                df["LiquidPrecipitationDepth"] = 0.0
            df["LiquidPrecipitationDepth"] = pd.to_numeric(df["LiquidPrecipitationDepth"], errors="coerce").fillna(0.0).clip(lower=0.0)
            df = df.sort_values("timestamp").reset_index(drop=True)
            if df.empty:
                st.error("No valid hourly rows were found in the EPW after parsing.")
                st.stop()
            df["ET0_mm_h"] = df.apply(hourly_et0_fao56, axis=1)

        ee_warning = None
        with st.spinner("Fetching NDVI and Kc from Earth Engine..."):
            try:
                stats_existing = sentinel_ndvi_kc_stats(aoi_geom, override_geom=None, satellite_year=int(satellite_year))
                stats_scenario = sentinel_ndvi_kc_stats(aoi_geom, override_geom=override_geom, satellite_year=int(satellite_year))
            except Exception as e:
                ee_warning = f"Satellite classification did not run because Earth Engine is not available or not authenticated: {e}"
                stats_existing = {
                    "site_kc": 0.45,
                    "site_ndvi": 0.25,
                    "tree_area_auto_m2": 0.0,
                    "grass_area_auto_m2": 0.0,
                    "water_area_auto_m2": 0.0,
                    "hard_area_auto_m2": aoi_area or 0.0,
                    "tree_frac_auto": 0.0,
                    "grass_frac_auto": 0.0,
                    "water_frac_auto": 0.0,
                    "hard_frac_auto": 1.0,
                    "tree_kc_auto": 0.95,
                    "grass_kc_auto": 0.65,
                    "water_kc_auto": 1.05,
                    "hard_kc_auto": 0.20,
                    "rem_kc": 0.45,
                    "rem_ndvi": 0.25,
                    "rem_tree_frac_auto": 0.0,
                    "rem_grass_frac_auto": 0.0,
                    "rem_water_frac_auto": 0.0,
                    "rem_hard_frac_auto": 1.0,
                    "rem_tree_area_auto_m2": 0.0,
                    "rem_grass_area_auto_m2": 0.0,
                    "rem_water_area_auto_m2": 0.0,
                    "rem_hard_area_auto_m2": rem_area,
                }
                stats_scenario = dict(stats_existing)

        rem_tree_area_auto = float(stats_scenario.get("rem_tree_area_auto_m2", 0.0))
        rem_grass_area_auto = float(stats_scenario.get("rem_grass_area_auto_m2", 0.0))
        rem_water_area_auto = float(stats_scenario.get("rem_water_area_auto_m2", 0.0))
        rem_hard_area_auto = float(stats_scenario.get("rem_hard_area_auto_m2", 0.0))
        existing_auto_green_water_area = float(stats_existing.get("tree_area_auto_m2", 0.0)) + float(stats_existing.get("grass_area_auto_m2", 0.0)) + float(stats_existing.get("water_area_auto_m2", 0.0))
        net_benchmark_area = tree_area + grass_area + water_area + rem_tree_area_auto + rem_grass_area_auto + rem_water_area_auto

        # Potential ET assumes enough water is available. The rain-fed option then limits
        # vegetation and wettable ground ET using rainfall from the EPW file.
        df["ET_existing_potential_mm_h"] = df["ET0_mm_h"] * stats_existing["site_kc"]
        df["ET_tree_potential_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Trees"] if tree_area > 0 else np.nan
        df["ET_grass_potential_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Grass / planting"] if grass_area > 0 else np.nan
        df["ET_water_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Water"] if water_area > 0 else np.nan
        df["ET_hard_potential_mm_h"] = df["ET0_mm_h"] * ZONE_KC["Hardscape"] if hard_area > 0 else np.nan
        df["ET_rem_potential_mm_h"] = df["ET0_mm_h"] * stats_scenario["rem_kc"]

        if use_rain_fed_model:
            df["ET_existing_mm_h"], df["Existing_Water_Storage_mm"] = apply_rain_fed_bucket(
                df["ET_existing_potential_mm_h"],
                df["LiquidPrecipitationDepth"],
                max_storage_mm=max_soil_storage_mm,
                initial_storage_fraction=initial_storage_fraction,
            )
            if tree_area > 0:
                df["ET_tree_mm_h"], df["Tree_Water_Storage_mm"] = apply_rain_fed_bucket(df["ET_tree_potential_mm_h"], df["LiquidPrecipitationDepth"], max_soil_storage_mm, initial_storage_fraction)
            else:
                df["ET_tree_mm_h"] = np.nan
            if grass_area > 0:
                df["ET_grass_mm_h"], df["Grass_Water_Storage_mm"] = apply_rain_fed_bucket(df["ET_grass_potential_mm_h"], df["LiquidPrecipitationDepth"], max_soil_storage_mm, initial_storage_fraction)
            else:
                df["ET_grass_mm_h"] = np.nan
            if hard_area > 0:
                # Hardscape evaporation is also rainfall-limited, with much smaller effective storage.
                df["ET_hard_mm_h"], df["Hardscape_Water_Storage_mm"] = apply_rain_fed_bucket(df["ET_hard_potential_mm_h"], df["LiquidPrecipitationDepth"], min(5.0, max_soil_storage_mm), initial_storage_fraction)
            else:
                df["ET_hard_mm_h"] = np.nan
            df["ET_rem_mm_h"], df["Remainder_Water_Storage_mm"] = apply_rain_fed_bucket(df["ET_rem_potential_mm_h"], df["LiquidPrecipitationDepth"], max_storage_mm=max_soil_storage_mm, initial_storage_fraction=initial_storage_fraction)
        else:
            df["ET_existing_mm_h"] = df["ET_existing_potential_mm_h"]
            df["ET_tree_mm_h"] = df["ET_tree_potential_mm_h"]
            df["ET_grass_mm_h"] = df["ET_grass_potential_mm_h"]
            df["ET_hard_mm_h"] = df["ET_hard_potential_mm_h"]
            df["ET_rem_mm_h"] = df["ET_rem_potential_mm_h"]

        df["Existing_ET_m3_h"] = df["ET_existing_mm_h"].apply(lambda x: mm_over_area_to_m3(x, aoi_area or 0.0))
        df["Existing_Cooling_kWh_h"] = df["ET_existing_mm_h"].apply(lambda x: et_mm_to_cooling_kwh(x, aoi_area or 0.0))
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
        df["ET_scenario_mm_h"] = df["Total_Weighted_ET_m3_h"].apply(lambda x: (x / (aoi_area or 1.0)) * 1000.0)
        df["Scenario_Cooling_kWh_h"] = df["Total_Weighted_Cooling_kWh_h"]
        df["Cooling_kWh_m2_h"] = df["Scenario_Cooling_kWh_h"] / net_benchmark_area if net_benchmark_area > 0 else np.nan

        st.session_state.results = {
            "df": df,
            "stats_existing": stats_existing,
            "stats_scenario": stats_scenario,
            "aoi_area": aoi_area,
            "tree_area": tree_area,
            "grass_area": grass_area,
            "water_area": water_area,
            "hard_area": hard_area,
            "rem_area": rem_area,
            "net_benchmark_area": net_benchmark_area,
            "existing_auto_green_water_area": existing_auto_green_water_area,
            "override_present": bool(override_geom),
            "ee_warning": ee_warning,
            "use_rain_fed_model": use_rain_fed_model,
            "max_soil_storage_mm": max_soil_storage_mm,
            "initial_storage_fraction": initial_storage_fraction,
            "satellite_year": int(satellite_year),
            "representative_weather_year": REPRESENTATIVE_WEATHER_YEAR,
        }

results = st.session_state.results
if results is not None and isinstance(results.get("df"), pd.DataFrame):
    results["df"] = results["df"].sort_values("timestamp").reset_index(drop=True)

required_result_keys = {
    "df", "stats_existing", "stats_scenario", "aoi_area", "tree_area",
    "grass_area", "water_area", "hard_area", "rem_area",
    "net_benchmark_area", "existing_auto_green_water_area", "override_present", "ee_warning", "use_rain_fed_model", "satellite_year", "representative_weather_year"
}
if results is not None and not required_result_keys.issubset(set(results.keys())):
    st.session_state.results = None
    results = None
    st.info("Previous saved results were from an older app version and were cleared. Please click Run model again.")

if results is not None:
    df = results["df"]
    period_start = pd.to_datetime(analysis_start_date)
    period_end = pd.to_datetime(analysis_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_period = df[(df["timestamp"] >= period_start) & (df["timestamp"] <= period_end)].copy()
    if df_period.empty:
        st.warning("No hourly weather rows were found for the selected period. Showing full-year results instead.")
        df_period = df.copy()
    stats_existing = results["stats_existing"]
    stats_scenario = results["stats_scenario"]
    aoi_area = results["aoi_area"]
    tree_area = results["tree_area"]
    grass_area = results["grass_area"]
    water_area = results["water_area"]
    hard_area = results["hard_area"]
    rem_area = results["rem_area"]
    net_benchmark_area = results["net_benchmark_area"]
    existing_auto_green_water_area = results["existing_auto_green_water_area"]
    override_present = results["override_present"]
    ee_warning = results["ee_warning"]

    rem_tree_area_auto = float(stats_scenario.get("rem_tree_area_auto_m2", 0.0))
    rem_grass_area_auto = float(stats_scenario.get("rem_grass_area_auto_m2", 0.0))
    rem_water_area_auto = float(stats_scenario.get("rem_water_area_auto_m2", 0.0))
    rem_hard_area_auto = float(stats_scenario.get("rem_hard_area_auto_m2", 0.0))

    if ee_warning:
        st.error(ee_warning)
        st.warning("The baseline/proposed site-composition charts are using fallback values, not real satellite classification. Configure Earth Engine credentials before trusting the land-cover split.")

    tab1, tab2, tab3 = st.tabs(["Summary", "Time Series Analysis", "Download / Print"])

    with tab1:
        st.header("Cooling Impact of Site Design")
        if ee_warning:
            st.info(f"Baseline = fallback land-cover assumption because Earth Engine did not run. Proposed = manual overrides plus fallback remainder. Weather and rainfall come from the uploaded EPW profile and are mapped to {results.get('representative_weather_year')} only for plotting/filtering.")
        else:
            st.info(f"Baseline = satellite-derived existing site condition using Sentinel-2 imagery from {results.get('satellite_year')} during May-August. Proposed = manual overrides plus satellite-derived remainder. Weather and rainfall come from the uploaded EPW profile and are mapped to {results.get('representative_weather_year')} only for plotting/filtering.")
            st.caption(f"Satellite imagery used: {stats_existing.get('sentinel_image_count', 0)} image(s), cloud-cover threshold <{stats_existing.get('cloud_threshold_used', 'N/A')}%.")
        rain_model_text = "Rain-fed ET enabled: rainfall is taken from the uploaded EPW file." if results.get("use_rain_fed_model") else "Potential ET mode: surfaces are assumed to have enough water available."
        st.caption(f"Results shown for selected representative period: {analysis_start_date.strftime('%d %b')} to {analysis_end_date.strftime('%d %b')}. Cooling intensity is normalised by total site area. {rain_model_text}")

        baseline_cooling_total = df_period["Existing_Cooling_kWh_h"].sum()
        proposed_cooling_total = df_period["Scenario_Cooling_kWh_h"].sum()
        baseline_kwh_m2 = baseline_cooling_total / aoi_area if aoi_area and aoi_area > 0 else np.nan
        proposed_kwh_m2 = proposed_cooling_total / aoi_area if aoi_area and aoi_area > 0 else np.nan
        change_kwh_m2 = proposed_kwh_m2 - baseline_kwh_m2 if pd.notna(proposed_kwh_m2) and pd.notna(baseline_kwh_m2) else np.nan
        change_percent = (change_kwh_m2 / baseline_kwh_m2 * 100.0) if pd.notna(change_kwh_m2) and baseline_kwh_m2 != 0 else np.nan

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Baseline cooling", f"{baseline_kwh_m2:,.2f} kWh/m2" if pd.notna(baseline_kwh_m2) else "-")
        kpi2.metric("Proposed cooling", f"{proposed_kwh_m2:,.2f} kWh/m2" if pd.notna(proposed_kwh_m2) else "-")
        kpi3.metric("Change", f"{change_kwh_m2:+,.2f} kWh/m2" if pd.notna(change_kwh_m2) else "-")
        kpi4.metric("Change", f"{change_percent:+,.1f}%" if pd.notna(change_percent) else "-")

        st.subheader("Baseline vs proposed cooling")
        comparison_df = pd.DataFrame({
            "Scenario": ["Baseline", "Proposed"],
            "Cooling (kWh/m2)": [baseline_kwh_m2, proposed_kwh_m2]
        })
        comparison_df["Label"] = comparison_df["Cooling (kWh/m2)"].map(lambda x: f"{x:,.2f} kWh/m2" if pd.notna(x) else "-")

        if ALTAIR_AVAILABLE:
            bar_chart = (
                alt.Chart(comparison_df)
                .mark_bar(size=34)
                .encode(
                    x=alt.X("Cooling (kWh/m2):Q", title="Cooling (kWh/m2)", axis=alt.Axis(labelFontSize=13, titleFontSize=14)),
                    y=alt.Y("Scenario:N", title="", sort=["Baseline", "Proposed"], axis=alt.Axis(labelFontSize=14)),
                    color=alt.Color(
                        "Scenario:N",
                        scale=alt.Scale(domain=["Baseline", "Proposed"], range=["#8A8A8A", "#2E7D32"]),
                        legend=None,
                    ),
                    tooltip=["Scenario", alt.Tooltip("Cooling (kWh/m2):Q", format=",.2f")],
                )
                .properties(height=150)
            )
            bar_text = (
                alt.Chart(comparison_df)
                .mark_text(align="left", dx=8, fontSize=13, fontWeight="bold")
                .encode(
                    x="Cooling (kWh/m2):Q",
                    y=alt.Y("Scenario:N", sort=["Baseline", "Proposed"]),
                    text="Label:N",
                )
            )
            st.altair_chart(bar_chart + bar_text, use_container_width=True)
        elif PLOTLY_AVAILABLE:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=comparison_df["Cooling (kWh/m2)"],
                y=comparison_df["Scenario"],
                orientation="h",
                text=comparison_df["Label"],
                textposition="outside",
                width=0.35,
                marker_color=["#8A8A8A", "#2E7D32"],
                showlegend=False,
            ))
            fig_compare.update_layout(height=220, margin=dict(l=20, r=80, t=10, b=30), xaxis_title="Cooling (kWh/m2)", yaxis_title="")
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.bar_chart(comparison_df.set_index("Scenario")[["Cooling (kWh/m2)"]])

        st.caption(f"Difference: {change_kwh_m2:+,.2f} kWh/m2 ({change_percent:+,.1f}%)" if pd.notna(change_kwh_m2) and pd.notna(change_percent) else "Difference: -")

        st.subheader("Site composition")
        baseline_comp_df = pd.DataFrame({
            "Scenario": "Baseline",
            "Surface type": ["Trees", "Grass", "Water", "Hardscape"],
            "Area (m2)": [
                float(stats_existing.get("tree_area_auto_m2", 0.0)),
                float(stats_existing.get("grass_area_auto_m2", 0.0)),
                float(stats_existing.get("water_area_auto_m2", 0.0)),
                float(stats_existing.get("hard_area_auto_m2", 0.0)),
            ],
        })
        proposed_comp_df = pd.DataFrame({
            "Scenario": "Proposed",
            "Surface type": ["Trees", "Grass", "Water", "Hardscape"],
            "Area (m2)": [
                tree_area + rem_tree_area_auto,
                grass_area + rem_grass_area_auto,
                water_area + rem_water_area_auto,
                hard_area + rem_hard_area_auto,
            ],
        })
        composition_df = pd.concat([baseline_comp_df, proposed_comp_df], ignore_index=True)
        composition_df = composition_df[composition_df["Area (m2)"] > 0].copy()
        composition_df["Percent"] = composition_df.groupby("Scenario")["Area (m2)"].transform(lambda s: s / s.sum() * 100 if s.sum() > 0 else 0)
        composition_df["Label"] = composition_df.apply(lambda r: f"{r['Area (m2)']:,.0f} m2 ({r['Percent']:.1f}%)", axis=1)

        surface_colors = alt.Scale(
            domain=["Trees", "Grass", "Water", "Hardscape"],
            range=["#2E7D32", "#7CB342", "#039BE5", "#9E9E9E"],
        )

        if ALTAIR_AVAILABLE and not composition_df.empty:
            c1, c2 = st.columns(2)
            for scenario_name, chart_col in [("Baseline", c1), ("Proposed", c2)]:
                scenario_df = composition_df[composition_df["Scenario"] == scenario_name]
                pie = (
                    alt.Chart(scenario_df)
                    .mark_arc(innerRadius=55, outerRadius=105)
                    .encode(
                        theta=alt.Theta("Area (m2):Q"),
                        color=alt.Color("Surface type:N", title="Surface type", scale=surface_colors),
                        tooltip=[
                            "Surface type",
                            alt.Tooltip("Area (m2):Q", format=",.0f", title="Area (m2)"),
                            alt.Tooltip("Percent:Q", format=".1f", title="Percent"),
                        ],
                    )
                    .properties(title=scenario_name, height=300)
                )
                chart_col.altair_chart(pie, use_container_width=True)

            display_comp_df = composition_df[["Scenario", "Surface type", "Area (m2)", "Percent"]].copy()
            display_comp_df["Area (m2)"] = display_comp_df["Area (m2)"].map(lambda x: f"{x:,.0f}")
            display_comp_df["Percent"] = display_comp_df["Percent"].map(lambda x: f"{x:.1f}%")
            st.dataframe(display_comp_df, use_container_width=True, hide_index=True)
        elif PLOTLY_AVAILABLE and not composition_df.empty:
            c1, c2 = st.columns(2)
            for scenario_name, chart_col in [("Baseline", c1), ("Proposed", c2)]:
                scenario_df = composition_df[composition_df["Scenario"] == scenario_name]
                fig_comp = go.Figure(data=[go.Pie(
                    labels=scenario_df["Surface type"],
                    values=scenario_df["Area (m2)"],
                    hole=0.35,
                    textinfo="none",
                    textfont_size=14,
                    marker=dict(colors=["#2E7D32", "#7CB342", "#039BE5", "#9E9E9E"]),
                    hovertemplate="%{label}<br>%{value:,.0f} m2<br>%{percent}<extra></extra>",
                )])
                fig_comp.update_layout(title=scenario_name, height=340, margin=dict(l=20, r=20, t=40, b=20), font=dict(size=14))
                chart_col.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.dataframe(composition_df[["Scenario", "Surface type", "Area (m2)", "Percent"]], use_container_width=True, hide_index=True)

        with st.expander("Show technical details and full summary"):
            st.subheader("Spatial coefficients")
            s1, s2, s3 = st.columns(3)
            s1.metric("Existing site NDVI", f"{stats_existing['site_ndvi']:.3f}")
            s2.metric("Existing site Kc", f"{stats_existing['site_kc']:.3f}")
            s3.metric("Mean ET0", f"{df_period['ET0_mm_h'].mean():.3f} mm/h")

            st.subheader("Remainder area and split")
            rr1, rr2, rr3, rr4, rr5 = st.columns(5)
            rr1.metric("Remainder area", f"{rem_area:,.0f} m2")
            rr2.metric("Remainder auto tree", f"{rem_tree_area_auto:,.0f} m2")
            rr3.metric("Remainder auto grass", f"{rem_grass_area_auto:,.0f} m2")
            rr4.metric("Remainder auto water", f"{rem_water_area_auto:,.0f} m2")
            rr5.metric("Remainder auto hard", f"{rem_hard_area_auto:,.0f} m2")

            summary = {
                "Selected period baseline cooling (kWh)": baseline_cooling_total,
                "Selected period proposed cooling (kWh)": proposed_cooling_total,
                "Baseline cooling (kWh/m2 site area)": baseline_kwh_m2,
                "Proposed cooling (kWh/m2 site area)": proposed_kwh_m2,
                "Change (kWh/m2 site area)": change_kwh_m2,
                "Change (%)": change_percent,
                "Site area (m2)": aoi_area or 0.0,
                "Manual tree area (m2)": tree_area,
                "Manual grass area (m2)": grass_area,
                "Manual water area (m2)": water_area,
                "Manual hardscape area (m2)": hard_area,
                "Remainder area (m2)": rem_area,
                "Remainder auto tree area (m2)": rem_tree_area_auto,
                "Remainder auto grass area (m2)": rem_grass_area_auto,
                "Remainder auto water area (m2)": rem_water_area_auto,
                "Remainder auto hard area (m2)": rem_hard_area_auto,
                "Selected period ET0 (mm)": df_period["ET0_mm_h"].sum(),
                "Selected period baseline ET depth (mm)": df_period["ET_existing_mm_h"].sum(),
                "Selected period proposed ET depth (mm)": df_period["ET_scenario_mm_h"].sum(),
                "Selected period weighted ET volume (m3)": df_period["Total_Weighted_ET_m3_h"].sum(),
                "Selected period rainfall from EPW (mm)": df_period["LiquidPrecipitationDepth"].sum(),
                "Satellite window start": stats_existing.get("satellite_window_start", ""),
                "Satellite window end": stats_existing.get("satellite_window_end", ""),
                "Satellite cloud threshold used (%)": stats_existing.get("cloud_threshold_used", np.nan),
                "Sentinel-2 images used": stats_existing.get("sentinel_image_count", np.nan),
                "Rain-fed ET model enabled": 1.0 if results.get("use_rain_fed_model") else 0.0,
                "Maximum rainfall storage (mm)": results.get("max_soil_storage_mm", np.nan),
            }
            summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
            summary_df["Value"] = summary_df["Value"].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notna(x) else str(x))
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Time Series Analysis")

        st.markdown("Select what to display:")
        show_baseline = st.checkbox("Baseline", value=True)
        show_proposed = st.checkbox("Proposed", value=True)

        # Build datasets
        ts_df = df_period.set_index("timestamp")

        # ET
        st.markdown("### Evapotranspiration (ET)")
        et_cols = []
        if show_baseline:
            et_cols.append("ET_existing_mm_h")
        if show_proposed:
            et_cols.append("ET_scenario_mm_h")
        if et_cols:
            st.line_chart(ts_df[et_cols])

        # Water volume
        st.markdown("### Water Volume")
        vol_cols = []
        if show_baseline:
            vol_cols.append("Existing_ET_m3_h")
        if show_proposed:
            vol_cols.append("Total_Weighted_ET_m3_h")
        if vol_cols:
            st.line_chart(ts_df[vol_cols])

        # Cooling
        st.markdown("### Cooling")
        cool_cols = []
        if show_baseline:
            cool_cols.append("Existing_Cooling_kWh_h")
        if show_proposed:
            cool_cols.append("Scenario_Cooling_kWh_h")
        if cool_cols:
            st.line_chart(ts_df[cool_cols])

        st.caption("All charts reflect the selected analysis period.")

    with tab3:
        st.markdown("### Download results")
        st.caption("Download the hourly ET, water volume, cooling, and benchmark results for the selected model run.")

        export_cols = [
            "timestamp", "ET0_mm_h", "ET_existing_mm_h", "ET_scenario_mm_h", "ET_rem_mm_h", "ET_tree_mm_h", "ET_grass_mm_h", "ET_water_mm_h", "ET_hard_mm_h",
            "Existing_ET_m3_h", "Tree_ET_m3_h", "Grass_ET_m3_h", "Water_ET_m3_h", "Hard_ET_m3_h", "Rem_ET_m3_h", "Total_Weighted_ET_m3_h",
            "Existing_Cooling_kWh_h", "Tree_Cooling_kWh_h", "Grass_Cooling_kWh_h", "Water_Cooling_kWh_h", "Hard_Cooling_kWh_h", "Rem_Cooling_kWh_h", "Total_Weighted_Cooling_kWh_h",
            "Cooling_kWh_m2_h"
        ]
        export_cols = [c for c in export_cols if c in df.columns]
        csv_data = df[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv_data, file_name="site_et_results.csv", mime="text/csv")

        st.markdown("### Print / export view")
        st.caption("Use your browser print command to save the current dashboard as a PDF. Collapse technical details first for a cleaner report.")
        if st.button("Prepare print view"):
            st.info("Print view is ready. Use Ctrl+P or your browser menu to save as PDF.")



