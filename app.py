# staffpilot.py
# Streamlit + Supabase single-file webapp
#
# ✅ Title branding: "StaffPilot AI - Architected by Gopi Chand"
# ✅ Mode 1: ⚡ Instant Forecast (ZIP/City + Lowest/Avg/Highest cars guardrails)
#    - Uses: day-of-week curve + weather suppression + bounce-back
#    - Shows: Expected Change % (weather-only) + Weather Driver
#    - Shows: transparency note about error due to missing promos/traffic/local events
#
# ✅ Mode 2: 🚀 Full AI Mode (Supabase sites)
#    - Uses: site baseline Mon–Sun + membership + traffic + weather + bounce-back
#    - Self-learning: bias + rain_sensitivity tuned from forecast vs actual
#    - Stores forecasts (per site, per date) + actuals (privacy-safe)
#    - Accuracy dashboard: rolling 7/14/30-day MAPE + table + chart
#
# ✅ Hard cap: max staff = 5
#
# Install:
#   pip install streamlit supabase requests pandas
#
# Streamlit secrets (.streamlit/secrets.toml):
#   SUPABASE_URL="YOUR_SUPABASE_URL"
#   SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
#
# -------------------- SUPABASE SQL (run once) --------------------
# -- Sites table
# create table if not exists sites (
#   id uuid primary key default gen_random_uuid(),
#   alias text,
#   location text,
#   lat double precision,
#   lon double precision,
#   open_hour int,
#   close_hour int,
#   traffic text,
#   min_staff int,
#   prep boolean,
#   one_loader boolean,
#   member_washes int,
#   baseline jsonb,
#   bias double precision default 1.0,
#   rain_sensitivity double precision default 1.0,
#   created_at timestamp default now()
# );
#
# -- Daily metrics: stores forecast and actual (privacy-safe)
# create table if not exists daily_metrics (
#   id uuid primary key default gen_random_uuid(),
#   site_id uuid references sites(id) on delete cascade,
#   day date not null,
#   forecast_cars int,
#   actual_cars int,
#   expected_change_pct int,
#   weather_driver text,
#   precip_prob int,
#   precip_in double precision,
#   temp_f double precision,
#   created_at timestamp default now(),
#   unique(site_id, day)
# );
# ---------------------------------------------------------------

import json
import math
from datetime import datetime, date, timedelta
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

# ---------------- CONFIG ----------------

st.set_page_config(page_title="StaffPilot AI", layout="wide")
st.title("🚀 StaffPilot AI - Architected by Gopi Chand")

MAX_STAFF = 5

TRAFFIC_MULT = {
    "Low": 0.90,
    "Medium": 1.00,
    "High": 1.10,
    "Very High": 1.20,
}

# Industry-ish weekday/weekend behavior (for Quick Mode)
DOW_MULT = {
    "Monday": 0.72,
    "Tuesday": 0.78,
    "Wednesday": 0.85,
    "Thursday": 0.92,
    "Friday": 1.05,
    "Saturday": 1.28,
    "Sunday": 1.15,
}

# Quick-mode staffing capacity (cars per staff-hour)
QUICK_CPSH = 9.5


# ---------------- SUPABASE ----------------

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------- WEATHER (Open-Meteo) ----------------

@st.cache_data(ttl=60 * 60)
def get_lat_lon(location: str) -> Tuple[float, float]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    res = requests.get(url, params={"name": location, "count": 1}, timeout=10)
    if res.status_code != 200:
        raise RuntimeError("Geocoding API failed.")
    data = res.json()
    results = data.get("results") or []
    if not results:
        raise RuntimeError("Location not found. Try ZIP or 'City, State'.")
    return float(results[0]["latitude"]), float(results[0]["longitude"])


@st.cache_data(ttl=60 * 30)
def get_weather_7d(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "forecast_days": 7,   # critical: force 7 days
        "timezone": "auto",
    }
    res = requests.get(url, params=params, timeout=10)
    if res.status_code != 200:
        raise RuntimeError("Weather API failed.")
    data = res.json()
    daily = data.get("daily")
    if not daily:
        raise RuntimeError("Weather returned no daily data.")
    required = ["time", "precipitation_probability_max", "precipitation_sum", "temperature_2m_max"]
    for k in required:
        if k not in daily or not daily[k]:
            raise RuntimeError(f"Weather missing field: {k}")
    if min(len(daily["time"]), len(daily["precipitation_probability_max"]), len(daily["precipitation_sum"]), len(daily["temperature_2m_max"])) < 7:
        raise RuntimeError("Weather did not return a full 7-day forecast.")
    return daily


# ---------------- EXPLAINABLE WEATHER IMPACT ----------------

def weather_impact_percent(prob: float, rain_in: float, temp_f: float) -> Tuple[int, str]:
    """
    Returns (impact_pct, reason). impact_pct is weather-only delta vs "neutral day".
    Negative = expected drop, Positive = expected lift.
    """
    impact = 0
    reasons = []

    # Rain impact
    if prob >= 80 or rain_in >= 0.25:
        impact -= 45
        reasons.append("Heavy Rain")
    elif prob >= 60 or rain_in >= 0.10:
        impact -= 25
        reasons.append("Moderate Rain")
    elif prob >= 40:
        impact -= 12
        reasons.append("Light Rain")
    else:
        reasons.append("Clear")

    # Temperature adjustments (tuned for car wash behavior)
    if temp_f < 40:
        impact -= 20
        reasons.append("Cold")
    elif temp_f < 50:
        impact -= 10
        reasons.append("Cool")
    elif 55 <= temp_f <= 82:
        impact += 6
        reasons.append("Ideal Temps")
    elif temp_f > 90:
        impact -= 5
        reasons.append("Extreme Heat")

    return int(impact), " + ".join(reasons)


def weather_factor(prob: float, rain_in: float, temp_f: float) -> float:
    """Model multiplier derived from weather (1.0 = neutral)."""
    factor = 1.0

    # Rain suppression
    if prob >= 80 or rain_in >= 0.25:
        factor *= 0.55
    elif prob >= 60 or rain_in >= 0.10:
        factor *= 0.75
    elif prob >= 40:
        factor *= 0.90

    # Temperature
    if temp_f < 40:
        factor *= 0.80
    elif temp_f < 50:
        factor *= 0.90
    elif 55 <= temp_f <= 82:
        factor *= 1.06
    elif temp_f > 90:
        factor *= 0.97

    return factor


def rebound_boost(prev_prob: float, prev_rain_in: float) -> float:
    """Bounce-back boost applied on a good wash day after a rainy day."""
    if prev_prob >= 80 or prev_rain_in >= 0.25:
        return 0.30
    if prev_prob >= 60 or prev_rain_in >= 0.10:
        return 0.20
    if prev_prob >= 40:
        return 0.10
    return 0.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_baseline(baseline: Any) -> Dict[str, int]:
    if baseline is None:
        return {}
    if isinstance(baseline, str):
        baseline = json.loads(baseline)
    return {str(k).strip().lower(): int(v) for k, v in baseline.items()}


# ---------------- FORECAST: QUICK MODE ----------------

def forecast_week_quick(location: str, min_cars: int, avg_cars: int, max_cars: int) -> List[Dict[str, Any]]:
    lat, lon = get_lat_lon(location)
    w = get_weather_7d(lat, lon)

    out = []
    prev_prob = 0.0
    prev_rain = 0.0

    for i in range(7):
        day_iso = w["time"][i]
        prob = float(w["precipitation_probability_max"][i])
        rain_in = float(w["precipitation_sum"][i])
        temp_f = float(w["temperature_2m_max"][i])

        dow = datetime.fromisoformat(day_iso).strftime("%A")
        dow_factor = DOW_MULT.get(dow, 1.0)

        impact_pct, reason = weather_impact_percent(prob, rain_in, temp_f)
        wf = weather_factor(prob, rain_in, temp_f)

        # Bounce-back on good wash day
        if prob < 30 and temp_f > 50:
            wf *= (1 + rebound_boost(prev_prob, prev_rain))

        raw = avg_cars * dow_factor * wf
        cars = int(max(min_cars, min(raw, max_cars)))

        peak_hour = cars * 0.12
        staff = math.ceil(peak_hour / QUICK_CPSH) if QUICK_CPSH > 0 else 0
        staff = min(MAX_STAFF, max(0, staff))

        out.append({
            "Day": dow,
            "Date": day_iso,
            "Forecast Cars": cars,
            "Peak Staff": staff,
            "Expected Change %": impact_pct,
            "Weather Driver": reason,
            "Rain %": int(prob),
            "Rain (in)": round(rain_in, 2),
            "Temp (F)": round(temp_f, 1),
        })

        prev_prob = prob
        prev_rain = rain_in

    return out


# ---------------- FORECAST: FULL MODE ----------------

def staff_needed_full(cars: int, prep: bool) -> int:
    peak_hour = cars * 0.12
    cpsh = 8 if prep else 11
    staff = math.ceil(peak_hour / cpsh) if cpsh > 0 else 0
    return min(MAX_STAFF, max(0, staff))


def forecast_week_full(site: Dict[str, Any]) -> List[Dict[str, Any]]:
    w = get_weather_7d(site["lat"], site["lon"])

    baseline = parse_baseline(site.get("baseline"))
    member_floor = int(site.get("member_washes", 0))
    traffic_mult = TRAFFIC_MULT.get(site.get("traffic", "Medium"), 1.0)

    bias = float(site.get("bias", 1.0))
    rain_sens = float(site.get("rain_sensitivity", 1.0))

    out = []
    prev_prob = 0.0
    prev_rain = 0.0

    for i in range(7):
        day_iso = w["time"][i]
        prob = float(w["precipitation_probability_max"][i])
        rain_in = float(w["precipitation_sum"][i])
        temp_f = float(w["temperature_2m_max"][i])

        dow_name = datetime.fromisoformat(day_iso).strftime("%A")
        dow_key = dow_name.lower()
        base = int(baseline.get(dow_key, 0))

        impact_pct, reason = weather_impact_percent(prob, rain_in, temp_f)

        retail = max(base - member_floor, 0)

        wf = weather_factor(prob, rain_in, temp_f)

        # Tune rain sensitivity only on rainy-ish days
        if prob >= 40 or rain_in >= 0.05:
            wf *= rain_sens

        # Bounce-back on good wash day
        if prob < 30 and temp_f > 50:
            wf *= (1 + rebound_boost(prev_prob, prev_rain))

        cars = int((member_floor + (retail * traffic_mult * wf)) * bias)
        cars = max(0, cars)

        staff = staff_needed_full(cars, bool(site.get("prep", False)))
        peak_staff = min(MAX_STAFF, max(int(site.get("min_staff", 1)), staff))

        out.append({
            "Day": dow_name,
            "Date": day_iso,
            "Forecast Cars": cars,
            "Peak Staff": peak_staff,
            "Expected Change %": impact_pct,
            "Weather Driver": reason,
            "Rain %": int(prob),
            "Rain (in)": round(rain_in, 2),
            "Temp (F)": round(temp_f, 1),
        })

        prev_prob = prob
        prev_rain = rain_in

    return out


# ---------------- SELF-LEARNING (FULL MODE) ----------------

def update_site_learning(site: Dict[str, Any], forecast: int, actual: int,
                         prob: float, rain_in: float) -> Tuple[float, float]:
    """
    Updates:
      - bias: overall correction factor (smooth)
      - rain_sensitivity: adjusts rain suppression in this market (bounded)
    """
    old_bias = float(site.get("bias", 1.0))
    old_rain = float(site.get("rain_sensitivity", 1.0))

    if forecast <= 0:
        return old_bias, old_rain

    ratio = actual / forecast  # >1 under-forecast; <1 over-forecast

    # Bias update: smooth and bounded
    new_bias = (old_bias * 0.80) + (ratio * 0.20)
    new_bias = clamp(new_bias, 0.70, 1.30)

    # Rain sensitivity update only on rainy days
    new_rain = old_rain
    rainy = (prob >= 50) or (rain_in >= 0.10)
    if rainy:
        if ratio > 1.00:
            # Rain hurt less than expected -> reduce suppression -> increase factor
            new_rain *= 1.03
        else:
            # Rain hurt more than expected -> increase suppression -> decrease factor
            new_rain *= 0.97
        new_rain = clamp(new_rain, 0.70, 1.30)

    return new_bias, new_rain


# ---------------- ACCURACY METRICS ----------------

def mape(df: pd.DataFrame) -> Optional[float]:
    """Mean Absolute Percentage Error in %."""
    if df.empty:
        return None
    df2 = df.dropna(subset=["forecast_cars", "actual_cars"]).copy()
    if df2.empty:
        return None
    # Avoid divide-by-zero
    df2 = df2[df2["forecast_cars"] > 0]
    if df2.empty:
        return None
    err = (df2["actual_cars"] - df2["forecast_cars"]).abs() / df2["forecast_cars"]
    return float(err.mean() * 100.0)


def confidence_label(df_forecast: pd.DataFrame) -> str:
    """Simple confidence score based on rain risk in the week."""
    avg_rain = float(df_forecast["Rain %"].mean()) if not df_forecast.empty else 0.0
    if avg_rain >= 60:
        return "MEDIUM (weather-heavy week)"
    if avg_rain >= 35:
        return "HIGH (some variability)"
    return "HIGH (stable week)"


# ---------------- SUMMARY ----------------

def generate_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    busiest = df.loc[df["Forecast Cars"].idxmax()]
    slowest = df.loc[df["Forecast Cars"].idxmin()]
    rain_days = int((df["Rain %"] >= 60).sum())
    peak_staff = int(df["Peak Staff"].max()) if "Peak Staff" in df.columns else None
    avg_impact = float(df["Expected Change %"].mean()) if "Expected Change %" in df.columns else 0.0

    lines = []
    lines.append("### 📈 Weekly Operational Summary")
    lines.append(f"• Busiest day: **{busiest['Day']}** (~{int(busiest['Forecast Cars'])} cars)")
    lines.append(f"• Slowest day: **{slowest['Day']}** (~{int(slowest['Forecast Cars'])} cars)")
    lines.append(f"• Rain-impacted days (≥60%): **{rain_days}**")
    lines.append(f"• Average weather impact: **{avg_impact:+.0f}%** (weather-only)")
    if peak_staff is not None:
        lines.append(f"• Peak staffing needed (cap {MAX_STAFF}): **{peak_staff}**")
    lines.append(f"• Forecast confidence: **{confidence_label(df)}**")
    return "\n".join(lines)


# ---------------- UI ----------------

mode = st.sidebar.radio("Mode", ["⚡ Instant Forecast", "🚀 Full AI Mode"])


# =========================
# ⚡ INSTANT FORECAST
# =========================
if mode == "⚡ Instant Forecast":
    st.header("⚡ Instant Weather-Based Forecast (No Site Setup)")

    st.caption(
        "This mode uses weather + a typical weekend/weekday curve. "
        "It does **not** know your promos, road traffic changes, construction, events, or peak-hour patterns."
    )

    location = st.text_input("ZIP or City (required for weather)", value="")

    st.subheader("Demand Guardrails (so forecasts never exceed your reality)")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_cars = st.number_input("Lowest cars/day", min_value=0, max_value=8000, value=80, step=10)
    with c2:
        avg_cars = st.number_input("Average cars/day", min_value=10, max_value=8000, value=150, step=10)
    with c3:
        max_cars = st.number_input("Highest cars/day", min_value=20, max_value=12000, value=280, step=10)

    if not (min_cars <= avg_cars <= max_cars):
        st.error("Ensure: Lowest ≤ Average ≤ Highest")
        st.stop()

    st.info(
        "Accuracy note: Weather-only forecasting can still be off because it doesn't include promos, "
        "local road traffic changes, nearby competition, or event-driven spikes. "
        "Use the **Expected Change %** as the main decision support."
    )

    if st.button("Generate Instant Forecast"):
        if not location.strip():
            st.warning("Please enter a ZIP or City.")
            st.stop()

        try:
            rows = forecast_week_quick(location.strip(), int(min_cars), int(avg_cars), int(max_cars))
            df = pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

        st.dataframe(df[["Day", "Forecast Cars", "Peak Staff", "Expected Change %", "Weather Driver", "Rain %", "Temp (F)"]],
                     use_container_width=True)
        st.line_chart(df.set_index("Day")["Forecast Cars"])
        st.markdown(generate_summary(df))


# =========================
# 🚀 FULL AI MODE
# =========================
else:
    if supabase is None:
        st.error("Supabase not connected. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")
        st.stop()

    st.header("🚀 Full AI Mode (Self-Learning + Accuracy Dashboard)")

    menu = st.sidebar.selectbox("Full Mode Menu", ["Create Site", "Forecast & Train", "Accuracy Dashboard"])

    # -------- Create Site --------
    if menu == "Create Site":
        st.subheader("Create a Site (privacy-safe inputs only)")

        alias = st.text_input("Site Alias (e.g., GA Site 1)", value="")
        location = st.text_input("ZIP or City (for weather)", value="")

        colA, colB = st.columns(2)
        with colA:
            open_hour = st.slider("Opening Hour", 0, 23, 8)
        with colB:
            close_hour = st.slider("Closing Hour", 1, 24, 20)

        traffic = st.selectbox("Street Traffic", list(TRAFFIC_MULT.keys()))
        min_staff = st.slider("Minimum Staff", 1, 5, 2)

        prep = st.checkbox("Prep every car?")
        one_loader = st.checkbox("Can operate with one loader at moderate flow?")

        member_washes = st.number_input("Estimated member washes/day", min_value=0, max_value=2000, value=80, step=10)

        st.markdown("**Normal week car counts (typical non-rain week)**")
        baseline = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        defaults = [120, 140, 130, 150, 180, 240, 210]
        for d, dv in zip(days, defaults):
            baseline[d] = st.number_input(d, min_value=0, max_value=12000, value=int(dv), step=10, key=f"base_{d}")

        if st.button("Create Site"):
            if not alias.strip():
                st.warning("Enter a Site Alias.")
                st.stop()
            if not location.strip():
                st.warning("Enter ZIP or City.")
                st.stop()

            try:
                lat, lon = get_lat_lon(location.strip())
            except Exception as e:
                st.error(f"Location error: {e}")
                st.stop()

            payload = {
                "alias": alias.strip(),
                "location": location.strip(),
                "lat": lat,
                "lon": lon,
                "open_hour": int(open_hour),
                "close_hour": int(close_hour),
                "traffic": traffic,
                "min_staff": int(min_staff),
                "prep": bool(prep),
                "one_loader": bool(one_loader),
                "member_washes": int(member_washes),
                "baseline": baseline,
                "bias": 1.0,
                "rain_sensitivity": 1.0,
            }

            try:
                supabase.table("sites").insert(payload).execute()
                st.success("Site created ✅")
            except Exception as e:
                st.error(f"Failed to create site: {e}")

    # -------- Forecast & Train --------
    if menu == "Forecast & Train":
        st.subheader("Generate 7-day plan (and save forecasts for accuracy tracking)")

        try:
            sites = supabase.table("sites").select("*").execute().data
        except Exception as e:
            st.error(f"Failed to load sites: {e}")
            st.stop()

        if not sites:
            st.info("Create a site first.")
            st.stop()

        site = st.selectbox("Choose Site", sites, format_func=lambda x: x.get("alias", "Unknown"))

        if st.button("Generate AI Staffing Plan"):
            try:
                rows = forecast_week_full(site)
                df = pd.DataFrame(rows)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

            # Display
            st.dataframe(df[["Day", "Forecast Cars", "Peak Staff", "Expected Change %", "Weather Driver", "Rain %", "Temp (F)"]],
                         use_container_width=True)
            st.line_chart(df.set_index("Day")["Forecast Cars"])
            st.markdown(generate_summary(df))

            # Save forecasts to Supabase daily_metrics (upsert)
            try:
                for _, r in df.iterrows():
                    payload = {
                        "site_id": site["id"],
                        "day": r["Date"],
                        "forecast_cars": int(r["Forecast Cars"]),
                        "expected_change_pct": int(r["Expected Change %"]),
                        "weather_driver": str(r["Weather Driver"]),
                        "precip_prob": int(r["Rain %"]),
                        "precip_in": float(r["Rain (in)"]),
                        "temp_f": float(r["Temp (F)"]),
                    }
                    supabase.table("daily_metrics").upsert(payload).execute()
                st.success("Saved 7-day forecasts ✅ (for accuracy tracking)")
            except Exception as e:
                st.warning(f"Forecast shown, but saving to Supabase failed: {e}")

            st.divider()
            st.subheader("Train StaffPilot (enter actual cars)")

            st.caption("Entering actuals updates bias + rain_sensitivity and improves future forecasts.")
            day_pick = st.date_input("Date", value=date.today())
            actual = st.number_input("Actual cars washed", min_value=0, max_value=20000, value=0, step=10)

            if st.button("Save Actual + Update Model"):
                day_str = day_pick.isoformat()

                # Pull the stored forecast for that day (preferred) so training doesn't depend on current df
                try:
                    recs = (
                        supabase.table("daily_metrics")
                        .select("*")
                        .eq("site_id", site["id"])
                        .eq("day", day_str)
                        .limit(1)
                        .execute()
                        .data
                    )
                except Exception as e:
                    st.error(f"Failed to fetch stored forecast for training: {e}")
                    st.stop()

                if not recs:
                    st.warning("No stored forecast found for that date. Generate forecast first, then enter actuals.")
                    st.stop()

                rec = recs[0]
                forecast = int(rec.get("forecast_cars") or 0)
                prob = float(rec.get("precip_prob") or 0)
                rain_in = float(rec.get("precip_in") or 0.0)

                new_bias, new_rain = update_site_learning(site, forecast, int(actual), prob, rain_in)

                # Update site parameters
                try:
                    supabase.table("sites").update({
                        "bias": new_bias,
                        "rain_sensitivity": new_rain
                    }).eq("id", site["id"]).execute()
                except Exception as e:
                    st.error(f"Failed to update learning params: {e}")
                    st.stop()

                # Update actual_cars in daily_metrics (upsert)
                try:
                    supabase.table("daily_metrics").upsert({
                        "site_id": site["id"],
                        "day": day_str,
                        "actual_cars": int(actual),
                    }).execute()
                except Exception as e:
                    st.error(f"Failed to save actual: {e}")
                    st.stop()

                st.success("Saved ✅ StaffPilot learned from today.")
                st.write(f"Bias: **{float(site.get('bias', 1.0)):.3f} → {new_bias:.3f}**")
                st.write(f"Rain sensitivity: **{float(site.get('rain_sensitivity', 1.0)):.3f} → {new_rain:.3f}**")

    # -------- Accuracy Dashboard --------
    if menu == "Accuracy Dashboard":
        st.subheader("Accuracy Dashboard (Forecast vs Actual)")

        try:
            sites = supabase.table("sites").select("*").execute().data
        except Exception as e:
            st.error(f"Failed to load sites: {e}")
            st.stop()

        if not sites:
            st.info("Create a site first.")
            st.stop()

        site = st.selectbox("Choose Site", sites, format_func=lambda x: x.get("alias", "Unknown"))

        days_back = st.slider("Lookback window (days)", min_value=7, max_value=90, value=30, step=1)
        start_day = (date.today() - timedelta(days=int(days_back))).isoformat()
        end_day = date.today().isoformat()

        try:
            recs = (
                supabase.table("daily_metrics")
                .select("day,forecast_cars,actual_cars,expected_change_pct,weather_driver,precip_prob,temp_f")
                .eq("site_id", site["id"])
                .gte("day", start_day)
                .lte("day", end_day)
                .order("day")
                .execute()
                .data
            )
        except Exception as e:
            st.error(f"Failed to load metrics: {e}")
            st.stop()

        if not recs:
            st.info("No forecasts/actuals yet. Generate a forecast, then enter actuals for a few days.")
            st.stop()

        dfm = pd.DataFrame(recs)
        dfm["day"] = pd.to_datetime(dfm["day"])
        dfm = dfm.sort_values("day")

        # Compute error %
        dfm["error_pct"] = None
        mask = dfm["forecast_cars"].notna() & dfm["actual_cars"].notna() & (dfm["forecast_cars"] > 0)
        dfm.loc[mask, "error_pct"] = (
            (dfm.loc[mask, "actual_cars"] - dfm.loc[mask, "forecast_cars"]).abs()
            / dfm.loc[mask, "forecast_cars"] * 100.0
        )

        # Rolling MAPE
        def window_mape(n: int) -> Optional[float]:
            cut = dfm.tail(n)
            return mape(cut.rename(columns={"forecast_cars": "forecast_cars", "actual_cars": "actual_cars"}))

        m7 = window_mape(7)
        m14 = window_mape(14)
        m30 = window_mape(30)

        c1, c2, c3 = st.columns(3)
        c1.metric("7-day MAPE", f"{m7:.1f}%" if m7 is not None else "—")
        c2.metric("14-day MAPE", f"{m14:.1f}%" if m14 is not None else "—")
        c3.metric("30-day MAPE", f"{m30:.1f}%" if m30 is not None else "—")

        st.caption("MAPE = average absolute % error. Lower is better. (Only days with both forecast + actual are included.)")

        # Chart: forecast vs actual
        chart_df = dfm[["day", "forecast_cars", "actual_cars"]].set_index("day")
        st.line_chart(chart_df)

        # Table: include errors
        show_cols = ["day", "forecast_cars", "actual_cars", "error_pct", "expected_change_pct", "weather_driver", "precip_prob", "temp_f"]
        pretty = dfm[show_cols].copy()
        pretty = pretty.rename(columns={
            "day": "Date",
            "forecast_cars": "Forecast",
            "actual_cars": "Actual",
            "error_pct": "Error (%)",
            "expected_change_pct": "Expected Change (%)",
            "weather_driver": "Weather Driver",
            "precip_prob": "Rain (%)",
            "temp_f": "Temp (F)",
        })
        st.dataframe(pretty, use_container_width=True)

        st.info(
            "Trust signal: As you log more actuals, StaffPilot tunes **bias** and **rain sensitivity**, "
            "and you can see accuracy improve over time."
        )
