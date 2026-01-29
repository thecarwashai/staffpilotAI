import json
import math
from datetime import datetime, date as dt_date

import pandas as pd
import requests
import streamlit as st
from supabase import create_client

# ---------------- CONFIG ----------------

st.set_page_config(page_title="StaffPilot AI - Architected by Gopi Chand", layout="centered")
APP_NAME = "StaffPilot"
MAX_STAFF = 5  # hard cap

TRAFFIC_MULT = {
    "Low": 0.90,
    "Medium": 1.00,
    "High": 1.10,
    "Very High": 1.20,
}

# For quick-mode staffing capacity (cars per staff per hour)
QUICK_CPSH = 9.5

# ---------------- SUPABASE ----------------

SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- WEATHER (Open-Meteo) ----------------

@st.cache_data(ttl=60 * 60)
def get_lat_lon(location: str):
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
def get_weather_7d(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "forecast_days": 7,  # critical to force 7 days
        "timezone": "auto",
    }
    res = requests.get(url, params=params, timeout=10)
    if res.status_code != 200:
        raise RuntimeError("Weather API failed.")
    data = res.json()
    daily = data.get("daily")
    if not daily:
        raise RuntimeError("Weather returned no daily data.")
    required = [
        "time",
        "precipitation_probability_max",
        "precipitation_sum",
        "temperature_2m_max",
    ]
    for k in required:
        if k not in daily or not daily[k]:
            raise RuntimeError(f"Weather missing field: {k}")
    if min(len(daily["time"]), len(daily["precipitation_probability_max"]), len(daily["precipitation_sum"]), len(daily["temperature_2m_max"])) < 7:
        raise RuntimeError("Weather did not return a full 7-day forecast.")
    return daily

# ---------------- FORECAST ENGINE ----------------

def weather_factor(prob: float, rain_in: float, temp_f: float) -> float:
    """Base weather suppression/boost factor (1.0 = neutral)."""
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
    elif 50 <= temp_f <= 85:
        factor *= 1.05
    else:
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

def parse_baseline(baseline):
    if baseline is None:
        return {}
    if isinstance(baseline, str):
        baseline = json.loads(baseline)
    # normalize keys to lowercase day names
    return {str(k).strip().lower(): int(v) for k, v in baseline.items()}

def forecast_week_full(site: dict) -> list[dict]:
    """7-day forecast using baseline + membership + traffic + weather + bounce-back + site-tuned params."""
    weather = get_weather_7d(site["lat"], site["lon"])

    baseline = parse_baseline(site.get("baseline"))
    member_floor = int(site.get("member_washes", 0))
    traffic_mult = TRAFFIC_MULT.get(site.get("traffic", "Medium"), 1.0)

    bias = float(site.get("bias", 1.0))
    rain_sens = float(site.get("rain_sensitivity", 1.0))

    out = []
    prev_prob = 0.0
    prev_rain = 0.0

    for i in range(7):
        day_iso = weather["time"][i]
        prob = float(weather["precipitation_probability_max"][i])
        rain_in = float(weather["precipitation_sum"][i])
        temp_f = float(weather["temperature_2m_max"][i])

        dow = datetime.fromisoformat(day_iso).strftime("%A").lower()
        base = int(baseline.get(dow, 0))

        retail = max(base - member_floor, 0)

        wf = weather_factor(prob, rain_in, temp_f)

        # Site-specific tuning: rain sensitivity nudges the weather factor only on rainy-ish days
        if prob >= 40 or rain_in >= 0.05:
            wf *= rain_sens

        # Bounce-back (only on "good wash day")
        if prob < 30 and temp_f > 50:
            wf *= (1 + rebound_boost(prev_prob, prev_rain))

        cars = int((member_floor + (retail * traffic_mult * wf)) * bias)
        cars = max(0, cars)

        out.append({
            "Day": dow.title(),
            "Date": day_iso,
            "Forecast Cars": cars,
            "Rain %": prob,
            "Rain (in)": rain_in,
            "Temp (F)": temp_f
        })

        prev_prob = prob
        prev_rain = rain_in

    return out

def forecast_week_quick(location: str, avg_cars: int) -> list[dict]:
    """7-day forecast using ONLY avg cars/day + weather + bounce-back."""
    lat, lon = get_lat_lon(location)
    weather = get_weather_7d(lat, lon)

    out = []
    prev_prob = 0.0
    prev_rain = 0.0

    for i in range(7):
        day_iso = weather["time"][i]
        prob = float(weather["precipitation_probability_max"][i])
        rain_in = float(weather["precipitation_sum"][i])
        temp_f = float(weather["temperature_2m_max"][i])

        wf = weather_factor(prob, rain_in, temp_f)
        if prob < 30 and temp_f > 50:
            wf *= (1 + rebound_boost(prev_prob, prev_rain))

        cars = int(avg_cars * wf)
        cars = max(0, cars)

        # quick staffing estimate from peak-hour assumption
        peak_hour_cars = cars * 0.12
        staff = math.ceil(peak_hour_cars / QUICK_CPSH) if QUICK_CPSH > 0 else 0
        staff = min(MAX_STAFF, max(0, staff))

        out.append({
            "Day": datetime.fromisoformat(day_iso).strftime("%A"),
            "Date": day_iso,
            "Forecast Cars": cars,
            "Peak Staff": staff,
            "Rain %": prob,
            "Rain (in)": rain_in,
            "Temp (F)": temp_f
        })

        prev_prob = prob
        prev_rain = rain_in

    return out

# ---------------- STAFFING (FULL MODE) ----------------

def staff_needed_full(cars: int, prep: bool) -> int:
    """Peak-hour heuristic → staff count; capped at MAX_STAFF."""
    peak_hour = cars * 0.12  # peak hour ~12% of day total
    cpsh = 8 if prep else 11  # cars per staff-hour
    staff = math.ceil(peak_hour / cpsh) if cpsh > 0 else 0
    return min(MAX_STAFF, max(0, staff))

# ---------------- SELF-LEARNING ----------------

def update_site_learning(site_id: str, old_bias: float, old_rain_sens: float,
                         forecast: int, actual: int,
                         today_prob: float, today_rain_in: float) -> tuple[float, float]:
    """
    Update:
      - bias: overall correction factor
      - rain_sensitivity: adjusts how strongly rain suppresses volume in this market
    """
    if forecast <= 0:
        return old_bias, old_rain_sens

    ratio = actual / forecast  # >1 means we under-forecasted; <1 means over-forecasted

    # Bias update (smooth + bounded)
    new_bias = (old_bias * 0.80) + (ratio * 0.20)
    new_bias = clamp(new_bias, 0.70, 1.30)

    # Rain sensitivity update only on rainy-ish days
    new_rain = old_rain_sens
    rainy = (today_prob >= 50) or (today_rain_in >= 0.10)
    if rainy:
        # If actual > forecast on a rainy day, rain hurt LESS than we expected → lighten suppression
        if ratio > 1.00:
            new_rain *= 1.03
        else:
            # Rain hurt MORE than expected → increase suppression
            new_rain *= 0.97
        new_rain = clamp(new_rain, 0.70, 1.30)

    return new_bias, new_rain

# ---------------- SUMMARY ----------------

def generate_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    busiest = df.loc[df["Forecast Cars"].idxmax()]
    slowest = df.loc[df["Forecast Cars"].idxmin()]

    rain_impacted = df[df["Rain %"] >= 60].shape[0]
    peak_staff = int(df["Peak Staff"].max()) if "Peak Staff" in df.columns else None

    lines = []
    lines.append("### 🧠 StaffPilot Weekly Summary")
    lines.append(f"• Busiest day: **{busiest['Day']}** (~{int(busiest['Forecast Cars'])} cars)")
    lines.append(f"• Slowest day: **{slowest['Day']}** (~{int(slowest['Forecast Cars'])} cars)")
    lines.append(f"• Rain-impacted days (≥60%): **{rain_impacted}**")

    if peak_staff is not None:
        lines.append(f"• Peak staffing needed (cap {MAX_STAFF}): **{peak_staff}** staff")

    # Simple confidence heuristic
    avg_rain = float(df["Rain %"].mean())
    if avg_rain >= 60:
        conf = "Medium (weather-heavy week)"
    elif avg_rain >= 35:
        conf = "High (some weather variability)"
    else:
        conf = "High (stable week)"
    lines.append(f"• Confidence: **{conf}**")

    return "\n".join(lines)

# ---------------- UI ----------------

st.title("🚀 StaffPilot AI")

mode = st.sidebar.radio(
    "Select Mode",
    ["⚡ Quick Weather Forecast", "🚀 Full StaffPilot AI"]
)

# ============ QUICK MODE ============
if mode == "⚡ Quick Weather Forecast":
    st.header("⚡ Quick Weather Forecast (No Setup)")
    st.caption("Enter a ZIP or City and your average cars/day. StaffPilot adjusts volume using weather + bounce-back.")

    location = st.text_input("ZIP or City (required for weather)", value="")
    avg_cars = st.number_input("Average Cars Per Day", min_value=10, max_value=5000, value=150, step=10)

    if st.button("Generate Instant Forecast"):
        if not location.strip():
            st.warning("Please enter a ZIP or City.")
            st.stop()

        try:
            rows = forecast_week_quick(location.strip(), int(avg_cars))
            df = pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

        st.dataframe(df[["Day", "Forecast Cars", "Peak Staff", "Rain %", "Temp (F)"]], use_container_width=True)
        st.line_chart(df.set_index("Day")["Forecast Cars"])
        st.markdown(generate_summary(df))

    st.info("Want higher accuracy + self-learning? Use **Full StaffPilot AI** in the sidebar.")

# ============ FULL MODE ============
else:
    if supabase is None:
        st.error("Supabase secrets missing. Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets to use Full Mode.")
        st.stop()

    menu = st.sidebar.selectbox("Full Mode Menu", ["Create Site", "View Forecast & Train"])

    # ---- Create Site ----
    if menu == "Create Site":
        st.header("New Operator Setup (Privacy-Safe)")

        alias = st.text_input("Site Name / Alias (e.g., GA Site 1)")
        location = st.text_input("ZIP or City (used only for weather)")

        col1, col2 = st.columns(2)
        with col1:
            open_hour = st.slider("Opening Hour", 0, 23, 8)
        with col2:
            close_hour = st.slider("Closing Hour", 1, 24, 20)

        traffic = st.selectbox("Street Traffic", list(TRAFFIC_MULT.keys()))
        member_washes = st.number_input("Estimated Member Washes / Day", min_value=0, max_value=1000, value=80, step=5)

        min_staff = st.slider("Minimum Staff", 1, 5, 2)
        prep = st.checkbox("Prep Every Car?")
        one_loader = st.checkbox("Can operate with one loader at moderate flow?")

        st.subheader("Normal Week Cars (Typical Non-Rain Week)")
        baseline = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        default_vals = [120, 140, 130, 150, 180, 240, 210]
        for d, dv in zip(days, default_vals):
            baseline[d] = st.number_input(d, min_value=0, max_value=8000, value=int(dv), step=10, key=f"b_{d}")

        if st.button("Create Site"):
            if not alias.strip():
                st.warning("Enter a Site Name / Alias.")
                st.stop()
            if not location.strip():
                st.warning("Enter ZIP or City for weather.")
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

    # ---- View Forecast + Train ----
    if menu == "View Forecast & Train":
        st.header("7-Day Forecast + Self-Learning")

        try:
            sites = supabase.table("sites").select("*").execute().data
        except Exception as e:
            st.error(f"Failed to load sites: {e}")
            st.stop()

        if not sites:
            st.warning("Create a site first.")
            st.stop()

        site = st.selectbox("Choose Site", sites, format_func=lambda x: x.get("alias", "Unknown"))

        colA, colB = st.columns(2)
        with colA:
            gen = st.button("Generate AI Staffing Plan")
        with colB:
            show_history = st.button("Show Last 30 Days History")

        if gen:
            try:
                rows = forecast_week_full(site)
                df = pd.DataFrame(rows)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

            # Staffing + cap
            peak_staff_list = []
            for _, r in df.iterrows():
                staff = staff_needed_full(int(r["Forecast Cars"]), bool(site.get("prep", False)))
                peak_staff = min(MAX_STAFF, max(int(site.get("min_staff", 1)), staff))
                peak_staff_list.append(peak_staff)
            df["Peak Staff"] = peak_staff_list

            st.dataframe(df[["Day", "Forecast Cars", "Peak Staff", "Rain %", "Temp (F)"]], use_container_width=True)
            st.line_chart(df.set_index("Day")["Forecast Cars"])
            st.markdown(generate_summary(df))

            st.divider()
            st.subheader("📊 Train StaffPilot (Enter Actual Cars)")
            st.caption("This updates site bias + rain sensitivity. Only daily totals are stored.")

            # Default: today; allow user to pick (helpful for backfilling)
            day_pick = st.date_input("Date", value=dt_date.today())
            actual_cars = st.number_input("Actual cars washed", min_value=0, max_value=8000, value=0, step=10)

            if st.button("Save Actual + Update Model"):
                # Find matching forecast row by date (if within next 7 days)
                day_str = day_pick.isoformat()
                match = df[df["Date"] == day_str]

                if match.empty:
                    st.warning("Selected date is not in the current 7-day forecast. Generate forecast again, or pick a date within the next 7 days.")
                    st.stop()

                frow = match.iloc[0]
                forecast = int(frow["Forecast Cars"])
                prob = float(frow["Rain %"])
                rain_in = float(frow["Rain (in)"])

                old_bias = float(site.get("bias", 1.0))
                old_rain = float(site.get("rain_sensitivity", 1.0))
                new_bias, new_rain = update_site_learning(site["id"], old_bias, old_rain, forecast, int(actual_cars), prob, rain_in)

                # Update site params
                try:
                    supabase.table("sites").update({
                        "bias": new_bias,
                        "rain_sensitivity": new_rain
                    }).eq("id", site["id"]).execute()
                except Exception as e:
                    st.error(f"Failed updating learning params: {e}")
                    st.stop()

                # Store actual (upsert on unique site_id+day)
                payload = {"site_id": site["id"], "day": day_str, "cars": int(actual_cars)}
                try:
                    # supabase-py supports upsert in most versions; if not, fallback to insert
                    supabase.table("actuals").upsert(payload).execute()
                except Exception:
                    try:
                        supabase.table("actuals").insert(payload).execute()
                    except Exception as e:
                        st.error(f"Saved learning params but failed to store actual: {e}")
                        st.stop()

                st.success("Saved ✅ StaffPilot got smarter.")
                st.write(f"Updated bias: **{old_bias:.3f} → {new_bias:.3f}**")
                st.write(f"Updated rain sensitivity: **{old_rain:.3f} → {new_rain:.3f}**")

        if show_history:
            st.subheader("📈 Forecast Accuracy History (Last 30 Days)")
            # Pull actuals and (optionally) show trend; since we don't store forecasts historically in this MVP,
            # we show actual counts trend. (Next upgrade: store forecast snapshot per day.)
            try:
                end = dt_date.today()
                start = end - pd.Timedelta(days=30)
                actuals = (
                    supabase.table("actuals")
                    .select("day,cars")
                    .eq("site_id", site["id"])
                    .gte("day", start.isoformat())
                    .lte("day", end.isoformat())
                    .order("day")
                    .execute()
                    .data
                )
            except Exception as e:
                st.error(f"Failed to load history: {e}")
                st.stop()

            if not actuals:
                st.info("No actuals saved yet. Train StaffPilot by entering daily actual cars.")
                st.stop()

            hdf = pd.DataFrame(actuals)
            hdf["day"] = pd.to_datetime(hdf["day"])
            hdf = hdf.sort_values("day")
            hdf = hdf.rename(columns={"day": "Date", "cars": "Actual Cars"})

            st.dataframe(hdf, use_container_width=True)
            st.line_chart(hdf.set_index("Date")["Actual Cars"])
            st.caption("Next upgrade: store daily forecast snapshots to show forecast-vs-actual accuracy over time.")
