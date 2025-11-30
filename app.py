"""
app.py - Streamlit UI
Features:
- upload image or enter local path
- run model prediction
- fetch NASA POWER and OpenWeather, blend results
- show numeric verdict, 0-100 sustainability score
- display remedy text
- show weather graphs (7-day forecast if OpenWeather available)
- color-coded verdict (green/yellow/red)
"""

import os
import sys
import time
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_utils import (
    load_model, preprocess_image, class_idx_to_name, disease_solutions,
    fetch_and_blend_weather, compute_sustainability_score, IMAGE_SIZE
)

st.set_page_config(page_title="PCNN — Disease + Sustainability", layout="centered")

st.title("PCNN — Plant disease recognition & sustainability (NASA + OpenWeather)")

st.sidebar.header("Settings & Weather API")
openweather_key = st.sidebar.text_input("OpenWeather API Key (optional)", type="password")
st.sidebar.markdown("Get API key: https://openweathermap.org/api")

mode_ip = st.sidebar.checkbox("Auto-detect my location (IP)", value=True)
manual_coords = st.sidebar.checkbox("Enter coordinates manually", value=False)

lat = None; lon = None
if manual_coords:
    lat = st.sidebar.number_input("Latitude", format="%.6f", value=13.0878)
    lon = st.sidebar.number_input("Longitude", format="%.6f", value=80.2785)
elif mode_ip:
    # try ip geolocation
    try:
        r = st.sidebar.button("Detect location now")
        if r:
            import requests
            j = requests.get("https://ipinfo.io/json", timeout=6).json()
            loc = j.get("loc","")
            if loc:
                lat, lon = map(float, loc.split(","))
                st.sidebar.success(f"Detected: {j.get('city','')} {j.get('region','')}")
            else:
                st.sidebar.warning("IP detection didn't return a location.")
    except Exception:
        st.sidebar.warning("IP detection unavailable. Enter coordinates manually.")

st.write("## Upload a leaf image or enter a local path")
uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])
local_path = st.text_input("Or enter local image path (optional)", "")

image = None
if uploaded:
    image = Image.open(uploaded).convert("RGB")
elif local_path:
    if os.path.exists(local_path):
        image = Image.open(local_path).convert("RGB")
    else:
        if local_path.strip():
            st.warning("Local path not found.")

if image is None:
    st.info("Please upload an image or provide a valid local path.")
    st.stop()

st.image(image, caption="Input image", use_column_width=True)

# load model
@st.cache_resource
def _load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(checkpoint_path="checkpoints/best.pth", device=device)
    return model, classes, device

try:
    model, CLASSES, device = _load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# run prediction
img_t = preprocess_image(image).unsqueeze(0).to(device)
with st.spinner("Running model..."):
    with torch.no_grad():
        logits = model(img_t, None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = class_idx_to_name(pred_idx, CLASSES)
        confidence = float(probs.max())

st.markdown(f"### Prediction: **{pred_class}**   (confidence {confidence:.2%})")

# solution block
sol = disease_solutions.get(pred_class)
if sol:
    st.markdown("#### Suggested treatment")
    st.write(sol["description"])
    for a in sol["actions"]:
        st.write("- " + a)
else:
    st.markdown("#### Suggested treatment")
    st.write("No specific remedy mapping found; follow general sanitation measures:")
    st.write("- Remove infected leaves and dispose safely")
    st.write("- Improve ventilation and avoid overhead watering")
    st.write("- Use recommended fungicides/bactericides per extension guidelines")

# get coordinates (if still None try IP detection quick)
if lat is None or lon is None:
    try:
        import requests
        j = requests.get("https://ipinfo.io/json", timeout=6).json()
        loc = j.get("loc","")
        if loc:
            lat, lon = map(float, loc.split(","))
    except Exception:
        pass

if lat is None or lon is None:
    st.error("No coordinates available. Enable IP detection or enter coordinates in sidebar.")
    st.stop()

st.markdown(f"**Location:** {lat:.6f}, {lon:.6f}")

# fetch and blend weather
with st.spinner("Fetching weather (NASA POWER + OpenWeather fallback)..."):
    try:
        weather_pack = fetch_and_blend_weather(lat, lon, openweather_key)
    except Exception as e:
        st.error(f"Weather fetch failed: {e}")
        weather_pack = {"blended": {}, "nasa": {}, "openweather": {}}

blended = weather_pack.get("blended", {})
nasa = weather_pack.get("nasa", {})
ow = weather_pack.get("openweather", {})

st.markdown("### Weather snapshot (blended)")
cols = st.columns(4)
cols[0].metric("Temperature (°C)", value=(f"{blended.get('temperature'):.2f}" if blended.get('temperature') is not None else "N/A"))
cols[1].metric("Humidity (%)", value=(f"{blended.get('humidity'):.1f}" if blended.get('humidity') is not None else "N/A"))
cols[2].metric("Precip (mm)", value=(f"{blended.get('precip_mm'):.2f}" if blended.get('precip_mm') is not None else "N/A"))
cols[3].metric("Wind (m/s)", value=(f"{blended.get('wind_m_s'):.2f}" if blended.get('wind_m_s') is not None else "N/A"))

# compute sustainability
score, reasons = compute_sustainability_score(pred_class, blended)

# color-coded verdict
if score >= 70:
    box_color = "#16a34a"  # green
    verdict = "✅ Likely sustainable"
elif score >= 40:
    box_color = "#f59e0b"  # orange
    verdict = "⚠️ Moderate risk"
else:
    box_color = "#dc2626"  # red
    verdict = "❌ Not sustainable / High risk"

st.markdown(f"## Sustainability: {verdict} (score {score}/100)")
with st.expander("Why?"):
    for r in reasons:
        st.write("- " + r)

# Show raw sources if user wants
with st.expander("Show raw weather sources (NASA / OpenWeather)"):
    st.write("NASA POWER returned:", nasa.get("_source"), nasa.get("_date_range"))
    st.json(nasa)
    st.write("OpenWeather returned (daily forecast):")
    st.json(ow)

# Weather graph: if OpenWeather daily exists plot 7-day forecast, else create small plot from NASA (if available)
st.markdown("### Weather graph / 7-day forecast")
fig = None
if ow and ow.get("daily"):
    days = ow["daily"]
    dates = [datetime.utcfromtimestamp(d["dt"]).date() for d in days]
    temps = [d["temp_day"] for d in days]
    humid = [d["humidity"] for d in days]
    fig, ax = plt.subplots(2, 1, figsize=(6,4), sharex=True)
    ax[0].plot(dates, temps, marker='o'); ax[0].set_ylabel("Temp (°C)")
    ax[1].plot(dates, humid, marker='o', color='tab:orange'); ax[1].set_ylabel("Humidity (%)")
    fig.autofmt_xdate()
    st.pyplot(fig)
elif nasa:
    # NASA doesn't provide forecast but we can plot last-days series if present
    # attempt to extract T2M series from nasa parameter block if present (predict_utils already consumed)
    st.info("OpenWeather forecast not available. Showing NASA snapshot values (no multi-day forecast).")
    # show simple bar for available NASA params
    keys = []
    vals = []
    for k in ["T2M","RH2M","PRECTOT","WS2M"]:
        if nasa.get(k) is not None:
            keys.append(k)
            vals.append(nasa.get(k))
    if vals:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(keys, vals); ax.set_ylabel("Value")
        st.pyplot(fig)
    else:
        st.write("No NASA detail series available to plot.")

st.markdown("---")
st.write("**Next steps & suggestions:**")
if score < 50:
    st.write("- Consider relocating plants, apply protective measures (covering), or fungicide sprays as per extension guidance.")
elif score < 70:
    st.write("- Monitor daily, reduce leaf wetness, improve airflow.")
else:
    st.write("- Conditions are acceptable; continue monitoring and good sanitation.")

st.success("Done — use the sidebar to change OpenWeather API key or coordinates.")
