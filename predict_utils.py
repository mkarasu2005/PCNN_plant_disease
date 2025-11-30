"""
predict_utils.py
Utilities:
- load_model (uses FusionModel from model.py)
- preprocessing
- NASA POWER query with smart fallback (finds most recent available daily data)
- NASA climatology fallback
- OpenWeatherMap current + 7-day forecast fallback (requires API key)
- blending of NASA + OpenWeather results
- compute 0-100 sustainability score + reasons
- disease_solutions mapping for 16 classes
"""

import os
import time
import math
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# import your model class
from model import FusionModel
import torch

IMAGE_SIZE = 224

# ---------------------------
# Model loading & preprocessing
# ---------------------------
def load_model(checkpoint_path="checkpoints/best.pth", device="cpu"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt.get('classes', None)
    env_features = ckpt.get('env_features', []) if 'env_features' in ckpt else []
    num_classes = len(classes) if classes else 1
    env_dim = len(env_features) if env_features else 0
    model = FusionModel(num_classes=num_classes, env_in_dim=env_dim, image_embed_dim=512, transformer_dim=256)
    # accept either 'state_dict' or 'state'
    state = ckpt.get('state_dict') or ckpt.get('state') or ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes

def preprocess_image(pil_img: Image.Image, image_size=IMAGE_SIZE):
    tf = transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(pil_img)

def class_idx_to_name(idx, classes):
    if classes is None:
        return str(idx)
    return classes[idx]

# ---------------------------
# Disease remedies (16 classes)
# ---------------------------
# Short human readable descriptions + recommended actions.
disease_solutions = {
    "Pepper__bell___Bacterial_spot": {
        "description":"Bacterial spot on pepper causes small dark lesions on leaves and fruit.",
        "actions":["Remove infected leaves","Use copper-based bactericides","Avoid overhead irrigation","Rotate crops and use disease-free seed/planting material"]
    },
    "Pepper__bell___healthy": {
        "description":"Healthy pepper leaf with no visible disease.",
        "actions":["No treatment required","Monitor regularly"]
    },
    "PlantVillage": { "description":"Generic healthy sample (PlantVillage).", "actions":["Monitor plant health"] },
    "Potato___Early_blight": {
        "description":"Early blight (Alternaria) causes concentric dark rings on leaves.",
        "actions":["Remove and destroy infected foliage","Apply appropriate fungicides (e.g., chlorothalonil)","Improve canopy airflow"]
    },
    "Potato___healthy": {"description":"Healthy potato leaf.","actions":["Monitor for pests and diseases"]},
    "Potato___Late_blight": {
        "description":"Late blight (Phytophthora) — fast-spreading; dark lesions and collapse.",
        "actions":["Remove infected plants immediately","Apply systemic fungicides","Avoid overhead watering and dense plantings"]
    },
    "Tomato__Target_Spot": {"description":"Target spot lesions on tomato.", "actions":["Remove affected leaves","Apply fungicide as advised","Practice crop rotation"]},
    "Tomato__Tomato_mosaic_virus": {"description":"Tomato mosaic virus causes mottling and distortion.", "actions":["Remove infected plants","Disinfect tools","Use resistant varieties if available"]},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"description":"TYLCV causes yellowing and curling of leaves.", "actions":["Remove infected plants","Control whiteflies (vectors)","Use certified seedlings and resistant varieties"]},
    "Tomato_Bacterial_spot": {"description":"Bacterial spot on tomato.", "actions":["Use copper sprays","Improve sanitation","Rotate crops"]},
    "Tomato_Early_blight": {"description":"Early blight on tomato.", "actions":["Prune infected foliage","Apply fungicide","Avoid overhead irrigation"]},
    "Tomato_healthy": {"description":"Healthy tomato leaf.", "actions":["No treatment required","Monitor weekly"]},
    "Tomato_Late_blight": {"description":"Late blight on tomato; severe under wet conditions.", "actions":["Remove plants rapidly","Apply systemic fungicides","Avoid overhead irrigation"]},
    "Tomato_Leaf_Mold": {"description":"Leaf mold causes fuzzy patches on tomato leaves.", "actions":["Improve ventilation","Apply fungicide if severe","Reduce humidity"]},
    "Tomato_Septoria_leaf_spot": {"description":"Septoria leaf spot causes small circular spots.", "actions":["Remove infected leaves","Apply fungicides","Rotate crops"]},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"description":"Spider mite infestation (tiny stippling / webbing).", "actions":["Use miticides or predatory mites","Increase humidity","Wash leaves with water spray"]},
}

# ---------------------------
# NASA POWER: robust daily query (finds most recent available)
# ---------------------------
def query_nasa_power(lat: float, lon: float, parameters: Optional[List[str]] = None, lookback_max_days: int = 14) -> Dict[str, Optional[float]]:
    """
    Query NASA POWER daily point endpoint and search backwards until valid data found.
    Returns dictionary with parameter -> float or None.
    Uses community=AG (agriculture).
    """
    if parameters is None:
        parameters = ["T2M", "RH2M", "PRECTOT", "ALLSKY_SFC_SW_DWN", "WS2M", "TSOIL1"]

    for days_back in range(2, lookback_max_days + 1):  # start 2 days ago
        end = datetime.utcnow().date() - timedelta(days=days_back)
        start = end - timedelta(days=3)
        start_date = start.strftime("%Y%m%d")
        end_date = end.strftime("%Y%m%d")
        params_str = ",".join(parameters)
        url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"community=AG&start={start_date}&end={end_date}"
            f"&latitude={lat}&longitude={lon}"
            f"&parameters={params_str}&format=JSON"
        )
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        props = data.get("properties", {})
        param_block = props.get("parameter", {})
        out = {}
        for p in parameters:
            series = param_block.get(p, {})
            if not series:
                out[p] = None
            else:
                last_key = sorted(series.keys())[-1]
                val = series[last_key]
                if val in [-999, -9999, None, "NULL", "null"]:
                    out[p] = None
                else:
                    try:
                        out[p] = float(val)
                    except Exception:
                        out[p] = None

        # return if ANY real number present (not all None)
        if any(v is not None for v in out.values()):
            out["_source"] = "NASA_POWER"
            out["_date_range"] = (start_date, end_date)
            return out

    # final fallback: attempt climatology endpoint (monthly/normal) to get plausible values
    try:
        clim_params = ",".join([p for p in parameters if p != "TSOIL1"])
        # climatology endpoint returns monthly averages; we'll take mean of months
        clim_url = (
            "https://power.larc.nasa.gov/api/temporal/climatology/point?"
            f"community=AG&start=19810101&end=20101231"
            f"&latitude={lat}&longitude={lon}"
            f"&parameters={clim_params}&format=JSON"
        )
        r = requests.get(clim_url, timeout=12)
        r.raise_for_status()
        data = r.json()
        props = data.get("properties", {})
        param_block = props.get("parameter", {})
        out = {}
        for p in parameters:
            series = param_block.get(p, {})
            if isinstance(series, dict) and series:
                # series keys are month numbers '01'..'12' or dates — average them
                vals = []
                for k, v in series.items():
                    try:
                        if v in [-999, -9999, None, "NULL", "null"]:
                            continue
                        vals.append(float(v))
                    except:
                        continue
                out[p] = float(np.mean(vals)) if vals else None
            else:
                out[p] = None
        out["_source"] = "NASA_CLIM"
        out["_date_range"] = ("climatology",)
        return out
    except Exception:
        # give back all None if nothing works
        return {p: None for p in parameters}

# ---------------------------
# OpenWeatherMap fallback (current + 7-day forecast)
# ---------------------------
def query_openweather(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    """
    Uses OpenWeather OneCall (or current + forecast). Returns:
    - current: temp (C), humidity (%), wind_speed (m/s), precipitation_mm (last hour or day)
    - daily: list of dicts with dt, temp_day, humidity, precipitation
    """
    if not api_key:
        return {}

    # OneCall 3.0 requires paid; we attempt current+forecast endpoints if available.
    # Use "onecall" if available; else use current and 7-day forecast endpoints.
    # Simpler: use One Call 2.5 (works with free key) via /data/2.5/onecall
    base = "https://api.openweathermap.org/data/2.5/onecall"
    params = {
        "lat": lat, "lon": lon, "exclude": "minutely,alerts", "units": "metric", "appid": api_key
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
    except Exception:
        return {}

    out = {}
    current = d.get("current", {})
    out["current_temp"] = current.get("temp")
    out["current_humidity"] = current.get("humidity")
    out["current_wind"] = current.get("wind_speed")
    # precipitation: check 'rain' or 'snow'
    precip = None
    rain = current.get("rain", {})
    if isinstance(rain, dict):
        precip = rain.get("1h") or rain.get("3h")
    out["current_precip_mm"] = precip
    # 7-day forecast
    daily = []
    for dd in d.get("daily", [])[:7]:
        daily.append({
            "dt": dd.get("dt"),
            "temp_day": dd.get("temp", {}).get("day"),
            "humidity": dd.get("humidity"),
            "precip": (dd.get("rain") or dd.get("snow") or 0)
        })
    out["daily"] = daily
    out["_source"] = "OPENWEATHER"
    return out

# ---------------------------
# Blend NASA and OpenWeather results
# ---------------------------
def blend_weather(nasa: Dict[str, Optional[float]], ow: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Create a blended summary:
    - prefer NASA values for historical daily averages (T2M, RH2M, PRECTOT)
    - if NASA missing and OpenWeather present, use OpenWeather current/daily
    - for forecast graph we return OpenWeather daily list if present
    """
    out = {}
    # Temperature: NASA T2M (C) or OW current_temp
    out["temperature"] = nasa.get("T2M") if nasa.get("T2M") is not None else ow.get("current_temp")
    out["humidity"] = nasa.get("RH2M") if nasa.get("RH2M") is not None else ow.get("current_humidity")
    # precipitation: NASA PRECTOT (mm) else OW current_precip_mm
    out["precip_mm"] = nasa.get("PRECTOT") if nasa.get("PRECTOT") is not None else ow.get("current_precip_mm")
    out["wind_m_s"] = nasa.get("WS2M") if nasa.get("WS2M") is not None else ow.get("current_wind")
    out["solar_w_m2"] = nasa.get("ALLSKY_SFC_SW_DWN") if nasa.get("ALLSKY_SFC_SW_DWN") is not None else None
    out["_nasa_source"] = nasa.get("_source")
    out["_ow_source"] = ow.get("_source")
    out["_ow_daily"] = ow.get("daily")
    return out

# ---------------------------
# Sustainability scoring: 0-100 (higher is better)
# ---------------------------
# disease-specific vulnerability ranges where disease thrives (if weather in that range -> lower score)
DISEASE_VULNERABILITIES = {
    # format: (temp_opt_min, temp_opt_max, hum_opt_min, hum_opt_max) => disease most favored in this box
    "Tomato_Early_blight": (15, 28, 70, 100),
    "Tomato_Late_blight": (8, 22, 80, 100),
    "Tomato_Bacterial_spot": (18, 32, 75, 100),
    "Tomato_Leaf_Mold": (18, 26, 85, 100),
    "Tomato_Septoria_leaf_spot": (15, 27, 80, 100),
    "Tomato_Spider_mites_Two_spotted_spider_mite": (25, 40, 10, 60),
    "Tomato_Early_blight": (15,28,70,100),
    "Tomato_healthy": (10,35,20,90),
    "Potato___Late_blight": (5,20,80,100),
    "Potato___Early_blight": (15,28,70,100),
    "Pepper__bell___Bacterial_spot": (18,32,70,100),
    # add others generically
}

def compute_sustainability_score(pred_class: str, blended: Dict[str, Optional[float]]) -> Tuple[int, List[str]]:
    """
    Return (score 0-100) and list of reasons.
    Score: 100 best (very sustainable), 0 worst (very risky).
    Logic:
    - If temperature/humidity missing -> moderate score 50 with reason
    - If both present: penalize if within disease-favorable ranges; otherwise reward
    - Factor in precipitation: heavy rainfall (>10 mm/day) increases risk for fungal diseases
    """
    reasons = []
    temp = blended.get("temperature")
    hum = blended.get("humidity")
    precip = blended.get("precip_mm")
    wind = blended.get("wind_m_s")

    # base score
    score = 70

    if temp is None or hum is None:
        reasons.append("Temperature or humidity missing; using conservative default.")
        return 50, reasons

    # clamp sensible ranges
    if temp < -30 or temp > 60:
        reasons.append(f"Extreme temperature ({temp}°C) — out of typical crop bounds.")
        return 10, reasons

    # disease vulnerability
    vuln = DISEASE_VULNERABILITIES.get(pred_class)
    if vuln:
        tmin, tmax, hmin, hmax = vuln
        # if current weather lies IN disease favorable box -> heavy penalty
        if tmin <= temp <= tmax and hmin <= hum <= hmax:
            reasons.append(f"Temperature {temp}°C and humidity {hum}% are favorable for {pred_class}.")
            score -= 45
        else:
            # attendance: distance from favorable box reduces risk
            # compute temperature distance penalty
            dt = 0
            if temp < tmin:
                dt = (tmin - temp) / max(1.0, abs(tmin))
            elif temp > tmax:
                dt = (temp - tmax) / max(1.0, abs(tmax))
            dh = 0
            if hum < hmin:
                dh = (hmin - hum) / max(1.0, hmin)
            elif hum > hmax:
                dh = (hum - hmax) / max(1.0, hmax)
            penalty = max(0, 25 - int((dt + dh) * 25))
            score -= penalty
            reasons.append(f"Current weather slightly away from disease-favorable range; penalty {penalty} applied.")
    else:
        # generic checks
        if hum > 90:
            reasons.append(f"High humidity ({hum}%) increases disease risk.")
            score -= 20
        if temp < 5 or temp > 40:
            reasons.append(f"Temperature ({temp}°C) is extreme for many crops.")
            score -= 20

    # precipitation penalties (fungal risk)
    if precip is not None:
        try:
            if precip > 10:
                reasons.append(f"High recent precipitation ({precip} mm) increases fungal disease risk.")
                score -= 20
            elif precip > 2:
                reasons.append(f"Moderate precipitation ({precip} mm).")
                score -= 8
        except:
            pass

    # wind increases risk for some pests/viruses spreading (small penalty)
    if wind is not None:
        if wind > 10:
            reasons.append(f"High wind speed ({wind} m/s) may spread pests/viruses.")
            score -= 8

    # clamp score to 0-100
    score = max(0, min(100, score))
    # better wording if score high
    if score >= 70:
        reasons.insert(0, "Overall conditions are not favorable for the detected disease.")
    else:
        reasons.insert(0, "Conditions increase disease risk.")

    return int(score), reasons

# ---------------------------
# Small helper to combine functions for app.py
# ---------------------------
def fetch_and_blend_weather(lat: float, lon: float, openweather_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns dict:
      - 'blended': blended weather summary
      - 'nasa': nasa raw dict
      - 'openweather': openweather raw dict (may be {})
    """
    nasa = query_nasa_power(lat, lon)
    ow = {}
    if openweather_key:
        try:
            ow = query_openweather(lat, lon, openweather_key)
        except Exception:
            ow = {}
    blended = blend_weather(nasa, ow)
    return {"blended": blended, "nasa": nasa, "openweather": ow}

# ---------------------------
# Expose everything
# ---------------------------
__all__ = [
    "load_model", "preprocess_image", "class_idx_to_name", "disease_solutions",
    "query_nasa_power", "query_openweather", "blend_weather", "fetch_and_blend_weather",
    "compute_sustainability_score", "IMAGE_SIZE", "class_idx_to_name"
]
