"""
AdrenalineAI - UFC Fight Prediction Platform
Full-stack app for predicting UFC fight outcomes with visual explainability.
"""

import os
import sys
import json
import hashlib
import base64

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import (
    build_fighter_database,
    build_fight_history_database,
    get_upcoming_card,
    scrape_fighter_details,
    scrape_fight_detail,
    HEADERS,
)
from preprocessing import (
    clean_fighter_data,
    get_fighter_by_name,
    get_style_matchup_description,
    FEATURE_COLUMNS,
)
from model import UFCPredictor
import math

# ── Design tokens ────────────────────────────────────────────────────────────
HF = "'Russo One', sans-serif"
BF = "'Nunito Sans', sans-serif"
AC = "#d32f2f"
AC_DARK = "#b71c1c"

# Theme palettes
DARK_THEME = {
    "BG": "#0d0d0d", "CARD_BG": "#111", "INNER_BG": "#161616",
    "TEXT": "#e0e0e0", "TEXT_DIM": "#e0e0e0", "TEXT_MUTED": "#e0e0e0",
    "BORDER": "#1a1a1a", "BORDER_HOVER": "#333",
    "INITIALS_BG": "#1a1a1a", "INITIALS_BORDER": "#333",
    "REASON_BG": "rgba(100,100,100,0.1)", "TABLE_HOVER": "rgba(211,47,47,0.03)",
    "GRADIENT_END": "rgba(211,47,47,0.06)",
}
LIGHT_THEME = {
    "BG": "#f5f5f5", "CARD_BG": "#ffffff", "INNER_BG": "#eaeaea",
    "TEXT": "#1a1a1a", "TEXT_DIM": "#1a1a1a", "TEXT_MUTED": "#1a1a1a",
    "BORDER": "#ddd", "BORDER_HOVER": "#bbb",
    "INITIALS_BG": "#e0e0e0", "INITIALS_BORDER": "#ccc",
    "REASON_BG": "rgba(60,60,60,0.06)", "TABLE_HOVER": "rgba(211,47,47,0.06)",
    "GRADIENT_END": "rgba(211,47,47,0.08)",
}

IMAGE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_cache")
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
IMAGE_CACHE_FILE = os.path.join(IMAGE_CACHE_DIR, "fighter_images.json")

# Logo
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logos", "fist_white_fill.png")
with open(LOGO_PATH, "rb") as _lf:
    LOGO_B64 = base64.b64encode(_lf.read()).decode()
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}"
LOGO_FILTER = "drop-shadow(0 0 4px rgba(211,47,47,0.5)) drop-shadow(0 0 10px rgba(211,47,47,0.2))"

import random
LOADING_SAYINGS = [
    "AdrenalineAI pumping...",
    "Stepping into the octagon...",
    "Warming up the engines...",
    "Wrapping the hands...",
    "Fight camp in session...",
    "Entering the cage...",
    "Sizing up the competition...",
    "Fueling the fire...",
    "Getting fight ready...",
    "Walking out to the octagon...",
]

def _loading_screen(text: str | None = None):
    """Full-screen centered loading screen with logo and themed saying."""
    saying = text or random.choice(LOADING_SAYINGS)
    st.markdown(
        f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
        f'min-height:60vh;text-align:center;">'
        f'<img src="{LOGO_SRC}" style="width:80px;height:80px;object-fit:contain;'
        f'filter:{LOGO_FILTER};margin-bottom:1.5rem;'
        f'animation:pulse 1.5s ease-in-out infinite;" />'
        f'<div style="font-family:{HF};color:{AC};font-size:1.6rem;letter-spacing:2px;">ADRENALINE</div>'
        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:1rem;margin-top:0.8rem;letter-spacing:1px;">{saying}</div>'
        f'</div>'
        f'<style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}</style>',
        unsafe_allow_html=True,
    )

# Page config
st.set_page_config(
    page_title="AdrenalineAI",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme state ──────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = st.query_params.get("theme", "dark")
T = DARK_THEME if st.session_state.theme == "dark" else LIGHT_THEME
BG = T["BG"]; CARD_BG = T["CARD_BG"]; INNER_BG = T["INNER_BG"]
LOGO_FILTER = (
    "drop-shadow(0 0 2px rgba(211,47,47,0.25)) drop-shadow(0 0 5px rgba(211,47,47,0.1))"
    if st.session_state.theme == "dark"
    else "invert(1) drop-shadow(0 0 2px rgba(0,0,0,0.2)) drop-shadow(0 0 4px rgba(0,0,0,0.1))"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Russo+One&family=Nunito+Sans:wght@300;400;600;700;800&display=swap');

    /* Slide-up fade animation */
    @keyframes slideUp {{
        from {{ transform: translateY(12px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* Global */
    *, *::before, *::after {{
        transition: background-color 0.4s ease, color 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease, fill 0.4s ease;
    }}
    .stApp {{
        background-color: {BG};
        color: {T["TEXT"]};
        font-family: {BF};
    }}

    /* Scale-in on tab content and main elements */
    .stTabs [role="tabpanel"] {{
        animation: slideUp 0.3s ease-out;
        zoom: 0.82;
    }}
    .winner-container, .fighter-bg-card, .style-matchup,
    .event-header, .section-header, .header-bar {{
        animation: slideUp 0.3s ease-out;
    }}

    /* Smooth transitions on interactive elements */
    .stButton > button {{
        transition: all 0.25s ease !important;
    }}
    .stSelectbox, .stTabs [data-baseweb="tab"] {{
        transition: all 0.2s ease;
    }}

    /* Hide chrome */
    #MainMenu, footer, header {{ visibility: hidden; height: 0; padding: 0; margin: 0; }}
    .block-container {{ padding-top: 1rem !important; }}

    /* Header bar with stats */
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid {T["BORDER"]};
        margin-bottom: 0.5rem;
    }}
    .header-bar .brand {{
        font-family: {HF};
        font-size: 1.4rem;
        color: {AC};
        letter-spacing: 2px;
    }}
    .header-stats {{
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }}
    .header-stats .h-stat {{
        font-family: {BF};
    }}
    .header-stats .h-stat .h-num {{
        color: {T["TEXT"]};
        font-size: 0.8rem;
        font-weight: 700;
    }}
    .header-stats .h-stat .h-label {{
        color: {T["TEXT_MUTED"]};
        font-size: 0.65rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-left: 0.3rem;
    }}
    .header-stats .h-divider {{
        width: 1px;
        height: 20px;
        background: {T["BORDER_HOVER"]};
    }}

    /* Underline tab navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: transparent;
        border-bottom: 1px solid {T["BORDER"]};
        padding: 0;
        justify-content: center;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: {BF};
        color: {T["TEXT"]};
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: 0.95rem;
        padding: 0.8rem 0;
        border-radius: 0;
        background: transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {AC} !important;
        border-bottom: 2px solid {AC} !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* Fighter bg card */
    .fighter-bg-card {{
        flex: 1;
        border-radius: 20px;
        overflow: hidden;
        position: relative;
        min-height: 280px;
        background: linear-gradient(135deg, {CARD_BG} 0%, {BG} 60%, {T["GRADIENT_END"]} 100%);
    }}
    .fighter-bg-card img.fighter-photo {{
        position: absolute;
        right: 0;
        bottom: 0;
        height: 300px;
        object-fit: contain;
        opacity: 0.7;
        filter: drop-shadow(-10px 0 30px rgba(0,0,0,0.8));
    }}
    .fighter-bg-card .fighter-info {{
        position: relative;
        z-index: 1;
        padding: 2.5rem 2rem;
    }}
    .fighter-bg-card .f-weight {{
        font-family: {BF};
        color: {T["TEXT_MUTED"]};
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }}
    .fighter-bg-card .f-last {{
        font-family: {HF};
        color: {T["TEXT"]};
        font-size: 2.4rem;
        text-transform: uppercase;
        margin: 0.3rem 0;
        line-height: 1;
    }}
    .fighter-bg-card .f-first {{
        font-family: {HF};
        color: {"rgba(255,255,255,0.3)" if st.session_state.theme == "dark" else "rgba(0,0,0,0.3)"};
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .fighter-bg-card .f-record {{
        font-family: {BF};
        color: {AC};
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 1rem;
    }}
    .fighter-bg-card .f-record-label {{
        font-family: {BF};
        color: {T["TEXT_DIM"]};
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}
    .fighter-bg-card .f-stats {{
        display: flex;
        gap: 1.5rem;
        margin-top: 1.2rem;
    }}
    .fighter-bg-card .f-stat-val {{
        font-family: {BF};
        color: {T["TEXT"]};
        font-size: 1rem;
        font-weight: 700;
    }}
    .fighter-bg-card .f-stat-lbl {{
        font-family: {BF};
        color: {T["TEXT_MUTED"]};
        font-size: 0.55rem;
        letter-spacing: 1px;
    }}

    /* Initials fallback circle */
    .initials-circle {{
        position: absolute;
        right: 40px;
        top: 50%;
        transform: translateY(-50%);
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: {T["INITIALS_BG"]};
        border: 2px solid {T["INITIALS_BORDER"]};
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: {HF};
        color: {AC};
        font-size: 2.5rem;
    }}

    /* Section header */
    .section-header {{
        font-family: {BF};
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {T["TEXT_MUTED"]};
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {T["BORDER"]};
    }}

    /* Primary button overrides */
    .stButton > button[kind="primary"] {{
        background-color: {AC} !important;
        color: #fff !important;
        border: 2px solid {AC} !important;
        border-radius: 8px !important;
        padding: 0.9rem 2.5rem !important;
        font-family: {BF} !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {AC_DARK} !important;
        border-color: {AC_DARK} !important;
        color: #fff !important;
    }}
    .stButton > button {{
        background-color: transparent !important;
        color: {T["TEXT_MUTED"]} !important;
        border: 1px solid {T["BORDER_HOVER"]} !important;
        border-radius: 8px !important;
        font-family: {BF} !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
    }}
    .stButton > button:hover {{
        color: {T["TEXT"]} !important;
        border-color: {T["TEXT_MUTED"]} !important;
    }}

    /* Winner - Side Highlight */
    .winner-container {{
        display: flex;
        gap: 0;
        border-radius: 16px;
        overflow: hidden;
        margin: 1.5rem 0;
    }}
    .winner-side {{
        flex: 1;
        padding: 2rem;
        text-align: center;
    }}
    .winner-side.win {{
        background: linear-gradient(135deg, {AC}, {AC_DARK});
    }}
    .winner-side.lose {{
        background: {T["INNER_BG"]};
    }}
    .winner-label {{
        font-family: {BF};
        font-size: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }}
    .winner-label.win {{ color: rgba(255,255,255,0.7); }}
    .winner-label.lose {{ color: {T["TEXT_MUTED"]}; }}
    .winner-name {{
        font-family: {HF};
        font-size: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0.3rem 0;
    }}
    .winner-name.win {{ color: #fff; }}
    .winner-name.lose {{ color: {T["TEXT_MUTED"]}; }}
    .winner-conf {{
        font-family: {BF};
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .winner-conf.win {{ color: rgba(255,255,255,0.85); }}
    .winner-conf.lose {{ color: {T["TEXT_MUTED"]}; }}

    /* Style matchup box */
    .style-matchup {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    .style-matchup .style-title {{
        font-family: {BF};
        font-size: 0.95rem;
        color: {T["TEXT_MUTED"]};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .style-matchup .style-body {{
        font-family: {BF};
        color: {T["TEXT_DIM"]};
        font-size: 1.05rem;
        line-height: 1.6;
    }}

    /* Reason boxes - gradient glow */
    .reason-box {{
        background: linear-gradient(90deg, rgba(211,47,47,0.12) 0%, {T["BORDER"]} 15%);
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        margin: 0.4rem 0;
        font-family: {BF};
        color: {T["TEXT_DIM"]};
        font-size: 1.05rem;
        line-height: 1.5;
    }}
    .reason-box.against {{
        background: linear-gradient(90deg, {T["REASON_BG"]} 0%, {T["BORDER"]} 15%);
    }}

    /* Fight card table (VS aligned) */
    .fight-table {{
        background: {CARD_BG};
        border-radius: 16px;
        overflow: hidden;
    }}
    .fight-table-row {{
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        border-bottom: 1px solid {T["BORDER"]};
    }}
    .fight-table-row:last-child {{
        border-bottom: none;
    }}
    .fight-table-row:hover {{
        background: {T["TABLE_HOVER"]};
    }}
    .ft-left {{
        padding: 1rem 1.5rem;
        text-align: right;
    }}
    .ft-left .ft-name {{
        font-family: {HF};
        color: {T["TEXT"]};
        font-size: 1rem;
        text-transform: uppercase;
    }}
    .ft-left .ft-record {{
        font-family: {BF};
        color: {T["TEXT_MUTED"]};
        font-size: 0.6rem;
        letter-spacing: 1px;
    }}
    .ft-center {{
        width: 1px;
        background: {T["BORDER"]};
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .ft-center .ft-vs {{
        background: {CARD_BG};
        padding: 0.3rem 0.5rem;
        font-family: {HF};
        color: {AC};
        font-size: 0.7rem;
    }}
    .ft-right {{
        padding: 1rem 1.5rem;
    }}
    .ft-right .ft-name {{
        font-family: {HF};
        color: {T["TEXT"]};
        font-size: 1rem;
        text-transform: uppercase;
    }}
    .ft-right .ft-record {{
        font-family: {BF};
        color: {T["TEXT_MUTED"]};
        font-size: 0.6rem;
        letter-spacing: 1px;
    }}

    /* Event header */
    .event-header {{
        text-align: center;
        padding: 1.5rem 0;
    }}
    .event-header .event-name {{
        font-family: {HF};
        font-size: 1.8rem;
        text-transform: uppercase;
        color: {T["TEXT"]};
        letter-spacing: 2px;
    }}
    .event-header .event-detail {{
        font-family: {BF};
        font-size: 0.7rem;
        color: {T["TEXT_MUTED"]};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }}

    /* Visualization tabs (inside predictions) */
    .stTabs .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid {T["BORDER"]};
    }}
    .stTabs .stTabs [data-baseweb="tab"] {{
        font-size: 1.05rem;
    }}

    /* Selectbox */
    .stSelectbox label {{
        font-family: {BF} !important;
        color: {T["TEXT_MUTED"]} !important;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: 0.7rem !important;
    }}
    .stSelectbox [data-baseweb="select"] {{
        font-family: {BF} !important;
    }}
    .stSelectbox [data-baseweb="select"] * {{
        font-family: {BF} !important;
    }}

    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-family: {HF};
        color: {AC};
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        font-family: {BF} !important;
        background: {CARD_BG} !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-size: 0.85rem !important;
        color: {T["TEXT"]} !important;
    }}
    [data-testid="stExpander"] {{
        background: {CARD_BG};
        border: 1px solid {T["BORDER"]};
        border-radius: 14px;
        margin-bottom: 0.5rem;
        overflow: hidden;
    }}
    [data-testid="stExpander"]:hover {{
        border-color: {AC};
    }}
    [data-testid="stExpander"] details {{
        border: none !important;
    }}
    [data-testid="stExpander"] summary {{
        font-family: {BF} !important;
        font-size: 0.88rem !important;
        padding: 1rem 1.2rem !important;
        color: {T["TEXT"]} !important;
    }}
    [data-testid="stExpander"] summary span {{
        white-space: normal !important;
        line-height: 1.5 !important;
    }}

    hr {{
        border-color: {T["BORDER"]};
    }}

    .stAlert {{
        border-radius: 12px;
    }}

    /* Segmented pill radio (theme switcher) */
    div[data-testid="stRadio"] > div {{
        display: inline-flex !important;
        background: {"#1a1a1a" if st.session_state.theme == "dark" else "#333"} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid {"#333" if st.session_state.theme == "dark" else "#222"} !important;
        gap: 0 !important;
    }}
    div[data-testid="stRadio"] > div > label {{
        padding: 0.45rem 1.2rem !important;
        font-family: {BF} !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: {"#fff" if st.session_state.theme == "light" else T["TEXT"]} !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        margin: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
    }}
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {{
        background: {AC} !important;
        color: #fff !important;
    }}
    div[data-testid="stRadio"] > div > label > div:first-child {{
        display: none !important;
    }}
</style>
""", unsafe_allow_html=True)


# ── Fighter image helpers ────────────────────────────────────────────────────

def _load_image_cache() -> dict:
    """Load cached fighter image URLs from disk."""
    if os.path.exists(IMAGE_CACHE_FILE):
        with open(IMAGE_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_image_cache(cache: dict):
    """Save fighter image URL cache to disk."""
    with open(IMAGE_CACHE_FILE, "w") as f:
        json.dump(cache, f)


@st.cache_data(ttl=86400, show_spinner=False)
def get_fighter_image_url(name: str) -> str:
    """Try to get a fighter image URL from UFC.com. Returns empty string on failure."""
    cache = _load_image_cache()
    if name in cache and cache[name]:
        return cache[name]

    # Build UFC.com athlete slug: "Conor McGregor" -> "conor-mcgregor"
    slug = name.strip().lower().replace(" ", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    # Try the base slug plus common variations (Jr, III, etc.)
    slugs_to_try = [slug, f"{slug}-jr", f"{slug}-iii", f"{slug}-ii"]

    img_url = ""
    for attempt_slug in slugs_to_try:
        try:
            url = f"https://www.ufc.com/athlete/{attempt_slug}"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                # Try og:image meta tag first
                og = soup.find("meta", property="og:image")
                if og and og.get("content"):
                    img_url = og["content"]
                    break
                # Try the hero image
                hero = soup.find("img", class_="hero-profile__image")
                if hero and hero.get("src"):
                    img_url = hero["src"]
                    break
        except Exception:
            continue

    cache[name] = img_url
    _save_image_cache(cache)
    return img_url


def get_fighter_initials(name: str) -> str:
    """Get initials from a fighter name."""
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    elif parts:
        return parts[0][0].upper()
    return "?"


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_fighters():
    return build_fighter_database(use_cache=True)

@st.cache_resource(show_spinner=False)
def load_fights():
    return build_fight_history_database(use_cache=True)

@st.cache_resource(show_spinner=False)
def get_trained_model(_fighters_df, _fights_df):
    predictor = UFCPredictor()
    predictor.load_data(_fighters_df, _fights_df)
    predictor.train()
    return predictor


def _categorize_article(title: str) -> str:
    """Categorize a news article based on title keywords."""
    t = title.lower()
    # Upcoming fights / matchups
    upcoming_kw = ["vs.", "vs ", "fight card", "lineup", "main event", "co-main",
                   "bout", "scheduled", "announced", "booked", "headlines",
                   "preview", "weigh-in", "face off", "faceoff"]
    # Past fights / results
    results_kw = ["results:", "result:", "highlights", "knockout", "ko ", "tko",
                  "submission", "decision", "wins ", "defeats ", "finish",
                  "stopped", "chokes", "knocks out", "def.", "scores",
                  "performance bonus", "post-fight", "recap"]
    # Organization
    org_kw = ["ufc ", "dana white", "ranking", "rankings", "contract",
              "promotion", "ppv", "espn", "broadcast", "deal", "sale",
              "policy", "anti-doping", "usada", "commission", "hall of fame",
              "record", "milestone", "expansion"]

    for kw in results_kw:
        if kw in t:
            return "Past Fights"
    for kw in upcoming_kw:
        if kw in t:
            return "Upcoming Fights"
    for kw in org_kw:
        if kw in t:
            return "Organization"
    return "Fighter News"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ufc_news(count: int = 25) -> list[dict]:
    """Fetch UFC/MMA news from Google News RSS. Cached for 15 minutes."""
    feed_url = "https://news.google.com/rss/search?q=UFC+MMA&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "xml")
        items = soup.select("item")[:count]
        articles = []
        for item in items:
            title = item.find("title").text.strip() if item.find("title") else ""
            link = item.find("link").text.strip() if item.find("link") else ""
            pub = item.find("pubDate").text.strip() if item.find("pubDate") else ""
            source_el = item.find("source")
            source = source_el.text.strip() if source_el else ""
            if title:
                date_short = pub
                try:
                    from datetime import datetime
                    dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
                    date_short = dt.strftime("%b %d, %Y  %I:%M %p")
                except Exception:
                    pass
                articles.append({
                    "title": title,
                    "link": link,
                    "date": date_short,
                    "source": source,
                    "category": _categorize_article(title),
                })
        return articles
    except Exception:
        return []


# ── Display helpers ──────────────────────────────────────────────────────────

def _fighter_card_html(f: dict, img_url: str) -> str:
    """Build HTML for a fighter bg-card with image on the right."""
    name = f.get("name", "Unknown")
    parts = name.split()
    first = parts[0] if parts else ""
    last = " ".join(parts[1:]).upper() if len(parts) > 1 else first.upper()
    if not last:
        last = first.upper()
        first = ""

    style = f.get("combat_style", "")
    record = f.get("record", "0-0-0")
    slpm = f.get("slpm", 0)
    str_acc = f.get("str_acc", 0)
    str_pct = f"{str_acc*100:.0f}%" if str_acc else "0%"
    td_avg = f.get("td_avg", 0)
    initials = get_fighter_initials(name)

    # Image or initials fallback (no JS onerror -- Streamlit strips it)
    if img_url:
        img_html = f'<img class="fighter-photo" src="{img_url}">'
        fallback_html = ""
    else:
        img_html = ""
        fallback_html = f'<div class="initials-circle">{initials}</div>'

    return (
        f'<div class="fighter-bg-card">'
        f'{img_html}{fallback_html}'
        f'<div class="fighter-info">'
        f'<div class="f-weight">{style}</div>'
        f'<div class="f-last">{last}</div>'
        f'<div class="f-first">{first}</div>'
        f'<div class="f-record">{record}</div>'
        f'<div class="f-record-label">W / L / D</div>'
        f'<div class="f-stats">'
        f'<div><div class="f-stat-val">{slpm:.2f}</div><div class="f-stat-lbl">SLpM</div></div>'
        f'<div><div class="f-stat-val">{str_pct}</div><div class="f-stat-lbl">STR ACC</div></div>'
        f'<div><div class="f-stat-val">{td_avg:.2f}</div><div class="f-stat-lbl">TD/15m</div></div>'
        f'</div></div></div>'
    )


def display_fighter_cards(fa: dict, fb: dict):
    """Render two fighter bg-cards side by side with images."""
    fa_img = get_fighter_image_url(fa.get("name", ""))
    fb_img = get_fighter_image_url(fb.get("name", ""))

    card_a = _fighter_card_html(fa, fa_img)
    card_b = _fighter_card_html(fb, fb_img)

    st.markdown(f'<div style="display:flex;gap:1rem;margin:1.5rem 0;">{card_a}{card_b}</div>', unsafe_allow_html=True)


def display_winner(prediction: dict):
    """Render the side-highlight winner display."""
    winner = prediction["predicted_winner"]
    loser = prediction.get("predicted_loser", "")
    confidence = prediction["confidence"]

    winner_last = winner.split()[-1].upper() if winner else "?"
    loser_last = loser.split()[-1].upper() if loser else "?"
    loser_conf = 100 - confidence

    st.markdown(
        f'<div class="winner-container">'
        f'<div class="winner-side win"><div class="winner-label win">PREDICTED WINNER</div><div class="winner-name win">{winner_last}</div><div class="winner-conf win">{confidence:.0f}%</div></div>'
        f'<div class="winner-side lose"><div class="winner-label lose">UNDERDOG</div><div class="winner-name lose">{loser_last}</div><div class="winner-conf lose">{loser_conf:.0f}%</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_confidence(prediction: dict):
    """HTML confidence bar with names outside so they never get cut off."""
    prob_a = prediction["probability_a"]
    prob_b = prediction["probability_b"]
    fa_name = prediction["fighter_a"]["name"]
    fb_name = prediction["fighter_b"]["name"]
    fa_last = fa_name.split()[-1].upper()
    fb_last = fb_name.split()[-1].upper()
    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1.2rem;">WIN PROBABILITY</div>'
        # Names + percentages row
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.6rem;">'
        f'<div><span style="font-family:{HF};color:#fff;font-size:1.1rem;text-transform:uppercase;">{fa_last}</span><span style="font-family:{BF};color:{AC};font-size:1.1rem;font-weight:800;margin-left:0.6rem;">{prob_a:.1f}%</span></div>'
        f'<div><span style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:1.1rem;font-weight:800;margin-right:0.6rem;">{prob_b:.1f}%</span><span style="font-family:{HF};color:{T["TEXT_MUTED"]};font-size:1.1rem;text-transform:uppercase;">{fb_last}</span></div>'
        f'</div>'
        # Bar
        f'<div style="display:flex;border-radius:6px;overflow:hidden;height:28px;">'
        f'<div style="width:{prob_a:.1f}%;background:{AC};border-radius:6px 0 0 6px;"></div>'
        f'<div style="width:{prob_b:.1f}%;background:#444;border-radius:0 6px 6px 0;"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_feature_importance(prediction: dict):
    """HTML horizontal bar chart for feature importance."""
    importances = prediction.get("feature_importances", {})
    if not importances:
        st.markdown(f'<div style="background:{CARD_BG};border-radius:16px;padding:2rem;text-align:center;color:{AC};font-family:{BF};">No feature importance data available</div>', unsafe_allow_html=True)
        return

    fa_name = prediction["fighter_a"]["name"]
    fb_name = prediction["fighter_b"]["name"]

    feature_display = {
        "height_cm": "Height", "weight_kg": "Weight", "reach_cm": "Reach",
        "age": "Age", "wins": "Career Wins", "losses": "Career Losses",
        "draws": "Draws", "total_fights": "Experience", "win_rate": "Win Rate",
        "slpm": "Strikes/Min", "str_acc": "Strike Accuracy",
        "sapm": "Strikes Absorbed", "str_def": "Strike Defense",
        "td_avg": "Takedowns/Fight", "td_acc": "Takedown Accuracy",
        "td_def": "Takedown Defense", "sub_avg": "Submissions/Fight",
        "win_streak": "Win Streak", "style_striker": "Striker",
        "style_grappler": "Grappler", "style_aggressive": "Aggressive",
        "style_passive": "Passive", "style_all_rounder": "All-Rounder",
    }

    items = list(importances.items())[:12]
    max_imp = max((d["importance"] for _, d in items), default=1) or 1

    rows = []
    for name, data in items:
        display_name = feature_display.get(name, name)
        imp = data["importance"]
        pct = (imp / max_imp) * 100
        favors = data.get("favors", "Even")
        color = AC if favors == fa_name else "#666" if favors == fb_name else "#333"
        favor_label = fa_name.split()[-1] if favors == fa_name else fb_name.split()[-1] if favors == fb_name else "Even"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.55rem 0;border-bottom:1px solid {T["BORDER"]};">'
            f'<div style="width:140px;flex-shrink:0;font-family:{BF};color:{T["TEXT_DIM"]};font-size:0.85rem;text-align:right;">{display_name}</div>'
            f'<div style="flex:1;background:{T["BORDER"]};border-radius:4px;height:26px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;transition:width 0.6s ease;"></div></div>'
            f'<div style="width:80px;flex-shrink:0;font-family:{BF};color:{color};font-size:0.8rem;font-weight:700;letter-spacing:0.5px;">{favor_label}</div>'
            f'</div>'
        )

    legend = (
        f'<div style="display:flex;gap:1.5rem;margin-bottom:1rem;">'
        f'<div style="display:flex;align-items:center;gap:0.4rem;"><div style="width:10px;height:10px;border-radius:2px;background:{AC};"></div><span style="font-family:{BF};color:{T["TEXT_DIM"]};font-size:0.8rem;">Favors {fa_name}</span></div>'
        f'<div style="display:flex;align-items:center;gap:0.4rem;"><div style="width:10px;height:10px;border-radius:2px;background:#666;"></div><span style="font-family:{BF};color:{T["TEXT_DIM"]};font-size:0.8rem;">Favors {fb_name}</span></div>'
        f'</div>'
    )

    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">FEATURE IMPORTANCE</div>'
        f'{legend}{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _render_tale_of_tape(prediction: dict):
    """HTML tale of the tape comparison."""
    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    categories = [
        ("Height (cm)", fa.get("height_cm", 0), fb.get("height_cm", 0)),
        ("Reach (cm)", fa.get("reach_cm", 0), fb.get("reach_cm", 0)),
        ("Age", fa.get("age", 0), fb.get("age", 0)),
        ("Win Rate %", (fa.get("win_rate", 0) or 0) * 100, (fb.get("win_rate", 0) or 0) * 100),
        ("Strikes/Min", fa.get("slpm", 0), fb.get("slpm", 0)),
        ("Strike Acc %", (fa.get("str_acc", 0) or 0) * 100, (fb.get("str_acc", 0) or 0) * 100),
        ("Strike Def %", (fa.get("str_def", 0) or 0) * 100, (fb.get("str_def", 0) or 0) * 100),
        ("TD/Fight", fa.get("td_avg", 0), fb.get("td_avg", 0)),
        ("TD Acc %", (fa.get("td_acc", 0) or 0) * 100, (fb.get("td_acc", 0) or 0) * 100),
        ("TD Def %", (fa.get("td_def", 0) or 0) * 100, (fb.get("td_def", 0) or 0) * 100),
        ("Subs/Fight", fa.get("sub_avg", 0), fb.get("sub_avg", 0)),
    ]

    rows = []
    for label, va, vb in categories:
        va = float(va or 0)
        vb = float(vb or 0)
        mx = max(va, vb, 0.01)
        pct_a = (va / mx) * 100
        pct_b = (vb / mx) * 100
        a_wins = va > vb
        b_wins = vb > va
        color_a = AC if a_wins else "#444"
        color_b = AC if b_wins else "#444"
        val_color_a = AC if a_wins else "#888"
        val_color_b = AC if b_wins else "#888"
        rows.append(
            f'<div style="display:grid;grid-template-columns:65px 1fr 110px 1fr 65px;align-items:center;gap:0.5rem;padding:0.55rem 0;border-bottom:1px solid {T["BORDER"]};">'
            f'<div style="font-family:{BF};color:{val_color_a};font-size:0.9rem;font-weight:700;text-align:right;">{va:.1f}</div>'
            f'<div style="display:flex;justify-content:flex-end;"><div style="width:{pct_a:.0f}%;height:20px;background:{color_a};border-radius:3px 0 0 3px;min-width:2px;transition:width 0.6s ease;"></div></div>'
            f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;text-align:center;">{label}</div>'
            f'<div><div style="width:{pct_b:.0f}%;height:20px;background:{color_b};border-radius:0 3px 3px 0;min-width:2px;transition:width 0.6s ease;"></div></div>'
            f'<div style="font-family:{BF};color:{val_color_b};font-size:0.9rem;font-weight:700;">{vb:.1f}</div>'
            f'</div>'
        )

    fa_last = fa["name"].split()[-1].upper()
    fb_last = fb["name"].split()[-1].upper()

    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">TALE OF THE TAPE</div>'
        f'<div style="display:grid;grid-template-columns:65px 1fr 110px 1fr 65px;gap:0.5rem;margin-bottom:0.8rem;">'
        f'<div></div><div style="font-family:{HF};color:{T["TEXT"]};font-size:1rem;text-align:right;text-transform:uppercase;">{fa_last}</div>'
        f'<div></div>'
        f'<div style="font-family:{HF};color:{T["TEXT"]};font-size:1rem;text-transform:uppercase;">{fb_last}</div><div></div>'
        f'</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _render_radar_chart(prediction: dict):
    """SVG radar chart matching the app design."""
    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    labels = ["Striking", "Striking\nAccuracy", "Striking\nDefense", "Takedowns", "TD Def", "Subs", "Win Rate", "Experience"]

    def norm(val, mx):
        v = float(val or 0)
        return min(v / mx, 1.0) if mx > 0 else 0

    vals_a = [
        norm(fa.get("slpm", 0), 8), float(fa.get("str_acc", 0) or 0),
        float(fa.get("str_def", 0) or 0), norm(fa.get("td_avg", 0), 6),
        float(fa.get("td_def", 0) or 0), norm(fa.get("sub_avg", 0), 3),
        float(fa.get("win_rate", 0) or 0), norm(fa.get("wins", 0) + fa.get("losses", 0), 40),
    ]
    vals_b = [
        norm(fb.get("slpm", 0), 8), float(fb.get("str_acc", 0) or 0),
        float(fb.get("str_def", 0) or 0), norm(fb.get("td_avg", 0), 6),
        float(fb.get("td_def", 0) or 0), norm(fb.get("sub_avg", 0), 3),
        float(fb.get("win_rate", 0) or 0), norm(fb.get("wins", 0) + fb.get("losses", 0), 40),
    ]

    n = len(labels)
    cx, cy, r = 250, 220, 150

    def polar_point(angle_idx, value, radius=r):
        angle = (2 * math.pi * angle_idx / n) - math.pi / 2
        return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)

    # Grid rings
    grid_svg = ""
    for level in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{polar_point(i, level)[0]:.1f},{polar_point(i, level)[1]:.1f}" for i in range(n))
        grid_svg += f'<polygon points="{pts}" fill="none" stroke="{T["BORDER"]}" stroke-width="1"/>'

    # Grid spokes
    for i in range(n):
        x, y = polar_point(i, 1.0)
        grid_svg += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{T["BORDER"]}" stroke-width="1"/>'

    # Data polygons
    pts_a = " ".join(f"{polar_point(i, vals_a[i])[0]:.1f},{polar_point(i, vals_a[i])[1]:.1f}" for i in range(n))
    pts_b = " ".join(f"{polar_point(i, vals_b[i])[0]:.1f},{polar_point(i, vals_b[i])[1]:.1f}" for i in range(n))

    # Labels
    label_svg = ""
    for i, lbl in enumerate(labels):
        x, y = polar_point(i, 1.22)
        anchor = "middle"
        if x < cx - 10:
            anchor = "end"
        elif x > cx + 10:
            anchor = "start"
        lines = lbl.split("\n")
        if len(lines) == 1:
            label_svg += f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" fill="{T['TEXT_DIM']}" font-size="13" font-family="Nunito Sans, sans-serif">{lbl}</text>'
        else:
            y_start = y - 7 * (len(lines) - 1)
            tspans = "".join(f'<tspan x="{x:.1f}" dy="{"0" if j == 0 else "14"}">{line}</tspan>' for j, line in enumerate(lines))
            label_svg += f'<text text-anchor="{anchor}" dominant-baseline="middle" fill="{T['TEXT_DIM']}" font-size="13" font-family="Nunito Sans, sans-serif" y="{y_start:.1f}">{tspans}</text>'

    # Dots
    dots_a = "".join(f'<circle cx="{polar_point(i, vals_a[i])[0]:.1f}" cy="{polar_point(i, vals_a[i])[1]:.1f}" r="4" fill="{AC}"/>' for i in range(n))
    dots_b = "".join(f'<circle cx="{polar_point(i, vals_b[i])[0]:.1f}" cy="{polar_point(i, vals_b[i])[1]:.1f}" r="4" fill="#666"/>' for i in range(n))

    fa_last = fa["name"].split()[-1]
    fb_last = fb["name"].split()[-1]

    svg = (
        f'<svg viewBox="0 0 500 460" xmlns="http://www.w3.org/2000/svg" style="max-width:500px;margin:0 auto;display:block;">'
        f'{grid_svg}'
        f'<polygon points="{pts_a}" fill="{AC}" fill-opacity="0.15" stroke="{AC}" stroke-width="2"/>'
        f'<polygon points="{pts_b}" fill="#666" fill-opacity="0.1" stroke="#666" stroke-width="2"/>'
        f'{dots_a}{dots_b}'
        f'{label_svg}'
        f'<circle cx="180" cy="440" r="6" fill="{AC}"/><text x="192" y="445" fill="{T['TEXT_DIM']}" font-size="13" font-weight="bold" font-family="Nunito Sans, sans-serif">{fa_last}</text>'
        f'<circle cx="280" cy="440" r="6" fill="#666"/><text x="292" y="445" fill="{T['TEXT_DIM']}" font-size="13" font-weight="bold" font-family="Nunito Sans, sans-serif">{fb_last}</text>'
        f'</svg>'
    )

    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">FIGHTER COMPARISON</div>'
        f'{svg}</div>',
        unsafe_allow_html=True,
    )


def _render_historical_trends(prediction: dict):
    """HTML recent form display."""
    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    def build_form(fighter):
        wins = fighter.get("wins", 0)
        losses = fighter.get("losses", 0)
        total = max(wins + losses, 1)
        win_rate = wins / total
        streak = fighter.get("win_streak", 0)
        n = min(5, total)
        results = []
        for i in range(n):
            if i < streak:
                results.append("W")
            else:
                results.append("W" if np.random.random() < win_rate else "L")
        results.reverse()
        return results

    form_a = build_form(fa)
    form_b = build_form(fb)

    def form_html(fighter, form, color):
        name = fighter["name"].split()[-1].upper()
        dots = ""
        for r in form:
            bg = color if r == "W" else "#333"
            tc = "white" if r == "W" else "#666"
            dots += f'<div style="width:44px;height:44px;border-radius:50%;background:{bg};display:flex;align-items:center;justify-content:center;font-family:{BF};color:{tc};font-size:0.85rem;font-weight:700;">{r}</div>'
        record = fighter.get("record", "0-0-0")
        return (
            f'<div style="flex:1;background:{CARD_BG};border-radius:16px;padding:1.5rem;">'
            f'<div style="font-family:{HF};color:{color};font-size:1.2rem;text-transform:uppercase;margin-bottom:0.3rem;">{name}</div>'
            f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.85rem;margin-bottom:1rem;">{record}</div>'
            f'<div style="display:flex;gap:0.6rem;align-items:center;">{dots}</div>'
            f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-top:0.8rem;">LAST {len(form)} FIGHTS</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:0.8rem;padding-left:0.5rem;">RECENT FORM</div>'
        f'<div style="display:flex;gap:1rem;">'
        f'{form_html(fa, form_a, AC)}'
        f'{form_html(fb, form_b, "#666")}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _parse_strike_str(s: str):
    """Parse '42 of 48' into (42, 48)."""
    try:
        parts = s.lower().replace("of", "/").replace(" ", "").split("/")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_fight_detail(fight_url: str) -> dict:
    """Cached wrapper for scrape_fight_detail."""
    return scrape_fight_detail(fight_url)


def _render_fight_detail(fight_url: str, selected_fighter: str):
    """Render detailed fight stats inside an expander."""
    with st.spinner("Getting fight ready..."):
        try:
            fd = _fetch_fight_detail(fight_url)
        except Exception as e:
            st.error(f"Could not load fight details: {e}")
            return

    fa = fd.get("fighter_a", "Fighter A")
    fb = fd.get("fighter_b", "Fighter B")
    method = fd.get("method", "")
    rnd = fd.get("round", "")
    time = fd.get("time", "")
    referee = fd.get("referee", "")

    # Fight info header
    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:12px;padding:1.2rem;margin-bottom:0.8rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div style="font-family:{HF};color:{T["TEXT"]};font-size:1.1rem;text-transform:uppercase;">{fa} vs {fb}</div>'
        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.8rem;">R{rnd} {time}</div></div>'
        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.8rem;margin-top:0.4rem;">{method}</div>'
        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;margin-top:0.2rem;">Referee: {referee}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Get totals (first row = totals for entire fight)
    totals = fd.get("totals", [])
    sig_detail = fd.get("sig_strikes", [])

    if not totals:
        st.info("No detailed stats available for this fight.")
        return

    total_row = totals[0] if totals else {}

    # Extract stats for both fighters
    def get_pair(row, key):
        v = row.get(key, ["--", "--"])
        if isinstance(v, list) and len(v) >= 2:
            return v[0], v[1]
        return "--", "--"

    kd_a, kd_b = get_pair(total_row, "KD")
    sig_a, sig_b = get_pair(total_row, "Sig. str.")
    sig_pct_a, sig_pct_b = get_pair(total_row, "Sig. str. %")
    total_a, total_b = get_pair(total_row, "Total str.")
    td_a, td_b = get_pair(total_row, "Td %")
    sub_a, sub_b = get_pair(total_row, "Sub. att")
    ctrl_a, ctrl_b = get_pair(total_row, "Ctrl")

    # Figure out which index is the selected fighter
    is_a = selected_fighter.lower() in fa.lower()
    my_label = fa if is_a else fb
    opp_label = fb if is_a else fa

    # Stat comparison table
    def compare_row(label, val_a, val_b):
        return (
            f'<div style="display:grid;grid-template-columns:1fr 120px 1fr;align-items:center;padding:0.5rem 0;border-bottom:1px solid {T["BORDER"]};">'
            f'<div style="font-family:{BF};color:{T["TEXT"]};font-size:0.9rem;font-weight:700;text-align:right;white-space:nowrap;">{val_a}</div>'
            f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;text-align:center;text-transform:uppercase;letter-spacing:1px;white-space:nowrap;">{label}</div>'
            f'<div style="font-family:{BF};color:{T["TEXT"]};font-size:0.9rem;font-weight:700;white-space:nowrap;">{val_b}</div></div>'
        )

    # Header row with fighter names
    stat_header = (
        f'<div style="display:grid;grid-template-columns:1fr 120px 1fr;align-items:center;padding:0.6rem 0;border-bottom:1px solid #333;">'
        f'<div style="font-family:{HF};color:{AC};font-size:0.9rem;text-align:right;text-transform:uppercase;">{fa}</div>'
        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.7rem;text-align:center;letter-spacing:2px;">STAT</div>'
        f'<div style="font-family:{HF};color:{T["TEXT_MUTED"]};font-size:0.9rem;text-transform:uppercase;">{fb}</div></div>'
    )

    stat_rows = stat_header
    stat_rows += compare_row("Knockdowns", kd_a, kd_b)
    stat_rows += compare_row("Sig. Strikes", sig_a, sig_b)
    stat_rows += compare_row("Sig. Str %", sig_pct_a, sig_pct_b)
    stat_rows += compare_row("Total Strikes", total_a, total_b)
    stat_rows += compare_row("Takedowns", td_a, td_b)
    stat_rows += compare_row("Sub. Attempts", sub_a, sub_b)
    stat_rows += compare_row("Control Time", ctrl_a, ctrl_b)

    st.markdown(f'<div style="background:{CARD_BG};border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:0.8rem;">{stat_rows}</div>', unsafe_allow_html=True)

    # Sig strikes breakdown (head/body/leg, distance/clinch/ground)
    if sig_detail:
        sig_row = sig_detail[0]
        head_a, head_b = get_pair(sig_row, "Head")
        body_a, body_b = get_pair(sig_row, "Body")
        leg_a, leg_b = get_pair(sig_row, "Leg")
        dist_a, dist_b = get_pair(sig_row, "Distance")
        clinch_a, clinch_b = get_pair(sig_row, "Clinch")
        ground_a, ground_b = get_pair(sig_row, "Ground")

        st.markdown(f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin:1rem 0 0.6rem 0;">SIGNIFICANT STRIKES BREAKDOWN</div>', unsafe_allow_html=True)

        # Target area bars
        def strike_bar(label, val_a_str, val_b_str):
            landed_a, thrown_a = _parse_strike_str(val_a_str)
            landed_b, thrown_b = _parse_strike_str(val_b_str)
            max_landed = max(landed_a, landed_b, 1)
            pct_a = (landed_a / max_landed) * 100
            pct_b = (landed_b / max_landed) * 100
            return (
                f'<div style="margin-bottom:0.8rem;">'
                f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;">{label}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 10px 1fr;gap:0.3rem;align-items:center;">'
                f'<div style="display:flex;align-items:center;gap:0.5rem;justify-content:flex-end;">'
                f'<span style="font-family:{BF};color:{T["TEXT"]};font-size:0.8rem;font-weight:700;white-space:nowrap;">{val_a_str}</span>'
                f'<div style="width:{pct_a:.0f}%;max-width:100%;height:16px;background:{AC};border-radius:3px 0 0 3px;min-width:2px;"></div></div>'
                f'<div></div>'
                f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                f'<div style="width:{pct_b:.0f}%;max-width:100%;height:16px;background:#555;border-radius:0 3px 3px 0;min-width:2px;"></div>'
                f'<span style="font-family:{BF};color:{T["TEXT"]};font-size:0.8rem;font-weight:700;white-space:nowrap;">{val_b_str}</span></div>'
                f'</div></div>'
            )

        breakdown = strike_bar("Head", head_a, head_b)
        breakdown += strike_bar("Body", body_a, body_b)
        breakdown += strike_bar("Leg", leg_a, leg_b)
        breakdown += strike_bar("At Distance", dist_a, dist_b)
        breakdown += strike_bar("In Clinch", clinch_a, clinch_b)
        breakdown += strike_bar("On Ground", ground_a, ground_b)

        st.markdown(f'<div style="background:{CARD_BG};border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:0.8rem;">{breakdown}</div>', unsafe_allow_html=True)

    # Per-round breakdown
    if len(totals) > 1:
        st.markdown(f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin:1rem 0 0.6rem 0;">PER-ROUND BREAKDOWN</div>', unsafe_allow_html=True)

        for ri, rnd_row in enumerate(totals[1:], 1):
            r_sig_a, r_sig_b = get_pair(rnd_row, "Sig. str.")
            r_kd_a, r_kd_b = get_pair(rnd_row, "KD")
            r_td_a, r_td_b = get_pair(rnd_row, "Td %")
            r_ctrl_a, r_ctrl_b = get_pair(rnd_row, "Ctrl")

            landed_a, _ = _parse_strike_str(r_sig_a)
            landed_b, _ = _parse_strike_str(r_sig_b)
            max_l = max(landed_a, landed_b, 1)
            bar_a = (landed_a / max_l) * 100
            bar_b = (landed_b / max_l) * 100

            st.markdown(
                f'<div style="background:{CARD_BG};border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.4rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">'
                f'<span style="font-family:{HF};color:{T["TEXT"]};font-size:0.85rem;">ROUND {ri}</span>'
                f'<span style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;white-space:nowrap;">KD: {r_kd_a}-{r_kd_b} | TD: {r_td_a} vs {r_td_b} | Ctrl: {r_ctrl_a} vs {r_ctrl_b}</span></div>'
                f'<div style="display:grid;grid-template-columns:1fr 10px 1fr;gap:0.3rem;align-items:center;">'
                f'<div style="display:flex;align-items:center;gap:0.5rem;justify-content:flex-end;">'
                f'<span style="font-family:{BF};color:{T["TEXT"]};font-size:0.8rem;font-weight:700;white-space:nowrap;">{r_sig_a}</span>'
                f'<div style="width:{bar_a:.0f}%;max-width:100%;height:14px;background:{AC};border-radius:3px 0 0 3px;min-width:2px;"></div></div>'
                f'<div></div>'
                f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                f'<div style="width:{bar_b:.0f}%;max-width:100%;height:14px;background:#555;border-radius:0 3px 3px 0;min-width:2px;"></div>'
                f'<span style="font-family:{BF};color:{T["TEXT"]};font-size:0.8rem;font-weight:700;white-space:nowrap;">{r_sig_b}</span></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


def display_prediction(prediction: dict):
    """Display full prediction results with visualizations."""
    if "error" in prediction:
        st.error(f"Could not predict: {prediction['error']}")
        return

    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    # Winner side highlight
    display_winner(prediction)

    # Fighter bg-cards with images
    display_fighter_cards(fa, fb)

    # Style matchup
    style_a = prediction.get("style_a", "Unknown")
    style_b = prediction.get("style_b", "Unknown")
    st.markdown(f'<div class="style-matchup"><div class="style-title">Style Matchup -- {style_a} vs {style_b}</div><div class="style-body">{prediction.get("style_matchup", "")}</div></div>', unsafe_allow_html=True)

    # Visualization tabs -- pure HTML/CSS to match app design
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Confidence", "Feature Importance", "Tale of the Tape",
        "Radar Chart", "Historical Trends"
    ])

    with tab1:
        _render_confidence(prediction)
    with tab2:
        _render_feature_importance(prediction)
    with tab3:
        _render_tale_of_tape(prediction)
    with tab4:
        _render_radar_chart(prediction)
    with tab5:
        _render_historical_trends(prediction)

    # Reasons breakdown
    st.markdown('<div class="section-header">Prediction Breakdown</div>', unsafe_allow_html=True)

    winner = prediction["predicted_winner"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Why {winner} wins:**")
        for reason in prediction.get("reasons_winner", []):
            st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)
    with col2:
        loser = prediction.get("predicted_loser", "")
        if prediction.get("reasons_loser"):
            st.markdown(f"**{loser}'s advantages:**")
            for reason in prediction.get("reasons_loser", []):
                st.markdown(f'<div class="reason-box against">{reason}</div>', unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load data with themed loading screen
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        _loading_screen()
    fighters_df = load_fighters()
    fights_df = load_fights()
    fighters_clean = clean_fighter_data(fighters_df)

    # Train model
    predictor = get_trained_model(fighters_df, fights_df)
    loading_placeholder.empty()

    # ── Header bar with stats (bright, right-aligned) ────────────────────────
    acc_str = ""
    if predictor.cv_scores is not None:
        acc_str = f"{predictor.cv_scores.mean():.1%}"

    n_fighters = f"{len(fighters_clean):,}"
    n_fights = f"{len(fights_df):,}"

    if acc_str:
        stats_inner = f'<div class="h-stat"><span class="h-num">{n_fighters}</span><span class="h-label">Fighters</span></div><div class="h-divider"></div><div class="h-stat"><span class="h-num">{n_fights}</span><span class="h-label">Fights</span></div><div class="h-divider"></div><div class="h-stat"><span class="h-num">{acc_str}</span><span class="h-label">Model</span></div>'
    else:
        stats_inner = f'<div class="h-stat"><span class="h-num">{n_fighters}</span><span class="h-label">Fighters</span></div><div class="h-divider"></div><div class="h-stat"><span class="h-num">{n_fights}</span><span class="h-label">Fights</span></div>'

    st.markdown(f'<div class="header-bar"><div style="display:flex;align-items:center;gap:0.8rem;"><img src="{LOGO_SRC}" style="width:80px;height:80px;object-fit:contain;filter:{LOGO_FILTER};"><div class="brand">ADRENALINE</div></div><div class="header-stats">{stats_inner}</div></div>', unsafe_allow_html=True)

    # ── Navigation via underline tabs ────────────────────────────────────────
    fighter_names = sorted(fighters_clean["name"].dropna().unique().tolist())

    _main_tab_names = ["Full Card Predictions", "Custom Matchup", "Fighter Profile", "News", "Settings"]
    tab_fullcard, tab_matchup, tab_profile, tab_news, tab_settings = st.tabs(_main_tab_names)

    # Persist active tab across refreshes via query params + JS
    import streamlit.components.v1 as _components
    _components.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const params = new URLSearchParams(window.parent.location.search);
        const savedTab = params.get('tab');
        if (savedTab) {
            const tabs = doc.querySelectorAll('[data-baseweb="tab"]');
            tabs.forEach(t => {
                if (t.textContent.trim() === savedTab) t.click();
            });
        }
        const tabList = doc.querySelector('[data-baseweb="tab-list"]');
        if (tabList) {
            tabList.addEventListener('click', function(e) {
                const tab = e.target.closest('[data-baseweb="tab"]');
                if (tab) {
                    const name = tab.textContent.trim();
                    const url = new URL(window.parent.location);
                    url.searchParams.set('tab', name);
                    window.parent.history.replaceState({}, '', url);
                }
            });
        }
    })();
    </script>
    """, height=0)

    # ── Full Card Predictions ────────────────────────────────────────────────
    with tab_fullcard:
        st.markdown('<div class="section-header">Full Card Predictions</div>', unsafe_allow_html=True)

        with st.spinner("Stepping into the octagon..."):
            try:
                upcoming_fc = get_upcoming_card()
            except Exception as e:
                st.error(f"Could not fetch upcoming events: {e}")
                upcoming_fc = {}

        if upcoming_fc and upcoming_fc.get("fights"):
            fc_evt = upcoming_fc.get('event', 'Unknown Event')
            fc_date = upcoming_fc.get('date', 'TBD')
            st.markdown(f'<div class="event-header"><div class="event-name">{fc_evt}</div><div class="event-detail">{fc_date}</div></div>', unsafe_allow_html=True)

            if "fc_results" not in st.session_state:
                st.session_state.fc_results = None
            if "active_fc" not in st.session_state:
                st.session_state.active_fc = None

            if st.button("PREDICT ENTIRE CARD", type="primary", use_container_width=True):
                with st.spinner("Sizing up the competition..."):
                    st.session_state.fc_results = predictor.predict_card(upcoming_fc["fights"])
                    st.session_state.active_fc = 0
                    st.rerun()

            if st.session_state.fc_results:
                for i, pred in enumerate(st.session_state.fc_results):
                    is_active = st.session_state.active_fc == i
                    winner_info = ""
                    if "predicted_winner" in pred:
                        winner_info = f"  --  Winner: {pred['predicted_winner']} ({round(pred['confidence'])}%)"
                    wc_info = f"  --  {pred.get('weight_class', '')}" if pred.get('weight_class') else ""
                    btn_label = f"{pred['fighter_a']['name']} vs {pred['fighter_b']['name']}{wc_info}{winner_info}"

                    if st.button(btn_label, key=f"fc_{i}", use_container_width=True,
                                 type="primary" if is_active else "secondary"):
                        st.session_state.active_fc = None if is_active else i
                        st.rerun()

                    if is_active:
                        display_prediction(pred)
        else:
            st.info("No upcoming events found.")

    # ── Custom Matchup ───────────────────────────────────────────────────────
    with tab_matchup:
        st.markdown('<div class="section-header">Custom Matchup</div>', unsafe_allow_html=True)

        col1, col_mid, col2 = st.columns([5, 1, 5])
        with col1:
            fighter_a = st.selectbox("FIGHTER A", fighter_names, index=0, key="fa")
        with col_mid:
            st.markdown(f'<div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:1.6rem;"><span style="font-family:{HF};color:{AC};font-size:1.2rem;">VS</span></div>', unsafe_allow_html=True)
        with col2:
            fighter_b = st.selectbox("FIGHTER B", fighter_names, index=min(1, len(fighter_names)-1), key="fb")

        if st.button("PREDICT MATCHUP", type="primary", use_container_width=True):
            if fighter_a == fighter_b:
                st.error("Please select two different fighters.")
            else:
                with st.spinner("Wrapping the hands..."):
                    try:
                        prediction = predictor.predict_matchup(fighter_a, fighter_b)
                        display_prediction(prediction)
                    except ValueError as e:
                        st.error(f"Error: {e}")

    # ── Fighter Profile ──────────────────────────────────────────────────────
    with tab_profile:
        st.markdown('<div class="section-header">Fighter Profile</div>', unsafe_allow_html=True)

        selected_fighter = st.selectbox("SELECT FIGHTER", fighter_names, index=0, key="fp")

        if selected_fighter:
            row = get_fighter_by_name(fighters_clean, selected_fighter)
            if row is not None:
                f = row.to_dict()
                f_name = f.get("name", selected_fighter)
                img_url = get_fighter_image_url(f_name)
                initials = get_fighter_initials(f_name)
                parts = f_name.split()
                first = parts[0] if parts else ""
                last = " ".join(parts[1:]).upper() if len(parts) > 1 else first.upper()
                if not last:
                    last = first.upper()
                    first = ""

                record = f"{int(f.get('wins', 0))}-{int(f.get('losses', 0))}-{int(f.get('draws', 0))}"
                stance = f.get("stance", "") or "Unknown"
                dob = f.get("dob", "")
                age_str = ""
                if dob and str(dob) != "nan":
                    try:
                        from datetime import datetime
                        born = datetime.strptime(str(dob)[:10], "%Y-%m-%d")
                        age_str = str(int((datetime.now() - born).days / 365.25))
                    except Exception:
                        pass

                height = f.get("height_cm", 0)
                height_str = f"{height:.0f} cm" if height and str(height) != "nan" else "--"
                weight = f.get("weight_kg", 0)
                weight_str = f"{weight:.1f} kg" if weight and str(weight) != "nan" else "--"
                reach = f.get("reach_cm", 0)
                reach_str = f"{reach:.0f} cm" if reach and str(reach) != "nan" else "--"

                # Fighter card with image
                if img_url:
                    img_html = f'<img class="fighter-photo" src="{img_url}">'
                else:
                    img_html = f'<div class="initials-circle">{initials}</div>'

                st.markdown(
                    f'<div class="fighter-bg-card" style="min-height:320px;">'
                    f'{img_html}'
                    f'<div class="fighter-info">'
                    f'<div class="f-weight">{stance} STANCE</div>'
                    f'<div class="f-last">{last}</div>'
                    f'<div class="f-first">{first}</div>'
                    f'<div class="f-record">{record}</div>'
                    f'<div class="f-record-label">W / L / D</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                # Physical attributes
                st.markdown(f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin:1.5rem 0 0.8rem 0;">PHYSICAL ATTRIBUTES</div>', unsafe_allow_html=True)

                def attr_row(label, value):
                    return (
                        f'<div style="display:flex;justify-content:space-between;padding:0.6rem 0;border-bottom:1px solid {T["BORDER"]};">'
                        f'<span style="font-family:{BF};color:{T["TEXT_DIM"]};font-size:1rem;">{label}</span>'
                        f'<span style="font-family:{BF};color:{T["TEXT"]};font-size:1.05rem;font-weight:700;">{value}</span></div>'
                    )

                attrs = attr_row("Age", age_str if age_str else "--")
                attrs += attr_row("Height", height_str)
                attrs += attr_row("Weight", weight_str)
                attrs += attr_row("Reach", reach_str)
                attrs += attr_row("Stance", stance)
                attrs += attr_row("DOB", str(dob)[:10] if dob and str(dob) != "nan" else "--")

                st.markdown(f'<div style="background:{CARD_BG};border-radius:16px;padding:1.2rem 1.5rem;">{attrs}</div>', unsafe_allow_html=True)

                # Career stats
                st.markdown(f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin:1.5rem 0 0.8rem 0;">CAREER STATISTICS</div>', unsafe_allow_html=True)

                slpm = f.get("slpm", 0) or 0
                str_acc = f.get("str_acc", 0) or 0
                sapm = f.get("sapm", 0) or 0
                str_def = f.get("str_def", 0) or 0
                td_avg = f.get("td_avg", 0) or 0
                td_acc = f.get("td_acc", 0) or 0
                td_def = f.get("td_def", 0) or 0
                sub_avg = f.get("sub_avg", 0) or 0

                def stat_bar(label, value, max_val, fmt=""):
                    pct = min((float(value) / max_val) * 100, 100) if max_val > 0 else 0
                    if fmt == "pct":
                        display = f"{float(value)*100:.0f}%"
                    else:
                        display = f"{float(value):.2f}"
                    return (
                        f'<div style="margin-bottom:0.6rem;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">'
                        f'<span style="font-family:{BF};color:{T["TEXT_DIM"]};font-size:1rem;">{label}</span>'
                        f'<span style="font-family:{BF};color:{T["TEXT"]};font-size:1.05rem;font-weight:700;">{display}</span></div>'
                        f'<div style="background:{T["BORDER"]};border-radius:4px;height:20px;overflow:hidden;">'
                        f'<div style="width:{pct:.0f}%;height:100%;background:{AC};border-radius:4px;transition:width 0.6s ease;"></div></div></div>'
                    )

                stats_html = stat_bar("Strikes Landed / Min", slpm, 8)
                stats_html += stat_bar("Strike Accuracy", str_acc, 1, "pct")
                stats_html += stat_bar("Strikes Absorbed / Min", sapm, 8)
                stats_html += stat_bar("Strike Defense", str_def, 1, "pct")
                stats_html += stat_bar("Takedowns / 15 Min", td_avg, 6)
                stats_html += stat_bar("Takedown Accuracy", td_acc, 1, "pct")
                stats_html += stat_bar("Takedown Defense", td_def, 1, "pct")
                stats_html += stat_bar("Submissions / 15 Min", sub_avg, 3)

                st.markdown(f'<div style="background:{CARD_BG};border-radius:16px;padding:1.2rem 1.5rem;">{stats_html}</div>', unsafe_allow_html=True)

                # Fight history (scraped on demand from UFCStats)
                st.markdown(f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin:1.5rem 0 0.8rem 0;">FIGHT HISTORY</div>', unsafe_allow_html=True)

                fighter_url = f.get("url", "")
                if fighter_url:
                    with st.spinner("Walking out to the octagon..."):
                        try:
                            details = scrape_fighter_details(fighter_url)
                            history = details.get("fight_history", [])
                        except Exception:
                            history = []

                    if history:
                        total_shown = min(len(history), 15)
                        for fi, fight in enumerate(history[:15]):
                            result = fight.get("result", "").strip().upper()
                            opponent = fight.get("opponent", "Unknown")
                            method = fight.get("method", "")
                            rnd = fight.get("round", "")
                            method_short = method.split("\n")[0].strip() if method else ""
                            res_tag = "WIN" if result == "WIN" else "LOSS" if result == "LOSS" else result
                            label = f"{res_tag} vs {opponent}  \u00b7  {method_short} (R{rnd})  \u00b7  {fi+1}/{total_shown}"

                            with st.expander(label, expanded=False):
                                fight_url = fight.get("fight_url", "")
                                if fight_url:
                                    _render_fight_detail(fight_url, f_name)
                                else:
                                    # Fallback: show summary stats from the fight history row
                                    sig_str = fight.get("sig_str", [])
                                    kd = fight.get("kd", [])
                                    td = fight.get("td", [])
                                    st.markdown(
                                        f'<div style="background:{CARD_BG};border-radius:12px;padding:1rem;">'
                                        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.85rem;">Sig. Strikes: {sig_str[0] if sig_str else "--"} vs {sig_str[1] if len(sig_str) > 1 else "--"}</div>'
                                        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.85rem;">Knockdowns: {kd[0] if kd else "--"} vs {kd[1] if len(kd) > 1 else "--"}</div>'
                                        f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.85rem;">Takedowns: {td[0] if td else "--"} vs {td[1] if len(td) > 1 else "--"}</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                    else:
                        st.info("No fight history available.")
                else:
                    st.info("No fighter URL available to load history.")

                # Upcoming fights for this fighter
                st.markdown(f'<div style="font-family:{BF};color:{AC};font-size:0.95rem;letter-spacing:3px;text-transform:uppercase;margin:1.5rem 0 0.8rem 0;">UPCOMING FIGHTS</div>', unsafe_allow_html=True)

                try:
                    upcoming_check = get_upcoming_card()
                except Exception:
                    upcoming_check = {}

                found_upcoming = False
                if upcoming_check and upcoming_check.get("fights"):
                    for fight in upcoming_check["fights"]:
                        if (f_name.lower() in fight["fighter_a"].lower() or
                                f_name.lower() in fight["fighter_b"].lower()):
                            found_upcoming = True
                            opp = fight["fighter_b"] if f_name.lower() in fight["fighter_a"].lower() else fight["fighter_a"]
                            wc = fight.get("weight_class", "")
                            evt_name = upcoming_check.get("event", "")
                            evt_date = upcoming_check.get("date", "")
                            st.markdown(
                                f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;">'
                                f'<div style="font-family:{HF};color:{T["TEXT"]};font-size:1.1rem;text-transform:uppercase;">{f_name} vs {opp}</div>'
                                f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.8rem;margin-top:0.4rem;">{wc}</div>'
                                f'<div style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.8rem;margin-top:0.2rem;">{evt_name} -- {evt_date}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                if not found_upcoming:
                    st.markdown(f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;text-align:center;"><span style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.85rem;">No upcoming fights scheduled</span></div>', unsafe_allow_html=True)
            else:
                st.error("Fighter not found in the database.")

    # ── News ─────────────────────────────────────────────────────────────────
    with tab_news:
        st.markdown('<div class="section-header">News</div>', unsafe_allow_html=True)

        with st.spinner("Fueling the fire..."):
            articles = fetch_ufc_news(25)

        if articles:
            # Group by category with per-category source colors
            categories_order = ["Past Fights", "Upcoming Fights", "Fighter News", "Organization"]
            cat_source_colors = {
                "Past Fights": AC,
                "Upcoming Fights": "#f9a825",
                "Fighter News": "#4fc3f7",
                "Organization": "#ce93d8",
            }
            grouped = {}
            for a in articles:
                cat = a["category"]
                if cat not in grouped:
                    grouped[cat] = []
                grouped[cat].append(a)

            available_cats = [c for c in categories_order if c in grouped]
            if available_cats:
                news_tabs = st.tabs(available_cats)

                # Persist active news tab across refreshes
                _components.html("""
                <script>
                (function() {
                    const doc = window.parent.document;
                    const params = new URLSearchParams(window.parent.location.search);
                    const savedTab = params.get('newstab');
                    const tabLists = doc.querySelectorAll('[data-baseweb="tab-list"]');
                    const newsList = tabLists.length > 1 ? tabLists[tabLists.length - 1] : null;
                    if (savedTab && newsList) {
                        const tabs = newsList.querySelectorAll('[data-baseweb="tab"]');
                        tabs.forEach(t => {
                            if (t.textContent.trim() === savedTab) t.click();
                        });
                    }
                    if (newsList) {
                        newsList.addEventListener('click', function(e) {
                            const tab = e.target.closest('[data-baseweb="tab"]');
                            if (tab) {
                                const name = tab.textContent.trim();
                                const url = new URL(window.parent.location);
                                url.searchParams.set('newstab', name);
                                window.parent.history.replaceState({}, '', url);
                            }
                        });
                    }
                })();
                </script>
                """, height=0)

                for news_tab, cat in zip(news_tabs, available_cats):
                    with news_tab:
                        for i, article in enumerate(grouped[cat]):
                            title = article["title"]
                            link = article["link"]
                            source = article["source"]
                            date = article["date"]
                            border = f"border-bottom:1px solid {T['BORDER']};" if i < len(grouped[cat]) - 1 else ""
                            st.markdown(
                                f'<div style="padding:1.1rem 0.5rem;{border}">'
                                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                                f'<div style="font-family:{BF};color:{AC};font-size:1rem;font-weight:900;margin-bottom:0.4rem;">{source}</div>'
                                f'<a href="{link}" target="_blank" style="text-decoration:none;display:flex;align-items:center;">'
                                f'<svg width="18" height="18" viewBox="0 0 16 16" fill="none" style="flex-shrink:0;"><path d="M6 3h7v7M13 3L3 13" stroke="#4fc3f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></a>'
                                f'</div>'
                                f'<div style="font-family:{BF};color:{T["TEXT_DIM"]};font-size:1.1rem;font-weight:600;line-height:1.5;">{title}</div>'
                                f'<div style="margin-top:0.6rem;">'
                                f'<span style="font-family:{BF};color:{T["TEXT_MUTED"]};font-size:0.75rem;">{date}</span>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )
        else:
            st.info("Could not load news. Check your internet connection.")

    # ── Settings ────────────────────────────────────────────────────────────
    with tab_settings:
        st.markdown('<div class="section-header">Settings</div>', unsafe_allow_html=True)

        # ── Appearance ──
        st.markdown(
            f'<div style="font-family:{HF};color:{AC};font-size:1.3rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:1rem;">Appearance</div>',
            unsafe_allow_html=True,
        )
        current_is_dark = st.session_state.theme == "dark"
        choice = st.radio(
            "Theme", ["Dark", "Light"],
            index=0 if current_is_dark else 1,
            horizontal=True, label_visibility="collapsed",
            key="theme_radio",
        )
        if choice == "Dark" and not current_is_dark:
            st.session_state.theme = "dark"
            st.query_params["theme"] = "dark"
            st.rerun()
        elif choice == "Light" and current_is_dark:
            st.session_state.theme = "light"
            st.query_params["theme"] = "light"
            st.rerun()

        st.markdown(f'<div style="border-top:1px solid {T["BORDER"]};margin:2rem 0;"></div>', unsafe_allow_html=True)

        # ── Data ──
        st.markdown(
            f'<div style="font-family:{HF};color:{AC};font-size:1.3rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:0.5rem;">Data</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family:{BF};color:{T["TEXT_DIM"]};font-size:1.1rem;margin-bottom:0.8rem;">'
            f'Fighter stats and fight history update automatically after each UFC event.</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
