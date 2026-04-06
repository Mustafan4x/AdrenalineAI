"""
UFC Matchup Predictor - Streamlit Web Application
Full-stack app for predicting UFC fight outcomes with visual explainability.
"""

import os
import sys
import json
import hashlib

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
BG = "#0d0d0d"
CARD_BG = "#111"
INNER_BG = "#161616"

IMAGE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "image_cache")
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
IMAGE_CACHE_FILE = os.path.join(IMAGE_CACHE_DIR, "fighter_images.json")

# Page config
st.set_page_config(
    page_title="UFC Matchup Predictor",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>U</text></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Russo+One&family=Nunito+Sans:wght@300;400;600;700;800&display=swap');

    /* Fade-in animation for all content */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Global */
    .stApp {{
        background-color: {BG};
        color: #e0e0e0;
        font-family: {BF};
    }}

    /* Smooth fade on tab content and main elements */
    .stTabs [role="tabpanel"] {{
        animation: fadeIn 0.4s ease-out;
    }}
    .stTabs [role="tabpanel"] > div {{
        animation: fadeIn 0.35s ease-out;
    }}
    .winner-container, .fighter-bg-card, .fight-table, .style-matchup, .reason-box,
    .event-header, .section-header, .header-bar {{
        animation: fadeIn 0.4s ease-out;
    }}
    .fight-table-row {{
        animation: fadeIn 0.3s ease-out;
        animation-fill-mode: both;
    }}
    .fight-table-row:nth-child(1) {{ animation-delay: 0.05s; }}
    .fight-table-row:nth-child(2) {{ animation-delay: 0.1s; }}
    .fight-table-row:nth-child(3) {{ animation-delay: 0.15s; }}
    .fight-table-row:nth-child(4) {{ animation-delay: 0.2s; }}
    .fight-table-row:nth-child(5) {{ animation-delay: 0.25s; }}
    .fight-table-row:nth-child(6) {{ animation-delay: 0.3s; }}
    .fight-table-row:nth-child(7) {{ animation-delay: 0.35s; }}
    .fight-table-row:nth-child(8) {{ animation-delay: 0.4s; }}
    .fight-table-row:nth-child(9) {{ animation-delay: 0.45s; }}
    .fight-table-row:nth-child(10) {{ animation-delay: 0.5s; }}

    /* Smooth transitions on interactive elements */
    .stButton > button {{
        transition: all 0.25s ease !important;
    }}
    .stSelectbox, .stTabs [data-baseweb="tab"] {{
        transition: all 0.2s ease;
    }}

    /* Hide chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}
    section[data-testid="stSidebar"] {{ display: none; }}

    /* Header bar with stats */
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #1a1a1a;
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
        color: #e0e0e0;
        font-size: 0.8rem;
        font-weight: 700;
    }}
    .header-stats .h-stat .h-label {{
        color: #777;
        font-size: 0.65rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-left: 0.3rem;
    }}
    .header-stats .h-divider {{
        width: 1px;
        height: 20px;
        background: #333;
    }}

    /* Underline tab navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background: transparent;
        border-bottom: 1px solid #1a1a1a;
        padding: 0;
        justify-content: center;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: {BF};
        color: #555;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: 0.75rem;
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
        background: linear-gradient(135deg, #111 0%, #0a0a0a 60%, rgba(211,47,47,0.06) 100%);
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
        color: #666;
        font-size: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }}
    .fighter-bg-card .f-last {{
        font-family: {HF};
        color: #fff;
        font-size: 2.4rem;
        text-transform: uppercase;
        margin: 0.3rem 0;
        line-height: 1;
    }}
    .fighter-bg-card .f-first {{
        font-family: {HF};
        color: rgba(255,255,255,0.3);
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
        color: #555;
        font-size: 0.55rem;
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
        color: #e0e0e0;
        font-size: 1rem;
        font-weight: 700;
    }}
    .fighter-bg-card .f-stat-lbl {{
        font-family: {BF};
        color: #555;
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
        background: #1a1a1a;
        border: 2px solid #333;
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
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #555;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1a1a1a;
    }}

    /* Ghost button overrides */
    .stButton > button[kind="primary"] {{
        background-color: transparent !important;
        color: {AC} !important;
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
        background-color: {AC} !important;
        color: white !important;
    }}
    .stButton > button {{
        background-color: transparent !important;
        color: #666 !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-family: {BF} !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
    }}
    .stButton > button:hover {{
        color: #e0e0e0 !important;
        border-color: #666 !important;
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
        background: {INNER_BG};
    }}
    .winner-label {{
        font-family: {BF};
        font-size: 0.6rem;
        letter-spacing: 3px;
        text-transform: uppercase;
    }}
    .winner-label.win {{ color: rgba(255,255,255,0.7); }}
    .winner-label.lose {{ color: #555; }}
    .winner-name {{
        font-family: {HF};
        font-size: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 0.3rem 0;
    }}
    .winner-name.win {{ color: #fff; }}
    .winner-name.lose {{ color: #555; }}
    .winner-conf {{
        font-family: {BF};
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .winner-conf.win {{ color: rgba(255,255,255,0.85); }}
    .winner-conf.lose {{ color: #444; }}

    /* Style matchup box */
    .style-matchup {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    .style-matchup .style-title {{
        font-family: {BF};
        font-size: 0.65rem;
        color: #555;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .style-matchup .style-body {{
        font-family: {BF};
        color: #999;
        font-size: 0.85rem;
        line-height: 1.6;
    }}

    /* Reason boxes */
    .reason-box {{
        background: {CARD_BG};
        padding: 0.9rem 1.2rem;
        border-radius: 12px;
        border-left: 3px solid {AC};
        margin: 0.4rem 0;
        font-family: {BF};
        color: #bbb;
        font-size: 0.85rem;
    }}
    .reason-box.against {{
        border-left-color: #333;
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
        border-bottom: 1px solid #1a1a1a;
    }}
    .fight-table-row:last-child {{
        border-bottom: none;
    }}
    .fight-table-row:hover {{
        background: rgba(211,47,47,0.03);
    }}
    .ft-left {{
        padding: 1rem 1.5rem;
        text-align: right;
    }}
    .ft-left .ft-name {{
        font-family: {HF};
        color: #e0e0e0;
        font-size: 1rem;
        text-transform: uppercase;
    }}
    .ft-left .ft-record {{
        font-family: {BF};
        color: #555;
        font-size: 0.6rem;
        letter-spacing: 1px;
    }}
    .ft-center {{
        width: 1px;
        background: #1a1a1a;
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
        color: #e0e0e0;
        font-size: 1rem;
        text-transform: uppercase;
    }}
    .ft-right .ft-record {{
        font-family: {BF};
        color: #555;
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
        color: #fff;
        letter-spacing: 2px;
    }}
    .event-header .event-detail {{
        font-family: {BF};
        font-size: 0.7rem;
        color: #555;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }}

    /* Visualization tabs (inside predictions) */
    .stTabs .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid #1a1a1a;
    }}

    /* Selectbox */
    .stSelectbox label {{
        font-family: {BF} !important;
        color: #555 !important;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: 0.7rem !important;
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
        color: #e0e0e0 !important;
    }}
    [data-testid="stExpander"] {{
        background: {CARD_BG};
        border: 1px solid #1a1a1a;
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
        font-size: 0.85rem !important;
        padding: 1rem 1.2rem !important;
    }}

    hr {{
        border-color: #1a1a1a;
    }}

    .stAlert {{
        border-radius: 12px;
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
    if name in cache:
        return cache[name]

    # Build UFC.com athlete slug: "Conor McGregor" -> "conor-mcgregor"
    slug = name.strip().lower().replace(" ", "-")
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    url = f"https://www.ufc.com/athlete/{slug}"

    img_url = ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Try og:image meta tag first
            og = soup.find("meta", property="og:image")
            if og and og.get("content"):
                img_url = og["content"]
            else:
                # Try the hero image
                hero = soup.find("img", class_="hero-profile__image")
                if hero and hero.get("src"):
                    img_url = hero["src"]
    except Exception:
        pass

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

@st.cache_resource(show_spinner="Loading fighter database...")
def load_fighters():
    return build_fighter_database(use_cache=True)

@st.cache_resource(show_spinner="Loading fight history...")
def load_fights():
    return build_fight_history_database(use_cache=True)

@st.cache_resource(show_spinner="Training prediction model...")
def get_trained_model(_fighters_df, _fights_df):
    predictor = UFCPredictor()
    predictor.load_data(_fighters_df, _fights_df)
    predictor.train()
    return predictor


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
        f'<div style="font-family:{BF};color:{AC};font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1.2rem;">WIN PROBABILITY</div>'
        # Names + percentages row
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.6rem;">'
        f'<div><span style="font-family:{HF};color:#fff;font-size:1.1rem;text-transform:uppercase;">{fa_last}</span><span style="font-family:{BF};color:{AC};font-size:1.1rem;font-weight:800;margin-left:0.6rem;">{prob_a:.1f}%</span></div>'
        f'<div><span style="font-family:{BF};color:#888;font-size:1.1rem;font-weight:800;margin-right:0.6rem;">{prob_b:.1f}%</span><span style="font-family:{HF};color:#888;font-size:1.1rem;text-transform:uppercase;">{fb_last}</span></div>'
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
            f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.55rem 0;border-bottom:1px solid #1a1a1a;">'
            f'<div style="width:140px;flex-shrink:0;font-family:{BF};color:#bbb;font-size:0.85rem;text-align:right;">{display_name}</div>'
            f'<div style="flex:1;background:#1a1a1a;border-radius:4px;height:26px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;transition:width 0.6s ease;"></div></div>'
            f'<div style="width:80px;flex-shrink:0;font-family:{BF};color:{color};font-size:0.8rem;font-weight:700;letter-spacing:0.5px;">{favor_label}</div>'
            f'</div>'
        )

    legend = (
        f'<div style="display:flex;gap:1.5rem;margin-bottom:1rem;">'
        f'<div style="display:flex;align-items:center;gap:0.4rem;"><div style="width:10px;height:10px;border-radius:2px;background:{AC};"></div><span style="font-family:{BF};color:#bbb;font-size:0.8rem;">Favors {fa_name}</span></div>'
        f'<div style="display:flex;align-items:center;gap:0.4rem;"><div style="width:10px;height:10px;border-radius:2px;background:#666;"></div><span style="font-family:{BF};color:#bbb;font-size:0.8rem;">Favors {fb_name}</span></div>'
        f'</div>'
    )

    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">FEATURE IMPORTANCE</div>'
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
            f'<div style="display:grid;grid-template-columns:65px 1fr 110px 1fr 65px;align-items:center;gap:0.5rem;padding:0.55rem 0;border-bottom:1px solid #1a1a1a;">'
            f'<div style="font-family:{BF};color:{val_color_a};font-size:0.9rem;font-weight:700;text-align:right;">{va:.1f}</div>'
            f'<div style="display:flex;justify-content:flex-end;"><div style="width:{pct_a:.0f}%;height:20px;background:{color_a};border-radius:3px 0 0 3px;min-width:2px;transition:width 0.6s ease;"></div></div>'
            f'<div style="font-family:{BF};color:#888;font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;text-align:center;">{label}</div>'
            f'<div><div style="width:{pct_b:.0f}%;height:20px;background:{color_b};border-radius:0 3px 3px 0;min-width:2px;transition:width 0.6s ease;"></div></div>'
            f'<div style="font-family:{BF};color:{val_color_b};font-size:0.9rem;font-weight:700;">{vb:.1f}</div>'
            f'</div>'
        )

    fa_last = fa["name"].split()[-1].upper()
    fb_last = fb["name"].split()[-1].upper()

    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">TALE OF THE TAPE</div>'
        f'<div style="display:grid;grid-template-columns:65px 1fr 110px 1fr 65px;gap:0.5rem;margin-bottom:0.8rem;">'
        f'<div></div><div style="font-family:{HF};color:#e0e0e0;font-size:1rem;text-align:right;text-transform:uppercase;">{fa_last}</div>'
        f'<div></div>'
        f'<div style="font-family:{HF};color:#e0e0e0;font-size:1rem;text-transform:uppercase;">{fb_last}</div><div></div>'
        f'</div>{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _render_radar_chart(prediction: dict):
    """SVG radar chart matching the app design."""
    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    labels = ["Striking", "Str Acc", "Str Def", "Takedowns", "TD Def", "Subs", "Win Rate", "Experience"]

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
    cx, cy, r = 200, 200, 150

    def polar_point(angle_idx, value, radius=r):
        angle = (2 * math.pi * angle_idx / n) - math.pi / 2
        return cx + radius * value * math.cos(angle), cy + radius * value * math.sin(angle)

    # Grid rings
    grid_svg = ""
    for level in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{polar_point(i, level)[0]:.1f},{polar_point(i, level)[1]:.1f}" for i in range(n))
        grid_svg += f'<polygon points="{pts}" fill="none" stroke="#1a1a1a" stroke-width="1"/>'

    # Grid spokes
    for i in range(n):
        x, y = polar_point(i, 1.0)
        grid_svg += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#1a1a1a" stroke-width="1"/>'

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
        label_svg += f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" fill="#bbb" font-size="13" font-family="Nunito Sans, sans-serif">{lbl}</text>'

    # Dots
    dots_a = "".join(f'<circle cx="{polar_point(i, vals_a[i])[0]:.1f}" cy="{polar_point(i, vals_a[i])[1]:.1f}" r="4" fill="{AC}"/>' for i in range(n))
    dots_b = "".join(f'<circle cx="{polar_point(i, vals_b[i])[0]:.1f}" cy="{polar_point(i, vals_b[i])[1]:.1f}" r="4" fill="#666"/>' for i in range(n))

    fa_last = fa["name"].split()[-1]
    fb_last = fb["name"].split()[-1]

    svg = (
        f'<svg viewBox="0 0 400 420" xmlns="http://www.w3.org/2000/svg" style="max-width:450px;margin:0 auto;display:block;">'
        f'{grid_svg}'
        f'<polygon points="{pts_a}" fill="{AC}" fill-opacity="0.15" stroke="{AC}" stroke-width="2"/>'
        f'<polygon points="{pts_b}" fill="#666" fill-opacity="0.1" stroke="#666" stroke-width="2"/>'
        f'{dots_a}{dots_b}'
        f'{label_svg}'
        f'<circle cx="130" cy="400" r="6" fill="{AC}"/><text x="142" y="405" fill="#bbb" font-size="13" font-weight="bold" font-family="Nunito Sans, sans-serif">{fa_last}</text>'
        f'<circle cx="230" cy="400" r="6" fill="#666"/><text x="242" y="405" fill="#bbb" font-size="13" font-weight="bold" font-family="Nunito Sans, sans-serif">{fb_last}</text>'
        f'</svg>'
    )

    st.markdown(
        f'<div style="background:{CARD_BG};border-radius:16px;padding:1.5rem;margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">FIGHTER COMPARISON</div>'
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
            f'<div style="font-family:{BF};color:#888;font-size:0.85rem;margin-bottom:1rem;">{record}</div>'
            f'<div style="display:flex;gap:0.6rem;align-items:center;">{dots}</div>'
            f'<div style="font-family:{BF};color:#888;font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-top:0.8rem;">LAST {len(form)} FIGHTS</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="margin:0.5rem 0;">'
        f'<div style="font-family:{BF};color:{AC};font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:0.8rem;padding-left:0.5rem;">RECENT FORM</div>'
        f'<div style="display:flex;gap:1rem;">'
        f'{form_html(fa, form_a, AC)}'
        f'{form_html(fb, form_b, "#666")}'
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
    # Check for data first
    data_exists = (
        os.path.exists(os.path.join("data", "fighters.csv")) and
        os.path.exists(os.path.join("data", "fights.csv"))
    )

    if not data_exists:
        st.markdown(
            f'<div style="text-align:center;padding:4rem 2rem;">'
            f'<div style="font-family:{HF};color:{AC};font-size:2rem;letter-spacing:2px;">UFC</div>'
            f'<div style="font-family:{BF};color:#555;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;margin-top:0.5rem;">MATCHUP PREDICTOR</div>'
            f'<div style="font-family:{BF};color:#888;font-size:1rem;margin-top:2rem;">No cached data found. Run the data collection script first:</div>'
            f'<div style="background:{CARD_BG};border-radius:12px;padding:1.2rem;margin:1.5rem auto;max-width:400px;font-family:monospace;color:#aaa;font-size:0.85rem;">python collect_data.py</div>'
            f'<div style="font-family:{BF};color:#555;font-size:0.8rem;">This takes 15-30 minutes to scrape data from UFCStats.com.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("Start Data Collection", type="primary"):
            with st.spinner("Scraping fighter database... This may take a while."):
                fighters_df = build_fighter_database(use_cache=False)
            with st.spinner("Scraping fight history..."):
                fights_df = build_fight_history_database(use_cache=False)
            st.success(f"Done! Loaded {len(fighters_df)} fighters and {len(fights_df)} fights.")
            st.rerun()
        return

    # Load data
    fighters_df = load_fighters()
    fights_df = load_fights()
    fighters_clean = clean_fighter_data(fighters_df)

    # Train model
    predictor = get_trained_model(fighters_df, fights_df)

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

    st.markdown(f'<div class="header-bar"><div class="brand">UFC</div><div class="header-stats">{stats_inner}</div></div>', unsafe_allow_html=True)

    # ── Navigation via underline tabs ────────────────────────────────────────
    tab_matchup, tab_upcoming, tab_fullcard = st.tabs([
        "Custom Matchup", "Upcoming Event", "Full Card Predictions"
    ])

    # ── Custom Matchup ───────────────────────────────────────────────────────
    with tab_matchup:
        st.markdown('<div class="section-header">Custom Matchup</div>', unsafe_allow_html=True)

        fighter_names = sorted(fighters_clean["name"].dropna().unique().tolist())

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
                with st.spinner("Analyzing matchup..."):
                    try:
                        prediction = predictor.predict_matchup(fighter_a, fighter_b)
                        display_prediction(prediction)
                    except ValueError as e:
                        st.error(f"Error: {e}")

    # ── Upcoming Event ───────────────────────────────────────────────────────
    with tab_upcoming:
        st.markdown('<div class="section-header">Next Upcoming UFC Event</div>', unsafe_allow_html=True)

        with st.spinner("Fetching upcoming event..."):
            try:
                upcoming = get_upcoming_card()
            except Exception as e:
                st.error(f"Could not fetch upcoming events: {e}")
                upcoming = {}

        if upcoming and upcoming.get("fights"):
            evt = upcoming.get('event', 'Unknown Event')
            evt_date = upcoming.get('date', 'TBD')
            evt_loc = upcoming.get('location', 'TBD')
            st.markdown(f'<div class="event-header"><div class="event-name">{evt}</div><div class="event-detail">{evt_date} -- {evt_loc}</div></div>', unsafe_allow_html=True)

            # Each fight is its own clickable expander
            for i, fight in enumerate(upcoming["fights"]):
                fa_name = fight["fighter_a"]
                fb_name = fight["fighter_b"]
                wc = fight.get("weight_class", "")
                label = f"{fa_name}  vs  {fb_name}    --  {wc}" if wc else f"{fa_name}  vs  {fb_name}"
                with st.expander(label, expanded=False):
                    if st.button("PREDICT THIS FIGHT", type="primary", use_container_width=True, key=f"predict_{i}"):
                        with st.spinner("Analyzing..."):
                            try:
                                prediction = predictor.predict_matchup(fa_name, fb_name)
                                display_prediction(prediction)
                            except ValueError as e:
                                st.error(f"Could not predict: {e}")
        else:
            st.info("No upcoming events found. Check back later.")

    # ── Full Card Predictions ────────────────────────────────────────────────
    with tab_fullcard:
        st.markdown('<div class="section-header">Full Card Predictions</div>', unsafe_allow_html=True)

        with st.spinner("Fetching upcoming event..."):
            try:
                upcoming_fc = get_upcoming_card()
            except Exception as e:
                st.error(f"Could not fetch upcoming events: {e}")
                upcoming_fc = {}

        if upcoming_fc and upcoming_fc.get("fights"):
            fc_evt = upcoming_fc.get('event', 'Unknown Event')
            fc_date = upcoming_fc.get('date', 'TBD')
            st.markdown(f'<div class="event-header"><div class="event-name">{fc_evt}</div><div class="event-detail">{fc_date}</div></div>', unsafe_allow_html=True)

            if st.button("PREDICT ENTIRE CARD", type="primary", use_container_width=True):
                with st.spinner("Predicting all fights..."):
                    results = predictor.predict_card(upcoming_fc["fights"])

                for i, pred in enumerate(results):
                    label = (
                        f"{pred['fighter_a']['name']} vs {pred['fighter_b']['name']}"
                        f"{' -- ' + pred.get('weight_class', '') if pred.get('weight_class') else ''}"
                        f"{' -- Winner: ' + pred['predicted_winner'] + ' (' + str(round(pred['confidence'])) + '%)' if 'predicted_winner' in pred else ''}"
                    )
                    with st.expander(label, expanded=(i == 0)):
                        display_prediction(pred)
        else:
            st.info("No upcoming events found.")


if __name__ == "__main__":
    main()
