"""
UFC Matchup Predictor - Streamlit Web Application
Full-stack app for predicting UFC fight outcomes with visual explainability.
"""

import os
import sys

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import (
    build_fighter_database,
    build_fight_history_database,
    get_upcoming_card,
    scrape_fighter_details,
)
from preprocessing import (
    clean_fighter_data,
    get_fighter_by_name,
    get_style_matchup_description,
    FEATURE_COLUMNS,
)
from model import UFCPredictor
from visualizations import (
    plot_feature_importance,
    plot_tale_of_tape,
    plot_radar_chart,
    plot_confidence_gauge,
    plot_historical_trends,
    plot_matchup_summary,
)

# Page config
st.set_page_config(
    page_title="UFC Matchup Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .fighter-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    .winner-banner {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stat-label {
        color: #aaa;
        font-size: 0.85rem;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .reason-box {
        background: #16213e;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
        margin: 0.3rem 0;
    }
    .reason-against {
        border-left-color: #e74c3c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading fighter database...")
def load_fighters():
    """Load and cache fighter database."""
    return build_fighter_database(use_cache=True)


@st.cache_resource(show_spinner="Loading fight history...")
def load_fights():
    """Load and cache fight history."""
    return build_fight_history_database(use_cache=True)


@st.cache_resource(show_spinner="Training prediction model...")
def get_trained_model(_fighters_df, _fights_df):
    """Train and cache the prediction model."""
    predictor = UFCPredictor()
    predictor.load_data(_fighters_df, _fights_df)
    predictor.train()
    return predictor


def display_fighter_card(fighter: dict, color: str):
    """Display a fighter info card."""
    style = fighter.get("combat_style", "Unknown")
    stance = fighter.get("stance", "Unknown") or "Unknown"
    record = fighter.get("record", "0-0-0")
    age = fighter.get("age", "?")
    height = fighter.get("height_cm", "?")
    reach = fighter.get("reach_cm", "?")
    slpm = fighter.get("slpm", 0)
    str_acc = fighter.get("str_acc", 0)
    td_avg = fighter.get("td_avg", 0)

    st.markdown(f"""
    <div class="fighter-card" style="border-left: 4px solid {color};">
        <h2 style="color: {color}; margin: 0;">{fighter['name']}</h2>
        <p style="margin: 0.3rem 0;"><b>Record:</b> {record} | <b>Age:</b> {age:.0f} | <b>Style:</b> {style}</p>
        <p style="margin: 0.3rem 0;"><b>Stance:</b> {stance} | <b>Height:</b> {height:.0f}cm | <b>Reach:</b> {reach:.0f}cm</p>
        <p style="margin: 0.3rem 0;"><b>Strikes/Min:</b> {slpm:.2f} | <b>Str Acc:</b> {str_acc*100 if str_acc else 0:.0f}% | <b>TD/Fight:</b> {td_avg:.2f}</p>
    </div>
    """, unsafe_allow_html=True)


def display_prediction(prediction: dict):
    """Display full prediction results with visualizations."""
    if "error" in prediction:
        st.error(f"Could not predict: {prediction['error']}")
        return

    winner = prediction["predicted_winner"]
    confidence = prediction["confidence"]
    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    # Winner banner
    st.markdown(f"""
    <div class="winner-banner">
        🏆 Predicted Winner: {winner} — {confidence:.1f}% Confidence
    </div>
    """, unsafe_allow_html=True)

    # Fighter cards
    col1, col2 = st.columns(2)
    with col1:
        display_fighter_card(fa, "#e74c3c")
    with col2:
        display_fighter_card(fb, "#3498db")

    # Style matchup
    st.markdown("---")
    style_a = prediction.get("style_a", "Unknown")
    style_b = prediction.get("style_b", "Unknown")
    st.markdown(f"**Style Matchup:** {style_a} vs {style_b}")
    st.info(prediction.get("style_matchup", ""))

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Confidence", "🎯 Feature Importance", "⚔️ Tale of the Tape",
        "🕸️ Radar Chart", "📈 Historical Trends"
    ])

    with tab1:
        fig = plot_confidence_gauge(prediction)
        st.pyplot(fig)

    with tab2:
        fig = plot_feature_importance(prediction)
        st.pyplot(fig)

    with tab3:
        fig = plot_tale_of_tape(prediction)
        st.pyplot(fig)

    with tab4:
        fig = plot_radar_chart(prediction)
        st.pyplot(fig)

    with tab5:
        fig = plot_historical_trends(prediction)
        st.pyplot(fig)

    # Reasons
    st.markdown("---")
    st.subheader("Prediction Breakdown")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Why {winner} wins:**")
        for reason in prediction.get("reasons_winner", []):
            st.markdown(f"""<div class="reason-box">✅ {reason}</div>""", unsafe_allow_html=True)

    with col2:
        loser = prediction.get("predicted_loser", "")
        if prediction.get("reasons_loser"):
            st.markdown(f"**{loser}'s advantages:**")
            for reason in prediction.get("reasons_loser", []):
                st.markdown(f"""<div class="reason-box reason-against">⚠️ {reason}</div>""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🥊 UFC Matchup Predictor</h1>
        <p style="color: #aaa;">AI-powered fight predictions with visual explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Check for data
    data_exists = (
        os.path.exists(os.path.join("data", "fighters.csv")) and
        os.path.exists(os.path.join("data", "fights.csv"))
    )

    if not data_exists:
        st.warning("⚠️ No cached data found. You need to scrape UFC data first.")
        st.markdown("""
        ### First-time Setup
        Run the data collection script to scrape fighter stats and fight history:
        ```bash
        python collect_data.py
        ```
        This will take 15-30 minutes to scrape all fighter data from UFCStats.com.
        The data will be cached in the `data/` directory for future use.
        """)

        if st.button("🔄 Start Data Collection", type="primary"):
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

    # Clean data
    fighters_clean = clean_fighter_data(fighters_df)

    st.sidebar.success(f"✅ {len(fighters_clean)} fighters loaded")
    st.sidebar.success(f"✅ {len(fights_df)} historical fights loaded")

    # Train model
    predictor = get_trained_model(fighters_df, fights_df)

    if predictor.cv_scores is not None:
        st.sidebar.metric("Model CV Accuracy", f"{predictor.cv_scores.mean():.1%}")

    # Navigation
    page = st.sidebar.radio("Navigate", [
        "🏠 Custom Matchup",
        "📅 Upcoming Event",
        "📊 Full Card Predictions",
    ])

    if page == "🏠 Custom Matchup":
        st.header("Custom Matchup Prediction")
        st.markdown("Select two fighters to predict their matchup outcome.")

        # Fighter selection
        fighter_names = sorted(fighters_clean["name"].dropna().unique().tolist())

        col1, col2 = st.columns(2)
        with col1:
            fighter_a = st.selectbox("🔴 Fighter A", fighter_names, index=0, key="fa")
        with col2:
            fighter_b = st.selectbox("🔵 Fighter B", fighter_names, index=min(1, len(fighter_names)-1), key="fb")

        if st.button("🥊 Predict Matchup", type="primary", use_container_width=True):
            if fighter_a == fighter_b:
                st.error("Please select two different fighters.")
            else:
                with st.spinner("Analyzing matchup..."):
                    try:
                        prediction = predictor.predict_matchup(fighter_a, fighter_b)
                        display_prediction(prediction)
                    except ValueError as e:
                        st.error(f"Error: {e}")

    elif page == "📅 Upcoming Event":
        st.header("Next Upcoming UFC Event")

        with st.spinner("Fetching upcoming event..."):
            try:
                upcoming = get_upcoming_card()
            except Exception as e:
                st.error(f"Could not fetch upcoming events: {e}")
                upcoming = {}

        if upcoming and upcoming.get("fights"):
            st.subheader(f"🏟️ {upcoming.get('event', 'Unknown Event')}")
            st.markdown(f"**Date:** {upcoming.get('date', 'TBD')} | **Location:** {upcoming.get('location', 'TBD')}")
            st.markdown("---")

            for i, fight in enumerate(upcoming["fights"]):
                col1, col2, col3 = st.columns([2, 1, 2])
                with col1:
                    st.markdown(f"### 🔴 {fight['fighter_a']}")
                with col2:
                    st.markdown("### VS")
                with col3:
                    st.markdown(f"### 🔵 {fight['fighter_b']}")

                wc = fight.get("weight_class", "")
                if wc:
                    st.caption(wc)

                if st.button(f"Predict: {fight['fighter_a']} vs {fight['fighter_b']}", key=f"predict_{i}"):
                    with st.spinner("Analyzing..."):
                        try:
                            prediction = predictor.predict_matchup(fight["fighter_a"], fight["fighter_b"])
                            display_prediction(prediction)
                        except ValueError as e:
                            st.error(f"Could not predict: {e}")
                st.markdown("---")
        else:
            st.info("No upcoming events found. Check back later!")

    elif page == "📊 Full Card Predictions":
        st.header("Full Card Predictions")

        with st.spinner("Fetching upcoming event..."):
            try:
                upcoming = get_upcoming_card()
            except Exception as e:
                st.error(f"Could not fetch upcoming events: {e}")
                upcoming = {}

        if upcoming and upcoming.get("fights"):
            st.subheader(f"🏟️ {upcoming.get('event', 'Unknown Event')}")
            st.markdown(f"**Date:** {upcoming.get('date', 'TBD')}")

            if st.button("🔮 Predict Entire Card", type="primary", use_container_width=True):
                with st.spinner("Predicting all fights..."):
                    results = predictor.predict_card(upcoming["fights"])

                for i, pred in enumerate(results):
                    with st.expander(
                        f"{'❌' if 'error' in pred else '✅'} "
                        f"{pred['fighter_a']['name']} vs {pred['fighter_b']['name']}"
                        f"{' — ' + pred.get('weight_class', '') if pred.get('weight_class') else ''}"
                        f"{' — Winner: ' + pred['predicted_winner'] + f' ({pred["\confidence\"]:.0f}%)' if 'predicted_winner' in pred else ''}",
                        expanded=(i == 0)
                    ):
                        display_prediction(pred)
        else:
            st.info("No upcoming events found.")


if __name__ == "__main__":
    main()
