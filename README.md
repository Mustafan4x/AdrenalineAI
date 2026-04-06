# UFC Matchup Predictor

An ML-powered web application that predicts UFC fight outcomes using real fighter statistics scraped from UFCStats.com. Select any two fighters or browse upcoming events to get AI-driven predictions with detailed breakdowns, stat comparisons, and confidence ratings.

## Goal

The goal of this project is to build a full-stack prediction tool that goes beyond simple win/loss guessing. It scrapes real fighter data (striking output, takedown accuracy, defense, reach, age, win streaks, and more), trains an XGBoost model on historical fight outcomes, and presents predictions through a clean, dark-themed UI inspired by professional fight apps. Every prediction comes with feature importance analysis, tale-of-the-tape comparisons, radar charts, and a breakdown of why the model favors one fighter over the other.

## Features

- Custom matchup predictions between any two fighters in the database
- Upcoming UFC event browser with per-fight predictions
- Full card prediction mode for entire events
- Fighter profile cards with images pulled from UFC.com
- HTML/CSS visualizations: confidence bars, feature importance, tale of the tape, radar charts, and recent form
- Style matchup analysis describing how fighting styles interact
- Split-screen fighter comparison layout
- Dark theme UI with smooth fade-in transitions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mustafan4x/UFC-Predictor.git
   cd UFC-Predictor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux / Mac
   # .venv\Scripts\activate          # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Collect Data

Run the data collection script to scrape fighter stats and fight history from UFCStats.com. This only needs to be done once (data is cached in the `data/` directory):

```bash
python collect_data.py
```

This takes roughly 15-30 minutes depending on your connection.

### Step 2: Launch the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Step 3: Make Predictions

- **Custom Matchup** -- Select any two fighters from the dropdown menus and click Predict Matchup.
- **Upcoming Event** -- Browse the next scheduled UFC event. Click on any fight to expand it, then hit Predict This Fight.
- **Full Card Predictions** -- Predict every fight on the upcoming card at once.

## Tech Stack

- **Data Collection**: requests, BeautifulSoup, lxml
- **ML Model**: XGBoost with scikit-learn cross-validation
- **Frontend**: Streamlit with custom HTML/CSS (Russo One + Nunito Sans fonts, dark theme)
- **Fighter Images**: Scraped from UFC.com and cached locally

## Project Structure

```
app.py              - Streamlit web application (UI + HTML visualizations)
scraper.py          - UFC data scraper (fighters, fights, upcoming events)
preprocessing.py    - Data cleaning, feature engineering, style classification
model.py            - XGBoost prediction model with feature importance
visualizations.py   - Matplotlib fallback charts (not used in current UI)
collect_data.py     - One-time data collection script
data/               - Cached fighter and fight CSVs, image cache
```
