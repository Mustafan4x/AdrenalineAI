"""
UFC Data Preprocessing Pipeline
Handles data cleaning, feature engineering, combat style classification,
and difference matrix creation for matchup prediction.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def classify_combat_style(row: pd.Series) -> str:
    """
    Classify a fighter's combat style based on their stats.
    Categories: Striker, Grappler, Aggressive, Passive, All-Rounder
    """
    slpm = row.get("slpm", 0) or 0
    td_avg = row.get("td_avg", 0) or 0
    sub_avg = row.get("sub_avg", 0) or 0
    str_acc = row.get("str_acc", 0) or 0
    td_acc = row.get("td_acc", 0) or 0
    sapm = row.get("sapm", 0) or 0

    grappling_score = td_avg * 2 + sub_avg * 3 + (td_acc * 2 if td_acc else 0)
    striking_score = slpm * 1.5 + (str_acc * 3 if str_acc else 0)

    # Aggressive: high output on both ends
    if slpm > 4.5 and sapm > 4.0:
        return "Aggressive"

    # Passive: low output
    if slpm < 2.5 and td_avg < 1.0 and sub_avg < 0.5:
        return "Passive"

    # Grappler: wrestling/submission focused
    if grappling_score > striking_score * 1.3:
        return "Grappler"

    # Striker: stand-up focused
    if striking_score > grappling_score * 1.3:
        return "Striker"

    return "All-Rounder"


def compute_age(dob: str, reference_date: str = None) -> float | None:
    """Compute age from date of birth."""
    if pd.isna(dob) or not dob:
        return None
    try:
        birth = datetime.strptime(str(dob), "%Y-%m-%d")
        ref = datetime.strptime(reference_date, "%Y-%m-%d") if reference_date else datetime.now()
        age = (ref - birth).days / 365.25
        return round(age, 1)
    except (ValueError, TypeError):
        return None


def compute_win_streak(fights_df: pd.DataFrame, fighter_name: str) -> int:
    """Compute current win streak for a fighter from fight history."""
    fighter_fights = fights_df[
        (fights_df["fighter_a"] == fighter_name) | (fights_df["fighter_b"] == fighter_name)
    ].copy()

    streak = 0
    for _, fight in fighter_fights.iterrows():
        if fight.get("winner") == fighter_name:
            streak += 1
        else:
            break
    return streak


def impute_reach(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing reach data using height correlation."""
    df = df.copy()
    mask = df["reach_cm"].isna() & df["height_cm"].notna()
    if mask.any() and df["reach_cm"].notna().sum() > 10:
        # Reach is typically ~1.04x height
        valid = df[df["reach_cm"].notna() & df["height_cm"].notna()]
        if len(valid) > 0:
            ratio = (valid["reach_cm"] / valid["height_cm"]).median()
            df.loc[mask, "reach_cm"] = df.loc[mask, "height_cm"] * ratio
    # Fill remaining with median
    if df["reach_cm"].isna().any():
        df["reach_cm"] = df["reach_cm"].fillna(df["reach_cm"].median())
    return df


def clean_fighter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare fighter data."""
    df = df.copy()

    # Compute age
    df["age"] = df["dob"].apply(compute_age)

    # Impute reach
    df = impute_reach(df)

    # Fill missing numeric columns with median
    numeric_cols = ["height_cm", "weight_kg", "reach_cm", "age",
                    "slpm", "str_acc", "sapm", "str_def",
                    "td_avg", "td_acc", "td_def", "sub_avg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Fill missing record values
    for col in ["wins", "losses", "draws"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Compute derived features
    df["total_fights"] = df["wins"] + df["losses"] + df["draws"]
    df["win_rate"] = df.apply(
        lambda r: r["wins"] / r["total_fights"] if r["total_fights"] > 0 else 0.5, axis=1
    )

    # Classify combat style
    df["combat_style"] = df.apply(classify_combat_style, axis=1)

    return df


def create_feature_vector(fighter: pd.Series) -> dict:
    """Create a feature dictionary for a single fighter."""
    features = {
        "height_cm": fighter.get("height_cm", 0),
        "weight_kg": fighter.get("weight_kg", 0),
        "reach_cm": fighter.get("reach_cm", 0),
        "age": fighter.get("age", 30),
        "wins": fighter.get("wins", 0),
        "losses": fighter.get("losses", 0),
        "draws": fighter.get("draws", 0),
        "total_fights": fighter.get("total_fights", 0),
        "win_rate": fighter.get("win_rate", 0.5),
        "slpm": fighter.get("slpm", 0),
        "str_acc": fighter.get("str_acc", 0),
        "sapm": fighter.get("sapm", 0),
        "str_def": fighter.get("str_def", 0),
        "td_avg": fighter.get("td_avg", 0),
        "td_acc": fighter.get("td_acc", 0),
        "td_def": fighter.get("td_def", 0),
        "sub_avg": fighter.get("sub_avg", 0),
        "win_streak": fighter.get("win_streak", 0),
        # One-hot encoded combat style
        "style_striker": 1 if fighter.get("combat_style") == "Striker" else 0,
        "style_grappler": 1 if fighter.get("combat_style") == "Grappler" else 0,
        "style_aggressive": 1 if fighter.get("combat_style") == "Aggressive" else 0,
        "style_passive": 1 if fighter.get("combat_style") == "Passive" else 0,
        "style_all_rounder": 1 if fighter.get("combat_style") == "All-Rounder" else 0,
    }
    return features


FEATURE_COLUMNS = [
    "height_cm", "weight_kg", "reach_cm", "age",
    "wins", "losses", "draws", "total_fights", "win_rate",
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
    "win_streak",
    "style_striker", "style_grappler", "style_aggressive",
    "style_passive", "style_all_rounder"
]


def create_difference_matrix(fighter_a: pd.Series, fighter_b: pd.Series) -> np.ndarray:
    """
    Create the difference matrix (X_A - X_B) for a matchup.
    Returns a 1D array of feature differences.
    """
    feats_a = create_feature_vector(fighter_a)
    feats_b = create_feature_vector(fighter_b)

    diff = []
    for col in FEATURE_COLUMNS:
        val_a = feats_a.get(col, 0) or 0
        val_b = feats_b.get(col, 0) or 0
        diff.append(float(val_a) - float(val_b))

    return np.array(diff).reshape(1, -1)


def build_training_data(fighters_df: pd.DataFrame, fights_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training data from historical fights.
    Returns X (difference matrices) and y (1 if fighter_a won, 0 if fighter_b won).
    """
    fighters_df = clean_fighter_data(fighters_df)

    # Create a lookup dict
    fighter_lookup = {}
    for _, row in fighters_df.iterrows():
        name = row.get("name", "").strip()
        if name:
            fighter_lookup[name.lower()] = row

    X_list = []
    y_list = []

    for _, fight in fights_df.iterrows():
        fa_name = str(fight.get("fighter_a", "")).strip().lower()
        fb_name = str(fight.get("fighter_b", "")).strip().lower()
        winner = fight.get("winner")

        if not winner or fa_name not in fighter_lookup or fb_name not in fighter_lookup:
            continue

        fa = fighter_lookup[fa_name]
        fb = fighter_lookup[fb_name]

        diff = create_difference_matrix(fa, fb)
        label = 1 if str(winner).strip().lower() == fa_name else 0

        X_list.append(diff.flatten())
        y_list.append(label)

    if not X_list:
        return np.array([]), np.array([])

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def get_fighter_by_name(fighters_df: pd.DataFrame, name: str) -> pd.Series | None:
    """Look up a fighter by name (case-insensitive, partial match)."""
    name_lower = name.strip().lower()
    # Exact match first
    exact = fighters_df[fighters_df["name"].str.lower().str.strip() == name_lower]
    if len(exact) > 0:
        return exact.iloc[0]

    # Partial match
    partial = fighters_df[fighters_df["name"].str.lower().str.contains(name_lower, na=False)]
    if len(partial) > 0:
        return partial.iloc[0]

    return None


def get_style_matchup_description(style_a: str, style_b: str) -> str:
    """Get a description of how two styles typically match up."""
    matchups = {
        ("Striker", "Grappler"): "Classic striker vs grappler matchup. The striker needs to keep distance and avoid takedowns. The grappler will try to close distance and take the fight to the ground.",
        ("Striker", "Striker"): "Stand-up war likely. Striking accuracy and defense will be key differentiators.",
        ("Grappler", "Grappler"): "Wrestling-heavy fight expected. The better chain wrestler or submission artist has the edge.",
        ("Aggressive", "Passive"): "Aggressive fighter will push the pace. The counter-striker needs to time shots and avoid getting overwhelmed.",
        ("Aggressive", "Aggressive"): "Fireworks expected! Both fighters will push forward, likely resulting in an exciting but risky fight for both.",
        ("All-Rounder", "Striker"): "The all-rounder can choose where the fight goes. They may try to exploit the striker's takedown defense.",
        ("All-Rounder", "Grappler"): "The all-rounder has options to keep it standing or match grappling. Versatility is the advantage.",
        ("All-Rounder", "All-Rounder"): "Evenly matched in style versatility. This fight will likely be decided by who executes their game plan better.",
    }
    key1 = (style_a, style_b)
    key2 = (style_b, style_a)
    if key1 in matchups:
        return matchups[key1]
    elif key2 in matchups:
        return matchups[key2]
    return f"{style_a} vs {style_b}: Unique style matchup with multiple possible game plans."
