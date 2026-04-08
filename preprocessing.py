"""
UFC Data Preprocessing Pipeline
Handles data cleaning, feature engineering, combat style classification,
and difference matrix creation for matchup prediction.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

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


def _build_fighter_fights_index(fights_df: pd.DataFrame) -> dict:
    """Build a lookup: fighter_name -> list of fight row indices (in df order)."""
    index = {}
    for idx, fight in fights_df.iterrows():
        for name in [fight.get("fighter_a", ""), fight.get("fighter_b", "")]:
            if name:
                index.setdefault(name, []).append(idx)
    return index


def compute_win_streak(fights_df: pd.DataFrame, fighter_name: str, fight_index: dict = None) -> int:
    """Compute current win streak for a fighter from fight history."""
    if fight_index:
        indices = fight_index.get(fighter_name, [])
        fighter_fights = fights_df.loc[indices]
    else:
        fighter_fights = fights_df[
            (fights_df["fighter_a"] == fighter_name) | (fights_df["fighter_b"] == fighter_name)
        ]

    streak = 0
    for _, fight in fighter_fights.iterrows():
        if fight.get("winner") == fighter_name:
            streak += 1
        else:
            break
    return streak


def compute_recent_form(fights_df: pd.DataFrame, fighter_name: str, fight_index: dict = None, n: int = 5) -> dict:
    """Compute stats weighted toward a fighter's last N fights.

    Returns recent win rate, recent finish rate, and a momentum score
    that captures whether the fighter is trending up or down.
    """
    if fight_index:
        indices = fight_index.get(fighter_name, [])
        fighter_fights = fights_df.loc[indices]
    else:
        fighter_fights = fights_df[
            (fights_df["fighter_a"] == fighter_name) | (fights_df["fighter_b"] == fighter_name)
        ]

    recent = fighter_fights.head(n)
    if len(recent) == 0:
        return {"recent_win_rate": 0.5, "recent_finish_rate": 0.0, "momentum": 0.0}

    wins = 0
    finishes = 0
    momentum = 0.0
    for i, (_, fight) in enumerate(recent.iterrows()):
        weight = (n - i) / n
        won = fight.get("winner") == fighter_name
        if won:
            wins += 1
            momentum += weight
            method = " ".join(str(fight.get("method", "")).split()).upper()
            if method.startswith("KO") or method.startswith("SUB"):
                finishes += 1
        else:
            momentum -= weight

    total = len(recent)
    return {
        "recent_win_rate": wins / total,
        "recent_finish_rate": finishes / total,
        "momentum": momentum / n,
    }


def compute_opponent_quality(fights_df: pd.DataFrame, fighter_lookup: dict, fighter_name: str, fight_index: dict = None, n: int = 5) -> float:
    """Compute average win rate of a fighter's recent opponents.

    Uses fights_df to compute opponent win rates from fight history,
    not from the static fighter lookup (which would leak future data).
    """
    if fight_index:
        indices = fight_index.get(fighter_name, [])
        fighter_fights = fights_df.loc[indices].head(n)
    else:
        fighter_fights = fights_df[
            (fights_df["fighter_a"] == fighter_name) | (fights_df["fighter_b"] == fighter_name)
        ].head(n)

    if len(fighter_fights) == 0:
        return 0.5

    opponent_win_rates = []
    for _, fight in fighter_fights.iterrows():
        opponent = fight["fighter_b"] if fight["fighter_a"] == fighter_name else fight["fighter_a"]
        # Compute opponent's win rate from the available fights_df (temporal-safe)
        opp_fights = fights_df[
            (fights_df["fighter_a"] == opponent) | (fights_df["fighter_b"] == opponent)
        ]
        if len(opp_fights) > 0:
            opp_wins = (opp_fights["winner"] == opponent).sum()
            opponent_win_rates.append(opp_wins / len(opp_fights))

    return np.mean(opponent_win_rates) if opponent_win_rates else 0.5


def compute_finish_rates(fights_df: pd.DataFrame, fighter_name: str) -> dict:
    """Compute KO rate, submission rate, and decision rate from fight history."""
    fighter_wins = fights_df[fights_df["winner"] == fighter_name]

    total_wins = len(fighter_wins)
    if total_wins == 0:
        return {"ko_rate": 0.0, "sub_rate": 0.0, "dec_rate": 0.0}

    ko_wins = 0
    sub_wins = 0
    dec_wins = 0
    for _, fight in fighter_wins.iterrows():
        method = " ".join(str(fight.get("method", "")).split()).upper()
        if method.startswith("KO") or method.startswith("TKO"):
            ko_wins += 1
        elif method.startswith("SUB"):
            sub_wins += 1
        else:
            dec_wins += 1

    return {
        "ko_rate": ko_wins / total_wins,
        "sub_rate": sub_wins / total_wins,
        "dec_rate": dec_wins / total_wins,
    }


# Weight class encoding (heavier = higher value, women's divisions offset)
WEIGHT_CLASS_ORDER = {
    "Women's Strawweight": 1, "Women's Flyweight": 2, "Women's Bantamweight": 3,
    "Women's Featherweight": 4, "Flyweight": 5, "Bantamweight": 6,
    "Featherweight": 7, "Lightweight": 8, "Welterweight": 9,
    "Middleweight": 10, "Light Heavyweight": 11, "Heavyweight": 12,
    "Catch Weight": 7,
}


def impute_reach(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing reach data using height correlation."""
    df = df.copy()
    mask = df["reach_cm"].isna() & df["height_cm"].notna()
    if mask.any() and df["reach_cm"].notna().sum() > 10:
        valid = df[df["reach_cm"].notna() & df["height_cm"].notna()]
        if len(valid) > 0:
            ratio = (valid["reach_cm"] / valid["height_cm"]).median()
            df.loc[mask, "reach_cm"] = df.loc[mask, "height_cm"] * ratio
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
        # Recent form features
        "recent_win_rate": fighter.get("recent_win_rate", 0.5),
        "recent_finish_rate": fighter.get("recent_finish_rate", 0.0),
        "momentum": fighter.get("momentum", 0.0),
        # Opponent quality
        "opponent_quality": fighter.get("opponent_quality", 0.5),
        # Finish rates
        "ko_rate": fighter.get("ko_rate", 0.0),
        "sub_rate": fighter.get("sub_rate", 0.0),
        "dec_rate": fighter.get("dec_rate", 0.0),
        # Weight class tier
        "weight_class_tier": fighter.get("weight_class_tier", 7),
        # Betting odds implied probability (0.5 = no odds available)
        "implied_prob": fighter.get("implied_prob", 0.5),
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
    "wins", "losses", "total_fights", "win_rate",
    "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
    "win_streak",
    "recent_win_rate", "recent_finish_rate", "momentum",
    "opponent_quality",
    "ko_rate", "sub_rate", "dec_rate",
    "weight_class_tier",
    "implied_prob",
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


def enrich_fighters(fighters_df: pd.DataFrame, fights_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (recent form, opponent quality, finish rates) to fighter data.

    Uses all available fights — intended for live prediction, not training.
    """
    fighters_df = fighters_df.copy()
    fight_index = _build_fighter_fights_index(fights_df)

    # Build a lookup for opponent quality computation
    fighter_lookup = {}
    for _, row in fighters_df.iterrows():
        name = row.get("name", "").strip()
        if name:
            fighter_lookup[name.lower()] = row

    # Pre-compute weight class tier
    fighter_weight_classes = {}
    for _, fight in fights_df.iterrows():
        wc = fight.get("weight_class", "")
        for name in [fight.get("fighter_a", ""), fight.get("fighter_b", "")]:
            if name:
                fighter_weight_classes.setdefault(name, []).append(wc)

    for idx, row in fighters_df.iterrows():
        name = row.get("name", "")

        fighters_df.at[idx, "win_streak"] = compute_win_streak(fights_df, name, fight_index)

        form = compute_recent_form(fights_df, name, fight_index)
        fighters_df.at[idx, "recent_win_rate"] = form["recent_win_rate"]
        fighters_df.at[idx, "recent_finish_rate"] = form["recent_finish_rate"]
        fighters_df.at[idx, "momentum"] = form["momentum"]

        fighters_df.at[idx, "opponent_quality"] = compute_opponent_quality(
            fights_df, fighter_lookup, name, fight_index
        )

        rates = compute_finish_rates(fights_df, name)
        fighters_df.at[idx, "ko_rate"] = rates["ko_rate"]
        fighters_df.at[idx, "sub_rate"] = rates["sub_rate"]
        fighters_df.at[idx, "dec_rate"] = rates["dec_rate"]

        wc_list = fighter_weight_classes.get(name, [])
        if wc_list:
            most_common = max(set(wc_list), key=wc_list.count)
            fighters_df.at[idx, "weight_class_tier"] = WEIGHT_CLASS_ORDER.get(most_common, 7)
        else:
            fighters_df.at[idx, "weight_class_tier"] = 7

    return fighters_df


def _compute_temporal_features(fighter_name: str, fights_before: pd.DataFrame, fighter_lookup: dict, fight_index: dict) -> dict:
    """Compute temporal features using only fights that happened before a given fight.

    This prevents data leakage during training — we only use past information.
    """
    form = compute_recent_form(fights_before, fighter_name, fight_index)
    rates = compute_finish_rates(fights_before, fighter_name)
    opp_quality = compute_opponent_quality(fights_before, fighter_lookup, fighter_name, fight_index)
    streak = compute_win_streak(fights_before, fighter_name, fight_index)

    return {
        "win_streak": streak,
        "recent_win_rate": form["recent_win_rate"],
        "recent_finish_rate": form["recent_finish_rate"],
        "momentum": form["momentum"],
        "opponent_quality": opp_quality,
        "ko_rate": rates["ko_rate"],
        "sub_rate": rates["sub_rate"],
        "dec_rate": rates["dec_rate"],
    }


def build_training_data(fighters_df: pd.DataFrame, fights_df: pd.DataFrame, odds_df: pd.DataFrame = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build training data from historical fights.
    Returns X (difference matrices) and y (1 if fighter_a won, 0 if fighter_b won).

    Temporal features are computed per-fight using only prior fights to prevent
    data leakage. Fighter order is randomized to prevent positional bias.

    Expects fighters_df to already be cleaned via clean_fighter_data().
    """

    # Build a lookup dict for base stats
    fighter_lookup = {}
    for _, row in fighters_df.iterrows():
        name = row.get("name", "").strip()
        if name:
            fighter_lookup[name.lower()] = row

    n_fights = len(fights_df)
    fight_rows = list(fights_df.iterrows())

    # Pre-compute weight class tiers per fight position (temporal-safe)
    # Build cumulative weight class info from the end backward
    fighter_wc_cumulative = [None] * (n_fights + 1)
    fighter_wc_cumulative[n_fights] = {}
    for j in range(n_fights - 1, -1, -1):
        _, fight_row = fight_rows[j]
        prev = {k: list(v) for k, v in fighter_wc_cumulative[j + 1].items()}
        wc = fight_row.get("weight_class", "")
        for name in [fight_row.get("fighter_a", ""), fight_row.get("fighter_b", "")]:
            if name:
                prev.setdefault(name, []).append(wc)
        fighter_wc_cumulative[j] = prev

    # Build odds lookup: (fighter_a_norm, fighter_b_norm) -> (prob_a, prob_b)
    odds_lookup = {}
    if odds_df is not None and len(odds_df) > 0:
        for _, row in odds_df.iterrows():
            fa = str(row.get("fighter_a", "")).strip().lower()
            fb = str(row.get("fighter_b", "")).strip().lower()
            odds_lookup[(fa, fb)] = (row["implied_prob_a"], row["implied_prob_b"])
            odds_lookup[(fb, fa)] = (row["implied_prob_b"], row["implied_prob_a"])

    # Pre-build fight indices for all positions (O(N) instead of O(N^2))
    cumulative_indices = [None] * (n_fights + 1)
    cumulative_indices[n_fights] = {}  # empty index for the last fight
    for j in range(n_fights - 1, -1, -1):
        idx, fight_row = fight_rows[j]
        prev = {k: list(v) for k, v in cumulative_indices[j + 1].items()}
        for name in [fight_row.get("fighter_a", ""), fight_row.get("fighter_b", "")]:
            if name:
                prev.setdefault(name, []).append(idx)
        cumulative_indices[j] = prev

    rng = np.random.RandomState(42)

    X_list = []
    y_list = []
    fight_ids = []

    for i, (_, fight) in enumerate(fights_df.iterrows()):
        fa_name = str(fight.get("fighter_a", "")).strip()
        fb_name = str(fight.get("fighter_b", "")).strip()
        winner = fight.get("winner")

        if not winner or fa_name.lower() not in fighter_lookup or fb_name.lower() not in fighter_lookup:
            continue

        # Use only fights before this one to compute temporal features
        fights_before = fights_df.iloc[i + 1:]  # fights are ordered most-recent-first
        fight_idx_before = cumulative_indices[i + 1]

        fa = fighter_lookup[fa_name.lower()].copy()
        fb = fighter_lookup[fb_name.lower()].copy()

        # Compute temporal features from only prior fights
        fa_temporal = _compute_temporal_features(fa_name, fights_before, fighter_lookup, fight_idx_before)
        fb_temporal = _compute_temporal_features(fb_name, fights_before, fighter_lookup, fight_idx_before)

        for key, val in fa_temporal.items():
            fa[key] = val
        for key, val in fb_temporal.items():
            fb[key] = val

        # Set weight class tier from only prior fights
        wc_before = fighter_wc_cumulative[i + 1]
        wc_a = wc_before.get(fa_name, [])
        wc_b = wc_before.get(fb_name, [])
        fa["weight_class_tier"] = WEIGHT_CLASS_ORDER.get(max(set(wc_a), key=wc_a.count), 7) if wc_a else 7
        fb["weight_class_tier"] = WEIGHT_CLASS_ORDER.get(max(set(wc_b), key=wc_b.count), 7) if wc_b else 7

        # Set betting odds implied probability
        odds_key = (fa_name.lower(), fb_name.lower())
        if odds_key in odds_lookup:
            fa["implied_prob"] = odds_lookup[odds_key][0]
            fb["implied_prob"] = odds_lookup[odds_key][1]
        else:
            fa["implied_prob"] = 0.5
            fb["implied_prob"] = 0.5

        winner_lower = str(winner).strip().lower()

        # Generate both orderings to fully eliminate positional bias and double data
        for swap in [False, True]:
            if swap:
                a, b = fb, fa
                a_name, b_name = fb_name.lower(), fa_name.lower()
            else:
                a, b = fa, fb
                a_name, b_name = fa_name.lower(), fb_name.lower()

            diff = create_difference_matrix(a, b)
            label = 1 if winner_lower == a_name else 0

            X_list.append(diff.flatten())
            y_list.append(label)
            fight_ids.append(i)  # same fight ID for both orderings

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(fight_ids)

    return X, y, groups


def get_fighter_by_name(fighters_df: pd.DataFrame, name: str) -> pd.Series | None:
    """Look up a fighter by name (case-insensitive, exact match only)."""
    name_lower = name.strip().lower()
    exact = fighters_df[fighters_df["name"].str.lower().str.strip() == name_lower]
    if len(exact) > 0:
        return exact.iloc[0]
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
