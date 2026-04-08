"""
UFC Matchup Prediction Model
Gradient Boosting classifier with feature importance and prediction explanations.
"""

import os

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

from preprocessing import (
    FEATURE_COLUMNS,
    build_training_data,
    clean_fighter_data,
    create_difference_matrix,
    get_fighter_by_name,
    get_style_matchup_description,
    enrich_fighters,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class UFCPredictor:
    """UFC fight outcome prediction model."""

    def __init__(self):
        self.model = None
        self.fighters_df = None
        self.fights_df = None
        self.is_trained = False
        self.cv_scores = None

    def load_data(self, fighters_df: pd.DataFrame, fights_df: pd.DataFrame):
        """Load fighter and fight data."""
        self.fighters_df = clean_fighter_data(fighters_df)
        self.fights_df = fights_df
        # Enrich with recent form, opponent quality, finish rates, weight class
        self.fighters_df = enrich_fighters(self.fighters_df, fights_df)

    def train(self, X: np.ndarray = None, y: np.ndarray = None):
        """Train the XGBoost model."""
        if X is None or y is None:
            if self.fighters_df is None or self.fights_df is None:
                raise ValueError("No data loaded. Call load_data() first.")
            X, y = build_training_data(self.fighters_df, self.fights_df)

        if len(X) == 0:
            raise ValueError("No valid training samples found.")

        print(f"Training on {len(X)} fight samples with {X.shape[1]} features...")

        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=3,
            random_state=42,
            eval_metric="logloss",
        )

        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring="accuracy")
        print(f"Cross-validation accuracy: {self.cv_scores.mean():.3f} (+/- {self.cv_scores.std():.3f})")

        # Final fit on all data
        self.model.fit(X, y)
        self.is_trained = True

    def save(self, path: str = None):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained yet.")
        path = path or os.path.join(MODEL_DIR, "ufc_model.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "cv_scores": self.cv_scores,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str = None):
        """Load a trained model."""
        path = path or os.path.join(MODEL_DIR, "ufc_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        data = joblib.load(path)
        self.model = data["model"]
        self.cv_scores = data.get("cv_scores")
        self.is_trained = True
        print("Model loaded successfully.")

    def predict_matchup(self, fighter_a_name: str, fighter_b_name: str) -> dict:
        """
        Predict the outcome of a matchup between two fighters.
        Returns prediction details including winner, confidence, and reasoning.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        if self.fighters_df is None:
            raise ValueError("No fighter data loaded.")

        # Look up fighters
        fa = get_fighter_by_name(self.fighters_df, fighter_a_name)
        fb = get_fighter_by_name(self.fighters_df, fighter_b_name)

        if fa is None:
            raise ValueError(f"Fighter not found: {fighter_a_name}")
        if fb is None:
            raise ValueError(f"Fighter not found: {fighter_b_name}")

        # Create difference matrix
        diff = create_difference_matrix(fa, fb)

        # Predict
        prediction = self.model.predict(diff)[0]
        probabilities = self.model.predict_proba(diff)[0]

        winner = fa["name"] if prediction == 1 else fb["name"]
        loser = fb["name"] if prediction == 1 else fa["name"]
        confidence = max(probabilities) * 100

        # Feature importance for this prediction
        feature_importances = self.model.feature_importances_
        feature_impacts = {}
        diff_flat = diff.flatten()
        for i, col in enumerate(FEATURE_COLUMNS):
            feature_impacts[col] = {
                "importance": float(feature_importances[i]),
                "difference": float(diff_flat[i]),
                "favors": fa["name"] if diff_flat[i] > 0 else fb["name"] if diff_flat[i] < 0 else "Even"
            }

        # Sort by importance
        sorted_features = sorted(feature_impacts.items(), key=lambda x: x[1]["importance"], reverse=True)

        # Generate reasons
        reasons = self._generate_reasons(fa, fb, sorted_features, winner)

        # Style matchup analysis
        style_a = fa.get("combat_style", "All-Rounder")
        style_b = fb.get("combat_style", "All-Rounder")
        style_analysis = get_style_matchup_description(style_a, style_b)

        return {
            "fighter_a": self._fighter_summary(fa),
            "fighter_b": self._fighter_summary(fb),
            "predicted_winner": winner,
            "predicted_loser": loser,
            "confidence": round(float(confidence), 1),
            "probability_a": round(float(probabilities[1]) * 100, 1),
            "probability_b": round(float(probabilities[0]) * 100, 1),
            "feature_importances": dict(sorted_features[:15]),
            "reasons_winner": reasons["winner"],
            "reasons_loser": reasons["loser"],
            "style_matchup": style_analysis,
            "style_a": style_a,
            "style_b": style_b,
            "diff_vector": diff_flat.tolist(),
            "feature_columns": FEATURE_COLUMNS,
        }

    def _fighter_summary(self, fighter: pd.Series) -> dict:
        """Create a summary dict for a fighter."""
        return {
            "name": fighter.get("name", "Unknown"),
            "height_cm": fighter.get("height_cm"),
            "weight_kg": fighter.get("weight_kg"),
            "reach_cm": fighter.get("reach_cm"),
            "age": fighter.get("age"),
            "record": f"{int(fighter.get('wins', 0))}-{int(fighter.get('losses', 0))}-{int(fighter.get('draws', 0))}",
            "wins": int(fighter.get("wins", 0)),
            "losses": int(fighter.get("losses", 0)),
            "draws": int(fighter.get("draws", 0)),
            "win_rate": fighter.get("win_rate"),
            "slpm": fighter.get("slpm"),
            "str_acc": fighter.get("str_acc"),
            "sapm": fighter.get("sapm"),
            "str_def": fighter.get("str_def"),
            "td_avg": fighter.get("td_avg"),
            "td_acc": fighter.get("td_acc"),
            "td_def": fighter.get("td_def"),
            "sub_avg": fighter.get("sub_avg"),
            "combat_style": fighter.get("combat_style"),
            "stance": fighter.get("stance"),
            "win_streak": fighter.get("win_streak", 0),
        }

    def _generate_reasons(self, fa: pd.Series, fb: pd.Series, sorted_features: list, winner: str) -> dict:
        """Generate human-readable reasons for the prediction."""
        winner_reasons = []
        loser_reasons = []

        fa_name = fa["name"]
        fb_name = fb["name"]
        winner_is_a = winner == fa_name

        feature_labels = {
            "height_cm": "height advantage",
            "weight_kg": "weight advantage",
            "reach_cm": "reach/wingspan advantage",
            "age": "youth/experience advantage",
            "wins": "more career wins",
            "losses": "more career losses (battle-tested)",
            "total_fights": "more experience (total fights)",
            "win_rate": "higher win rate",
            "slpm": "higher striking output (sig. strikes per minute)",
            "str_acc": "better striking accuracy",
            "sapm": "higher strike absorption rate",
            "str_def": "better striking defense",
            "td_avg": "more takedown attempts per fight",
            "td_acc": "better takedown accuracy",
            "td_def": "better takedown defense",
            "sub_avg": "more submission attempts",
            "win_streak": "longer current win streak",
            "recent_win_rate": "better recent form (last 5 fights)",
            "recent_finish_rate": "higher recent finish rate",
            "momentum": "stronger momentum (trending upward)",
            "opponent_quality": "faced tougher opponents",
            "ko_rate": "higher knockout finish rate",
            "sub_rate": "higher submission finish rate",
            "dec_rate": "more decision wins",
            "weight_class_tier": "weight class factor",
            "style_striker": "striking style",
            "style_grappler": "grappling style",
            "style_aggressive": "aggressive approach",
            "style_passive": "passive/counter-fighting approach",
            "style_all_rounder": "well-rounded skill set",
        }

        for feat_name, feat_data in sorted_features[:8]:
            imp = feat_data["importance"]
            diff = feat_data["difference"]
            if imp < 0.01:
                continue

            label = feature_labels.get(feat_name, feat_name)
            favors = feat_data["favors"]

            if favors == winner:
                winner_reasons.append(f"{winner} has {label} ({feat_name}: {diff:+.2f} diff)")
            elif favors != "Even":
                loser_name = fb_name if winner_is_a else fa_name
                loser_reasons.append(f"{loser_name} has {label} ({feat_name}: {abs(diff):.2f})")

        if not winner_reasons:
            winner_reasons.append(f"{winner} has a slight overall statistical edge")

        return {"winner": winner_reasons, "loser": loser_reasons}

    def predict_card(self, card: list[dict]) -> list[dict]:
        """Predict outcomes for an entire fight card."""
        results = []
        for fight in card:
            try:
                prediction = self.predict_matchup(fight["fighter_a"], fight["fighter_b"])
                prediction["weight_class"] = fight.get("weight_class", "")
                results.append(prediction)
            except ValueError as e:
                results.append({
                    "fighter_a": {"name": fight["fighter_a"]},
                    "fighter_b": {"name": fight["fighter_b"]},
                    "error": str(e),
                    "weight_class": fight.get("weight_class", ""),
                })
        return results
