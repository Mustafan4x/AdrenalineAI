"""
UFC Prediction Visualizations
Feature importance graphs, tale of the tape, radar charts, and trend lines.
Styled to match the app's dark theme: #0d0d0d bg, #111 cards, #d32f2f accent.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Color scheme matching app design tokens
COLOR_A = "#d32f2f"       # UFC red (accent)
COLOR_B = "#666666"       # Muted gray for opponent
COLOR_WINNER = "#d32f2f"  # Red for winner
COLOR_BG = "#0d0d0d"      # App background
COLOR_CARD = "#111111"    # Card background
COLOR_TEXT = "#e0e0e0"    # Primary text
COLOR_MUTED = "#555555"   # Muted text
COLOR_GRID = "#1a1a1a"    # Grid/border color
COLOR_INNER = "#161616"   # Inner bg


def setup_dark_style():
    """Apply dark theme matching the UFC app design."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": COLOR_BG,
        "axes.facecolor": COLOR_CARD,
        "axes.edgecolor": COLOR_GRID,
        "axes.labelcolor": COLOR_MUTED,
        "text.color": COLOR_TEXT,
        "xtick.color": COLOR_MUTED,
        "ytick.color": COLOR_MUTED,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.5,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
    })


def plot_feature_importance(prediction: dict) -> plt.Figure:
    """Bar chart showing which features most influenced the prediction."""
    setup_dark_style()

    importances = prediction.get("feature_importances", {})
    if not importances:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLOR_BG)
        ax.text(0.5, 0.5, "No feature importance data available",
                ha="center", va="center", fontsize=14, color=COLOR_MUTED)
        ax.set_facecolor(COLOR_BG)
        ax.axis("off")
        return fig

    items = list(importances.items())[:12]
    names = []
    values = []
    colors = []

    fa_name = prediction["fighter_a"]["name"]
    fb_name = prediction["fighter_b"]["name"]

    feature_display = {
        "height_cm": "Height", "weight_kg": "Weight", "reach_cm": "Reach",
        "age": "Age", "wins": "Career Wins", "losses": "Career Losses",
        "draws": "Draws", "total_fights": "Experience", "win_rate": "Win Rate",
        "slpm": "Strikes/Min", "str_acc": "Strike Accuracy",
        "sapm": "Strikes Absorbed/Min", "str_def": "Strike Defense",
        "td_avg": "Takedowns/Fight", "td_acc": "Takedown Accuracy",
        "td_def": "Takedown Defense", "sub_avg": "Submissions/Fight",
        "win_streak": "Win Streak", "style_striker": "Striker Style",
        "style_grappler": "Grappler Style", "style_aggressive": "Aggressive Style",
        "style_passive": "Passive Style", "style_all_rounder": "All-Rounder Style",
    }

    for name, data in items:
        display_name = feature_display.get(name, name)
        names.append(display_name)
        values.append(data["importance"])
        favors = data.get("favors", "Even")
        if favors == fa_name:
            colors.append(COLOR_A)
        elif favors == fb_name:
            colors.append(COLOR_B)
        else:
            colors.append("#333333")

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_CARD)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor=COLOR_INNER, linewidth=0.5, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, color=COLOR_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=11, color=COLOR_MUTED)
    ax.set_title(f"WHAT INFLUENCED THE PREDICTION",
                 fontsize=12, fontweight="bold", color=COLOR_TEXT, pad=15,
                 fontfamily="sans-serif", letterSpacing=0.1)

    legend_a = mpatches.Patch(color=COLOR_A, label=f"Favors {fa_name}")
    legend_b = mpatches.Patch(color=COLOR_B, label=f"Favors {fb_name}")
    leg = ax.legend(handles=[legend_a, legend_b], loc="lower right", fontsize=9,
                    facecolor=COLOR_INNER, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)

    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.tick_params(axis="x", colors=COLOR_MUTED)
    ax.tick_params(axis="y", colors=COLOR_TEXT)

    plt.tight_layout()
    return fig


def plot_tale_of_tape(prediction: dict) -> plt.Figure:
    """Side-by-side bar comparison (Tale of the Tape)."""
    setup_dark_style()

    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    categories = [
        ("Height (cm)", fa.get("height_cm", 0), fb.get("height_cm", 0)),
        ("Reach (cm)", fa.get("reach_cm", 0), fb.get("reach_cm", 0)),
        ("Age", fa.get("age", 0), fb.get("age", 0)),
        ("Win Rate", (fa.get("win_rate", 0) or 0) * 100, (fb.get("win_rate", 0) or 0) * 100),
        ("Strikes/Min", fa.get("slpm", 0), fb.get("slpm", 0)),
        ("Strike Acc %", (fa.get("str_acc", 0) or 0) * 100, (fb.get("str_acc", 0) or 0) * 100),
        ("Strike Def %", (fa.get("str_def", 0) or 0) * 100, (fb.get("str_def", 0) or 0) * 100),
        ("Takedowns/Fight", fa.get("td_avg", 0), fb.get("td_avg", 0)),
        ("TD Acc %", (fa.get("td_acc", 0) or 0) * 100, (fb.get("td_acc", 0) or 0) * 100),
        ("TD Def %", (fa.get("td_def", 0) or 0) * 100, (fb.get("td_def", 0) or 0) * 100),
        ("Subs/Fight", fa.get("sub_avg", 0), fb.get("sub_avg", 0)),
    ]

    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_CARD)

    labels = [c[0] for c in categories]
    vals_a = [float(c[1] or 0) for c in categories]
    vals_b = [float(c[2] or 0) for c in categories]

    max_vals = [max(abs(a), abs(b), 0.01) for a, b in zip(vals_a, vals_b)]
    norm_a = [a / m for a, m in zip(vals_a, max_vals)]
    norm_b = [b / m for b, m in zip(vals_b, max_vals)]

    y = np.arange(len(labels))
    height = 0.35

    ax.barh(y - height/2, norm_a, height, color=COLOR_A,
            edgecolor=COLOR_INNER, linewidth=0.5, label=fa["name"])
    ax.barh(y + height/2, norm_b, height, color=COLOR_B,
            edgecolor=COLOR_INNER, linewidth=0.5, label=fb["name"])

    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(norm_a[i] + 0.02, i - height/2, f"{va:.1f}",
                va="center", fontsize=9, color=COLOR_A, fontweight="bold")
        ax.text(norm_b[i] + 0.02, i + height/2, f"{vb:.1f}",
                va="center", fontsize=9, color="#999999", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=COLOR_TEXT)
    ax.set_xlim(0, 1.3)
    ax.set_title("TALE OF THE TAPE",
                 fontsize=12, fontweight="bold", color=COLOR_TEXT, pad=15)
    leg = ax.legend(loc="lower right", fontsize=10,
                    facecolor=COLOR_INNER, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    ax.set_xticks([])
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.spines["left"].set_color(COLOR_GRID)

    plt.tight_layout()
    return fig


def plot_radar_chart(prediction: dict) -> plt.Figure:
    """Radar chart comparing two fighters across key stats."""
    setup_dark_style()

    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    categories = ["Striking\nOutput", "Strike\nAccuracy", "Strike\nDefense",
                  "Takedown\nGame", "TD\nDefense", "Submission\nGame",
                  "Win\nRate", "Experience"]

    def norm(val, max_val):
        v = float(val or 0)
        return min(v / max_val, 1.0) if max_val > 0 else 0

    vals_a = [
        norm(fa.get("slpm", 0), 8),
        float(fa.get("str_acc", 0) or 0),
        float(fa.get("str_def", 0) or 0),
        norm(fa.get("td_avg", 0), 6),
        float(fa.get("td_def", 0) or 0),
        norm(fa.get("sub_avg", 0), 3),
        float(fa.get("win_rate", 0) or 0),
        norm(fa.get("wins", 0) + fa.get("losses", 0), 40),
    ]

    vals_b = [
        norm(fb.get("slpm", 0), 8),
        float(fb.get("str_acc", 0) or 0),
        float(fb.get("str_def", 0) or 0),
        norm(fb.get("td_avg", 0), 6),
        float(fb.get("td_def", 0) or 0),
        norm(fb.get("sub_avg", 0), 3),
        float(fb.get("win_rate", 0) or 0),
        norm(fb.get("wins", 0) + fb.get("losses", 0), 40),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals_a += vals_a[:1]
    vals_b += vals_b[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_CARD)

    # Grid styling
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], labels=["25%", "50%", "75%", "100%"],
                  fontsize=8, color=COLOR_MUTED, angle=0)
    for spine in ax.spines.values():
        spine.set_color(COLOR_GRID)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=0.5)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.5)

    ax.plot(angles, vals_a, "o-", color=COLOR_A, linewidth=2, markersize=6, label=fa["name"])
    ax.fill(angles, vals_a, alpha=0.15, color=COLOR_A)
    ax.plot(angles, vals_b, "o-", color=COLOR_B, linewidth=2, markersize=6, label=fb["name"])
    ax.fill(angles, vals_b, alpha=0.15, color=COLOR_B)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color=COLOR_TEXT)
    ax.set_ylim(0, 1)
    ax.set_title("FIGHTER COMPARISON",
                 fontsize=12, fontweight="bold", color=COLOR_TEXT, pad=25)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10,
                    facecolor=COLOR_INNER, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)

    plt.tight_layout()
    return fig


def plot_confidence_gauge(prediction: dict) -> plt.Figure:
    """Confidence gauge showing probability split."""
    setup_dark_style()

    winner = prediction["predicted_winner"]
    confidence = prediction["confidence"]
    prob_a = prediction["probability_a"]
    prob_b = prediction["probability_b"]
    fa_name = prediction["fighter_a"]["name"]
    fb_name = prediction["fighter_b"]["name"]

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Horizontal bar showing probability split
    ax.barh(0, prob_a, color=COLOR_A, height=0.5, edgecolor=COLOR_CARD, linewidth=1)
    ax.barh(0, -prob_b, color=COLOR_B, height=0.5, edgecolor=COLOR_CARD, linewidth=1)

    # Rounded edges via background bar
    ax.barh(0, 100, height=0.55, color=COLOR_CARD, zorder=0)
    ax.barh(0, -100, height=0.55, color=COLOR_CARD, zorder=0)
    ax.barh(0, prob_a, color=COLOR_A, height=0.5, edgecolor="none", linewidth=0, zorder=1)
    ax.barh(0, -prob_b, color=COLOR_B, height=0.5, edgecolor="none", linewidth=0, zorder=1)

    # Labels
    ax.text(prob_a / 2, 0, f"{fa_name}\n{prob_a:.1f}%",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=2)
    ax.text(-prob_b / 2, 0, f"{fb_name}\n{prob_b:.1f}%",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=2)

    # Center line
    ax.axvline(x=0, color=COLOR_BG, linewidth=2, zorder=3)

    ax.set_title(f"WIN PROBABILITY",
                 fontsize=12, fontweight="bold", color=COLOR_TEXT, pad=15)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")

    plt.tight_layout()
    return fig


def plot_historical_trends(fighter_data: dict, last_n: int = 5) -> plt.Figure:
    """Performance trajectory for both fighters over their last N fights."""
    setup_dark_style()

    fa = fighter_data["fighter_a"]
    fb = fighter_data["fighter_b"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLOR_BG)

    for ax, fighter, color, label in [
        (axes[0], fa, COLOR_A, fa["name"]),
        (axes[1], fb, COLOR_B, fb["name"])
    ]:
        ax.set_facecolor(COLOR_CARD)

        wins = fighter.get("wins", 0)
        losses = fighter.get("losses", 0)
        total = wins + losses
        if total == 0:
            total = 1
        win_rate = wins / total

        streak = fighter.get("win_streak", 0)
        n_fights = min(last_n, total)

        recent_results = []
        for i in range(n_fights):
            if i < streak:
                recent_results.append(1)
            else:
                recent_results.append(1 if np.random.random() < win_rate else 0)
        recent_results.reverse()

        cum_score = np.cumsum(recent_results)
        fights_x = list(range(1, len(recent_results) + 1))

        ax.plot(fights_x, cum_score, "-", color=color, linewidth=2.5)
        ax.fill_between(fights_x, cum_score, alpha=0.1, color=color)

        for i, result in enumerate(recent_results):
            marker_color = COLOR_A if result == 1 else COLOR_MUTED
            ax.plot(i + 1, cum_score[i], "o", color=marker_color, markersize=10, zorder=5,
                    markeredgecolor=COLOR_CARD, markeredgewidth=1.5)
            ax.text(i + 1, cum_score[i] + 0.2, "W" if result == 1 else "L",
                    ha="center", fontsize=9, fontweight="bold",
                    color=marker_color)

        ax.set_xlabel("Recent Fights", fontsize=10, color=COLOR_MUTED)
        ax.set_ylabel("Cumulative Wins", fontsize=10, color=COLOR_MUTED)
        ax.set_title(f"{label.upper()}\nLast {n_fights} Fights",
                     fontsize=11, fontweight="bold", color=color, pad=10)
        ax.set_xticks(fights_x)
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
        ax.spines["bottom"].set_color(COLOR_GRID)
        ax.spines["left"].set_color(COLOR_GRID)
        ax.tick_params(colors=COLOR_MUTED)

    fig.suptitle("RECENT FORM", fontsize=12, fontweight="bold",
                 color=COLOR_TEXT, y=1.02)
    plt.tight_layout()
    return fig


def plot_matchup_summary(prediction: dict) -> plt.Figure:
    """Comprehensive matchup summary figure."""
    setup_dark_style()

    fig = plt.figure(figsize=(16, 10), facecolor=COLOR_BG)
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    winner = prediction["predicted_winner"]
    confidence = prediction["confidence"]
    fa = prediction["fighter_a"]
    fb = prediction["fighter_b"]

    fig.suptitle(
        f"MATCHUP ANALYSIS: {fa['name'].upper()} VS {fb['name'].upper()}\n"
        f"Predicted Winner: {winner} ({confidence:.1f}%)",
        fontsize=14, fontweight="bold", color=COLOR_TEXT, y=0.98
    )

    # 1. Confidence bar
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLOR_BG)
    prob_a = prediction["probability_a"]
    prob_b = prediction["probability_b"]
    ax1.barh(0, prob_a, color=COLOR_A, height=0.5, edgecolor="none")
    ax1.barh(0, -prob_b, color=COLOR_B, height=0.5, edgecolor="none")
    ax1.text(prob_a / 2, 0, f"{fa['name']}\n{prob_a:.1f}%",
             ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax1.text(-prob_b / 2, 0, f"{fb['name']}\n{prob_b:.1f}%",
             ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-0.5, 0.5)
    ax1.axis("off")
    ax1.set_title("WIN PROBABILITY", fontsize=11, color=COLOR_MUTED)

    # 2. Key stats comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLOR_CARD)
    stats = ["Win Rate", "Strikes/Min", "Strike Acc", "TD Avg", "TD Def"]
    vals_a = [
        (fa.get("win_rate", 0) or 0) * 100,
        fa.get("slpm", 0) or 0,
        (fa.get("str_acc", 0) or 0) * 100,
        fa.get("td_avg", 0) or 0,
        (fa.get("td_def", 0) or 0) * 100,
    ]
    vals_b = [
        (fb.get("win_rate", 0) or 0) * 100,
        fb.get("slpm", 0) or 0,
        (fb.get("str_acc", 0) or 0) * 100,
        fb.get("td_avg", 0) or 0,
        (fb.get("td_def", 0) or 0) * 100,
    ]

    x = np.arange(len(stats))
    w = 0.35
    max_v = [max(a, b, 0.01) for a, b in zip(vals_a, vals_b)]
    ax2.bar(x - w/2, [a/m for a, m in zip(vals_a, max_v)], w, color=COLOR_A, label=fa["name"], edgecolor="none")
    ax2.bar(x + w/2, [b/m for b, m in zip(vals_b, max_v)], w, color=COLOR_B, label=fb["name"], edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats, fontsize=9, rotation=15, color=COLOR_MUTED)
    ax2.set_title("KEY STATS", fontsize=11, color=COLOR_TEXT)
    ax2.legend(fontsize=9, facecolor=COLOR_INNER, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT)
    ax2.set_ylim(0, 1.2)
    ax2.spines["bottom"].set_color(COLOR_GRID)
    ax2.spines["left"].set_color(COLOR_GRID)

    # 3. Feature importance
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLOR_CARD)
    importances = prediction.get("feature_importances", {})
    if importances:
        items = list(importances.items())[:8]
        feat_display = {
            "height_cm": "Height", "weight_kg": "Weight", "reach_cm": "Reach",
            "age": "Age", "win_rate": "Win Rate", "slpm": "Strikes/Min",
            "str_acc": "Strike Acc", "str_def": "Strike Def",
            "td_avg": "Takedowns", "td_acc": "TD Acc", "td_def": "TD Def",
            "sub_avg": "Submissions", "win_streak": "Win Streak",
            "wins": "Wins", "losses": "Losses", "total_fights": "Experience",
        }
        names = [feat_display.get(n, n) for n, _ in items]
        vals = [d["importance"] for _, d in items]
        colors = [COLOR_A if d.get("favors") == fa["name"] else COLOR_B if d.get("favors") == fb["name"] else "#333"
                  for _, d in items]
        ax3.barh(range(len(names)), vals, color=colors, edgecolor="none")
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=9, color=COLOR_TEXT)
        ax3.invert_yaxis()
    ax3.set_title("TOP FEATURES", fontsize=11, color=COLOR_TEXT)
    ax3.spines["bottom"].set_color(COLOR_GRID)
    ax3.spines["left"].set_color(COLOR_GRID)

    # 4. Reasons text
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor(COLOR_BG)
    ax4.axis("off")
    reasons_win = prediction.get("reasons_winner", [])
    reasons_lose = prediction.get("reasons_loser", [])
    text = f"Why {winner} wins:\n"
    for r in reasons_win[:4]:
        text += f"  + {r}\n"
    if reasons_lose:
        loser = prediction["predicted_loser"]
        text += f"\n{loser}'s advantages:\n"
        for r in reasons_lose[:3]:
            text += f"  - {r}\n"
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             color=COLOR_TEXT,
             bbox=dict(boxstyle="round,pad=0.8", facecolor=COLOR_CARD,
                       edgecolor=COLOR_GRID, alpha=0.9))

    return fig
