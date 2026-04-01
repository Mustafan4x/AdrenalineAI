#!/usr/bin/env python3
"""
UFC Data Collection Script
Run this first to scrape and cache all fighter data and fight history.
Usage: python collect_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import build_fighter_database, build_fight_history_database, get_upcoming_card


def main():
    print("=" * 60)
    print("  UFC Data Collection Script")
    print("=" * 60)

    print("\n[1/3] Building fighter database...")
    print("  This scrapes all fighter profiles from UFCStats.com")
    print("  (This may take 15-30 minutes on first run)\n")
    fighters_df = build_fighter_database(use_cache=True)
    print(f"  -> {len(fighters_df)} fighters loaded\n")

    print("[2/3] Building fight history database...")
    print("  This scrapes recent completed events for training data\n")
    fights_df = build_fight_history_database(use_cache=True)
    print(f"  -> {len(fights_df)} fights loaded\n")

    print("[3/3] Fetching upcoming events...")
    try:
        upcoming = get_upcoming_card()
        if upcoming and upcoming.get("fights"):
            print(f"  -> Next event: {upcoming.get('event', 'Unknown')}")
            print(f"  -> Date: {upcoming.get('date', 'TBD')}")
            print(f"  -> {len(upcoming['fights'])} fights on the card")
        else:
            print("  -> No upcoming events found")
    except Exception as e:
        print(f"  -> Could not fetch upcoming events: {e}")

    print("\n" + "=" * 60)
    print("  Data collection complete!")
    print(f"  Cached files in: {os.path.join(os.path.dirname(__file__), 'data')}")
    print("\n  To start the app, run:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
