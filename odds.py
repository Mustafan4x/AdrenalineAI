"""
UFC Betting Odds Integration
Scrapes historical odds from BestFightOdds.com and fetches live odds
from The Odds API for upcoming fights.
"""

import os
import re
import time
import json

import requests
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ODDS_CACHE_PATH = os.path.join(DATA_DIR, "odds.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

BFO_BASE = "https://www.bestfightodds.com"


def american_to_implied_prob(odds: int) -> float:
    """Convert American moneyline odds to implied probability (0-1).

    Returns 0.5 (neutral) for invalid odds (0 or None).
    """
    if not odds or odds == 0:
        return 0.5
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _parse_american_odds(text: str) -> int | None:
    """Parse an American odds string like '+108' or '-145' to int."""
    text = text.strip()
    match = re.match(r"([+-]?\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _normalize_name(name: str) -> str:
    """Normalize a fighter name for matching across data sources."""
    return " ".join(name.strip().lower().split())


def scrape_bfo_archive(max_pages: int = 1) -> list[dict]:
    """Scrape event list from BestFightOdds archive page."""
    events = []
    try:
        resp = requests.get(f"{BFO_BASE}/archive", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for row in soup.select("table.content-list tr"):
            date_td = row.select_one("td.content-list-date")
            title_td = row.select_one("td.content-list-title a")
            if date_td and title_td:
                events.append({
                    "date": date_td.text.strip(),
                    "name": title_td.text.strip(),
                    "url": BFO_BASE + title_td["href"],
                })
    except Exception as e:
        print(f"Error fetching BFO archive: {e}")

    return events


def search_bfo_event(query: str) -> str | None:
    """Search BestFightOdds for an event and return its URL.

    Returns the best matching event URL from search results.
    Prefers events with higher IDs (more recent).
    """
    try:
        resp = requests.get(
            f"{BFO_BASE}/search",
            params={"query": query},
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Collect all event links with their IDs
        best_url = None
        best_id = -1

        for a in soup.select("a[href^='/events/']"):
            href = a["href"]
            # Extract event ID from URL like /events/ufc-326-4065
            id_match = re.search(r"-(\d+)$", href)
            if id_match:
                event_id = int(id_match.group(1))
                # Pick the highest ID (most recent event)
                if event_id > best_id:
                    best_id = event_id
                    best_url = BFO_BASE + href

        if best_url:
            return best_url

        # Also check if we landed directly on an event page
        table_div = soup.select_one("div.table-div")
        if table_div:
            return resp.url

    except Exception as e:
        print(f"  BFO search error for '{query}': {e}")

    return None


def _event_search_queries(event_name: str) -> list[str]:
    """Generate search queries from a UFC event name.

    BFO names events differently than UFCStats, so we try multiple strategies.
    """
    queries = []

    # "UFC 326: Holloway vs. Oliveira 2" -> try "UFC 326" first (most specific)
    if ":" in event_name:
        prefix = event_name.split(":")[0].strip()
        # Only use numbered events (UFC 320, UFC 326, etc.)
        if any(c.isdigit() for c in prefix):
            queries.append(prefix)

    # Try full event name
    queries.append(event_name)

    # "UFC Fight Night: X vs. Y" -> try fighter last names
    if "vs." in event_name:
        fighters = event_name.split("vs.")
        if len(fighters) == 2:
            a = fighters[0].split()[-1].strip().rstrip(":") if fighters[0].split() else ""
            b = fighters[1].strip().split()[0] if fighters[1].strip().split() else ""
            if a and b:
                queries.append(f"{a} vs {b}")

    return queries


def scrape_bfo_event_odds(event_url: str) -> list[dict]:
    """Scrape fight odds from a BestFightOdds event page.

    Returns a list of dicts with fighter_a, fighter_b, odds_a, odds_b
    (consensus/average American odds across available books).
    """
    fights = []
    try:
        resp = requests.get(event_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Find the scrollable odds table (second table in each table-div)
        for table_div in soup.select("div.table-div"):
            scroller = table_div.select_one("div.table-scroller table")
            if not scroller:
                continue

            rows = scroller.select("tbody tr")
            i = 0
            while i < len(rows) - 1:
                row_a = rows[i]
                row_b = rows[i + 1]

                # Skip prop bet rows
                if "pr" in row_a.get("class", []) or "pr" in row_b.get("class", []):
                    i += 1
                    continue

                name_a_el = row_a.select_one("th span.t-b-fcc")
                name_b_el = row_b.select_one("th span.t-b-fcc")

                if not name_a_el or not name_b_el:
                    i += 2
                    continue

                name_a = name_a_el.text.strip()
                name_b = name_b_el.text.strip()

                # Collect all odds for each fighter across sportsbooks
                odds_a_list = []
                odds_b_list = []

                for cell in row_a.select("td.but-sg"):
                    span = cell.select_one("span[id^='oID']")
                    if span:
                        val = _parse_american_odds(span.text)
                        if val is not None:
                            odds_a_list.append(val)

                for cell in row_b.select("td.but-sg"):
                    span = cell.select_one("span[id^='oID']")
                    if span:
                        val = _parse_american_odds(span.text)
                        if val is not None:
                            odds_b_list.append(val)

                if odds_a_list and odds_b_list:
                    # Use median odds as consensus
                    odds_a = int(sorted(odds_a_list)[len(odds_a_list) // 2])
                    odds_b = int(sorted(odds_b_list)[len(odds_b_list) // 2])

                    fights.append({
                        "fighter_a": name_a,
                        "fighter_b": name_b,
                        "odds_a": odds_a,
                        "odds_b": odds_b,
                        "implied_prob_a": round(american_to_implied_prob(odds_a), 4),
                        "implied_prob_b": round(american_to_implied_prob(odds_b), 4),
                    })

                i += 2

    except Exception as e:
        print(f"Error scraping BFO event {event_url}: {e}")

    return fights


def _match_fighter_name(bfo_name: str, fights_df_names: set) -> str | None:
    """Try to match a BFO fighter name to a name in our fights CSV."""
    norm = _normalize_name(bfo_name)
    if norm in fights_df_names:
        return norm

    # Try last-name, first-name swap (BFO sometimes uses different ordering)
    parts = norm.split()
    if len(parts) >= 2:
        swapped = " ".join(parts[1:]) + " " + parts[0]
        if swapped in fights_df_names:
            return swapped

    # Fuzzy: check if bfo name is contained in any fights name or vice versa
    # Require minimum length to avoid false matches like "li" matching "ali"
    if len(norm) >= 6:
        for name in fights_df_names:
            if len(name) >= 6 and (norm in name or name in norm):
                return name

    return None


def build_odds_database(fights_df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Build a database of odds matched to existing fights.

    Searches BestFightOdds for each event in our fights data and scrapes odds.
    """
    if use_cache and os.path.exists(ODDS_CACHE_PATH):
        cached = pd.read_csv(ODDS_CACHE_PATH)
        if len(cached) > 0:
            print(f"Loaded {len(cached)} cached odds entries")
            return cached

    # Get all unique fighter names from fights_df (normalized)
    all_names = set()
    for _, fight in fights_df.iterrows():
        all_names.add(_normalize_name(str(fight.get("fighter_a", ""))))
        all_names.add(_normalize_name(str(fight.get("fighter_b", ""))))

    # Get unique events from our fights data
    our_events = fights_df["event"].dropna().unique().tolist()

    print(f"Searching BestFightOdds for {len(our_events)} events...")

    all_odds = []
    events_found = 0

    for event_name in our_events:
        queries = _event_search_queries(event_name)
        event_url = None

        for query in queries:
            event_url = search_bfo_event(query)
            if event_url:
                break
            time.sleep(0.5)

        if not event_url:
            print(f"  Not found: {event_name}")
            continue

        events_found += 1
        print(f"  Found: {event_name}")
        event_odds = scrape_bfo_event_odds(event_url)

        for fight_odds in event_odds:
            fa_match = _match_fighter_name(fight_odds["fighter_a"], all_names)
            fb_match = _match_fighter_name(fight_odds["fighter_b"], all_names)

            if fa_match and fb_match:
                all_odds.append({
                    "fighter_a": fa_match,
                    "fighter_b": fb_match,
                    "odds_a": fight_odds["odds_a"],
                    "odds_b": fight_odds["odds_b"],
                    "implied_prob_a": fight_odds["implied_prob_a"],
                    "implied_prob_b": fight_odds["implied_prob_b"],
                    "event": event_name,
                })

        time.sleep(1)  # Be respectful

    print(f"Found {events_found}/{len(our_events)} events, matched {len(all_odds)} fights")

    odds_df = pd.DataFrame(all_odds)
    if len(odds_df) > 0:
        odds_df.to_csv(ODDS_CACHE_PATH, index=False)
        print(f"Saved odds to {ODDS_CACHE_PATH}")

    return odds_df


def fetch_upcoming_odds(api_key: str = None) -> list[dict]:
    """Fetch live odds for upcoming UFC fights from The Odds API.

    Requires an API key (free tier: 500 requests/month).
    Set via ODDS_API_KEY environment variable or pass directly.
    """
    if not api_key:
        # Try Streamlit secrets first, then environment variable
        try:
            import streamlit as st
            api_key = st.secrets.get("ODDS_API_KEY", "")
        except Exception:
            pass
        if not api_key:
            api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return []

    url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        fights = []
        for event in data:
            if len(event.get("bookmakers", [])) == 0:
                continue

            # Gather odds across bookmakers
            fighter_odds = {}
            for bookmaker in event["bookmakers"]:
                for market in bookmaker.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    for outcome in market["outcomes"]:
                        name = outcome["name"]
                        price = outcome["price"]
                        fighter_odds.setdefault(name, []).append(price)

            names = list(fighter_odds.keys())
            if len(names) >= 2:
                # Median odds
                odds_a_list = fighter_odds[names[0]]
                odds_b_list = fighter_odds[names[1]]
                odds_a = int(sorted(odds_a_list)[len(odds_a_list) // 2])
                odds_b = int(sorted(odds_b_list)[len(odds_b_list) // 2])

                fights.append({
                    "fighter_a": names[0],
                    "fighter_b": names[1],
                    "odds_a": odds_a,
                    "odds_b": odds_b,
                    "implied_prob_a": round(american_to_implied_prob(odds_a), 4),
                    "implied_prob_b": round(american_to_implied_prob(odds_b), 4),
                })

        return fights

    except Exception as e:
        print(f"Error fetching odds from The Odds API: {e}")
        return []


if __name__ == "__main__":
    print("=== UFC Odds Scraper ===\n")

    # Test BFO scraping
    events = scrape_bfo_archive()
    print(f"Found {len(events)} events")
    if events:
        print(f"\nFirst event: {events[0]['name']}")
        odds = scrape_bfo_event_odds(events[0]["url"])
        for fight in odds[:3]:
            print(f"  {fight['fighter_a']} ({fight['odds_a']:+d}) vs "
                  f"{fight['fighter_b']} ({fight['odds_b']:+d})")
