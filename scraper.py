"""
UFC Data Scraper
Scrapes fighter stats, fight history, and upcoming events from UFCStats.com
"""

import re
import time
import json
import os
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "http://www.ufcstats.com/statistics/fighters"
EVENT_URL = "http://www.ufcstats.com/statistics/events/upcoming"
COMPLETED_EVENTS_URL = "http://www.ufcstats.com/statistics/events/completed"
FIGHTER_DETAIL_URL = "http://www.ufcstats.com/fighter-details/"
EVENT_DETAIL_URL = "http://www.ufcstats.com/event-details/"
FIGHT_DETAIL_URL = "http://www.ufcstats.com/fight-details/"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def _parse_height_to_cm(height_str: str) -> float | None:
    """Convert height string like '5\\' 11\"' to cm."""
    if not height_str or height_str.strip() == "--":
        return None
    match = re.search(r"(\d+)'\s*(\d+)\"", height_str)
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return round((feet * 12 + inches) * 2.54, 1)
    return None


def _parse_reach_to_cm(reach_str: str) -> float | None:
    """Convert reach string like '74\"' to cm."""
    if not reach_str or reach_str.strip() == "--":
        return None
    match = re.search(r"([\d.]+)", reach_str)
    if match:
        return round(float(match.group(1)) * 2.54, 1)
    return None


def _parse_weight_to_kg(weight_str: str) -> float | None:
    """Convert weight string like '185 lbs.' to kg."""
    if not weight_str or weight_str.strip() == "--":
        return None
    match = re.search(r"([\d.]+)", weight_str)
    if match:
        return round(float(match.group(1)) * 0.453592, 1)
    return None


def _parse_pct(pct_str: str) -> float | None:
    """Parse '54%' to 0.54."""
    if not pct_str or pct_str.strip() == "--":
        return None
    match = re.search(r"([\d.]+)", pct_str)
    if match:
        return round(float(match.group(1)) / 100, 4)
    return None


def _parse_float(s: str) -> float | None:
    """Parse a numeric string to float."""
    if not s or s.strip() == "--":
        return None
    match = re.search(r"([\d.]+)", s)
    if match:
        return float(match.group(1))
    return None


def _parse_dob(dob_str: str) -> str | None:
    """Parse DOB string to ISO format."""
    if not dob_str or dob_str.strip() == "--":
        return None
    try:
        dt = datetime.strptime(dob_str.strip(), "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_record(record_str: str) -> tuple[int, int, int]:
    """Parse record like 'Record: 22-6-0' to (wins, losses, draws)."""
    match = re.search(r"(\d+)-(\d+)-(\d+)", record_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return 0, 0, 0


def scrape_all_fighter_urls() -> list[dict]:
    """Scrape all fighter URLs from the alphabetical listing."""
    fighters = []
    for char in "abcdefghijklmnopqrstuvwxyz":
        url = f"{BASE_URL}?char={char}&page=all"
        print(f"  Fetching fighters: {char.upper()}...")
        try:
            soup = _get_soup(url)
            rows = soup.select("tbody tr.b-statistics__table-row")
            for row in rows:
                link = row.select_one("a.b-link")
                if link and link.get("href"):
                    name_parts = row.select("td a.b-link")
                    first = name_parts[0].text.strip() if len(name_parts) > 0 else ""
                    last = name_parts[1].text.strip() if len(name_parts) > 1 else ""
                    name = f"{first} {last}".strip()
                    fighters.append({
                        "name": name,
                        "url": link["href"].strip()
                    })
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error fetching letter {char}: {e}")
    return fighters


def scrape_fighter_details(fighter_url: str) -> dict:
    """Scrape detailed stats for a single fighter."""
    soup = _get_soup(fighter_url)

    info = {}
    # Name
    name_el = soup.select_one("span.b-content__title-highlight")
    info["name"] = name_el.text.strip() if name_el else ""

    # Record
    record_el = soup.select_one("span.b-content__title-record")
    if record_el:
        wins, losses, draws = _parse_record(record_el.text)
        info["wins"] = wins
        info["losses"] = losses
        info["draws"] = draws

    # Bio box items
    bio_items = soup.select("li.b-list__box-list-item")
    for item in bio_items:
        text = item.text.strip()
        if "Height:" in text:
            info["height_cm"] = _parse_height_to_cm(text.split("Height:")[-1])
        elif "Weight:" in text:
            info["weight_kg"] = _parse_weight_to_kg(text.split("Weight:")[-1])
        elif "Reach:" in text:
            info["reach_cm"] = _parse_reach_to_cm(text.split("Reach:")[-1])
        elif "STANCE:" in text.upper():
            stance = text.split(":")[-1].strip()
            info["stance"] = stance if stance != "--" else None
        elif "DOB:" in text.upper():
            info["dob"] = _parse_dob(text.split(":")[-1])

    # Career statistics
    stat_boxes = soup.select("div.b-list__info-box-left li, div.b-list__info-box li")
    for box in stat_boxes:
        text = box.text.strip()
        if "SLpM:" in text:
            info["slpm"] = _parse_float(text.split(":")[-1])
        elif "Str. Acc.:" in text:
            info["str_acc"] = _parse_pct(text.split(":")[-1])
        elif "SApM:" in text:
            info["sapm"] = _parse_float(text.split(":")[-1])
        elif "Str. Def:" in text or "Str. Def.:" in text:
            info["str_def"] = _parse_pct(text.split(":")[-1])
        elif "TD Avg.:" in text:
            info["td_avg"] = _parse_float(text.split(":")[-1])
        elif "TD Acc.:" in text:
            info["td_acc"] = _parse_pct(text.split(":")[-1])
        elif "TD Def.:" in text:
            info["td_def"] = _parse_pct(text.split(":")[-1])
        elif "Sub. Avg.:" in text:
            info["sub_avg"] = _parse_float(text.split(":")[-1])

    # Fight history
    info["fight_history"] = scrape_fighter_fight_history(soup)
    info["url"] = fighter_url

    return info


def scrape_fighter_fight_history(soup: BeautifulSoup) -> list[dict]:
    """Extract fight history from a fighter detail page."""
    fights = []
    rows = soup.select("tbody tr.b-fight-details__table-row")
    for row in rows:
        cols = row.select("td")
        if len(cols) < 8:
            continue
        link = row.get("data-link", "")
        result = cols[0].text.strip()
        fighters_in_fight = [a.text.strip() for a in cols[1].select("a")]
        method = cols[7].text.strip() if len(cols) > 7 else ""
        rnd = cols[8].text.strip() if len(cols) > 8 else ""

        # Get sig strikes and takedowns from fight row
        kd = [p.text.strip() for p in cols[2].select("p")]
        sig_str = [p.text.strip() for p in cols[3].select("p")]
        td = [p.text.strip() for p in cols[5].select("p")]

        fights.append({
            "result": result,
            "opponent": fighters_in_fight[1] if len(fighters_in_fight) > 1 else "",
            "method": method,
            "round": rnd,
            "kd": kd,
            "sig_str": sig_str,
            "td": td,
            "fight_url": link
        })
    return fights


def scrape_upcoming_events() -> list[dict]:
    """Scrape upcoming UFC events."""
    soup = _get_soup(EVENT_URL)
    events = []
    rows = soup.select("tbody tr.b-statistics__table-row")
    for row in rows:
        link = row.select_one("a.b-link")
        if link:
            name = link.text.strip()
            url = link["href"].strip()
            date_el = row.select_one("span.b-statistics__date")
            date = date_el.text.strip() if date_el else ""
            location_el = row.select_one("td.b-statistics__table-col_type_second")
            location = location_el.text.strip() if location_el else ""
            events.append({
                "name": name,
                "url": url,
                "date": date,
                "location": location
            })
    return events


def scrape_event_fights(event_url: str) -> list[dict]:
    """Scrape fight card from an event detail page."""
    soup = _get_soup(event_url)
    fights = []
    rows = soup.select("tbody tr.b-fight-details__table-row")
    for row in rows:
        cols = row.select("td")
        if len(cols) < 2:
            continue
        fighters = [a.text.strip() for a in cols[1].select("a") if a.text.strip()]
        if len(fighters) >= 2:
            weight_class = cols[6].text.strip() if len(cols) > 6 else ""
            fights.append({
                "fighter_a": fighters[0],
                "fighter_b": fighters[1],
                "weight_class": weight_class
            })
    return fights


def scrape_completed_events(max_events: int = 50) -> list[dict]:
    """Scrape recent completed events for training data."""
    soup = _get_soup(COMPLETED_EVENTS_URL)
    events = []
    rows = soup.select("tbody tr.b-statistics__table-row")
    for row in rows[:max_events]:
        link = row.select_one("a.b-link")
        if link:
            events.append({
                "name": link.text.strip(),
                "url": link["href"].strip()
            })
    return events


def scrape_completed_fight_details(event_url: str) -> list[dict]:
    """Scrape fight results from a completed event."""
    soup = _get_soup(event_url)
    fights = []
    rows = soup.select("tbody tr.b-fight-details__table-row")
    for row in rows:
        cols = row.select("td")
        if len(cols) < 8:
            continue
        result_col = cols[0].text.strip()
        fighters = [a.text.strip() for a in cols[1].select("a") if a.text.strip()]
        if len(fighters) < 2:
            continue

        # In UFCStats, the winner is always listed first with "win"
        # If it's a draw, result_col will say "draw"
        winner = None
        if "win" in result_col.lower():
            winner = fighters[0]

        method = cols[7].text.strip() if len(cols) > 7 else ""
        weight_class = cols[6].text.strip() if len(cols) > 6 else ""

        fights.append({
            "fighter_a": fighters[0],
            "fighter_b": fighters[1],
            "winner": winner,
            "method": method,
            "weight_class": weight_class
        })
    return fights


def build_fighter_database(use_cache: bool = True) -> pd.DataFrame:
    """Build a complete fighter database. Uses cache if available."""
    cache_path = os.path.join(DATA_DIR, "fighters.csv")
    if use_cache and os.path.exists(cache_path):
        print("Loading cached fighter database...")
        return pd.read_csv(cache_path)

    print("Scraping all fighter URLs...")
    fighter_list = scrape_all_fighter_urls()
    print(f"Found {len(fighter_list)} fighters")

    all_fighters = []
    for i, f in enumerate(fighter_list):
        if i % 100 == 0:
            print(f"  Scraping fighter details: {i}/{len(fighter_list)}...")
        try:
            details = scrape_fighter_details(f["url"])
            # Don't include fight_history in the main DataFrame
            fight_hist = details.pop("fight_history", [])
            all_fighters.append(details)
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error scraping {f['name']}: {e}")

    df = pd.DataFrame(all_fighters)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} fighters to cache")
    return df


def build_fight_history_database(use_cache: bool = True) -> pd.DataFrame:
    """Build a database of completed fights for training."""
    cache_path = os.path.join(DATA_DIR, "fights.csv")
    if use_cache and os.path.exists(cache_path):
        print("Loading cached fight database...")
        return pd.read_csv(cache_path)

    print("Scraping completed events...")
    events = scrape_completed_events(max_events=80)
    print(f"Found {len(events)} events")

    all_fights = []
    for i, event in enumerate(events):
        print(f"  Scraping event {i+1}/{len(events)}: {event['name']}...")
        try:
            fights = scrape_completed_fight_details(event["url"])
            for fight in fights:
                fight["event"] = event["name"]
            all_fights.extend(fights)
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error scraping event: {e}")

    df = pd.DataFrame(all_fights)
    df.to_csv(cache_path, index=False)
    print(f"Saved {len(df)} fights to cache")
    return df


def get_upcoming_card() -> list[dict]:
    """Get the next upcoming UFC event card."""
    cache_path = os.path.join(DATA_DIR, "upcoming.json")

    events = scrape_upcoming_events()
    if not events:
        # Try cache
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)
        return []

    # Get the first event with fights listed
    for event in events:
        fights = scrape_event_fights(event["url"])
        if fights:
            result = {
                "event": event["name"],
                "date": event["date"],
                "location": event["location"],
                "fights": fights
            }
            with open(cache_path, "w") as f:
                json.dump(result, f, indent=2)
            return result

    return {}


if __name__ == "__main__":
    print("=== UFC Data Scraper ===\n")
    print("Fetching upcoming events...")
    upcoming = get_upcoming_card()
    if upcoming:
        print(f"\nNext event: {upcoming.get('event', 'Unknown')}")
        print(f"Date: {upcoming.get('date', 'TBD')}")
        for fight in upcoming.get("fights", []):
            print(f"  {fight['fighter_a']} vs {fight['fighter_b']} ({fight.get('weight_class', '')})")
    else:
        print("No upcoming events found.")
