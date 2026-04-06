from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_HEADERS = {
    "Accept": "application/json",
}


@dataclass(frozen=True)
class League:
    country: str
    name: str
    api_id: int


LEAGUES: dict[str, League] = {
    "super_lig": League("Turkey", "Super Lig", 203),
    "premier_league": League("England", "Premier League", 39),
    "bundesliga": League("Germany", "Bundesliga", 78),
    "laliga": League("Spain", "LaLiga", 140),
    "serie_a": League("Italy", "Serie A", 135),
    "ligue_1": League("France", "Ligue 1", 61),
    "liga_portugal": League("Portugal", "Liga Portugal", 94),
    "jupiler_pro_league": League("Belgium", "Jupiler Pro League", 144),
    "eredivisie": League("Netherlands", "Eredivisie", 88),
}


class APIFootballClient:
    def __init__(self, api_key: str, delay: float = 1.0, timeout: float = 60.0) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.session.headers.update({"x-apisports-key": api_key})
        self.delay = delay
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{BASE_URL}{endpoint}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise requests.RequestException(f"Rate-limited by API-Football (429). Retry-After={retry_after}")
        response.raise_for_status()
        payload = response.json()
        time.sleep(self.delay)
        return payload

    def get_paginated(self, endpoint: str, params: dict) -> list[dict]:
        page = 1
        rows: list[dict] = []
        while True:
            payload = self._get(endpoint, {**params, "page": page})
            response_rows = payload.get("response", [])
            rows.extend(response_rows)
            paging = payload.get("paging", {})
            current = int(paging.get("current", page))
            total = int(paging.get("total", current))
            if current >= total:
                break
            page += 1
        return rows

    def get_fixtures(self, league_id: int, season_start_year: int) -> list[dict]:
        return self.get_paginated("/fixtures", {"league": league_id, "season": season_start_year})

    def get_teams(self, league_id: int, season_start_year: int) -> list[dict]:
        return self.get_paginated("/teams", {"league": league_id, "season": season_start_year})


class APIFootballCollector:
    def __init__(self, client: APIFootballClient) -> None:
        self.client = client

    def collect_matches(self, league_key: str, league: League, season_start_year: int) -> list[dict]:
        fixtures = self.client.get_fixtures(league.api_id, season_start_year)
        records: list[dict] = []

        for item in fixtures:
            fixture = item.get("fixture") or {}
            teams = item.get("teams") or {}
            goals = item.get("goals") or {}

            home_team = (teams.get("home") or {}).get("name")
            away_team = (teams.get("away") or {}).get("name")
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            match_date = fixture.get("date")

            if not home_team or not away_team or home_goals is None or away_goals is None or not match_date:
                continue

            records.append(
                {
                    "season": f"{season_start_year}/{season_start_year + 1}",
                    "season_start_year": season_start_year,
                    "league_key": league_key,
                    "league_name": league.name,
                    "country": league.country,
                    "match_date": match_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "attendance": fixture.get("attendance"),
                    "stadium": ((fixture.get("venue") or {}).get("name")),
                }
            )

        return records

    def collect_team_context(self, league_key: str, league: League, season_start_year: int) -> list[dict]:
        teams = self.client.get_teams(league.api_id, season_start_year)
        rows: list[dict] = []
        for item in teams:
            team = item.get("team") or {}
            venue = item.get("venue") or {}
            team_name = team.get("name")
            if not team_name:
                continue
            rows.append(
                {
                    "season": f"{season_start_year}/{season_start_year + 1}",
                    "season_start_year": season_start_year,
                    "data_reference_season": f"{season_start_year}/{season_start_year + 1}",
                    "league_key": league_key,
                    "team_name": team_name,
                    "squad_market_value_eur": None,
                    "top_players": json.dumps([]),
                    "injured_players": json.dumps([]),
                    "injured_top_player_count": 0,
                    "stadium_capacity": venue.get("capacity"),
                }
            )
        return rows


def season_years(window: int, end_year: int | None) -> list[int]:
    if end_year is None:
        today = datetime.utcnow()
        end_year = today.year if today.month >= 7 else today.year - 1
    return list(range(end_year - window + 1, end_year + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch league and team context data from API-Football.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES.keys()), choices=list(LEAGUES.keys()))
    parser.add_argument("--seasons", type=int, default=5)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--api-key", default=None, help="API-Football key. Fallback: API_FOOTBALL_KEY env variable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set API_FOOTBALL_KEY environment variable.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = APIFootballClient(api_key=api_key, delay=args.delay, timeout=args.timeout)
    collector = APIFootballCollector(client)
    seasons = season_years(args.seasons, args.end_season)
    all_matches: list[dict] = []
    all_team_context: list[dict] = []
    errors: list[dict] = []
    rows_per_league_season: list[dict] = []

    for league_key in args.leagues:
        league = LEAGUES[league_key]
        for season_start_year in seasons:
            print(f"Fetching {league.name} {season_start_year}/{season_start_year + 1} from API-Football")
            try:
                match_rows = collector.collect_matches(league_key, league, season_start_year)
                team_rows = collector.collect_team_context(league_key, league, season_start_year)
                all_matches.extend(match_rows)
                all_team_context.extend(team_rows)
                rows_per_league_season.append(
                    {
                        "league_key": league_key,
                        "season": f"{season_start_year}/{season_start_year + 1}",
                        "match_rows": len(match_rows),
                        "team_rows": len(team_rows),
                    }
                )
            except requests.RequestException as exc:
                errors.append(
                    {
                        "league_key": league_key,
                        "season": f"{season_start_year}/{season_start_year + 1}",
                        "error": str(exc),
                    }
                )
                print(f"Skipping {league.name} {season_start_year}: {exc}")

    match_columns = [
        "season",
        "season_start_year",
        "league_key",
        "league_name",
        "country",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "attendance",
        "stadium",
    ]
    team_columns = [
        "season",
        "season_start_year",
        "data_reference_season",
        "league_key",
        "team_name",
        "squad_market_value_eur",
        "top_players",
        "injured_players",
        "injured_top_player_count",
        "stadium_capacity",
    ]

    matches_df = pd.DataFrame(all_matches)
    if matches_df.empty:
        matches_df = pd.DataFrame(columns=match_columns)
    else:
        matches_df = matches_df[match_columns].drop_duplicates()

    team_df = pd.DataFrame(all_team_context)
    if team_df.empty:
        team_df = pd.DataFrame(columns=team_columns)
    else:
        team_df = team_df[team_columns].drop_duplicates(subset=["season", "team_name", "league_key"])

    matches_path = out_dir / "matches.csv"
    team_path = out_dir / "team_context.csv"
    metadata_path = out_dir / "api_football_metadata.json"

    matches_df.to_csv(matches_path, index=False)
    team_df.to_csv(team_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.utcnow().isoformat(),
                "source": "api-football",
                "league_count": len(args.leagues),
                "season_start_years": seasons,
                "match_rows": len(matches_df),
                "team_rows": len(team_df),
                "rows_per_league_season": rows_per_league_season,
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {matches_path}")
    print(f"Wrote {team_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
