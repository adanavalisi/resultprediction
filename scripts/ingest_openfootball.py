from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_RAW_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"


@dataclass(frozen=True)
class League:
    country: str
    name: str
    openfootball_code: str


LEAGUES: dict[str, League] = {
    "premier_league": League("England", "Premier League", "en.1"),
    "bundesliga": League("Germany", "Bundesliga", "de.1"),
    "laliga": League("Spain", "LaLiga", "es.1"),
    "serie_a": League("Italy", "Serie A", "it.1"),
    "ligue_1": League("France", "Ligue 1", "fr.1"),
    "liga_portugal": League("Portugal", "Liga Portugal", "pt.1"),
    "jupiler_pro_league": League("Belgium", "Jupiler Pro League", "be.1"),
    "eredivisie": League("Netherlands", "Eredivisie", "nl.1"),
    "super_lig": League("Turkey", "Super Lig", "tr.1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build raw training data from openfootball/football.json.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES.keys()), choices=list(LEAGUES.keys()))
    parser.add_argument("--seasons", type=int, default=5, help="Number of seasons to fetch (most recent N).")
    parser.add_argument("--end-season", type=int, default=None, help="Season start year for last season. Example: 2025 => 2025/2026")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--base-url", default=BASE_RAW_URL)
    return parser.parse_args()


def season_years(window: int, end_year: int | None) -> list[int]:
    if end_year is None:
        today = datetime.utcnow()
        end_year = today.year if today.month >= 7 else today.year - 1
    return list(range(end_year - window + 1, end_year + 1))


def season_slug(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def score_from_match(match: dict[str, Any]) -> tuple[int | None, int | None]:
    score = match.get("score") or {}
    if isinstance(score, dict):
        ft = score.get("ft")
        if isinstance(ft, list) and len(ft) == 2:
            return int(ft[0]), int(ft[1])
        full_time = score.get("full_time")
        if isinstance(full_time, list) and len(full_time) == 2:
            return int(full_time[0]), int(full_time[1])
        if isinstance(full_time, dict):
            home = full_time.get("home")
            away = full_time.get("away")
            if home is not None and away is not None:
                return int(home), int(away)
    result = match.get("result")
    if isinstance(result, dict):
        home = result.get("goals1")
        away = result.get("goals2")
        if home is not None and away is not None:
            return int(home), int(away)
    return None, None


def fetch_season_matches(base_url: str, league: League, league_key: str, season_start_year: int, timeout: float) -> tuple[list[dict], str]:
    slug = season_slug(season_start_year)
    url = f"{base_url}/{slug}/{league.openfootball_code}.json"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    rows: list[dict] = []
    rounds = payload.get("rounds") or []
    for round_item in rounds:
        for match in round_item.get("matches") or []:
            home_team = match.get("team1")
            away_team = match.get("team2")
            match_date = match.get("date")
            home_goals, away_goals = score_from_match(match)
            if not home_team or not away_team or not match_date or home_goals is None or away_goals is None:
                continue
            rows.append(
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
                    "attendance": None,
                    "stadium": None,
                }
            )
    return rows, url


def build_team_context(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df.empty:
        return pd.DataFrame(
            columns=[
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
        )

    unique_rows = []
    season_team = matches_df[["season", "season_start_year", "league_key", "home_team"]].drop_duplicates()
    season_team = season_team.rename(columns={"home_team": "team_name"})
    away_team = matches_df[["season", "season_start_year", "league_key", "away_team"]].drop_duplicates()
    away_team = away_team.rename(columns={"away_team": "team_name"})
    teams = pd.concat([season_team, away_team], ignore_index=True).drop_duplicates()

    for row in teams.itertuples(index=False):
        unique_rows.append(
            {
                "season": row.season,
                "season_start_year": row.season_start_year,
                "data_reference_season": row.season,
                "league_key": row.league_key,
                "team_name": row.team_name,
                "squad_market_value_eur": None,
                "top_players": json.dumps([]),
                "injured_players": json.dumps([]),
                "injured_top_player_count": 0,
                "stadium_capacity": None,
            }
        )

    return pd.DataFrame(unique_rows)


def main() -> None:
    args = parse_args()
    seasons = season_years(args.seasons, args.end_season)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_matches: list[dict] = []
    errors: list[dict] = []
    fetched_urls: list[str] = []

    for league_key in args.leagues:
        league = LEAGUES[league_key]
        for season_start in seasons:
            print(f"Fetching {league.name} {season_start}/{season_start + 1} from openfootball")
            try:
                rows, url = fetch_season_matches(args.base_url, league, league_key, season_start, args.timeout)
                all_matches.extend(rows)
                fetched_urls.append(url)
                print(f"  -> rows={len(rows)}")
            except requests.RequestException as exc:
                errors.append(
                    {
                        "league_key": league_key,
                        "season": f"{season_start}/{season_start + 1}",
                        "error": str(exc),
                    }
                )
                print(f"  -> skipped: {exc}")

    matches_columns = [
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

    matches_df = pd.DataFrame(all_matches)
    if matches_df.empty:
        matches_df = pd.DataFrame(columns=matches_columns)
    else:
        matches_df = matches_df[matches_columns].drop_duplicates()

    team_context_df = build_team_context(matches_df)

    matches_path = output_dir / "matches.csv"
    team_context_path = output_dir / "team_context.csv"
    metadata_path = output_dir / "openfootball_metadata.json"

    matches_df.to_csv(matches_path, index=False)
    team_context_df.to_csv(team_context_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.utcnow().isoformat(),
                "source": "openfootball",
                "league_count": len(args.leagues),
                "season_start_years": seasons,
                "match_rows": len(matches_df),
                "team_rows": len(team_context_df),
                "fetched_urls": fetched_urls,
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {matches_path}")
    print(f"Wrote {team_context_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
