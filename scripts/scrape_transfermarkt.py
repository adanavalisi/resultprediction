from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.transfermarkt.com"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class League:
    country: str
    name: str
    code: str
    slug: str

    @property
    def competition_url(self) -> str:
        return f"{BASE_URL}/{self.slug}/startseite/wettbewerb/{self.code}"


LEAGUES: dict[str, League] = {
    "super_lig": League("Turkey", "Super Lig", "TR1", "super-lig"),
    "premier_league": League("England", "Premier League", "GB1", "premier-league"),
    "bundesliga": League("Germany", "Bundesliga", "L1", "bundesliga"),
    "laliga": League("Spain", "LaLiga", "ES1", "laliga"),
    "serie_a": League("Italy", "Serie A", "IT1", "serie-a"),
    "ligue_1": League("France", "Ligue 1", "FR1", "ligue-1"),
    "liga_portugal": League("Portugal", "Liga Portugal", "PO1", "liga-portugal"),
    "jupiler_pro_league": League("Belgium", "Jupiler Pro League", "BE1", "jupiler-pro-league"),
    "eredivisie": League("Netherlands", "Eredivisie", "NL1", "eredivisie"),
}


class TransfermarktScraper:
    def __init__(self, delay: float = 3.0) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay = delay

    def fetch_html(self, url: str, params: dict | None = None) -> str:
        response = self.session.get(url, params=params, timeout=120)
        response.raise_for_status()
        time.sleep(self.delay)
        return response.text

    def soup(self, url: str, params: dict | None = None) -> BeautifulSoup:
        return BeautifulSoup(self.fetch_html(url, params=params), "lxml")

    def scrape_matches(self, league: League, season_start_year: int) -> list[dict]:
        url = f"{BASE_URL}/{league.slug}/gesamtspielplan/wettbewerb/{league.code}"
        html = self.fetch_html(url, params={"saison_id": season_start_year})
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"class": "items"})
        records: list[dict] = []

        if not table:
            return records

        for row in table.select("tbody tr"):
            row_data = self._extract_match_row_data(row, season_start_year)
            if row_data is None:
                continue
            home_team, away_team, home_goals, away_goals, match_date, attendance, stadium = row_data
            records.append(
                {
                    "season": f"{season_start_year}/{season_start_year + 1}",
                    "season_start_year": season_start_year,
                    "league_key": self._league_key(league),
                    "league_name": league.name,
                    "country": league.country,
                    "match_date": match_date.isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "attendance": attendance,
                    "stadium": stadium,
                }
            )
        return records

    def _extract_match_row_data(self, row: BeautifulSoup, season_start_year: int) -> tuple[str, str, int, int, datetime, int | None, str | None] | None:
        text_cells = [self._clean_text(td.get_text(" ", strip=True)) for td in row.find_all("td")]
        clean_cells = [cell for cell in text_cells if cell]
        if not clean_cells:
            return None

        result_text = next((cell for cell in clean_cells if re.search(r"\d+\s*:\s*\d+", cell)), None)
        if result_text is None:
            return None

        team_links = []
        for link in row.select("a[href*='/verein/']"):
            name = self._clean_text(link.get_text(" ", strip=True))
            if name and name not in team_links:
                team_links.append(name)
        if len(team_links) < 2:
            return None
        home_team, away_team = team_links[0], team_links[1]

        match_date = None
        for cell in clean_cells:
            match_date = self._parse_date(cell, season_start_year)
            if match_date is not None:
                break
        if match_date is None:
            return None

        home_goals, away_goals = self._parse_score(result_text)
        attendance = self._extract_attendance(clean_cells)
        stadium = self._extract_stadium(row)
        return home_team, away_team, home_goals, away_goals, match_date, attendance, stadium

    def scrape_team_context(self, league: League, season_start_year: int) -> list[dict]:
        soup = self.soup(league.competition_url, params={"saison_id": season_start_year})
        links = soup.select("td.hauptlink a[title]")
        team_pages: dict[str, str] = {}
        for link in links:
            href = link.get("href")
            title = self._clean_text(link.get("title"))
            if not href or not title or "/verein/" not in href:
                continue
            team_pages[title] = urljoin(BASE_URL, href)

        contexts: list[dict] = []
        for team_name, team_url in tqdm(team_pages.items(), desc=f"{league.name} squads", leave=False):
            contexts.append(self._scrape_single_team_context(league, season_start_year, team_name, team_url))
        return contexts

    def _scrape_single_team_context(
        self,
        league: League,
        season_start_year: int,
        team_name: str,
        team_url: str,
    ) -> dict:
        squad_url = team_url.replace("/startseite/", "/kader/") if "/startseite/" in team_url else team_url
        injuries_url = team_url.replace("/startseite/", "/verletzungen/") if "/startseite/" in team_url else team_url
        venue_url = team_url.replace("/startseite/", "/stadion/") if "/startseite/" in team_url else team_url

        squad_soup = self.soup(squad_url, params={"saison_id": season_start_year})
        injuries_soup = self.soup(injuries_url)
        venue_soup = self.soup(venue_url)

        top_players = self._extract_top_players(squad_soup, limit=5)
        injured_players = self._extract_injuries(injuries_soup)
        injured_top_players = [player["name"] for player in top_players if player["name"] in injured_players]

        return {
            "season": f"{season_start_year}/{season_start_year + 1}",
            "season_start_year": season_start_year,
            "league_key": self._league_key(league),
            "team_name": team_name,
            "squad_market_value_eur": self._extract_market_value(squad_soup),
            "top_players": top_players,
            "injured_players": injured_players,
            "injured_top_player_count": len(injured_top_players),
            "stadium_capacity": self._extract_stadium_capacity(venue_soup),
        }

    def _extract_market_value(self, soup: BeautifulSoup) -> float | None:
        values = []
        for tag in soup.select(".data-header__market-value-wrapper, .content-box-headline"):
            value = self._parse_money_to_eur(self._clean_text(tag.get_text(" ", strip=True)))
            if value is not None:
                values.append(value)
        return max(values) if values else None

    def _extract_top_players(self, soup: BeautifulSoup, limit: int) -> list[dict]:
        players: list[dict] = []
        for row in soup.select("table.items tbody tr"):
            name_tag = row.select_one("td.hauptlink a")
            if name_tag is None:
                continue
            name = self._clean_text(name_tag.get_text(" ", strip=True))
            if not name:
                continue
            row_text = self._clean_text(row.get_text(" ", strip=True)) or ""
            market_values = re.findall(r"([\d\.,]+)\s*(m|k)", row_text.lower())
            value = None
            if market_values:
                amount, unit = market_values[-1]
                value = self._parse_money_to_eur(f"{amount}{unit}")
            players.append({"name": name, "market_value_eur": value})
        players.sort(key=lambda item: item["market_value_eur"] or 0, reverse=True)
        return players[:limit]

    def _extract_injuries(self, soup: BeautifulSoup) -> list[str]:
        names = []
        for row in soup.select("table.items tbody tr"):
            link = row.select_one("td.hauptlink a")
            if link is None:
                continue
            name = self._clean_text(link.get_text(" ", strip=True))
            if name:
                names.append(name)
        return names

    def _extract_stadium_capacity(self, soup: BeautifulSoup) -> int | None:
        text = soup.get_text(" ", strip=True)
        match = re.search(r"Capacity\s*([\d\.,]+)", text, flags=re.IGNORECASE)
        return self._to_int(match.group(1)) if match else None

    def _league_key(self, league: League) -> str:
        return next(key for key, value in LEAGUES.items() if value == league)

    @staticmethod
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _parse_score(result_text: str) -> tuple[int, int]:
        match = re.search(r"(\d+)\s*:\s*(\d+)", result_text)
        if not match:
            raise ValueError(f"Score parse failed: {result_text}")
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _parse_date(value: object, season_start_year: int) -> datetime | None:
        if value is None:
            return None
        text = str(value).strip()
        for fmt in ("%a, %b %d, %Y", "%b %d, %Y", "%d/%m/%Y", "%d.%m.%Y", "%d %b %Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        partial = re.match(r"(\d{1,2})/(\d{1,2})$", text)
        if partial:
            day = int(partial.group(1))
            month = int(partial.group(2))
            year = season_start_year if month >= 7 else season_start_year + 1
            return datetime(year, month, day)
        return None

    @staticmethod
    def _to_int(value: object) -> int | None:
        if value is None:
            return None
        digits = re.sub(r"[^\d]", "", str(value))
        return int(digits) if digits else None

    @staticmethod
    def _parse_money_to_eur(value: str | None) -> float | None:
        if value is None:
            return None
        text = value.lower().replace("eur", "").replace("€", "").strip()
        match = re.search(r"([\d\.,]+)\s*([mk]|million|th\.)?", text)
        if not match:
            return None
        amount = float(match.group(1).replace(".", "").replace(",", "."))
        unit = (match.group(2) or "").strip()
        if unit in {"m", "million"}:
            return amount * 1_000_000
        if unit in {"k", "th."}:
            return amount * 1_000
        return amount

    @staticmethod
    def _extract_attendance(values: list[str]) -> int | None:
        candidates = []
        for value in values:
            if ":" in value or "/" in value or "." in value and not re.search(r"\d{1,2}\.\d{1,2}\.\d{2,4}", value):
                continue
            parsed = TransfermarktScraper._to_int(value)
            if parsed and parsed >= 1_000:
                candidates.append(parsed)
        return max(candidates) if candidates else None

    @staticmethod
    def _extract_stadium(row: BeautifulSoup) -> str | None:
        venue_link = row.select_one("a[href*='/stadion/']")
        if venue_link:
            return TransfermarktScraper._clean_text(venue_link.get_text(" ", strip=True))
        title_link = row.select_one("a[title*='Stadium'], a[title*='Arena']")
        if title_link:
            return TransfermarktScraper._clean_text(title_link.get("title"))
        return None


def season_years(window: int, end_year: int | None) -> list[int]:
    if end_year is None:
        today = datetime.utcnow()
        end_year = today.year if today.month >= 7 else today.year - 1
    return list(range(end_year - window + 1, end_year + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape league and team context data from Transfermarkt.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUES.keys()), choices=list(LEAGUES.keys()))
    parser.add_argument("--seasons", type=int, default=5)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scraper = TransfermarktScraper(delay=args.delay)
    seasons = season_years(args.seasons, args.end_season)
    all_matches: list[dict] = []
    all_team_context: list[dict] = []

    for league_key in args.leagues:
        league = LEAGUES[league_key]
        for season_start_year in seasons:
            print(f"Scraping {league.name} {season_start_year}/{season_start_year + 1}")
            try:
                all_matches.extend(scraper.scrape_matches(league, season_start_year))
                all_team_context.extend(scraper.scrape_team_context(league, season_start_year))
            except requests.RequestException as exc:
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
    metadata_path = out_dir / "scrape_metadata.json"

    matches_df.to_csv(matches_path, index=False)
    team_df.to_csv(team_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.utcnow().isoformat(),
                "league_count": len(args.leagues),
                "season_start_years": seasons,
                "match_rows": len(matches_df),
                "team_rows": len(team_df),
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
