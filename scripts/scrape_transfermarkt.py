from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://www.transfermarkt.com"
SELECTOR_CONFIG = {
    "selector_version": "v1_primary_fallback",
    "matches_table": [
        "table.items",
        "div.responsive-table table.items",
        "div.box table.items",
        "main table.items",
    ],
    "match_rows": [
        "tbody tr",
        "tr",
    ],
    "team_links": [
        "td.hauptlink a[title]",
        "a[href*='/verein/'][title]",
        "a[href*='/verein/']",
    ],
}
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
    def __init__(
        self,
        delay: float = 3.0,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        debug_dir: str | Path = "data/raw/debug",
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.debug_dir = Path(debug_dir)
        self.parse_warnings: list[dict] = []

        retry_strategy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            backoff_factor=0,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_html(
        self,
        url: str,
        params: dict | None = None,
        *,
        expected_selector: str | None = None,
        league_key: str | None = None,
        season_start_year: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                html = response.text
                bot_reason = self._detect_bot_signature(html, expected_selector=expected_selector)
                if bot_reason is not None:
                    self._save_debug_snapshot(
                        html=html,
                        league_key=league_key,
                        season_start_year=season_start_year,
                        reason=bot_reason,
                        url=url,
                    )
                    raise RuntimeError(f"Bot/CAPTCHA signature detected: {bot_reason}")
                time.sleep(self.delay)
                return html
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_time = self._calculate_backoff(attempt)
                time.sleep(sleep_time)
        raise requests.RequestException(f"Failed to fetch {url} after {self.max_retries + 1} attempts: {last_error}")

    def soup(self, url: str, params: dict | None = None) -> BeautifulSoup:
        return BeautifulSoup(self.fetch_html(url, params=params), "lxml")

    @staticmethod
    def _detect_bot_signature(html: str, expected_selector: str | None = None) -> str | None:
        lowered = html.lower()
        blocked_markers = [
            "captcha",
            "access denied",
            "too many requests",
            "cloudflare",
            "forbidden",
            "verify you are human",
        ]
        for marker in blocked_markers:
            if marker in lowered:
                return f"marker_{marker.replace(' ', '_')}"
        if expected_selector:
            soup = BeautifulSoup(html, "lxml")
            if soup.select_one(expected_selector) is None:
                return "missing_expected_selector"
        return None

    def _calculate_backoff(self, attempt: int) -> float:
        jitter = random.uniform(0, self.delay)
        return self.delay * (self.backoff_base**attempt) + jitter

    def _save_debug_snapshot(
        self,
        *,
        html: str,
        league_key: str | None,
        season_start_year: int | None,
        reason: str,
        url: str,
    ) -> None:
        league_label = league_key or "unknown_league"
        if season_start_year is not None:
            season_label = f"{season_start_year}_{season_start_year + 1}"
        else:
            season_label = "unknown_season"
        snapshot_dir = self.debug_dir / league_label / season_label
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        snapshot_path = snapshot_dir / f"{reason}_{ts}.html"
        snapshot_path.write_text(f"<!-- url: {url} -->\n{html}", encoding="utf-8")

    @staticmethod
    def _detect_reason_code(table: BeautifulSoup | None, row_count: int, html: str) -> str | None:
        if table is not None and row_count > 0:
            return None
        lowered = html.lower()
        blocked_markers = [
            "captcha",
            "cloudflare",
            "access denied",
            "forbidden",
            "too many requests",
            "bot",
        ]
        if any(marker in lowered for marker in blocked_markers):
            return "blocked"
        if table is None:
            return "selector_miss"
        return "no_data"

    @staticmethod
    def _select_first(soup: BeautifulSoup, selectors: list[str]) -> tuple[BeautifulSoup | None, str | None, bool]:
        for index, selector in enumerate(selectors):
            match = soup.select_one(selector)
            if match is not None:
                return match, selector, index > 0
        return None, None, False

    @staticmethod
    def _normalize_team_name(raw_name: str | None, href: str | None) -> str | None:
        cleaned = TransfermarktScraper._clean_text(raw_name)
        if cleaned:
            return cleaned
        if not href:
            return None
        parts = [part for part in href.split("/") if part]
        if not parts:
            return None
        slug = next((part for part in parts if part != "verein"), parts[-1])
        normalized = re.sub(r"[-_]+", " ", slug).strip()
        return TransfermarktScraper._clean_text(normalized.title())

    def _record_parse_warning(self, *, dataset: str, league_key: str, season_start_year: int, message: str, strategy_used: str) -> None:
        self.parse_warnings.append(
            {
                "type": "selector_fallback",
                "dataset": dataset,
                "league_key": league_key,
                "season": f"{season_start_year}/{season_start_year + 1}",
                "selector_version": SELECTOR_CONFIG["selector_version"],
                "strategy_used": strategy_used,
                "message": message,
            }
        )

    def scrape_matches(self, league: League, season_start_year: int) -> tuple[list[dict], str | None, dict]:
        url = f"{BASE_URL}/{league.slug}/gesamtspielplan/wettbewerb/{league.code}"
        html = self.fetch_html(
            url,
            params={"saison_id": season_start_year},
            expected_selector="table.items",
            league_key=self._league_key(league),
            season_start_year=season_start_year,
        )
        soup = BeautifulSoup(html, "lxml")
        table, table_strategy, used_table_fallback = self._select_first(soup, SELECTOR_CONFIG["matches_table"])
        parse_metadata = {
            "selector_version": SELECTOR_CONFIG["selector_version"],
            "strategy_used": {
                "table_selector": table_strategy,
                "row_selector": None,
            },
        }
        records: list[dict] = []

        if not table:
            return records, "selector_strategy_exhausted_matches_table", parse_metadata

        if used_table_fallback and table_strategy is not None:
            self._record_parse_warning(
                dataset="matches",
                league_key=self._league_key(league),
                season_start_year=season_start_year,
                message="Primary match table selector failed, fallback selector used.",
                strategy_used=table_strategy,
            )

        row_selector = None
        rows = []
        for candidate in SELECTOR_CONFIG["match_rows"]:
            rows = table.select(candidate)
            if rows:
                row_selector = candidate
                break
        parse_metadata["strategy_used"]["row_selector"] = row_selector

        for row in rows:
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
                    "selector_version": SELECTOR_CONFIG["selector_version"],
                    "strategy_used": json.dumps(parse_metadata["strategy_used"], ensure_ascii=False),
                }
            )
        return records, self._detect_reason_code(table, len(records), html), parse_metadata

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

    def scrape_team_context(self, league: League, season_start_year: int) -> tuple[list[dict], str | None, dict]:
        league_key = self._league_key(league)
        html = self.fetch_html(
            league.competition_url,
            params={"saison_id": season_start_year},
            expected_selector="table.items",
            league_key=league_key,
            season_start_year=season_start_year,
        )
        soup = BeautifulSoup(html, "lxml")
        table, table_strategy, used_table_fallback = self._select_first(soup, SELECTOR_CONFIG["matches_table"])
        parse_metadata = {
            "selector_version": SELECTOR_CONFIG["selector_version"],
            "strategy_used": {
                "table_selector": table_strategy,
                "team_link_selector": None,
            },
        }

        if used_table_fallback and table_strategy is not None:
            self._record_parse_warning(
                dataset="team_context",
                league_key=league_key,
                season_start_year=season_start_year,
                message="Primary team-context table selector failed, fallback selector used.",
                strategy_used=table_strategy,
            )

        team_pages: dict[str, str] = {}
        used_link_selector = None
        for link_selector in SELECTOR_CONFIG["team_links"]:
            links = soup.select(link_selector)
            local_team_pages: dict[str, str] = {}
            for link in links:
                href = link.get("href")
                if not href or "/verein/" not in href:
                    continue
                team_name = self._normalize_team_name(link.get("title") or link.get_text(" ", strip=True), href)
                if not team_name:
                    continue
                local_team_pages[team_name] = urljoin(BASE_URL, href)
            if local_team_pages:
                team_pages = local_team_pages
                used_link_selector = link_selector
                break
        parse_metadata["strategy_used"]["team_link_selector"] = used_link_selector

        if not team_pages:
            return [], "selector_strategy_exhausted_team_links", parse_metadata

        if used_link_selector and used_link_selector != SELECTOR_CONFIG["team_links"][0]:
            self._record_parse_warning(
                dataset="team_context",
                league_key=league_key,
                season_start_year=season_start_year,
                message="Primary team link selector failed, fallback URL-pattern selector used.",
                strategy_used=used_link_selector,
            )

        contexts: list[dict] = []
        for team_name, team_url in tqdm(team_pages.items(), desc=f"{league.name} squads", leave=False):
            contexts.append(
                self._scrape_single_team_context(
                    league,
                    season_start_year,
                    team_name,
                    team_url,
                    league_key=league_key,
                    parse_metadata=parse_metadata,
                )
            )
        return contexts, self._detect_reason_code(table, len(contexts), html), parse_metadata

    def _scrape_single_team_context(
        self,
        league: League,
        season_start_year: int,
        team_name: str,
        team_url: str,
        league_key: str,
        parse_metadata: dict,
    ) -> dict:
        squad_url = team_url.replace("/startseite/", "/kader/") if "/startseite/" in team_url else team_url
        injuries_url = team_url.replace("/startseite/", "/verletzungen/") if "/startseite/" in team_url else team_url
        venue_url = team_url.replace("/startseite/", "/stadion/") if "/startseite/" in team_url else team_url

        squad_soup = BeautifulSoup(
            self.fetch_html(
                squad_url,
                params={"saison_id": season_start_year},
                expected_selector="table.items",
                league_key=league_key,
                season_start_year=season_start_year,
            ),
            "lxml",
        )
        injuries_soup = self.soup(injuries_url, params={"saison_id": season_start_year})
        venue_soup = BeautifulSoup(
            self.fetch_html(
                venue_url,
                params={"saison_id": season_start_year},
                league_key=league_key,
                season_start_year=season_start_year,
            ),
            "lxml",
        )

        top_players = self._extract_top_players(squad_soup, limit=5)
        injured_players = self._extract_injuries(injuries_soup)
        injured_top_players = [player["name"] for player in top_players if player["name"] in injured_players]

        return {
            "season": f"{season_start_year}/{season_start_year + 1}",
            "season_start_year": season_start_year,
            "data_reference_season": f"{season_start_year}/{season_start_year + 1}",
            "league_key": self._league_key(league),
            "team_name": team_name,
            "squad_market_value_eur": self._extract_market_value(squad_soup),
            "top_players": top_players,
            "injured_players": injured_players,
            "injured_top_player_count": len(injured_top_players),
            "stadium_capacity": self._extract_stadium_capacity(venue_soup),
            "selector_version": SELECTOR_CONFIG["selector_version"],
            "strategy_used": json.dumps(parse_metadata.get("strategy_used", {}), ensure_ascii=False),
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scraper = TransfermarktScraper(
        delay=args.delay,
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
    )
    seasons = season_years(args.seasons, args.end_season)
    all_matches: list[dict] = []
    all_team_context: list[dict] = []
    rows_per_league_season: list[dict] = []
    errors: list[dict] = []
    warnings: list[dict] = []

    min_match_rows = 100
    min_team_rows = 10

    for league_key in args.leagues:
        league = LEAGUES[league_key]
        for season_start_year in seasons:
            print(f"Scraping {league.name} {season_start_year}/{season_start_year + 1}")
            season_label = f"{season_start_year}/{season_start_year + 1}"
            try:
                match_records, match_reason, match_parse_metadata = scraper.scrape_matches(league, season_start_year)
                team_context_records, team_reason, team_parse_metadata = scraper.scrape_team_context(league, season_start_year)

                all_matches.extend(match_records)
                all_team_context.extend(team_context_records)

                rows_per_league_season.append(
                    {
                        "league_key": league_key,
                        "league_name": league.name,
                        "season": season_label,
                        "matches": len(match_records),
                        "team_context": len(team_context_records),
                        "parse_metadata": {
                            "matches": match_parse_metadata,
                            "team_context": team_parse_metadata,
                        },
                    }
                )

                if match_reason is not None:
                    errors.append(
                        {
                            "type": "matches_scrape_validation",
                            "league_key": league_key,
                            "season": season_label,
                            "reason_code": match_reason,
                            "rows": len(match_records),
                        }
                    )

                if team_reason is not None:
                    errors.append(
                        {
                            "type": "team_context_scrape_validation",
                            "league_key": league_key,
                            "season": season_label,
                            "reason_code": team_reason,
                            "rows": len(team_context_records),
                        }
                    )

                if len(match_records) < min_match_rows:
                    errors.append(
                        {
                            "type": "threshold_breach",
                            "dataset": "matches",
                            "league_key": league_key,
                            "season": season_label,
                            "minimum_required": min_match_rows,
                            "actual": len(match_records),
                            "reason_code": "min_rows_not_met",
                        }
                    )
                if len(team_context_records) < min_team_rows:
                    errors.append(
                        {
                            "type": "threshold_breach",
                            "dataset": "team_context",
                            "league_key": league_key,
                            "season": season_label,
                            "minimum_required": min_team_rows,
                            "actual": len(team_context_records),
                            "reason_code": "min_rows_not_met",
                        }
                    )
            except requests.RequestException as exc:
                errors.append(
                    {
                        "type": "request_exception",
                        "league_key": league_key,
                        "season": season_label,
                        "reason_code": "request_exception",
                        "message": str(exc),
                    }
                )
            finally:
                if scraper.parse_warnings:
                    warnings.extend(scraper.parse_warnings)
                    scraper.parse_warnings = []

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
        "selector_version",
        "strategy_used",
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
        "selector_version",
        "strategy_used",
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
                "rows_per_league_season": rows_per_league_season,
                "errors": errors,
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {matches_path}")
    print(f"Wrote {team_path}")
    print(f"Wrote {metadata_path}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
