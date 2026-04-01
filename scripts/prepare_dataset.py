from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

TARGET_MAP = {"H": 0, "D": 1, "A": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training data for football match outcome prediction.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def safe_literal(value: object) -> list[dict] | list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []


def build_team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[
        ["match_id", "match_date", "season", "season_start_year", "league_key", "home_team", "away_team", "home_goals", "away_goals"]
    ].copy()
    home.columns = ["match_id", "match_date", "season", "season_start_year", "league_key", "team", "opponent", "goals_for", "goals_against"]
    home["is_home"] = 1
    home["points"] = np.select(
        [home["goals_for"] > home["goals_against"], home["goals_for"] == home["goals_against"]],
        [3, 1],
        default=0,
    )

    away = matches[
        ["match_id", "match_date", "season", "season_start_year", "league_key", "away_team", "home_team", "away_goals", "home_goals"]
    ].copy()
    away.columns = ["match_id", "match_date", "season", "season_start_year", "league_key", "team", "opponent", "goals_for", "goals_against"]
    away["is_home"] = 0
    away["points"] = np.select(
        [away["goals_for"] > away["goals_against"], away["goals_for"] == away["goals_against"]],
        [3, 1],
        default=0,
    )
    return pd.concat([home, away], ignore_index=True).sort_values(["league_key", "team", "match_date"])


def add_rolling_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    team_rows = team_rows.sort_values(["league_key", "team", "match_date"]).copy()
    team_rows["goal_diff"] = team_rows["goals_for"] - team_rows["goals_against"]

    def transform(group: pd.DataFrame) -> pd.DataFrame:
        prior = group.shift(1)
        weighted = np.array([5, 4, 3, 2, 1], dtype=float)
        group["form_points_last_5_weighted"] = prior["points"].rolling(5).apply(
            lambda values: float(np.dot(values, weighted[-len(values) :]) / weighted[-len(values) :].sum()),
            raw=True,
        )
        group["form_points_last_10"] = prior["points"].rolling(10, min_periods=3).mean()
        group["goals_for_last_5"] = prior["goals_for"].rolling(5, min_periods=2).mean()
        group["goals_against_last_5"] = prior["goals_against"].rolling(5, min_periods=2).mean()
        group["goal_diff_last_10"] = prior["goal_diff"].rolling(10, min_periods=3).mean()
        group["wins_last_5"] = prior["points"].rolling(5, min_periods=2).apply(lambda values: float(np.sum(values == 3)), raw=True)
        return group

    return team_rows.groupby(["league_key", "team"], group_keys=False).apply(transform)


def add_head_to_head_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("match_date").copy()
    home_points: list[float] = []
    away_points: list[float] = []
    home_goal_diff: list[float] = []
    history: dict[tuple[str, str], list[dict]] = {}

    for row in matches.itertuples(index=False):
        key = tuple(sorted([row.home_team, row.away_team]))
        past = history.get(key, [])[-5:]
        relevant = []
        for item in past:
            if item["home_team"] == row.home_team:
                relevant.append(item)
            else:
                relevant.append(
                    {
                        "home_goals": item["away_goals"],
                        "away_goals": item["home_goals"],
                    }
                )

        if relevant:
            points = []
            diffs = []
            for item in relevant:
                if item["home_goals"] > item["away_goals"]:
                    points.append(3)
                elif item["home_goals"] == item["away_goals"]:
                    points.append(1)
                else:
                    points.append(0)
                diffs.append(item["home_goals"] - item["away_goals"])
            home_points.append(float(np.mean(points)))
            away_points.append(float(np.mean([3 if p == 0 else 1 if p == 1 else 0 for p in points])))
            home_goal_diff.append(float(np.mean(diffs)))
        else:
            home_points.append(np.nan)
            away_points.append(np.nan)
            home_goal_diff.append(np.nan)

        history.setdefault(key, []).append(
            {
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_goals": row.home_goals,
                "away_goals": row.away_goals,
            }
        )

    matches["h2h_home_points_last_5"] = home_points
    matches["h2h_away_points_last_5"] = away_points
    matches["h2h_goal_diff_last_5"] = home_goal_diff
    return matches


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        matches = pd.read_csv(raw_dir / "matches.csv")
    except EmptyDataError as exc:
        raise ValueError(
            "matches.csv has no usable columns/rows. First fix raw data fetch (scripts/scrape_transfermarkt.py) and rerun it."
        ) from exc
    team_context = pd.read_csv(raw_dir / "team_context.csv")

    if matches.empty:
        raise ValueError(
            "matches.csv is empty. First fix the raw data fetch step (scripts/scrape_transfermarkt.py) and rerun fetching."
        )

    matches["match_date"] = pd.to_datetime(matches["match_date"], utc=False)
    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")
    matches["attendance"] = pd.to_numeric(matches["attendance"], errors="coerce")
    matches = matches.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"]).copy()
    matches["match_id"] = np.arange(len(matches))
    matches["result_code"] = np.select(
        [matches["home_goals"] > matches["away_goals"], matches["home_goals"] == matches["away_goals"]],
        ["H", "D"],
        default="A",
    )
    matches["target"] = matches["result_code"].map(TARGET_MAP)

    team_context["top_players"] = team_context["top_players"].apply(safe_literal)
    team_context["injured_players"] = team_context["injured_players"].apply(safe_literal)
    team_context["top_player_total_value_eur"] = team_context["top_players"].apply(
        lambda players: float(sum((player.get("market_value_eur") or 0) for player in players if isinstance(player, dict)))
    )

    team_rows = add_rolling_features(build_team_match_rows(matches))
    home_features = team_rows[team_rows["is_home"] == 1].copy().add_prefix("home_").rename(columns={"home_match_id": "match_id"})
    away_features = team_rows[team_rows["is_home"] == 0].copy().add_prefix("away_").rename(columns={"away_match_id": "match_id"})

    dataset = matches.merge(home_features, on="match_id", how="left").merge(away_features, on="match_id", how="left")
    dataset = add_head_to_head_features(dataset)

    home_context = team_context.add_prefix("home_").rename(
        columns={"home_team_name": "home_team", "home_season": "season", "home_league_key": "league_key"}
    )
    away_context = team_context.add_prefix("away_").rename(
        columns={"away_team_name": "away_team", "away_season": "season", "away_league_key": "league_key"}
    )
    dataset = dataset.merge(home_context, on=["season", "league_key", "home_team"], how="left")
    dataset = dataset.merge(away_context, on=["season", "league_key", "away_team"], how="left")

    home_mismatch_mask = (
        dataset["home_data_reference_season"].notna() & (dataset["season"] != dataset["home_data_reference_season"])
    )
    away_mismatch_mask = (
        dataset["away_data_reference_season"].notna() & (dataset["season"] != dataset["away_data_reference_season"])
    )
    season_mismatch_rows = dataset[home_mismatch_mask | away_mismatch_mask].copy()
    if not season_mismatch_rows.empty:
        mismatch_summary = (
            season_mismatch_rows.groupby(["season", "league_key"], dropna=False)
            .size()
            .reset_index(name="mismatch_rows")
            .sort_values(["season", "league_key"])
        )
        print(
            "WARNING: season-context mismatch detected "
            f"(home={int(home_mismatch_mask.sum())}, away={int(away_mismatch_mask.sum())}, total={len(season_mismatch_rows)})."
        )
        print(mismatch_summary.to_string(index=False))
    else:
        print("Season-context consistency check passed: no mismatched rows after merge.")

    dataset["attendance_ratio"] = dataset["attendance"] / dataset["home_stadium_capacity"].replace({0: np.nan})
    dataset["home_squad_value_diff"] = dataset["home_squad_market_value_eur"] - dataset["away_squad_market_value_eur"]
    dataset["home_top_player_value_diff"] = dataset["home_top_player_total_value_eur"] - dataset["away_top_player_total_value_eur"]
    dataset["home_injured_top_player_diff"] = dataset["home_injured_top_player_count"] - dataset["away_injured_top_player_count"]
    dataset["home_form_diff_last_5"] = dataset["home_form_points_last_5_weighted"] - dataset["away_form_points_last_5_weighted"]
    dataset["home_form_diff_last_10"] = dataset["home_form_points_last_10"] - dataset["away_form_points_last_10"]
    dataset["home_goal_diff_edge"] = dataset["home_goal_diff_last_10"] - dataset["away_goal_diff_last_10"]
    dataset["home_attack_edge"] = dataset["home_goals_for_last_5"] - dataset["away_goals_for_last_5"]
    dataset["home_defense_edge"] = dataset["away_goals_against_last_5"] - dataset["home_goals_against_last_5"]
    dataset["home_recent_win_edge"] = dataset["home_wins_last_5"] - dataset["away_wins_last_5"]
    dataset["is_home_match"] = 1

    feature_columns = [
        "attendance",
        "attendance_ratio",
        "home_form_diff_last_5",
        "home_form_diff_last_10",
        "home_goal_diff_edge",
        "home_attack_edge",
        "home_defense_edge",
        "home_recent_win_edge",
        "h2h_home_points_last_5",
        "h2h_away_points_last_5",
        "h2h_goal_diff_last_5",
        "home_squad_value_diff",
        "home_top_player_value_diff",
        "home_injured_top_player_diff",
        "home_stadium_capacity",
        "away_stadium_capacity",
        "is_home_match",
    ]

    dataset = dataset.sort_values("match_date").reset_index(drop=True)
    dataset[feature_columns] = dataset[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = dataset[feature_columns].median(numeric_only=True)
    dataset[feature_columns] = dataset[feature_columns].fillna(medians).fillna(0.0)

    dataset_path = out_dir / "training_dataset.parquet"
    features_path = out_dir / "feature_columns.json"
    dataset.to_parquet(dataset_path, index=False)
    features_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    print(f"Wrote {dataset_path}")
    print(f"Wrote {features_path}")
    print(f"Rows: {len(dataset)}")


if __name__ == "__main__":
    main()
