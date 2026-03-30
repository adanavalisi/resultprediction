from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass(frozen=True)
class PredictionArtifacts:
    model: tf.keras.Model
    scaler_bundle: dict
    matches: pd.DataFrame
    team_context: pd.DataFrame
    training_dataset: pd.DataFrame
    feature_columns: list[str]


class MatchPredictor:
    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        model_dir: str = "models",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.model_dir = Path(model_dir)
        self.artifacts = self._load()

    def available_leagues(self) -> list[str]:
        leagues = (
            self.artifacts.matches[["league_name", "league_key"]]
            .dropna()
            .drop_duplicates()
            .sort_values("league_name")
        )
        return leagues["league_name"].tolist()

    def teams_for_league(self, league_name: str) -> list[str]:
        league_matches = self.artifacts.matches[self.artifacts.matches["league_name"] == league_name]
        teams = sorted(set(league_matches["home_team"]).union(set(league_matches["away_team"])))
        return teams

    def predict_match(self, league_name: str, home_team: str, away_team: str) -> dict:
        if home_team == away_team:
            raise ValueError("Home and away team must be different.")

        league_key = self._league_key_from_name(league_name)
        latest_match_date = self.artifacts.matches[
            (self.artifacts.matches["league_key"] == league_key)
            & ((self.artifacts.matches["home_team"] == home_team) | (self.artifacts.matches["away_team"] == home_team))
        ]["match_date"].max()

        if pd.isna(latest_match_date):
            raise ValueError("No recent match history found for the selected home team.")

        home_snapshot = self._team_snapshot(league_key, home_team)
        away_snapshot = self._team_snapshot(league_key, away_team)
        h2h_snapshot = self._head_to_head_snapshot(home_team, away_team)
        context_snapshot = self._context_snapshot(league_key, home_team, away_team)

        feature_row = {
            "attendance": context_snapshot["estimated_attendance"],
            "attendance_ratio": context_snapshot["attendance_ratio"],
            "home_form_diff_last_5": home_snapshot["form_points_last_5_weighted"] - away_snapshot["form_points_last_5_weighted"],
            "home_form_diff_last_10": home_snapshot["form_points_last_10"] - away_snapshot["form_points_last_10"],
            "home_goal_diff_edge": home_snapshot["goal_diff_last_10"] - away_snapshot["goal_diff_last_10"],
            "home_attack_edge": home_snapshot["goals_for_last_5"] - away_snapshot["goals_for_last_5"],
            "home_defense_edge": away_snapshot["goals_against_last_5"] - home_snapshot["goals_against_last_5"],
            "home_recent_win_edge": home_snapshot["wins_last_5"] - away_snapshot["wins_last_5"],
            "h2h_home_points_last_5": h2h_snapshot["h2h_home_points_last_5"],
            "h2h_away_points_last_5": h2h_snapshot["h2h_away_points_last_5"],
            "h2h_goal_diff_last_5": h2h_snapshot["h2h_goal_diff_last_5"],
            "home_squad_value_diff": context_snapshot["home_squad_market_value_eur"] - context_snapshot["away_squad_market_value_eur"],
            "home_top_player_value_diff": context_snapshot["home_top_player_total_value_eur"] - context_snapshot["away_top_player_total_value_eur"],
            "home_injured_top_player_diff": context_snapshot["home_injured_top_player_count"] - context_snapshot["away_injured_top_player_count"],
            "home_stadium_capacity": context_snapshot["home_stadium_capacity"],
            "away_stadium_capacity": context_snapshot["away_stadium_capacity"],
            "is_home_match": 1,
        }

        feature_frame = pd.DataFrame([feature_row])[self.artifacts.feature_columns]
        medians = self.artifacts.training_dataset[self.artifacts.feature_columns].median(numeric_only=True)
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(medians)

        scaler = self.artifacts.scaler_bundle["scaler"]
        probabilities = self.artifacts.model.predict(scaler.transform(feature_frame), verbose=0)[0]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "league_name": league_name,
            "match_date_basis": str(latest_match_date.date()),
            "probabilities": {
                "Home Win": float(probabilities[0]),
                "Draw": float(probabilities[1]),
                "Away Win": float(probabilities[2]),
            },
            "predicted_label": ["1", "X", "2"][int(np.argmax(probabilities))],
            "feature_row": feature_row,
        }

    def _load(self) -> PredictionArtifacts:
        matches_path = self.raw_dir / "matches.csv"
        context_path = self.raw_dir / "team_context.csv"
        training_path = self.processed_dir / "training_dataset.parquet"
        feature_columns_path = self.processed_dir / "feature_columns.json"
        scaler_path = self.model_dir / "feature_scaler.joblib"
        model_path = self.model_dir / "football_outcome_dnn.keras"

        missing = [
            str(path)
            for path in [matches_path, context_path, training_path, feature_columns_path, scaler_path, model_path]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required artifacts. Run scraping, dataset preparation and training first: "
                + ", ".join(missing)
            )

        matches = pd.read_csv(matches_path)
        matches["match_date"] = pd.to_datetime(matches["match_date"], utc=False)
        matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="coerce")
        matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="coerce")
        matches["attendance"] = pd.to_numeric(matches["attendance"], errors="coerce")

        team_context = pd.read_csv(context_path)
        team_context["season_start_year"] = pd.to_numeric(team_context["season_start_year"], errors="coerce")
        team_context["squad_market_value_eur"] = pd.to_numeric(team_context["squad_market_value_eur"], errors="coerce")
        team_context["injured_top_player_count"] = pd.to_numeric(team_context["injured_top_player_count"], errors="coerce")
        team_context["stadium_capacity"] = pd.to_numeric(team_context["stadium_capacity"], errors="coerce")
        team_context["top_players"] = team_context["top_players"].apply(self._safe_literal)
        team_context["top_player_total_value_eur"] = team_context["top_players"].apply(
            lambda players: float(sum((player.get("market_value_eur") or 0) for player in players if isinstance(player, dict)))
        )

        training_dataset = pd.read_parquet(training_path)
        feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
        scaler_bundle = joblib.load(scaler_path)
        model = tf.keras.models.load_model(model_path)

        return PredictionArtifacts(
            model=model,
            scaler_bundle=scaler_bundle,
            matches=matches,
            team_context=team_context,
            training_dataset=training_dataset,
            feature_columns=feature_columns,
        )

    def _league_key_from_name(self, league_name: str) -> str:
        match = self.artifacts.matches.loc[self.artifacts.matches["league_name"] == league_name, "league_key"]
        if match.empty:
            raise ValueError("League not found.")
        return str(match.iloc[0])

    def _team_snapshot(self, league_key: str, team_name: str) -> dict:
        team_matches = self.artifacts.matches[
            (self.artifacts.matches["league_key"] == league_key)
            & ((self.artifacts.matches["home_team"] == team_name) | (self.artifacts.matches["away_team"] == team_name))
        ].sort_values("match_date")

        if team_matches.empty:
            raise ValueError(f"No match history found for {team_name}.")

        rows = []
        for row in team_matches.itertuples(index=False):
            is_home = row.home_team == team_name
            goals_for = row.home_goals if is_home else row.away_goals
            goals_against = row.away_goals if is_home else row.home_goals
            if goals_for > goals_against:
                points = 3
            elif goals_for == goals_against:
                points = 1
            else:
                points = 0
            rows.append(
                {
                    "match_date": row.match_date,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_diff": goals_for - goals_against,
                    "points": points,
                }
            )

        history = pd.DataFrame(rows).sort_values("match_date")
        last_five = history.tail(5)
        last_ten = history.tail(10)
        weights = np.array([5, 4, 3, 2, 1], dtype=float)[-len(last_five) :]

        return {
            "form_points_last_5_weighted": float(np.dot(last_five["points"], weights) / weights.sum()) if not last_five.empty else np.nan,
            "form_points_last_10": float(last_ten["points"].mean()) if not last_ten.empty else np.nan,
            "goals_for_last_5": float(last_five["goals_for"].mean()) if not last_five.empty else np.nan,
            "goals_against_last_5": float(last_five["goals_against"].mean()) if not last_five.empty else np.nan,
            "goal_diff_last_10": float(last_ten["goal_diff"].mean()) if not last_ten.empty else np.nan,
            "wins_last_5": float((last_five["points"] == 3).sum()) if not last_five.empty else np.nan,
        }

    def _head_to_head_snapshot(self, home_team: str, away_team: str) -> dict:
        matches = self.artifacts.matches[
            ((self.artifacts.matches["home_team"] == home_team) & (self.artifacts.matches["away_team"] == away_team))
            | ((self.artifacts.matches["home_team"] == away_team) & (self.artifacts.matches["away_team"] == home_team))
        ].sort_values("match_date")

        recent = matches.tail(5)
        if recent.empty:
            return {
                "h2h_home_points_last_5": np.nan,
                "h2h_away_points_last_5": np.nan,
                "h2h_goal_diff_last_5": np.nan,
            }

        home_points = []
        away_points = []
        goal_diffs = []
        for row in recent.itertuples(index=False):
            if row.home_team == home_team:
                home_goals = row.home_goals
                away_goals = row.away_goals
            else:
                home_goals = row.away_goals
                away_goals = row.home_goals

            if home_goals > away_goals:
                home_points.append(3)
                away_points.append(0)
            elif home_goals == away_goals:
                home_points.append(1)
                away_points.append(1)
            else:
                home_points.append(0)
                away_points.append(3)
            goal_diffs.append(home_goals - away_goals)

        return {
            "h2h_home_points_last_5": float(np.mean(home_points)),
            "h2h_away_points_last_5": float(np.mean(away_points)),
            "h2h_goal_diff_last_5": float(np.mean(goal_diffs)),
        }

    def _context_snapshot(self, league_key: str, home_team: str, away_team: str) -> dict:
        home_context = self._latest_team_context(league_key, home_team)
        away_context = self._latest_team_context(league_key, away_team)

        home_matches = self.artifacts.matches[
            (self.artifacts.matches["league_key"] == league_key) & (self.artifacts.matches["home_team"] == home_team)
        ].sort_values("match_date")
        estimated_attendance = float(home_matches["attendance"].dropna().tail(10).mean()) if not home_matches.empty else np.nan
        home_capacity = home_context["stadium_capacity"]
        attendance_ratio = estimated_attendance / home_capacity if pd.notna(estimated_attendance) and pd.notna(home_capacity) and home_capacity else np.nan

        return {
            "estimated_attendance": estimated_attendance,
            "attendance_ratio": attendance_ratio,
            "home_squad_market_value_eur": home_context["squad_market_value_eur"],
            "away_squad_market_value_eur": away_context["squad_market_value_eur"],
            "home_top_player_total_value_eur": home_context["top_player_total_value_eur"],
            "away_top_player_total_value_eur": away_context["top_player_total_value_eur"],
            "home_injured_top_player_count": home_context["injured_top_player_count"],
            "away_injured_top_player_count": away_context["injured_top_player_count"],
            "home_stadium_capacity": home_capacity,
            "away_stadium_capacity": away_context["stadium_capacity"],
        }

    def _latest_team_context(self, league_key: str, team_name: str) -> pd.Series:
        rows = self.artifacts.team_context[
            (self.artifacts.team_context["league_key"] == league_key) & (self.artifacts.team_context["team_name"] == team_name)
        ].sort_values("season_start_year")
        if rows.empty:
            raise ValueError(f"No context data found for {team_name}.")
        return rows.iloc[-1]

    @staticmethod
    def _safe_literal(value: object) -> list[dict]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return value
        try:
            return ast.literal_eval(str(value))
        except Exception:
            return []
