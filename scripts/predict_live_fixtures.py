from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prediction_service import MatchPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict live/day fixtures using API-Football and a trained model.")
    parser.add_argument("--league", required=True, help="League display name (e.g. 'Premier League').")
    parser.add_argument("--date", default=None, help="Fixture date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--api-key", default=None, help="API-Football key. Fallback: API_FOOTBALL_KEY env variable.")
    parser.add_argument("--output", default="data/predictions/live_predictions.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set API_FOOTBALL_KEY environment variable.")

    predictor = MatchPredictor(raw_dir=args.raw_dir, processed_dir=args.processed_dir, model_dir=args.model_dir)
    predictions = predictor.predict_live_fixtures(league_name=args.league, api_key=api_key, match_date=args.date)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"Predicted fixtures: {len(predictions)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
