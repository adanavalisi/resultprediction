from __future__ import annotations

import pandas as pd
import streamlit as st

from prediction_service import MatchPredictor


st.set_page_config(page_title="Result Prediction", page_icon="VS", layout="wide")


@st.cache_resource(show_spinner=False)
def load_predictor() -> MatchPredictor:
    return MatchPredictor()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(255,255,255,0.08), transparent 28%),
                linear-gradient(180deg, #151515 0%, #0d0d0d 100%);
            color: #ffffff;
        }
        .block-container {
            max-width: 1200px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }
        .selector-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 24px;
            min-height: 220px;
            backdrop-filter: blur(8px);
        }
        .vs-mark {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 220px;
            font-size: 5.2rem;
            font-weight: 900;
            color: #ffffff;
            letter-spacing: 0.18em;
            text-shadow: 0 0 30px rgba(255,255,255,0.14);
        }
        .chart-title {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 1.5rem;
            margin-bottom: 0.4rem;
        }
        .helper-text {
            text-align: center;
            color: rgba(255,255,255,0.68);
            margin-bottom: 1.8rem;
        }
        .stButton > button {
            background: #ffffff;
            color: #0d0d0d;
            border: none;
            border-radius: 999px;
            padding: 0.9rem 1.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            width: 100%;
        }
        .stButton > button:hover {
            background: #e9e9e9;
            color: #000000;
        }
        div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.12);
            color: #ffffff;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 1.4rem;
        }
        .metric-box {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px;
            text-align: center;
        }
        .metric-label {
            color: rgba(255,255,255,0.68);
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def selector_panel(title: str, leagues: list[str], prefix: str) -> tuple[str, str]:
    with st.container(border=False):
        st.markdown(f"<div class='selector-card'><h3>{title}</h3>", unsafe_allow_html=True)
        league = st.selectbox("League", leagues, key=f"{prefix}_league")
        team = st.selectbox("Team", predictor.teams_for_league(league), key=f"{prefix}_team")
        st.markdown("</div>", unsafe_allow_html=True)
    return league, team


inject_styles()

st.markdown("<div class='helper-text'>Select the league and teams, then run the trained model.</div>", unsafe_allow_html=True)

try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

league_options = predictor.available_leagues()

left_col, center_col, right_col = st.columns([1.35, 0.7, 1.35], gap="large")

with left_col:
    home_league, home_team = selector_panel("Home Side", league_options, "home")

with center_col:
    st.markdown("<div class='vs-mark'>VS</div>", unsafe_allow_html=True)

with right_col:
    away_league, away_team = selector_panel("Away Side", league_options, "away")

if home_league != away_league:
    st.warning("Both teams must be selected from the same league.")

trigger_col = st.columns([1, 1.2, 1])[1]
with trigger_col:
    submitted = st.button("show result prediction", use_container_width=True)

if submitted:
    if home_league != away_league:
        st.error("Select both teams from the same league.")
    elif home_team == away_team:
        st.error("Choose two different teams.")
    else:
        try:
            result = predictor.predict_match(home_league, home_team, away_team)
        except ValueError as exc:
            st.error(str(exc))
        else:
            chart_df = pd.DataFrame(
                {
                    "Outcome": ["Home Win", "Draw", "Away Win"],
                    "Probability": [
                        round(result["probabilities"]["Home Win"] * 100, 2),
                        round(result["probabilities"]["Draw"] * 100, 2),
                        round(result["probabilities"]["Away Win"] * 100, 2),
                    ],
                }
            )

            st.markdown("<div class='chart-title'>Prediction Output</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='helper-text'>{result['league_name']} | model basis date {result['match_date_basis']}</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart(chart_df.set_index("Outcome"), color="#ffffff")

            st.markdown(
                f"""
                <div class="metric-strip">
                    <div class="metric-box">
                        <div class="metric-label">Home Win</div>
                        <div class="metric-value">{chart_df.iloc[0]['Probability']}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Draw</div>
                        <div class="metric-value">{chart_df.iloc[1]['Probability']}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Away Win</div>
                        <div class="metric-value">{chart_df.iloc[2]['Probability']}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
