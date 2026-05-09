from pathlib import Path
import argparse
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# GDELT uses FIPS-style country codes in ActionGeo_CountryCode.
# These are approximate country centroids for visualization.
# Add more countries if needed.
# ---------------------------------------------------------------------
COUNTRY_INFO = {
    "US": {"name": "United States", "lat": 39.8, "lon": -98.6},
    "CH": {"name": "China", "lat": 35.9, "lon": 104.2},
    "TW": {"name": "Taiwan", "lat": 23.7, "lon": 121.0},
    "JA": {"name": "Japan", "lat": 36.2, "lon": 138.2},
    "KS": {"name": "South Korea", "lat": 36.5, "lon": 127.8},
    "RS": {"name": "Russia", "lat": 61.5, "lon": 105.3},
    "UP": {"name": "Ukraine", "lat": 49.0, "lon": 31.0},
    "IR": {"name": "Iran", "lat": 32.4, "lon": 53.7},
    "IZ": {"name": "Iraq", "lat": 33.2, "lon": 43.7},
    "SA": {"name": "Saudi Arabia", "lat": 23.9, "lon": 45.1},
    "AE": {"name": "United Arab Emirates", "lat": 24.3, "lon": 54.4},
    "KU": {"name": "Kuwait", "lat": 29.3, "lon": 47.5},
    "QA": {"name": "Qatar", "lat": 25.4, "lon": 51.2},
    "VE": {"name": "Venezuela", "lat": 6.4, "lon": -66.6},
    "SY": {"name": "Syria", "lat": 35.0, "lon": 38.5},
    "YM": {"name": "Yemen", "lat": 15.6, "lon": 48.5},
    "IS": {"name": "Israel", "lat": 31.0, "lon": 35.0},
    "UK": {"name": "United Kingdom", "lat": 55.4, "lon": -3.4},
    "FR": {"name": "France", "lat": 46.2, "lon": 2.2},
    "GM": {"name": "Germany", "lat": 51.2, "lon": 10.4},
    "CA": {"name": "Canada", "lat": 56.1, "lon": -106.3},
    "IN": {"name": "India", "lat": 20.6, "lon": 78.9},
    "PK": {"name": "Pakistan", "lat": 30.4, "lon": 69.3},
    "TU": {"name": "Turkey", "lat": 39.0, "lon": 35.2},
    "EI": {"name": "Ireland", "lat": 53.1, "lon": -8.2},
    "AS": {"name": "Australia", "lat": -25.3, "lon": 133.8},
    "SF": {"name": "South Africa", "lat": -30.6, "lon": 22.9},
    "RP": {"name": "Philippines", "lat": 12.9, "lon": 121.8},
    "NI": {"name": "Nigeria", "lat": 9.1, "lon": 8.7},
}


ASSET_THEMES = {
    "Gold safe-haven risk": {
        "asset": "Gold",
        "countries": ["US", "RS", "UP", "IR", "IS", "SY", "SA", "CH", "UK"],
        "root_codes": ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
        "color": "#D4A017",
        "asset_lon": 158,
        "asset_lat": 52,
        "asset_label": "Gold",
    },
    "Oil geopolitical risk": {
        "asset": "WTI Oil",
        "countries": ["US", "RS", "UP", "IR", "IZ", "SA", "AE", "KU", "QA", "VE", "SY", "YM"],
        "root_codes": ["15", "16", "17", "18", "19", "20"],
        "color": "#8B4513",
        "asset_lon": 158,
        "asset_lat": 27,
        "asset_label": "WTI Oil",
    },
    "China-Taiwan attention": {
        "asset": "QQQ",
        "countries": ["CH", "TW"],
        "root_codes": None,  # all root codes
        "color": "#1F77B4",
        "asset_lon": 158,
        "asset_lat": 4,
        "asset_label": "QQQ",
    },
}


ROOT_GROUP_NAMES = {
    "01": "Make public statement",
    "02": "Appeal",
    "03": "Express intent to cooperate",
    "04": "Consult",
    "05": "Engage in diplomatic cooperation",
    "06": "Engage in material cooperation",
    "07": "Provide aid",
    "08": "Yield",
    "09": "Investigate",
    "10": "Demand",
    "11": "Disapprove",
    "12": "Reject",
    "13": "Threaten",
    "14": "Protest",
    "15": "Exhibit force posture",
    "16": "Reduce relations",
    "17": "Coerce",
    "18": "Assault",
    "19": "Fight",
    "20": "Use unconventional mass violence",
}


def month_file_from_date(event_long_dir: Path, selected_date: str) -> Path:
    month = selected_date.replace("-", "")[:6]
    path = event_long_dir / f"events_market_day_long_{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find event_long file for month {month}: {path}")
    return path


def read_selected_day(event_long_dir: Path, selected_date: str) -> pd.DataFrame:
    path = month_file_from_date(event_long_dir, selected_date)
    df = pd.read_parquet(path)

    required = [
        "market_day_ny",
        "EventRootCode",
        "ActionGeo_CountryCode",
        "n_events",
        "goldstein_neg_abs_sum",
        "tone_mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    day = df[df["market_day_ny"].astype(str) == selected_date].copy()

    if day.empty:
        available = sorted(df["market_day_ny"].astype(str).unique())[:10]
        raise ValueError(
            f"No rows found for selected_date={selected_date} in {path}. "
            f"Example available dates in this file: {available}"
        )

    day["EventRootCode"] = day["EventRootCode"].fillna("UNK").astype(str)
    day["ActionGeo_CountryCode"] = day["ActionGeo_CountryCode"].fillna("UNK").astype(str)
    day["n_events"] = pd.to_numeric(day["n_events"], errors="coerce").fillna(0.0)

    if "goldstein_neg_abs_sum" in day.columns:
        day["goldstein_neg_abs_sum"] = pd.to_numeric(
            day["goldstein_neg_abs_sum"], errors="coerce"
        ).fillna(0.0)

    if "tone_mean" in day.columns:
        day["tone_mean"] = pd.to_numeric(day["tone_mean"], errors="coerce").fillna(0.0)

    return day


def summarize_country_events(day: pd.DataFrame, top_n: int) -> pd.DataFrame:
    country = (
        day.groupby("ActionGeo_CountryCode", as_index=False)
        .agg(
            n_events=("n_events", "sum"),
            neg_goldstein=("goldstein_neg_abs_sum", "sum"),
            avg_tone=("tone_mean", "mean"),
        )
        .sort_values("n_events", ascending=False)
    )

    country = country[country["ActionGeo_CountryCode"].isin(COUNTRY_INFO)].copy()
    country = country.head(top_n).copy()

    country["name"] = country["ActionGeo_CountryCode"].map(lambda c: COUNTRY_INFO[c]["name"])
    country["lat"] = country["ActionGeo_CountryCode"].map(lambda c: COUNTRY_INFO[c]["lat"])
    country["lon"] = country["ActionGeo_CountryCode"].map(lambda c: COUNTRY_INFO[c]["lon"])

    return country


def summarize_theme(day: pd.DataFrame, theme_name: str, spec: dict) -> pd.DataFrame:
    mask = day["ActionGeo_CountryCode"].isin(spec["countries"])

    if spec["root_codes"] is not None:
        mask = mask & day["EventRootCode"].isin(spec["root_codes"])

    sub = day[mask].copy()

    if sub.empty:
        return pd.DataFrame()

    out = (
        sub.groupby("ActionGeo_CountryCode", as_index=False)
        .agg(
            n_events=("n_events", "sum"),
            neg_goldstein=("goldstein_neg_abs_sum", "sum"),
            avg_tone=("tone_mean", "mean"),
        )
    )

    out = out[out["ActionGeo_CountryCode"].isin(COUNTRY_INFO)].copy()

    if out.empty:
        return out

    out["theme"] = theme_name
    out["asset"] = spec["asset"]
    out["color"] = spec["color"]
    out["name"] = out["ActionGeo_CountryCode"].map(lambda c: COUNTRY_INFO[c]["name"])
    out["lat"] = out["ActionGeo_CountryCode"].map(lambda c: COUNTRY_INFO[c]["lat"])
    out["lon"] = out["ActionGeo_CountryCode"].map(lambda c: COUNTRY_INFO[c]["lon"])

    return out


def marker_size(values, min_size=8, max_size=34):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values

    x = np.log1p(values)

    if np.nanmax(x) == np.nanmin(x):
        return np.full_like(x, (min_size + max_size) / 2)

    scaled = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    return min_size + scaled * (max_size - min_size)


def weighted_centroid(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return None, None

    w = df["n_events"].astype(float).values
    if np.sum(w) <= 0:
        return float(df["lat"].mean()), float(df["lon"].mean())

    lat = float(np.average(df["lat"].values, weights=w))
    lon = float(np.average(df["lon"].values, weights=w))
    return lat, lon


def build_map(day: pd.DataFrame, selected_date: str, top_country_n: int, output_dir: Path):
    country_summary = summarize_country_events(day, top_n=top_country_n)

    theme_parts = []
    for theme_name, spec in ASSET_THEMES.items():
        theme_df = summarize_theme(day, theme_name, spec)
        if not theme_df.empty:
            theme_parts.append(theme_df)

    theme_summary = pd.concat(theme_parts, ignore_index=True) if theme_parts else pd.DataFrame()

    fig = go.Figure()

    # -----------------------------------------------------------------
    # Layer 1: top countries by total event volume on selected day
    # -----------------------------------------------------------------
    if not country_summary.empty:
        fig.add_trace(
            go.Scattergeo(
                lon=country_summary["lon"],
                lat=country_summary["lat"],
                mode="markers+text",
                text=country_summary["ActionGeo_CountryCode"],
                textposition="top center",
                marker=dict(
                    size=marker_size(country_summary["n_events"], min_size=8, max_size=28),
                    color="rgba(120,120,120,0.45)",
                    line=dict(width=0.5, color="black"),
                ),
                customdata=np.stack(
                    [
                        country_summary["name"],
                        country_summary["n_events"],
                        country_summary["neg_goldstein"],
                        country_summary["avg_tone"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Total events: %{customdata[1]:,.0f}<br>"
                    "Negative Goldstein: %{customdata[2]:,.1f}<br>"
                    "Avg tone: %{customdata[3]:.2f}<extra></extra>"
                ),
                name="Top countries by event volume",
            )
        )

    # -----------------------------------------------------------------
    # Layer 2: theme-specific event countries
    # -----------------------------------------------------------------
    if not theme_summary.empty:
        for theme_name, spec in ASSET_THEMES.items():
            sub = theme_summary[theme_summary["theme"] == theme_name].copy()

            if sub.empty:
                continue

            fig.add_trace(
                go.Scattergeo(
                    lon=sub["lon"],
                    lat=sub["lat"],
                    mode="markers+text",
                    text=sub["ActionGeo_CountryCode"],
                    textposition="bottom center",
                    marker=dict(
                        size=marker_size(sub["n_events"], min_size=12, max_size=38),
                        color=spec["color"],
                        opacity=0.82,
                        line=dict(width=1.5, color="white"),
                    ),
                    customdata=np.stack(
                        [
                            sub["name"],
                            sub["theme"],
                            sub["n_events"],
                            sub["neg_goldstein"],
                            sub["avg_tone"],
                        ],
                        axis=1,
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Theme: %{customdata[1]}<br>"
                        "Theme events: %{customdata[2]:,.0f}<br>"
                        "Negative Goldstein: %{customdata[3]:,.1f}<br>"
                        "Avg tone: %{customdata[4]:.2f}<extra></extra>"
                    ),
                    name=theme_name,
                )
            )

    # -----------------------------------------------------------------
    # Layer 3: pseudo asset nodes and theme-to-asset lines
    # -----------------------------------------------------------------
    for theme_name, spec in ASSET_THEMES.items():
        if theme_summary.empty:
            continue

        sub = theme_summary[theme_summary["theme"] == theme_name].copy()
        if sub.empty:
            continue

        lat0, lon0 = weighted_centroid(sub)
        lat1, lon1 = spec["asset_lat"], spec["asset_lon"]

        fig.add_trace(
            go.Scattergeo(
                lon=[lon0, lon1],
                lat=[lat0, lat1],
                mode="lines",
                line=dict(width=2.5, color=spec["color"]),
                opacity=0.65,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lon=[lon1],
                lat=[lat1],
                mode="markers+text",
                text=[spec["asset_label"]],
                textposition="middle right",
                marker=dict(
                    size=20,
                    color=spec["color"],
                    line=dict(width=1.5, color="black"),
                ),
                name=f"{spec['asset_label']} asset node",
                hovertemplate=f"<b>{spec['asset_label']}</b><br>{theme_name}<extra></extra>",
            )
        )

    title = (
        f"Selected-day GDELT event map: {selected_date}<br>"
        "<sup>Bubble size reflects event volume; colored bubbles show asset-specific event themes.</sup>"
    )

    fig.update_layout(
        title=title,
        width=1400,
        height=760,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="v",
            x=0.02,
            y=0.05,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(235,235,235)",
            showcountries=True,
            countrycolor="rgb(180,180,180)",
            showocean=True,
            oceancolor="rgb(245,248,252)",
            showlakes=True,
            lakecolor="rgb(245,248,252)",
            showframe=False,
            lataxis=dict(range=[-55, 80]),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    html_file = output_dir / f"selected_day_event_map_{selected_date}.html"
    png_file = output_dir / f"selected_day_event_map_{selected_date}.png"

    fig.write_html(html_file)
    print(f"Saved HTML: {html_file}")

    try:
        fig.write_image(png_file, scale=2)
        print(f"Saved PNG:  {png_file}")
    except Exception as e:
        print("[WARNING] PNG export failed. HTML was still saved.")
        print("Reason:", e)
        print("Install kaleido if needed: pip install kaleido")

    # Save the exact data behind the map.
    country_file = output_dir / f"selected_day_country_summary_{selected_date}.csv"
    theme_file = output_dir / f"selected_day_theme_summary_{selected_date}.csv"

    country_summary.to_csv(country_file, index=False)
    if not theme_summary.empty:
        theme_summary.to_csv(theme_file, index=False)

    print(f"Saved country summary: {country_file}")
    if not theme_summary.empty:
        print(f"Saved theme summary:   {theme_file}")

    print("\nTheme summary:")
    if theme_summary.empty:
        print("No theme rows found.")
    else:
        show_cols = [
            "theme",
            "asset",
            "ActionGeo_CountryCode",
            "name",
            "n_events",
            "neg_goldstein",
            "avg_tone",
        ]
        print(theme_summary[show_cols].sort_values(["theme", "n_events"], ascending=[True, False]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_long_dir",
        default="/project/hrao/GDELT/features/event_long",
    )
    parser.add_argument(
        "--selected_date",
        default="2022-02-24",
        help="Trading/event market day to plot, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--top_country_n",
        type=int,
        default=25,
        help="Number of top countries by total event volume to show as gray bubbles.",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/ppt_figures_v1/maps",
    )

    args = parser.parse_args()

    event_long_dir = Path(args.event_long_dir)
    output_dir = Path(args.output_dir)

    day = read_selected_day(event_long_dir, args.selected_date)

    print("Selected date:", args.selected_date)
    print("Rows on selected day:", day.shape)
    print("Total events:", day["n_events"].sum())

    build_map(
        day=day,
        selected_date=args.selected_date,
        top_country_n=args.top_country_n,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()