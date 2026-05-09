import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gdelt_risk.features import parse_ddfe_root_country


def save_heatmap(df, metric, out_path):
    pivot = df.pivot_table(index="method", columns="asset_target", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.45)))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(metric)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default="/project/hrao/GDELT/results/ddfe_event_fusion_summary_v1")
    parser.add_argument("--output_dir", default="/project/hrao/GDELT/results/ddfe_event_fusion_summary_v1/figures")
    args = parser.parse_args()
    summary_dir = Path(args.summary_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta = pd.read_csv(summary_dir / "delta_vs_price_only.csv")
    delta = delta[delta["method"] != "price_only"].copy()
    delta["asset_target"] = delta["asset"].astype(str) + " | " + delta["target"].astype(str)
    save_heatmap(delta, "delta_auc", out_dir / "delta_auc_heatmap.png")
    save_heatmap(delta, "delta_average_precision", out_dir / "delta_average_precision_heatmap.png")
    save_heatmap(delta, "delta_precision_at_top20pct", out_dir / "delta_precision_top20_heatmap.png")

    summary = pd.read_csv(summary_dir / "all_methods_summary.csv")
    summary["asset_target"] = summary["asset"].astype(str) + " | " + summary["target"].astype(str)
    for metric in ["auc", "average_precision", "precision_at_top20pct"]:
        pivot = summary.pivot_table(index="asset_target", columns="method", values=metric, aggfunc="mean")
        ax = pivot.plot(kind="bar", figsize=(max(10, len(pivot) * 1.2), 5))
        ax.set_ylabel(metric)
        ax.set_title(f"Method comparison: {metric}")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"method_comparison_{metric}.png", dpi=180)
        plt.close()

    freq_path = summary_dir / "selected_event_feature_frequency.csv"
    if freq_path.exists():
        freq = pd.read_csv(freq_path)
        if not freq.empty:
            fam = freq.groupby("family")["selection_count"].sum().sort_values(ascending=False)
            ax = fam.plot(kind="bar", figsize=(8, 4))
            ax.set_ylabel("selection count")
            ax.set_title("Selected event family frequency")
            plt.tight_layout()
            plt.savefig(out_dir / "selected_event_family_frequency.png", dpi=180)
            plt.close()

            parsed = freq["event_column"].apply(parse_ddfe_root_country)
            freq["root_code"] = [x[0] for x in parsed]
            freq["country"] = [x[1] for x in parsed]
            root_country = (
                freq.dropna(subset=["root_code", "country"])
                .assign(root_country=lambda x: x["root_code"] + "_" + x["country"])
                .groupby("root_country")["selection_count"]
                .sum()
                .sort_values(ascending=False)
                .head(25)
            )
            if not root_country.empty:
                ax = root_country.plot(kind="bar", figsize=(10, 4))
                ax.set_ylabel("selection count")
                ax.set_title("Selected root-code/country frequency")
                plt.tight_layout()
                plt.savefig(out_dir / "selected_root_country_frequency.png", dpi=180)
                plt.close()


if __name__ == "__main__":
    main()
