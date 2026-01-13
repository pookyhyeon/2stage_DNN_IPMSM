import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0) 논문용 스타일 (여기만 계속 만지면 됨)
# =========================================================
STYLE = {
    # save
    "dpi": 600,
    "save_formats": ["png"],

    # figure sizes
    "figsize_wide": (7.2, 3.8),
    "figsize_square": (4.6, 4.6),

    # fonts
    "title": 18,
    "label": 18,
    "tick": 18,
    "legend": 18,

    # lines / markers
    "lw": 2.2,
    "ms": 5,

    # grid
    "grid_alpha": 0.25,

    # axis
    "rpm_unit": "rpm",  # "rpm" or "krpm"
}

OUT_DIR = "paper_figs"
DEFAULT_CSV = os.path.join("machine", "metrics_torque.csv")
DEFAULT_JSON = os.path.join("machine", "metrics_torque.json")
HISTORY_PATH = os.path.join("machine", "history_torque.npz")
VAL_PRED_PATH = os.path.join("machine", "val_pred_torque.npz")


# =========================================================
# Utils
# =========================================================
def apply_axes_style(ax, title=None, xlabel=None, ylabel=None, legend=True):
    if title:
        ax.set_title(title, fontsize=STYLE["title"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE["label"])

    ax.tick_params(axis="both", labelsize=STYLE["tick"])
    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle="--")
    if legend:
        ax.legend(fontsize=STYLE["legend"], frameon=True)


def save_fig(fig, outbase):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in STYLE["save_formats"]:
        fig.savefig(os.path.join(OUT_DIR, f"{outbase}.{ext}"),
                    dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def rpm_x(rpm_array):
    rpm_array = np.asarray(rpm_array, dtype=float)
    if STYLE["rpm_unit"].lower() == "krpm":
        return rpm_array / 1000.0, "Speed (krpm)"
    return rpm_array, "Speed (rpm)"


def load_metrics(csv_path, json_path):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["rpm"].astype(str).str.upper() != "MEAN"].copy()
        df["rpm"] = df["rpm"].astype(int)
        return df.sort_values("rpm").reset_index(drop=True)

    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        df = pd.DataFrame(js["per_rpm"])
        df["rpm"] = df["rpm"].astype(int)
        return df.sort_values("rpm").reset_index(drop=True)

    raise FileNotFoundError(f"metrics 파일을 못 찾음: {csv_path} / {json_path}")


# =========================================================
# A) Metrics Figures
# =========================================================
def fig_metric_curves(df):
    x, xlabel = rpm_x(df["rpm"].values)

    fig, ax = plt.subplots(figsize=STYLE["figsize_wide"])
    ax.plot(x, df["rmse_Nm"].values, marker="o", linewidth=STYLE["lw"],
            markersize=STYLE["ms"], label="RMSE")
    apply_axes_style(ax, title="RMSE over Speed", xlabel=xlabel, ylabel="RMSE (Nm)")
    save_fig(fig, "metric_rmse_over_speed")

    fig, ax = plt.subplots(figsize=STYLE["figsize_wide"])
    ax.plot(x, df["mae_Nm"].values, marker="o", linewidth=STYLE["lw"],
            markersize=STYLE["ms"], label="MAE")
    apply_axes_style(ax, title="MAE over Speed", xlabel=xlabel, ylabel="MAE (Nm)")
    save_fig(fig, "metric_mae_over_speed")

    fig, ax = plt.subplots(figsize=STYLE["figsize_wide"])
    ax.plot(x, df["r2"].values, marker="o", linewidth=STYLE["lw"],
            markersize=STYLE["ms"], label="$R^2$")
    ax.set_ylim(-0.1, 1.05)
    apply_axes_style(ax, title="$R^2$ over Speed", xlabel=xlabel, ylabel="$R^2$")
    save_fig(fig, "metric_r2_over_speed")


def fig_metric_summary(df):
    x, xlabel = rpm_x(df["rpm"].values)

    fig, ax = plt.subplots(figsize=STYLE["figsize_wide"])
    ax.plot(x, df["rmse_Nm"].values, marker="o", linewidth=STYLE["lw"],
            markersize=STYLE["ms"], label="RMSE (Nm)")
    ax.plot(x, df["mae_Nm"].values, marker="s", linewidth=STYLE["lw"],
            markersize=STYLE["ms"], label="MAE (Nm)")

    ax2 = ax.twinx()
    ax2.plot(x, df["r2"].values, marker="^", linewidth=STYLE["lw"],
             markersize=STYLE["ms"], label="$R^2$")
    ax2.set_ylim(-0.1, 1.05)
    ax2.tick_params(axis="y", labelsize=STYLE["tick"])

    ax.set_title("Per-speed Metrics (Torque Prediction)", fontsize=STYLE["title"])
    ax.set_xlabel(xlabel, fontsize=STYLE["label"])
    ax.set_ylabel("Error (Nm)", fontsize=STYLE["label"])
    ax2.set_ylabel("$R^2$", fontsize=STYLE["label"])

    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle="--")
    ax.tick_params(axis="both", labelsize=STYLE["tick"])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=STYLE["legend"], frameon=True, loc="best")

    save_fig(fig, "metric_summary_rmse_mae_r2")


def fig_worst_rpms(df, topk=3):
    worst = df.sort_values("rmse_Nm", ascending=False).head(topk)
    x, xlabel = rpm_x(worst["rpm"].values)

    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    ax.bar(x, worst["rmse_Nm"].values)
    apply_axes_style(ax, title=f"Worst-{topk} Speeds (by RMSE)", xlabel=xlabel, ylabel="RMSE (Nm)", legend=False)
    save_fig(fig, f"metric_worst{topk}_rmse_bar")


# =========================================================
# B) Validation Figures
# =========================================================
def plot_learning_curve():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"history 파일이 없음: {HISTORY_PATH}")

    hist = np.load(HISTORY_PATH)
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]

    fig, ax = plt.subplots(figsize=STYLE["figsize_wide"])
    ax.plot(train_loss, label="Train", linewidth=STYLE["lw"])
    ax.plot(val_loss, label="Validation", linewidth=STYLE["lw"])
    apply_axes_style(ax, title="Learning Curve", xlabel="Epoch", ylabel="MSE Loss")
    save_fig(fig, "learning_curve_train_val")


def plot_parity():
    if not os.path.exists(VAL_PRED_PATH):
        raise FileNotFoundError(f"val_pred 파일이 없음: {VAL_PRED_PATH}")

    data = np.load(VAL_PRED_PATH)
    y_true = data["y_true"].reshape(-1)
    y_pred = data["y_pred"].reshape(-1)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    ax.plot(lims, lims, "k--", linewidth=1.5)
    apply_axes_style(ax, title="Parity Plot (Validation)", xlabel="FEA Torque (Nm)", ylabel="Predicted Torque (Nm)", legend=False)
    ax.set_xlim(lims); ax.set_ylim(lims)
    save_fig(fig, "parity_plot_validation")


def plot_residual():
    if not os.path.exists(VAL_PRED_PATH):
        raise FileNotFoundError(f"val_pred 파일이 없음: {VAL_PRED_PATH}")

    data = np.load(VAL_PRED_PATH)
    y_true = data["y_true"].reshape(-1)
    y_pred = data["y_pred"].reshape(-1)
    residual = y_pred - y_true

    fig, ax = plt.subplots(figsize=STYLE["figsize_square"])
    ax.scatter(y_true, residual, s=10, alpha=0.5)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    apply_axes_style(ax, title="Residual Plot (Validation)", xlabel="FEA Torque (Nm)", ylabel="Prediction Error (Nm)", legend=False)
    save_fig(fig, "residual_plot_validation")


# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--json", type=str, default=DEFAULT_JSON)
    parser.add_argument("--only", type=str, default="all", choices=["all", "metrics", "val"])
    parser.add_argument("--krpm", action="store_true")
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--title", type=int, default=None)
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--tick", type=int, default=None)
    parser.add_argument("--legend", type=int, default=None)
    args = parser.parse_args()

    if args.krpm:
        STYLE["rpm_unit"] = "krpm"
    if args.dpi is not None:
        STYLE["dpi"] = args.dpi
    for k in ["title", "label", "tick", "legend"]:
        v = getattr(args, k)
        if v is not None:
            STYLE[k] = v

    if args.only in ("all", "metrics"):
        df = load_metrics(args.csv, args.json)
        fig_metric_curves(df)
        fig_metric_summary(df)
        fig_worst_rpms(df, topk=3)

    if args.only in ("all", "val"):
        plot_learning_curve()
        plot_parity()
        plot_residual()

    print(f"[OK] Saved figures to: {OUT_DIR}/")
    print(f"     dpi={STYLE['dpi']} fonts(title/label/tick/legend)={STYLE['title']}/{STYLE['label']}/{STYLE['tick']}/{STYLE['legend']}")


if __name__ == "__main__":
    main()
