# ================================================================
# 📚 DNN 기반 모터 Torque Curve 예측 (Torque Only, 18 outputs)
# - 입력(11개 설계변수) -> 1000~18000rpm 토크(18개) 예측
# - 변수명 통일: X_cols, y_cols
# - 논문용 지표 저장: RMSE/MAE/R2 (RPM별 + 평균) -> CSV/JSON
# - 학습 곡선/잔차/Actual-vs-Pred 산점도 저장 + 폰트 크기 조절
# - Train loss가 더 큰 현상(Dropout 등) 점검용: eval 모드에서 train/val loss 재계산
# - GPU 옵션 포함
# - 스케일러 누수 방지: train set으로만 fit (논문 친화적)
# ================================================================

import os
import json
from io import StringIO

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# 0) 사용자 설정 (여기만 만지면 됨)
# ================================================================
DATA_PATH = "data/a.csv"

# GPU 사용 여부 (CUDA 있으면 자동 사용)
USE_GPU = True

# 학습 설정
NUM_EPOCHS = 2000
PATIENCE = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5

# K-Fold (시간 많이 걸림)
USE_CV = True
N_FOLDS = 5
CV_EPOCHS = 200

# 그래프 폰트/해상도 설정
PLOT_FONTS = {
    "title": 18,
    "label":18,
    "tick": 18,
    "legend": 18,
}
PLOT_DPI = 300

# 대표 RPM scatter (논문용)
REP_RPMS = [3000, 9000, 15000, 18000]

# 재현성
SEED = 42


# ================================================================
# 1) 기본 세팅 + 출력 폴더
# ================================================================
np.random.seed(SEED)
torch.manual_seed(SEED)

RPMS = np.arange(1000, 18001, 1000)  # 1000~18000 step 1000 => 18개
N_SPEEDS = len(RPMS)

OUT_DIR_MACHINE = "machine"
OUT_DIR_GRAPH = "graph"
OUT_DIR_RESID = "residual_plots"
os.makedirs(OUT_DIR_MACHINE, exist_ok=True)
os.makedirs(OUT_DIR_GRAPH, exist_ok=True)
os.makedirs(OUT_DIR_RESID, exist_ok=True)

device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print(f"\n🧠 Device: {device} (USE_GPU={USE_GPU})")


# ================================================================
# 2) 데이터 로딩 및 정제
# ================================================================
try:
    with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
        text = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"❌ 오류: '{DATA_PATH}' 파일을 찾을 수 없습니다.")

# 탭이 문자열 "\\t"로 들어온 케이스 대응 + 따옴표 제거
text = text.replace('"', "").replace("\\t", "\t")

df = pd.read_csv(StringIO(text), sep="\t")
df.columns = df.columns.str.replace("#", "", regex=False).str.strip()

print("\n📊 데이터 로딩 완료")
print(f"   전체 행/열: {df.shape}")


# ================================================================
# 3) 입력/출력 컬럼 정의 (변수명 통일!)
# ================================================================
X_cols = [
    "L1_Pole_V_Angle",
    "L1_Bridge_Thickness",
    "DC_Link_Voltage",
    "Airgap",
    "Turns",
    "Active_Length",
    "Tooth_Width_Ratio",
    "Stator_Bore_Ratio",
    "Slot_Depth_Ratio",
    "Max_Phase_Current",
    "L1_Magnet_Bar_Width_Ratio",
]

# Torque only targets (18개)
y_cols = [f"peak_Shaft_Torque_{rpm}rpm" for rpm in RPMS]

print("\n📌 컬럼 정보")
print(f"   입력 X: {len(X_cols)}개")
print(f"   출력 y (Torque only): {len(y_cols)}개")


# ================================================================
# 4) 컬럼 존재 체크 + 유효 데이터 필터링 + 결측 처리
# ================================================================
missing_x = [c for c in X_cols if c not in df.columns]
missing_y = [c for c in y_cols if c not in df.columns]
if missing_x or missing_y:
    msg = "❌ 데이터 컬럼이 부족합니다.\n"
    if missing_x:
        msg += f"   누락된 입력 컬럼: {missing_x}\n"
    if missing_y:
        msg += f"   누락된 출력 컬럼: {missing_y}\n"
    raise KeyError(msg)

# 출력 중 하나라도 0이면(또는 NaN을 0으로 보고) 제거 -> 해석 실패 제거
valid = np.ones(len(df), dtype=bool)
for col in y_cols:
    valid &= (df[col].fillna(0) != 0)

removed = len(df) - valid.sum()
df = df[valid].copy()

print("\n🧹 무효 데이터 제거")
print(f"   제거: {removed}개 -> 남음: {len(df)}개")

# X, y 분리
X = df[X_cols].copy()
y = df[y_cols].copy()

# 결측치 처리(남아있다면 평균 대체)
if X.isnull().values.any():
    print("⚠️ 입력 X에 NaN 발견 -> 평균으로 대체")
    X = X.fillna(X.mean())

if y.isnull().values.any():
    print("⚠️ 출력 y에 NaN 발견 -> 평균으로 대체")
    y = y.fillna(y.mean())


# ================================================================
# 5) Train/Test Split (스케일러 누수 방지: split 후 train으로만 fit)
# ================================================================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

x_scaler = StandardScaler().fit(X_train_raw)
y_scaler = StandardScaler().fit(y_train_raw)

X_train = x_scaler.transform(X_train_raw)
X_test = x_scaler.transform(X_test_raw)
y_train = y_scaler.transform(y_train_raw)
y_test = y_scaler.transform(y_test_raw)

joblib.dump(x_scaler, os.path.join(OUT_DIR_MACHINE, "x_scaler_torque.gz"))
joblib.dump(y_scaler, os.path.join(OUT_DIR_MACHINE, "y_scaler_torque.gz"))
print("\n✅ 스케일러 저장 완료 (누수 방지: train으로만 fit)")
print("   machine/x_scaler_torque.gz")
print("   machine/y_scaler_torque.gz")

# Torch tensor 변환 + device 이동
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

print("\n📦 데이터 분할")
print(f"   학습: {len(X_train)}개")
print(f"   검증: {len(X_test)}개")


# ================================================================
# 6) 모델 정의
# ================================================================
class MultiOutputNN_Regulated(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


model = MultiOutputNN_Regulated(in_dim=len(X_cols), out_dim=len(y_cols)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("\n🏗️ 모델 구조")
print(f"   in_dim={len(X_cols)}, out_dim={len(y_cols)} (Torque only)")


# ================================================================
# 7) (선택) K-Fold CV (누수 방지 버전)
# ================================================================
if USE_CV:
    print(f"\n🔁 K-Fold 시작: {N_FOLDS} folds, fold당 최대 {CV_EPOCHS} epochs")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_rmse_means, cv_r2_means = [], []

    # KFold는 원본 X,y(DataFrame) 기준으로 split 후 fold마다 스케일러 fit
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr_raw, X_va_raw = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr_raw, y_va_raw = y.iloc[tr_idx], y.iloc[va_idx]

        x_sc = StandardScaler().fit(X_tr_raw)
        y_sc = StandardScaler().fit(y_tr_raw)

        X_tr = x_sc.transform(X_tr_raw)
        X_va = x_sc.transform(X_va_raw)
        y_tr = y_sc.transform(y_tr_raw)
        y_va = y_sc.transform(y_va_raw)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
        y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

        m = MultiOutputNN_Regulated(len(X_cols), len(y_cols)).to(device)
        opt = optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best = float("inf")
        patience_cv = 20
        pc = 0
        best_state = None

        for epoch in range(CV_EPOCHS):
            m.train()
            opt.zero_grad()
            out = m(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            opt.step()

            m.eval()
            with torch.no_grad():
                va_out = m(X_va_t)
                va_loss = criterion(va_out, y_va_t)

            if va_loss.item() < best:
                best = va_loss.item()
                pc = 0
                best_state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            else:
                pc += 1
                if pc >= patience_cv:
                    break

        if best_state is not None:
            m.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        m.eval()
        with torch.no_grad():
            pred_va_s = m(X_va_t).detach().cpu().numpy()

        pred_va = y_sc.inverse_transform(pred_va_s)
        y_va_true = y_va_raw.values

        rmse_each = np.sqrt(mean_squared_error(y_va_true, pred_va, multioutput="raw_values"))
        r2_each = r2_score(y_va_true, pred_va, multioutput="raw_values")

        cv_rmse_means.append(float(np.mean(rmse_each)))
        cv_r2_means.append(float(np.mean(r2_each)))

        print(f"Fold {fold+1}: mean RMSE={np.mean(rmse_each):.4f}, mean R2={np.mean(r2_each):.4f}")

    print("\n✅ CV 평균")
    print(f"   RMSE={np.mean(cv_rmse_means):.4f}")
    print(f"   R2  ={np.mean(cv_r2_means):.4f}")


# ================================================================
# 8) 학습 (Early Stopping)
# ================================================================
best_val = float("inf")
pc = 0
history = {"train_loss": [], "val_loss": []}
best_path = os.path.join(OUT_DIR_MACHINE, "motor_model_best_temp_torque.pth")

print("\n" + "=" * 60)
print("🚀 학습 시작 (Torque only, Early Stopping)")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    # ---- train ----
    model.train()  # Dropout ON
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = criterion(pred, y_train_t)
    loss.backward()
    optimizer.step()

    # ---- val ----
    model.eval()   # Dropout OFF
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t)

    history["train_loss"].append(loss.item())
    history["val_loss"].append(val_loss.item())

    # early stopping
    if val_loss.item() < best_val:
        best_val = val_loss.item()
        pc = 0
        torch.save(model.state_dict(), best_path)
    else:
        pc += 1
        if pc >= PATIENCE:
            print(f"\n🛑 Early Stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 100 == 0:
        print(f"[{epoch+1:4d}] train={loss.item():.6f} | val={val_loss.item():.6f}")

# 최적 가중치 로드
model.load_state_dict(torch.load(best_path, map_location=device))
try:
    os.remove(best_path)
except Exception:
    pass
print("\n✅ 최적 모델 로드 완료")


# ================================================================
# 9) (중요) Dropout 영향 점검: 같은 모드(eval)에서 train/val loss 재측정
# ================================================================
model.eval()
with torch.no_grad():
    train_eval_loss = criterion(model(X_train_t), y_train_t).item()
    val_eval_loss = criterion(model(X_test_t), y_test_t).item()

print("\n🧪 Loss check (same mode: eval)")
print(f"   Train(eval) loss: {train_eval_loss:.6f}")
print(f"   Val  (eval) loss: {val_eval_loss:.6f}")
print("   ※ train(loss) > val(loss)가 Dropout 때문인지 판단할 때 이 값이 중요함")


# ================================================================
# 10) 학습 곡선 저장 (폰트 크기 지정)
# ================================================================
plt.figure(figsize=(12, 6))
plt.plot(history["train_loss"], label="Training Loss (train mode)", linewidth=2, alpha=0.85)
plt.plot(history["val_loss"], label="Validation Loss (eval mode)", linewidth=2, alpha=0.85, linestyle="--")
plt.yscale("log")

plt.title("Training Convergence (MSE, log scale)", fontsize=PLOT_FONTS["title"])
plt.xlabel("Epoch", fontsize=PLOT_FONTS["label"])
plt.ylabel("Loss", fontsize=PLOT_FONTS["label"])
plt.xticks(fontsize=PLOT_FONTS["tick"])
plt.yticks(fontsize=PLOT_FONTS["tick"])
plt.grid(alpha=0.35)
plt.legend(fontsize=PLOT_FONTS["legend"])
plt.tight_layout()

lc_path = os.path.join(OUT_DIR_GRAPH, "learning_curve_torque.png")
plt.savefig(lc_path, dpi=PLOT_DPI)
plt.close()
print(f"\n📈 학습 곡선 저장: {lc_path}")


# ================================================================
# 11) 평가: RMSE/R2/MAE 저장 + residual plots (Torque only)
# ================================================================
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_t).detach().cpu().numpy()

preds = y_scaler.inverse_transform(preds_scaled)
y_true = y_scaler.inverse_transform(y_test)  # y_test는 numpy array

residuals = y_true - preds  # (n_samples, 18)

rmse_each = np.sqrt(mean_squared_error(y_true, preds, multioutput="raw_values"))  # (18,)
r2_each = r2_score(y_true, preds, multioutput="raw_values")                       # (18,)
mae_each = np.mean(np.abs(y_true - preds), axis=0)                                # (18,)

mean_rmse = float(np.mean(rmse_each))
mean_mae = float(np.mean(mae_each))
mean_r2 = float(np.mean(r2_each))

print("\n" + "=" * 60)
print("📊 모델 평가 (Torque only)")
print("=" * 60)
print(f"평균 R²  : {mean_r2:.4f}")
print(f"평균 RMSE: {mean_rmse:.4f} Nm")
print(f"평균 MAE : {mean_mae:.4f} Nm\n")

print("-" * 64)
print(f"{'RPM':>6s} | {'R²':>8s} | {'RMSE(Nm)':>10s} | {'MAE(Nm)':>10s}")
print("-" * 64)
for i, rpm in enumerate(RPMS):
    print(f"{rpm:>6d} | {r2_each[i]:>8.4f} | {rmse_each[i]:>10.4f} | {mae_each[i]:>10.4f}")


# ---- 논문용 성능지표 CSV 저장 (RPM별 + 평균 행) ----
metrics_df = pd.DataFrame({
    "rpm": RPMS,
    "rmse_Nm": rmse_each,
    "mae_Nm": mae_each,
    "r2": r2_each,
})
metrics_mean = pd.DataFrame([{
    "rpm": "MEAN",
    "rmse_Nm": mean_rmse,
    "mae_Nm": mean_mae,
    "r2": mean_r2,
}])
metrics_df_out = pd.concat([metrics_df, metrics_mean], ignore_index=True)

metrics_csv_path = os.path.join(OUT_DIR_MACHINE, "metrics_torque.csv")
metrics_df_out.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
print(f"\n📄 성능 지표 CSV 저장: {metrics_csv_path}")

# ---- 논문/자동화용 JSON 저장 ----
metrics_json_path = os.path.join(OUT_DIR_MACHINE, "metrics_torque.json")
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "mean_rmse_Nm": mean_rmse,
        "mean_mae_Nm": mean_mae,
        "mean_r2": mean_r2,
        "train_eval_loss": float(train_eval_loss),
        "val_eval_loss": float(val_eval_loss),
        "per_rpm": metrics_df.to_dict(orient="records"),
        "config": {
            "DATA_PATH": DATA_PATH,
            "USE_GPU": USE_GPU,
            "device": str(device),
            "NUM_EPOCHS": NUM_EPOCHS,
            "PATIENCE": PATIENCE,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "X_cols": X_cols,
            "y_cols": y_cols,
        }
    }, f, ensure_ascii=False, indent=2)
print(f"🧾 성능 지표 JSON 저장: {metrics_json_path}")


# ================================================================
# 12) Residual plots 저장
# ================================================================
# 전체 residual histogram
plt.figure(figsize=(10, 4))
plt.hist(residuals.flatten(), bins=80, alpha=0.85)
plt.title("Residuals Distribution (Torque only, all outputs)", fontsize=PLOT_FONTS["title"])
plt.xlabel("Actual - Predicted (Nm)", fontsize=PLOT_FONTS["label"])
plt.ylabel("Count", fontsize=PLOT_FONTS["label"])
plt.xticks(fontsize=PLOT_FONTS["tick"])
plt.yticks(fontsize=PLOT_FONTS["tick"])
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_RESID, "residuals_hist_all_torque.png"), dpi=PLOT_DPI)
plt.close()

# rpm별 residual scatter
for i, rpm in enumerate(RPMS):
    x = preds[:, i]
    y_r = residuals[:, i]

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y_r, s=12, alpha=0.6)
    plt.hlines(0, xmin=x.min(), xmax=x.max(), colors="r", linestyles="--", linewidth=1)
    plt.title(f"Torque Residuals @ {rpm} rpm", fontsize=PLOT_FONTS["title"])
    plt.xlabel("Predicted Torque (Nm)", fontsize=PLOT_FONTS["label"])
    plt.ylabel("Actual - Predicted (Nm)", fontsize=PLOT_FONTS["label"])
    plt.xticks(fontsize=PLOT_FONTS["tick"])
    plt.yticks(fontsize=PLOT_FONTS["tick"])
    plt.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_RESID, f"residual_torque_{rpm}rpm.png"), dpi=PLOT_DPI)
    plt.close()

print(f"\n✅ Residual plots 저장 완료: {OUT_DIR_RESID}/")


# ================================================================
# 13) 논문용 Actual vs Predicted scatter (대표 RPM)
# ================================================================
for r in REP_RPMS:
    if r not in RPMS:
        continue
    idx = int(np.where(RPMS == r)[0][0])

    plt.figure(figsize=(5.5, 5))
    plt.scatter(y_true[:, idx], preds[:, idx], s=12, alpha=0.55)

    mn = min(y_true[:, idx].min(), preds[:, idx].min())
    mx = max(y_true[:, idx].max(), preds[:, idx].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)

    plt.title(f"Actual vs Predicted Torque @ {r} rpm", fontsize=PLOT_FONTS["title"])
    plt.xlabel("Actual Torque (Nm)", fontsize=PLOT_FONTS["label"])
    plt.ylabel("Predicted Torque (Nm)", fontsize=PLOT_FONTS["label"])
    plt.xticks(fontsize=PLOT_FONTS["tick"])
    plt.yticks(fontsize=PLOT_FONTS["tick"])
    plt.grid(alpha=0.35)
    plt.tight_layout()

    path = os.path.join(OUT_DIR_GRAPH, f"scatter_actual_vs_pred_{r}rpm.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()

print(f"✅ 대표 RPM scatter 저장 완료: {OUT_DIR_GRAPH}/")


# ================================================================
# 14) 예측 함수 + 예측 시범
# ================================================================
def predict_torque_curve(input_values_11, model, x_scaler, y_scaler, X_cols, device):
    """
    input_values_11: 길이 11 리스트/배열 (X_cols 순서대로)
    return: torque curve length 18 (1000~18000rpm)
    """
    model.eval()
    arr = np.array(input_values_11).reshape(1, -1)

    # scaler feature-name warning 방지
    inp_df = pd.DataFrame(arr, columns=X_cols)
    inp_s = x_scaler.transform(inp_df)

    inp_t = torch.tensor(inp_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_s = model(inp_t).detach().cpu().numpy()

    pred = y_scaler.inverse_transform(pred_s)[0]
    return pred


print("\n" + "=" * 60)
print("🎯 예측 시범 (데이터셋 첫 번째 샘플)")
print("=" * 60)

example_input = X_train_raw.iloc[0].values
torque_curve = predict_torque_curve(example_input, model, x_scaler, y_scaler, X_cols, device)

print("-" * 55)
print(f"{'RPM':>6s} | {'Torque (Nm)':>12s}")
print("-" * 55)
for i, rpm in enumerate(RPMS):
    print(f"{rpm:>6d} | {torque_curve[i]:>12.2f}")

np.savez(
    "machine/history_torque.npz",
    train_loss=np.array(history["train_loss"]),
    val_loss=np.array(history["val_loss"])
)
print("[OK] Saved: machine/history_torque.npz")

# 2) Validation prediction 저장 (parity / residual plot용)
np.savez(
    "machine/val_pred_torque.npz",
    y_true=y_test_raw.values.astype(np.float32),  # 반드시 원스케일
    y_pred=preds.astype(np.float32),             # inverse_transform된 값
    rpms=RPMS.astype(np.int32)
)
print("[OK] Saved: machine/val_pred_torque.npz")

# ================================================================
# 15) 최종 모델 저장
# ================================================================
final_path = os.path.join(OUT_DIR_MACHINE, "optimal_torque_model.pth")
torch.save(model.state_dict(), final_path)

print("\n" + "=" * 60)
print("💾 모델 저장 완료")
print("=" * 60)
print(f"✅ 저장 경로: {final_path}")
print("🎉 완료!")
