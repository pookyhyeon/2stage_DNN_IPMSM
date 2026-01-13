# ================================================================
# 📚 머신러닝 분류 모델 학습 코드 - Random Forest 사용
# ================================================================
# 목표: 주어진 입력 변수(X)로부터 모터 설계가 유효한지 판별하는 분류 모델 구축

import os
import joblib
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# =========================================================
# 1️⃣ 데이터 로딩 및 기본 전처리
# =========================================================
file_name = "data/a.csv"

try:
    with open(file_name, "r", encoding="utf-8-sig") as f:
        text = f.read()
except FileNotFoundError:
    print(f"오류: {file_name} 파일을 찾을 수 없습니다.")
    exit()

text = text.replace('"', '').replace("\\t", "\t")
df = pd.read_csv(StringIO(text), sep="\t")
df.columns = df.columns.str.replace('#', '').str.strip()

# =========================================================
# 2️⃣ 분류용 데이터 전처리
# =========================================================

X_cols = [
    'L1_Pole_V_Angle',             # 극 V자 각도
    'L1_Bridge_Thickness',         # 다리 두께
    'DC_Link_Voltage',             # DC 링크 전압 (기존 DC_Bus_Voltage 수정)
    'Airgap',                      # 에어갭
    'Turns',                       # 턴 수
    'Active_Length',               # 유효 길이
    'Tooth_Width_Ratio',           # 치아 너비 비율
    'Stator_Bore_Ratio',           # 스테이터 구멍 비율
    'Slot_Depth_Ratio',            # 슬롯 깊이 비율
    'Max_Phase_Current',           # 최대 상전류 (추가됨)
    'L1_Magnet_Bar_Width_Ratio'    # 자석 바 너비 비율
]

# 📌 STEP 2: 출력 변수(y) 정의 - 토크 1000~18000rpm (Power 컬럼 없음)
# RPM 범위: 1000 ~ 18000 (1000 단위)
RPMS = np.arange(1000, 18001, 1000)
print(df["Turns"].describe())
print(df["Turns"].value_counts().head(10))

y_output_cols = []
for rpm in RPMS:
    # a.csv에는 peak_Shaft_Torque 컬럼만 존재함
    y_output_cols.append(f"peak_Shaft_Torque_{rpm}rpm")

# 📌 STEP 3: 분류 라벨(y_class) 생성
valid_indices = np.ones(len(df), dtype=bool)

for col in y_output_cols:
    if col in df.columns:
        # NaN을 0으로 간주하고, 값이 0이 아니면 유효하다고 판단
        valid_indices &= (df[col].fillna(0) != 0)
    else:
        print(f"경고: 라벨 생성 중 {col} 컬럼을 찾지 못했습니다.")

df['y_class'] = valid_indices.astype(int)

print(f"\n📊 분류 작업용 데이터: 총 {len(df)}개 행")
print("class 비율:")
print(df['y_class'].value_counts(normalize=True).to_frame(name="비율"))

# 📌 STEP 4: 입력(X)과 라벨(y) 분리
X = df[X_cols]
y_class = df['y_class']

if X.isnull().values.any():
    print("⚠️ 경고: X 데이터에 NaN 값이 포함되어 평균으로 대체합니다.")
    X = X.fillna(X.mean())

# =========================================================
# 3️⃣ 분류 모델 학습
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

x_scaler_class = StandardScaler()
X_train_scaled = x_scaler_class.fit_transform(X_train)
X_test_scaled = x_scaler_class.transform(X_test)

print("\n" + "="*50)
print("🚀 STEP 1: 분류 모델(Random Forest) 학습 시작")
print("="*50)

classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
classifier.fit(X_train_scaled, y_train)
print("✅ 분류 모델 학습 완료!")

# =========================================================
# 4️⃣ 분류 모델 평가 및 저장
# =========================================================
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n▶ Accuracy (정확도): {accuracy * 100:.2f}%")
print(f"▶ F1-Score (유효 클래스): {f1:.4f}")
print(classification_report(y_test, y_pred, target_names=['Class 0 (무효)', 'Class 1 (유효)']))

joblib.dump(classifier, "test/classifier_model.joblib")
joblib.dump(x_scaler_class, "test/x_scaler_class.gz")

print("\n💾 모델 저장 완료: classifier_model.joblib, x_scaler_class.gz")