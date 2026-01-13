# 2stage_DNN_IPMSM
# ⚡ 2-Stage DNN-based IPMSM Design Optimization Framework
> **Reliability-Aware Performance Prediction & Real-time Optimization for Tesla Model 3 Motor**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
본 프로젝트는 **Tesla Model 3 IPMSM**의 전 구간 성능 최적화를 위한 2-Stage 딥러닝 프레임워크를 제안합니다. 유한요소해석(FEM)의 높은 연산 비용을 해결하기 위해, 설계 유효성을 판별하는 **분류 모델**과 성능을 정밀 예측하는 **회귀 모델**을 결합한 대리 모델(Surrogate Model)을 구축하였습니다.



## 🚀 Key Features
- **Reliability-Aware Classifier**: 설계 변수의 기하학적 유효성을 98.5% 정확도로 판별하여 무효 설계안을 차단합니다.
- **Whole-Speed Range Prediction**: 1,000 ~ 18,000 RPM 전 구간의 T-N 커브를 0.001초 내에 예측합니다 ($R^2 = 0.9943$).
- **Real-time Optimization**: 유전 알고리즘(GA)과 연동하여 **1분 이내(수 초 내외)**에 최적 설계안을 도출합니다.
- **Interactive GUI**: Python Tkinter 기반의 인터페이스를 통해 직관적인 최적 설계 환경을 제공합니다.



## 📊 Performance
### 1. Regression Accuracy
| Metric | Value |
| :--- | :--- |
| **Mean $R^2$** | **0.9943** |
| **Mean $RMSE$** | **1.332 Nm** |
| **Mean $MAE$** | **0.989 Nm** |



### 2. Optimization Efficiency
- **Traditional FEM GA**: 수 일(Days) 소요
- **Proposed 2-Stage GA**: **1분 미만(Seconds)** 소요

## 🛠️ System Architecture
1. **Stage 1 (Classifier)**: Random Forest 기반의 설계 타당성 체크
2. **Stage 2 (Regressor)**: Multi-Output DNN을 이용한 전 구간 토크 맵 복원
3. **Stage 3 (Optimizer)**: Genetic Algorithm (PyGAD) 기반 최적 형상 도출

## 💻 Usage
```bash
# Clone the repository
git clone [https://github.com/YourUsername/2Stage-DNN-Motor-Optimizer.git](https://github.com/YourUsername/2Stage-DNN-Motor-Optimizer.git)

# Run the GUI application
python 3_gui_with_validation.py
