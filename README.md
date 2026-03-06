# 📊 Semiconductor Portfolio Market Risk Analysis
## Value-at-Risk Based Risk Quantification for Korean Semiconductor Portfolio

## 📌 Project Overview

본 프로젝트는 **한국 반도체 기업 포트폴리오의 시장 위험(Market Risk)**을 정량적으로 분석하기 위한 데이터 분석 프로젝트입니다.

Python 기반 데이터 분석 파이프라인을 구축하여 포트폴리오 수익률을 계산하고 다양한 Value at Risk (VaR) 모델을 적용하여 하방 리스크를 분석하였습니다.

또한 **Backtesting (Kupiec Test)**을 통해 VaR 모델의 적합성을 검증하고 Stress Testing을 통해 극단적 시장 상황에서의 포트폴리오 손실을 평가하였습니다.

## 🎯 Project Objectives

본 프로젝트의 주요 목적은 다음과 같습니다.

    - 금융 데이터 기반 시장 위험(Market Risk) 정량 분석
    - 포트폴리오 수익률 계산 및 리스크 지표 분석
    - 다양한 VaR 모델 비교
        - Historical VaR
        - Student-t VaR
        - GARCH VaR
    - **VaR Backtesting (Kupiec Test)**을 통한 모델 검증
    - 금융 데이터 분석 Pipeline 구축 및 시각화

## 📂 Dataset
### 분석 대상 기업 (Korean Semiconductor Companies)

    - Samsung Electronics
    - SK Hynix
    - Samsung SDI

### Data Source
    Yahoo Finance

### Data Period
    2016 – 2026

## 🧠 Methodology

본 프로젝트에서는 다음과 같은 리스크 분석 모델을 적용하였습니다.

### 1️⃣ Historical VaR

과거 수익률 분포를 기반으로 VaR를 계산하는 방법입니다.

특징

    - 구현이 간단
    - 과거 데이터 기반 분석
    - 시장 충격에 대한 반응 속도가 상대적으로 느림

### 2️⃣ Student-t VaR

Student-t 분포를 이용하여 fat-tail 특성을 반영한 VaR 모델입니다.

특징

    - 금융 데이터의 두꺼운 꼬리 분포 반영
    - 극단적인 손실 가능성을 보다 보수적으로 추정

### 3️⃣ GARCH VaR

GARCH(1,1) 모델을 이용하여 **조건부 변동성 (Conditional Volatility)**을 반영한 VaR 모델입니다.

특징

    - 시장 변동성 변화 반영
    - volatility clustering 모델링 가능
    - 시장 충격 발생 시 VaR 값이 빠르게 확대


## 🔄 Project Pipeline

프로젝트 분석 과정

    Data Download
            ↓
    Data Preprocessing
            ↓
    Log Return Calculation
            ↓
    Portfolio Return Construction
            ↓
    VaR / CVaR Calculation
            ↓
    Rolling VaR Estimation
            ↓
    Backtesting (Kupiec Test)
            ↓
    Stress Testing
            ↓
    Visualization
            ↓
    Insight Extraction


## 📈 Risk Metrics

본 프로젝트에서 사용한 주요 리스크 지표

    - VaR (Value at Risk)
    - CVaR (Expected Shortfall)
    - Rolling VaR
    - Kupiec Backtesting
    - Stress Testing

## 📊 Visualization
### Return vs VaR

포트폴리오 수익률과 VaR 변화를 비교하여
시장 변동성과 리스크 변화를 시각적으로 분석합니다.

Example

    results/figures/return_vs_var_zoom_95.png


### VaR Model Comparison

다음 VaR 모델을 비교 분석합니다.

    - Historical VaR
    - Student-t VaR
    - GARCH VaR

Example:

    results/figures/var_models_only_95.png    

## 📉 Backtesting Result (Kupiec Test)

VaR 모델의 적합성을 평가하기 위해 Kupiec POF Test를 수행하였습니다.   

| Model          | Confidence Level | Violation Rate | Result |
| -------------- | ---------------- | -------------- | ------ |
| Historical VaR | 95%              | ~5.78%         | PASS   |
| Historical VaR | 99%              | ~1.48%         | FAIL   |
| Student-t VaR  | 95%              | ~5.4%          | PASS   |
| GARCH VaR      | 95%              | ~4.7%          | PASS   |

결과적으로 Student-t VaR와 GARCH VaR가 Historical VaR보다 안정적인 성능을 보였습니다.


## 💡 Key Insights

분석 결과 다음과 같은 특징을 확인할 수 있었습니다.

### 1️⃣ Volatility Clustering

시장 변동성이 급격히 증가하는 구간에서 VaR 값이 크게 확대되는 volatility clustering 현상이 나타났습니다.

특히 2020년 COVID 시장 충격 구간에서 포트폴리오 수익률이 VaR를 크게 하회하는 extreme loss가 발생했습니다.

### 2️⃣ Historical VaR 특징

Historical VaR 모델은 과거 데이터 기반이기 때문에
시장 충격에 대한 반응 속도가 상대적으로 느린 특징을 보였습니다.

### 3️⃣ Student-t VaR 특징

Student-t 분포는 fat-tail 특성을 반영하여
극단적인 손실 가능성을 보다 보수적으로 추정하는 특징을 보였습니다.

### 4️⃣ GARCH VaR 특징

GARCH VaR는 조건부 변동성을 반영하기 때문에
시장 변동성 증가 구간에서 VaR 값이 빠르게 확대되는 특징을 보였습니다.


## 🛠 Tech Stack

사용 기술
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - SciPy
    - ARCH (GARCH Model)
    
## 📁 Project Structure
semiconductor-portfolio-risk
│
├─ data
│   ├─ raw
│   └─ processed
│
├─ results
│   ├─ tables
│   └─ figures
│
├─ src
│   ├─ config.py
│   ├─ data_download.py
│   ├─ preprocess.py
│   ├─ portfolio.py
│   ├─ risk_metrics.py
│   ├─ visualization.py
│   └─ run_analysis.py
│
└─ README.md

## 🚀 Future Improvements

향후 다음과 같은 분석 확장을 고려할 수 있습니다.

    - Monte Carlo VaR

    - EVT (Extreme Value Theory)

    - Multi-factor Risk Model

    - Portfolio Optimization


## ⭐ Portfolio Note

이 프로젝트는 금융 데이터 분석 역량 강화를 위한 개인 프로젝트입니다.

본 프로젝트를 통해 다음 역량을 보여주고자 하였습니다.

    - 금융 데이터 분석 파이프라인 구축
    - VaR 리스크 모델 구현
    - 데이터 기반 리스크 인사이트 도출

본 연구는 한국 반도체 포트폴리오의 시장 리스크를 VaR 기반으로 정량적으로 분석하였다.
분석 결과 금융 데이터에서는 변동성 군집과 Fat Tail 현상이 나타났으며, Normal VaR 모델은 극단적 손실을 과소추정하는 경향이 있음을 확인하였다.
따라서 시장 변동성이 확대되는 구간에서는 포트폴리오 노출도를 조절하는 리스크 관리 전략이 필요하다.    

이 프로젝트의 목적은 반도체 포트폴리오의 하방 리스크를 VaR 기반으로 측정하고, 시장 변동성이 확대되는 구간에서 리스크 관리 전략이 필요한지를 분석하는 것입니다.


시장 상황이 악화될 경우,
→ 포트폴리오 손실은 얼마나 될까요?
→ 그리고 위험 감소는 언제 고려해야 할까요?
