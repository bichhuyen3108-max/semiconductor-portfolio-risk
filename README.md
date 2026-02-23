📊 한국 반도체 포트폴리오 데이터 분석
변동성 및 하방 리스크 인사이트 도출 (VaR / CVaR 기반)
📌 프로젝트 소개

본 프로젝트는 한국 반도체 주요 기업(Samsung Electronics, SK Hynix, Samsung SDI)의 주가 데이터를 기반으로 포트폴리오 변동성과 하방 리스크를 정량적으로 분석한 데이터 분석 프로젝트입니다.

단순 리스크 수치 계산을 넘어,
데이터 수집 → 전처리 → 분석 → 해석 → 인사이트 도출의 전체 분석 흐름을 구조화하는 데 목적이 있습니다.

🎯 프로젝트 목표

금융 데이터 기반 분석 프로세스 정리

로그 수익률 기반 포트폴리오 리스크 측정

VaR / CVaR 비교 분석

변동성 및 Drawdown 패턴 관찰

데이터 기반 의사결정 인사이트 도출

📂 데이터 개요

데이터 출처: Yahoo Finance (yfinance)

기간: 2016-01-01 ~ 2025-12-31

데이터 유형: 일별 종가 (Adj Close)

포트폴리오 구성: 동일가중(Equal Weight)

⚙️ 분석 프로세스

데이터 수집 및 전처리

로그 수익률(Log Return) 계산

동일가중 포트폴리오 구성

변동성(Volatility) 및 최대 낙폭(Drawdown) 분석

Historical VaR (95%, 99%) 산출

CVaR(Expected Shortfall) 계산

Stress Scenario 기반 리스크 관찰

(선택) Backtesting을 통한 모델 검증

📊 주요 지표

Rolling Volatility

Annualized Volatility

Maximum Drawdown

Historical VaR

Parametric VaR

Monte Carlo Simulation

CVaR (Expected Shortfall)

💡 주요 인사이트

위기 구간에서 변동성이 급격히 확대되는 패턴 관찰

Tail Risk가 특정 시점에 집중적으로 발생

VaR 단일 지표만으로는 극단 손실을 충분히 설명하기 어려움

CVaR를 함께 고려할 경우 보다 보수적인 리스크 평가 가능

🛠 Tech Stack

Python

Pandas

Numpy

Matplotlib / Seaborn

yfinance
