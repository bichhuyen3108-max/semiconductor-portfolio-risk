# 🇰🇷 한국 반도체 포트폴리오 시장위험 분석 (Rolling VaR 기반)

## 1. 프로젝트 개요
- **목표**: 한국 반도체 포트폴리오의 **하방 리스크(Downside Risk)**를 정량화하고,  
  **시장 레짐 변화(Regime Shift)** 및 **변동성 국면(Volatility Regime)**을 관찰한다.
- **핵심 방법**: Historical Rolling VaR (95%, 99%), Violation 분석(간단 Backtesting), 시각화

## 2. 데이터
- **데이터 종류**: 개별 종목 일간 로그수익률 (Daily Log Return)
- **포트폴리오 구성**: 동일가중(Equal-weight) 포트폴리오
- **기간**: (예: 2017 ~ 2026-02-25)

## 3. 분석 방법 (Methodology)
### 3.1 Portfolio Return
- 동일가중 방식으로 포트폴리오 수익률을 계산:
  - `Portfolio_Return = mean(stock_returns)`

### 3.2 Rolling VaR (95%, 99%)
- Rolling window 기반으로 VaR를 계산하여 시간에 따라 변하는 리스크를 추적:
  - `Rolling_VaR_95`, `Rolling_VaR_99`

### 3.3 Violation 정의 (간단 Backtesting 지표)
- **Violation**: `Portfolio_Return < Rolling_VaR`
- 연도별 Violation Rate를 계산하여 위험 국면을 정량 비교:
  - 기대 위반율(Expected rate): 95% → 5%, 99% → 1%

## 4. 주요 결과 (Key Findings)
### 4.1 Volatility Regime & Tail Regime
- 본 포트폴리오 수익률은 **변동성 군집(Volatility Clustering)**이 관찰되며,
  특정 구간에서 하방 변동성이 급격히 확대되는 **레짐 변화**가 발생했다.

### 4.2 Structural Shock vs Prolonged Stress
- **Structural Shock(구조적 충격)**: 짧은 기간에 **극단적 손실(-8%~-10%)**이 집중 발생  
  → 예: 2020년 구간에서 VaR 위반이 증가하며 tail risk가 크게 확대됨
- **Prolonged Stress(지속적 스트레스)**: 극단적 -10% 충격은 적지만,  
  **-4%~-6% 수준의 손실이 장기간 반복**되어 95% VaR 위반이 누적됨  
  → 예: 2024~2025년 구간에서 변동성 확대 국면이 지속

## 5. 결과 시각화 (Figures)
- Rolling VaR & Violation Plot:
  - `results/figures/rolling_var_violation_95.png`
  - `results/figures/rolling_var_violation_99.png`

## 6. 파일 구조 (Project Structure)
```text
src/
  visualization.py   # plotting & yearly violation tables
results/
  figures/
  tables/
data/
  processed/