import os  # 폴더 생성 등
import numpy as np  # 분위수(quantile) 계산용
import pandas as pd  # 데이터 처리

from src.config import CONF_LEVELS  # 예: [0.95, 0.99]


def load_returns(path: str = "data/processed/log_returns.csv") -> pd.DataFrame:
    """로그 수익률 CSV를 불러오는 함수"""
    rets = pd.read_csv(path, index_col=0, parse_dates=True)  # 날짜 인덱스 처리
    rets = rets.sort_index()  # 날짜 정렬
    return rets


def make_equal_weights(columns) -> pd.Series:
    """동일가중치(1/n) 생성 - 컬럼명 기준으로 안전하게 매핑"""
    n = len(columns)  # 종목 수
    w = pd.Series(1 / n, index=columns)  # 각 종목에 1/n 비중
    return w


def portfolio_return(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """포트폴리오 수익률 계산 (라벨 정렬 기반 dot)"""
    # pandas는 index(티커명) 기준으로 자동 정렬하여 곱하므로 순서가 바뀌어도 안전
    port = returns.dot(weights)
    port.name = "portfolio"
    return port


def historical_var(x: pd.Series, alpha: float) -> float:
    """
    Historical VaR 계산
    alpha=0.95 -> 좌측 5% 분위수(손실 구간)
    """
    q = np.quantile(x, 1 - alpha)
    return float(q)


def cvar(x: pd.Series, alpha: float) -> float:
    """
    CVaR(Expected Shortfall) 계산
    VaR보다 더 나쁜 손실 구간의 평균
    """
    var = historical_var(x, alpha)
    tail = x[x <= var]
    return float(tail.mean())


def summarize_var_cvar(port: pd.Series, alphas=None) -> pd.DataFrame:
    """VaR/CVaR 요약 테이블 생성"""
    if alphas is None:
        alphas = CONF_LEVELS

    rows = []
    for a in alphas:
        rows.append({
            "alpha": a,
            "VaR": historical_var(port, a),
            "CVaR": cvar(port, a),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 결과 저장 폴더 생성
    os.makedirs("results/tables", exist_ok=True)

    # 1) 로그 수익률 로드
    rets = load_returns()

    # 2) 동일가중치 생성
    w = make_equal_weights(rets.columns)

    # 3) 포트폴리오 수익률 계산
    port = portfolio_return(rets, w)

    # 4) VaR/CVaR 요약
    summary = summarize_var_cvar(port)

    # 5) 저장
    out_path = "results/tables/var_summary.csv"
    summary.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(summary)