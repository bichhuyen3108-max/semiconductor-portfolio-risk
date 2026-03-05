import os  # 운영체제(OS) 관련 기능 사용 (폴더 생성 등)
import numpy as np  # 수치 계산 라이브러리 (로그 계산용)
import pandas as pd  # 데이터프레임 처리 라이브러리
from .config import TICKERS


def load_prices(path: str = "data/raw/adj_close.csv") -> pd.DataFrame:
    """
    원시 종가 데이터(Adj Close)를 불러오는 함수
    :param path: CSV 파일 경로
    :return: 날짜 인덱스를 가진 DataFrame
    """
    
    # CSV 파일을 읽고, 첫 번째 열을 날짜 인덱스로 설정
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # 날짜 기준으로 오름차순 정렬 (시계열 데이터 정렬 필수)
    prices = prices.sort_index()
    missing = set(TICKERS.keys()) - set(prices.columns)
    if missing:
        raise ValueError(f"CSV에 없는 ticker가 있습니다. 먼저 다운로드하세요: {sorted(missing)}")

    # config 기준으로 컬럼 순서 재정렬 (다운로드 파일 변경 대비)
    prices = prices[list(TICKERS.keys())]  
    
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    로그 수익률(Log Return)을 계산하는 함수
    로그수익률 = ln(P_t / P_(t-1))
    :param prices: 종가 데이터
    :return: 로그 수익률 DataFrame
    """
    prices = prices.dropna(how="any")  # align to full-available days

    # 이전 날짜 가격 대비 현재 가격 비율 계산 후 자연로그 적용
    log_ret = np.log(prices / prices.shift(1))
    
    # 첫 행은 NaN이므로 제거
    log_ret = log_ret.dropna()
    
    return log_ret


if __name__ == "__main__":
    
    # data/processed 폴더가 없으면 생성
    os.makedirs("data/processed", exist_ok=True)
    
    # 종가 데이터 불러오기
    prices = load_prices()
    
    # 로그 수익률 계산
    log_ret = compute_log_returns(prices)
    
    # CSV 파일로 저장
    log_ret.to_csv("data/processed/log_returns.csv")
    
    # 저장 결과 출력 (행 개수, 열 개수 확인)
    print("Saved: data/processed/log_returns.csv", log_ret.shape)