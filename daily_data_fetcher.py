"""
daily_data_fetcher.py
─────────────────────────────────────────────────────────────────
미국 주식 시장 마감 직후(한국시간 오전 5시, 썸머타임 기준) 자동 실행되는
데이터 수집 스크립트입니다.

수집 항목:
  1. 나스닥 100 전 종목 어제 종가 및 등락률
  2. S&P500 이번 주 상승률 상위 10개 종목
  3. 주요 지수 (SPY, QQQ, DIA, VIX)
  4. 공포탐욕지수 (Fear & Greed Index)
  5. 미국 국채 10년 금리 (^TNX)
  6. 주요 섹터 ETF 등락률
  7. 핫한 테마주 / 주목받는 종목
  8. 이번 주 핫한 글로벌 금융 뉴스
  9. 어닝 서프라이즈 예정 종목 (이번 주)

결과는 data/daily_cache.json 에 저장됩니다.
─────────────────────────────────────────────────────────────────
"""

import os
import json
import datetime
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ── 저장 경로 설정 ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_PATH = os.path.join(DATA_DIR, "daily_cache.json")

# ── API 클라이언트 설정 ──
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Groq는 환경변수(GROQ_API_KEY)를 자동으로 인식하므로 
# 별도의 전역 설정 코드가 필요하지 않습니다.


# ────────────────────────────────────────────
# 헬퍼 함수
# ────────────────────────────────────────────

def safe_get_info(ticker_sym: str) -> dict:
    """yfinance에서 주가 정보를 안전하게 가져옵니다."""
    try:
        t = yf.Ticker(ticker_sym)
        hist = t.history(period="5d")
        if hist.empty:
            return {"ticker": ticker_sym, "error": "데이터 없음"}
        
        close_prices = hist["Close"].dropna()
        if len(close_prices) < 2:
            return {"ticker": ticker_sym, "error": "가격 데이터 부족"}
        
        prev_close = float(close_prices.iloc[-2])
        last_close = float(close_prices.iloc[-1])
        change_pct = ((last_close - prev_close) / prev_close) * 100
        
        # 주간 수익률 (5거래일)
        week_start = float(close_prices.iloc[0])
        week_change_pct = ((last_close - week_start) / week_start) * 100
        
        return {
            "ticker": ticker_sym,
            "close": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "week_change_pct": round(week_change_pct, 2),
        }
    except Exception as e:
        return {"ticker": ticker_sym, "error": str(e)}


def fetch_news(query: str, days: int = 3, max_results: int = 3) -> list:
    """Tavily 뉴스 검색 (항상 news 토픽)."""
    try:
        resp = tavily.search(query=query, topic="news", days=days, max_results=max_results)
        results = []
        for r in resp.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:300],
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ────────────────────────────────────────────
# 1. 나스닥 100 주요 종목 어제 종가
# ────────────────────────────────────────────
NDX_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","TSLA","GOOGL","GOOG","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","TMUS","AMAT","INTC","INTU","CSCO","PEP",
    "ISRG","AMGN","MU","LRCX","KLAC","ADI","MRVL","REGN","PANW","FTNT",
    "SNPS","CRWD","CDNS","MELI","PYPL","CTAS","DXCM","CEG","SMCI","MNST",
    "ABNB","ORLY","ADP","ASML","FAST","PAYX","KDP","CTSH","ROST","IDXX",
    "ODFL","TEAM","DDOG","EBAY","ZS","VRSK","GEHC","TTD","SGEN","ANSS",
    "ILMN","WBD","BIDU","JD","PDD","LCID","RIVN","ZM","PTON","DOCU"
]

def collect_nasdaq100():
    print("  [1/8] 나스닥 100 주요 종목 수집 중...")
    results = []
    for sym in NDX_TICKERS:
        results.append(safe_get_info(sym))
    # 등락률 기준 내림차순 정렬
    results.sort(key=lambda x: x.get("change_pct", -999), reverse=True)
    return results


# ────────────────────────────────────────────
# 2. S&P500 이번 주 상승률 상위 10개
# ────────────────────────────────────────────
SP500_SAMPLE = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AVGO","JPM","V",
    "MA","UNH","XOM","JNJ","WMT","PG","HD","ORCL","CRM","BAC",
    "LLY","MRK","ABBV","PFE","TMO","CVX","KO","PEP","NFLX","DIS",
    "INTC","AMD","QCOM","TXN","GS","MS","BRK-B","RTX","CAT","DE",
    "GE","HON","LMT","MMM","BA","UPS","FDX","NKE","SBUX","MCD"
]

def collect_sp500_weekly_top():
    print("  [2/8] S&P500 주간 상승 상위 10개 수집 중...")
    results = []
    for sym in SP500_SAMPLE:
        info = safe_get_info(sym)
        if "error" not in info:
            results.append(info)
    results.sort(key=lambda x: x.get("week_change_pct", -999), reverse=True)
    return results[:10]


# ────────────────────────────────────────────
# 3. 주요 지수 및 기준 지표
# ────────────────────────────────────────────
INDEX_TICKERS = {
    "S&P500":        "^GSPC",
    "나스닥":         "^IXIC",
    "다우존스":       "^DJI",
    "VIX(공포지수)":  "^VIX",
    "미국채10년금리":  "^TNX",
    "달러인덱스":     "DX-Y.NYB",
    "금":             "GC=F",
    "유가(WTI)":      "CL=F",
}

def collect_major_indices():
    print("  [3/8] 주요 지수 및 거시 지표 수집 중...")
    results = {}
    for name, sym in INDEX_TICKERS.items():
        info = safe_get_info(sym)
        results[name] = info
    return results


# ────────────────────────────────────────────
# 4. 주요 섹터 ETF 등락률
# ────────────────────────────────────────────
SECTOR_ETFS = {
    "기술(XLK)":      "XLK",
    "헬스케어(XLV)":  "XLV",
    "금융(XLF)":      "XLF",
    "에너지(XLE)":    "XLE",
    "소비재(XLY)":    "XLY",
    "필수소비(XLP)":  "XLP",
    "산업재(XLI)":    "XLI",
    "반도체(SOXX)":   "SOXX",
    "AI테마(BOTZ)":   "BOTZ",
    "청정에너지(ICLN)":"ICLN",
}

def collect_sector_etfs():
    print("  [4/8] 섹터 ETF 수집 중...")
    results = {}
    for name, sym in SECTOR_ETFS.items():
        info = safe_get_info(sym)
        results[name] = info
    return results


# ────────────────────────────────────────────
# 5. 공포탐욕지수 (뉴스 기반 추정)
# ────────────────────────────────────────────
def collect_fear_greed():
    print("  [5/8] 공포탐욕지수 뉴스 수집 중...")
    news = fetch_news("CNN Fear and Greed Index today stock market sentiment", days=1, max_results=2)
    return {
        "description": "CNN Fear & Greed Index (뉴스 기반 요약)",
        "news": news
    }


# ────────────────────────────────────────────
# 6. 이번 주 핫한 글로벌 금융 뉴스
# ────────────────────────────────────────────
def collect_global_news():
    print("  [6/8] 글로벌 금융 뉴스 수집 중...")
    queries = [
        "global financial markets news this week",
        "US stock market major news today",
        "Federal Reserve interest rate news",
        "미국 주식시장 이번주 주요 뉴스",
    ]
    all_news = []
    for q in queries:
        all_news.extend(fetch_news(q, days=3, max_results=2))
    return all_news


# ────────────────────────────────────────────
# 7. 이번 주 어닝 서프라이즈 예정 종목
# ────────────────────────────────────────────
def collect_earnings_calendar():
    print("  [7/8] 이번 주 어닝 캘린더 뉴스 수집 중...")
    news = fetch_news("earnings report this week stock surprise beat miss", days=5, max_results=5)
    return news


# ────────────────────────────────────────────
# 8. 현재 가장 핫한 테마 / 주목받는 종목 뉴스
# ────────────────────────────────────────────
def collect_hot_themes():
    print("  [8/8] 핫한 테마 및 주목받는 종목 뉴스 수집 중...")
    queries = [
        "most trending stocks today Wall Street",
        "hot sector theme stocks AI semiconductor",
        "stocks making big moves today market movers",
    ]
    all_news = []
    for q in queries:
        all_news.extend(fetch_news(q, days=2, max_results=2))
    return all_news


# ────────────────────────────────────────────
# 9. LLM 기반 오늘의 맞춤형 질문 자동 생성
# ────────────────────────────────────────────
def generate_daily_ai_summary(cache_data):
    print("  [9/9] 수집된 데이터를 바탕으로 Groq LLM이 마스터 브리핑과 질문을 동시 생성 중입니다...")
    
    default_q = {
        "master_briefing": "현재 시장 브리핑 데이터가 업데이트되지 않았습니다. 실시간 검색을 활용해주세요.",
        "main_question": "시장이 닫힌 후, 오늘의 주요 기술주 흐름은 어떠한가요?",
        "sub_questions": ["최근 나스닥 등락 원인", "금리 인하 수혜주", "지금 핫한 테마", "오늘 주목할 특징주"]
    }
    
    # 환경변수에서 Groq API 키 확인
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("  ❌ GROQ_API_KEY가 설정되지 않았습니다.")
        return default_q
        
    try:
        # agent.py와 동일하게 빠르고 넉넉한 Llama 3.1 모델을 호출합니다.
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        
        sp500_tickers = [x.get('ticker') for x in cache_data.get('sp500_weekly_top', [])][:10]
        hot_themes = [x.get('title') for x in cache_data.get('hot_themes', [])][:5]
        global_news = [x.get('title') for x in cache_data.get('global_news', [])][:3]
        
        prompt = f"""
        당신은 예리한 통찰력을 가진 월스트리트 수석 애널리스트입니다. 
        아래는 오늘 아침 수집된 시장 데이터의 핵심 키워드들입니다.
        
        - 주간 급등 S&P500 종목: {sp500_tickers}
        - 핫한 테마 뉴스: {hot_themes}
        - 글로벌 금융 뉴스: {global_news}
        
        이 데이터를 바탕으로 두 가지 임무를 한 번에 수행하세요.
        
        임무 1: 에이전트가 백그라운드 지식으로 사용할 3~4문장 분량의 밀도 높은 '마스터 브리핑' 작성. (수치, 거시경제 트렌드 위주로 매우 건조하게)
        임무 2: 개인 투자자가 가장 클릭하고 싶어할 만한 날카로운 '메인 질문' 1개와, 클릭을 유도할 수 있는 흥미롭고 창의적인 '서브 질문' 4개 작성. 
        임무 3: 서브 질문 4개의 구성 원칙:
          - 1번 질문: 오늘 발표된 글로벌 뉴스나 매크로 지표(금리 등)를 반영한 거시적 질문
          - 2번 질문: 이번 주 가장 핫한 테마나 급등락 섹터의 원인을 묻는 질문
          - 3번 질문: 실적 발표(어닝) 예정이거나 최근 변동성이 컸던 구체적인 '기업명'을 포함한 날카로운 질문
          - 4번 질문: 기존의 뻔한 질문에서 벗어나 시장의 숨겨진 리스크나 새로운 기회를 묻는 창의적이고 도발적인 질문
        
        [매우 중요한 지시사항]
        - 마스터 브리핑 내용, 메인 질문, 4개의 서브 질문은 **반드시 서로 다른 주제와 내용**이어야 합니다. 
        - 서브 질문은 "어떤 기업이 좋은가요?" 같은 모호한 형태가 아니라, "테슬라의 자율주행 소식이 주가에 미칠 영향은?" 처럼 매우 구체적이어야 합니다.
        - 절대 똑같은 문장을 반복하지 마세요. (중복 출력 시 시스템 오류 발생)

        반드시 아래 JSON 포맷으로만 응답해야 합니다. 마크다운(`) 등은 절대 포함하지 마세요.
        {{
            "master_briefing": "마스터 브리핑 내용",
            "main_question": "메인 질문 내용",
            "sub_questions": ["서브질문1", "서브질문2", "서브질문3", "서브질문4"]
        }}
        """
        
        # Groq LLM에 프롬프트 전송 및 답변 받기
        response = llm.invoke(prompt)
        text = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"  ❌ Groq AI 요약 생성 실패 (기본값 사용): {e}")
        return default_q

# ────────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────────
def run_daily_fetch():
    now_kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    print(f"\n{'='*60}")
    print(f"  AI 금융 에이전트 - 데일리 데이터 수집 시작")
    print(f"  실행 시각 (KST): {now_kst.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    cache = {
        "fetched_at": now_kst.isoformat(),
        "nasdaq100":         collect_nasdaq100(),
        "sp500_weekly_top":  collect_sp500_weekly_top(),
        "major_indices":     collect_major_indices(),
        "sector_etfs":       collect_sector_etfs(),
        "fear_greed":        collect_fear_greed(),
        "global_news":       collect_global_news(),
        "earnings_calendar": collect_earnings_calendar(),
        "hot_themes":        collect_hot_themes(),
    }
    
    # 캐시 데이터를 기반으로 LLM 동적 질문 생성 추가
    cache["ai_summary"] = generate_daily_ai_summary(cache)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ 완료! 저장 경로: {CACHE_PATH}")
    print(f"{'='*60}\n")
    return cache


if __name__ == "__main__":
    run_daily_fetch()
