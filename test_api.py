import os
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

def test_yfinance():
    print("--- 1. yfinance 주가 데이터 테스트 (AAPL) ---")
    try:
        aapl = yf.Ticker("AAPL")
        hist = aapl.history(period="1d")
        print(f"애플(AAPL) 최근 1일 데이터:\n{hist[['Close', 'Volume']]}")
        print("✅ yfinance 테스트 성공!\n")
    except Exception as e:
        print(f"❌ yfinance 테스트 실패: {e}\n")

def test_tavily():
    print("--- 2. Tavily 뉴스 검색 API 테스트 ---")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key or "tvly-" not in tavily_key:
        print("❌ TAVILY_API_KEY가 올바르게 설정되지 않았습니다.")
        return

    try:
        client = TavilyClient(api_key=tavily_key)
        # 애플 관련 최신 뉴스 3개 검색
        response = client.search(
            query="애플 주가 향후 전망", 
            search_depth="basic", 
            max_results=3
        )
        
        print("검색 결과:")
        for i, result in enumerate(response.get('results', [])):
            print(f"[{i+1}] {result['title']}")
            print(f"    - URL: {result['url']}")
        print("\n✅ Tavily API 테스트 성공!\n")
    except Exception as e:
        print(f"❌ Tavily API 테스트 실패: {e}\n")

def check_gemini_key():
    print("--- 3. Gemini API 키 확인 ---")
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "여기에_제미나이_API_키를_넣어주세요":
        print("✅ Gemini API 키가 .env에 등록되어 있습니다.\n")
    else:
        print("⚠️ Gemini API 키가 아직 .env에 등록되지 않았습니다. (테스트 자체는 문제 없음)\n")

if __name__ == "__main__":
    print("🚀 금융 에이전트 핵심 도구 테스트를 시작합니다...\n")
    test_yfinance()
    test_tavily()
    check_gemini_key()
