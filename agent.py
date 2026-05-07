import os
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv

# LangChain & LangGraph 도구들
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 환경변수 로드 (.env)
load_dotenv()

# --- 1. 에이전트에게 쥐어줄 도구(Tools) 정의 ---

@tool
def get_stock_info(ticker: str) -> str:
    """특정 주식의 티커(예: AAPL, NVDA, TSLA)를 입력받아 최근 주가 정보를 반환합니다."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return f"데이터를 찾을 수 없습니다: {ticker}"
        close_price = hist['Close'].iloc[-1]
        return f"[{ticker}] 최근 종가: ${close_price:.2f}"
    except Exception as e:
        return f"주가 검색 오류: {e}"

@tool
def search_news(query: str) -> str:
    """최신 금융 뉴스를 검색합니다. 검색어(query)를 입력받아 관련 뉴스 요약을 반환합니다."""
    try:
        tavily_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=tavily_key)
        # 일반 웹 검색이 아닌 '뉴스(news)' 전용으로 검색하고 최근 3일(days=3)로 제한합니다.
        response = client.search(
            query=query, 
            topic="news", 
            days=3, 
            max_results=3
        )
        
        results = []
        for res in response.get('results', []):
            results.append(f"- 제목: {res['title']}\n  내용: {res['content']}\n  출처: {res['url']}")
        return "\n".join(results)
    except Exception as e:
        return f"뉴스 검색 오류: {e}"

# --- 2. 에이전트 두뇌(LLM)와 워크플로우 설정 ---

# 무료 티어 한도가 넉넉하고 속도가 아주 빠른 최신 Flash 모델 사용 (Gemini 2.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 에이전트가 사용할 도구 목록
tools = [get_stock_info, search_news]

# 환각을 막기 위한 강력한 시스템 프롬프트 (족쇄)
system_prompt = """
당신은 최고의 금융 분석 에이전트입니다.
사용자의 질문에 대해 반드시 제공된 도구(get_stock_info, search_news)를 사용하여 최신 데이터를 확인한 뒤 답변하세요.
절대 당신의 사전 지식을 기반으로 소설을 쓰거나 지어내지 마세요. 데이터가 없으면 '정보가 없습니다'라고 답하세요.
답변의 마지막에는 사용자가 더 궁금해할 만한 '후속 질문(Follow-up Query)' 3가지를 추천해 주세요.
"""

# LangGraph를 이용해 도구와 LLM을 결합한 에이전트 생성 (버전 호환성을 위해 추가 인자 제거)
agent_executor = create_react_agent(llm, tools)

# --- 3. 실행 및 테스트 함수 ---

def chat_with_agent(user_message: str):
    print(f"\n🗣️ 사용자 질문: {user_message}")
    print("-" * 50)
    print("🤖 에이전트가 생각 중입니다...\n")
    
    # 에이전트에게 시스템 프롬프트와 사용자 메시지를 함께 전달
    response = agent_executor.invoke({
        "messages": [
            ("system", system_prompt),
            ("user", user_message)
        ]
    })
    
    # 디버깅을 위해 에이전트의 중간 과정(도구 사용)을 출력합니다.
    for msg in response["messages"]:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"🔧 [에이전트 결정] 도구 실행: {tc['name']} (검색어: {tc['args']})")
        elif msg.type == "tool":
            # 결과가 너무 길면 잘라서 보여줍니다.
            content_preview = msg.content[:150].replace('\n', ' ') + "..." if len(msg.content) > 150 else msg.content.replace('\n', ' ')
            print(f"📥 [도구 결과] {msg.name}: {content_preview}")
    
    print("\n" + "=" * 50)
    # 에이전트의 최종 답변 출력
    print("✅ 에이전트의 최종 답변:")
    raw_answer = response["messages"][-1].content
    if isinstance(raw_answer, list):
        # 구글 API 특성상 리스트로 반환될 경우 텍스트 부분만 추출
        final_answer = "".join([part.get("text", "") for part in raw_answer if isinstance(part, dict) and "text" in part])
    else:
        final_answer = str(raw_answer)
    print(final_answer)
    print("=" * 50)

if __name__ == "__main__":
    # 간단한 테스트 질문
    chat_with_agent("인텔(INTC) 최신 주가 알려주고, 관련된 최신 뉴스도 요약해 줘.")
