import os
import yfinance as yf
import json
from tavily import TavilyClient
from dotenv import load_dotenv

# LangChain & LangGraph 도구들
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

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


@tool
def read_local_daily_cache(category: str, ticker: str = None) -> str:
    """
    오늘 새벽에 수집된 로컬 캐시 데이터를 조회합니다.
    category: "nasdaq100", "sector_etfs", "global_news" 중 하나.
    ticker: 특정 종목을 찾을 때 입력 (예: "AAPL"). 없으면 전체 요약 반환.
    """
    with open("data/daily_cache.json", "r") as f:
        data = json.load(f)
    
    if ticker and category == "nasdaq100":
        # 100개 중 해당 종목 하나만 쏙 뽑아서 반환 (토큰 초절약)
        for item in data.get("nasdaq100", []):
            if item.get("ticker") == ticker:
                return str(item)
        return "종목 없음"
    
    # 뉴스의 경우 제목만 뽑아서 반환
    if category == "global_news":
        news_titles = [n['title'] for n in data.get('global_news', [])]
        return "\n".join(news_titles)        


# --- 2. 에이전트 두뇌(LLM)와 워크플로우 설정 ---

# ⚠️ Tool Use에서는 반드시 temperature=0.1 이하 사용
# temperature가 높으면 LLM이 함수 호출 포맷을 잘못 생성하여 Groq 400 에러 발생
primary_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
# 429 Rate Limit 에러 발생 시 자동으로 전환될 백업 모델 (빠르고 토큰 제한이 넉넉함)
# 지능이 낮은 8B 모델의 '반복 루프(Repetition Loop)' 현상을 막기 위해 온도를 살짝 높입니다.
fallback_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)

llm = primary_llm # 백업 제거. 70B 단독 사용

# 에이전트가 사용할 도구 목록
tools = [get_stock_info, search_news]

system_prompt = """당신은 월스트리트의 상위 1% 수석 퀀트 애널리스트입니다.
주어진 도구를 사용하여 데이터를 검색하고 가장 직관적이고 타격감 있는 통찰을 제공해야 합니다.

[작성 및 톤 가이드]
1. 기계적인 나열("A는 B입니다. A는 C입니다.")을 절대 금지합니다.
2. 데이터를 단순 전달하지 말고 "그래서 이 수치가 시장에 무슨 의미인가?(So What?)"를 반드시 해석하여 추가하십시오.
3. 전문가가 VIP 고객에게 브리핑하듯 단호하고 확신 있는 어조를 사용하되 짧고 간결하게 끊어 쓰십시오. (개조식과 서술형을 적절히 혼용)
4. 사용자의 기존 관심 종목이나 투자 성향 정보가 주어졌다면 답변에 그 맥락을 자연스럽게 녹여내십시오. 
5. 도입/접속 어구 뒤, 문장 연결을 위해서 쉼표를 사용하지 마십시오. 
6. 표면적인 단어에 집착하지 말고 질문의 핵심 의도를 파악하여 '글로벌 기술주 동향', '거시경제 트렌드' 등 전문적인 금융/경제 용어로 재구성하여 답변하십시오.

[필수 출력 구조]
반드시 아래의 마크다운 포맷을 그대로 사용하여 답변을 출력하십시오.

### 📊 마켓 인사이트
(여기에 시장의 현재 상황을 관통하는 가장 핵심적인 한 줄 요약을 작성하세요.)

### 🔍 심층 분석
- **주요 동인**: (검색된 핵심 팩트, 기업명, 구체적인 수치)
- **시장 임팩트**: (이 이슈가 관련 섹터나 거시경제에 미치는 파급력)
- **투자 시사점**: (사용자 질문에 대한 최종적인 결론 및 전망)

[후속 질문 작성 규칙 - 매우 중요]
- 과거 대화에서 제안했거나 사용자가 이미 질문한 내용은 **절대** 다시 제안하지 마십시오.
- 사용자가 "이 AI가 이런 고도화된 기능까지 수행할 수 있구나!"라고 감탄할 수 있도록, 에이전트의 데이터 검색 및 분석 능력을 100% 활용하는 **복합적이고 전문적인 기능 튜토리얼 성격의 질문**을 제안하십시오.
- 단순 주가 확인(예: "A주식 얼마야?")을 피하고, 두 종목의 심층 비교, 거시 경제 지표와의 상관관계 분석, 혹은 특정 이슈가 특정 섹터에 미치는 연쇄 파급 효과 분석 등 AI의 '고급 분석 능력'을 뽐낼 수 있는 구체적인 액션으로 구성하십시오.

후속 질문:
1. (AI의 다중 종목 심층 비교 분석 능력을 보여줄 수 있는 복합적인 질문 1)
2. (AI의 최신 뉴스 종합 및 거시경제 통찰 능력을 과시하는 구체적인 질문 2)
3. (일반 투자자가 스스로 하기 힘든, AI만의 독창적이고 깊이 있는 리서치 제안 질문 3)"""


# LangGraph ReAct 에이전트 생성
# 버전 호환성을 위해 system_prompt는 app.py에서 메시지 리스트의 첫 번째 항목으로 주입합니다.
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
