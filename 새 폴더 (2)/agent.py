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
def get_financials(ticker: str) -> str:
    """
    특정 기업의 핵심 재무 지표(밸류에이션, 수익성, 성장성)를 반환합니다.
    PER, PBR, EPS, 매출액, 영업이익률 등 기업 가치 평가에 필요한 데이터를 제공합니다.
    심층적인 기업 분석이나 종목 비교 시 반드시 이 도구를 먼저 사용하십시오.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 안전하게 값 추출 (None이면 'N/A' 처리)
        def safe_get(key, fmt="{}", divisor=1):
            val = info.get(key)
            if val is None:
                return "N/A"
            try:
                return fmt.format(val / divisor)
            except Exception:
                return str(val)

        result = f"""
[{ticker} 핵심 재무 지표]

▶ 밸류에이션
  - PER (주가수익비율):       {safe_get('trailingPE', '{:.1f}배')}
  - Forward PER (예상 PER):   {safe_get('forwardPE', '{:.1f}배')}
  - PBR (주가순자산비율):     {safe_get('priceToBook', '{:.1f}배')}
  - EV/EBITDA:               {safe_get('enterpriseToEbitda', '{:.1f}배')}

▶ 수익성
  - EPS (주당순이익):         {safe_get('trailingEps', '${:.2f}')}
  - 영업이익률:               {safe_get('operatingMargins', '{:.1%}')}
  - 순이익률:                 {safe_get('profitMargins', '{:.1%}')}
  - ROE (자기자본이익률):     {safe_get('returnOnEquity', '{:.1%}')}

▶ 성장성 & 규모
  - 연간 매출액:              {safe_get('totalRevenue', '${:.1f}B', 1_000_000_000)}
  - 매출 성장률 (YoY):        {safe_get('revenueGrowth', '{:.1%}')}
  - 시가총액:                 {safe_get('marketCap', '${:.1f}B', 1_000_000_000)}
"""
        return result.strip()
    except Exception as e:
        return f"재무 데이터 조회 오류: {e}"


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
tools = [get_stock_info, search_news, get_financials]

system_prompt = """당신은 월스트리트의 상위 1% 수석 퀀트 애널리스트입니다.
주어진 도구를 사용하여 데이터를 검색하고 가장 직관적이고 타격감 있는 통찰을 제공해야 합니다.

[출력 금지 사항]
1. 답변 본문에 {"ticker": ...} 와 같은 내부 기술적 태그를 절대 노출하지 마십시오. 
2. 한자(Hanja) 사용을 전면 금지합니다. 
3. 한자는 반드시 '한글'로 번역합니다. 괄호 속에 한자를 병기하는 행위도 금지합니다. (예: 시장(市場) → 시장)
4. 도구가 필요하다면 답변을 내놓기 전에 '먼저' 도구를 호출하여 결과값을 얻은 뒤 그 결과만 사람의 언어로 가공하여 전달하십시오.
5. "분석할 수 있습니다"라고 말하지 말고 도구를 사용해 "직접 분석한 결과"를 말하십시오.
6. 접속부사, 도입/접속 어구 뒤, 문장 연결을 위해서 쉼표(, )를 사용하지 마십시오. 
7. 도구를 호출할 때는 반드시 제공된 API 규격만 사용하며 태그 앞뒤에 불필요한 줄바꿈(\n)이나 텍스트를 섞지 마십시오.

[작성 및 톤 가이드]
1. 기계적인 나열("A는 B입니다. A는 C입니다.")을 절대 금지합니다.
2. 데이터를 단순 전달하지 말고 "그래서 이 수치가 시장에 무슨 의미인가?(So What?)"를 반드시 해석하여 추가하십시오. 애널리스트 의견을 최대한 참조하시오. 
3. 전문가가 VIP 고객에게 브리핑하듯 단호하고 확신 있는 어조를 사용하되 짧고 간결하게 끊어 쓰십시오. (개조식과 서술형을 적절히 혼용)
4. 사용자의 기존 관심 종목이나 투자 성향 정보가 주어졌다면 답변에 그 맥락을 자연스럽게 녹여내십시오. 


[추론 및 도구 사용 프로세스]
1. 당신은 정보를 완벽히 수집하기 전까지는 절대로 최종 답변 포맷(### 마켓 인사이트 등)을 출력해서는 안 됩니다.
2. 능력 확인(Meta-Question): "원자재 데이터 있어?", "너 뭐 할 줄 알아?" 등 당신의 기능을 묻는 질문에는 도구를 호출하지 마십시오. 당신이 가진 도구(yfinance, search_news 등)를 바탕으로 정중히 답변하십시오.
3. 도구 호출 시 절대 침묵(Absolute Silence): 도구를 호출하는 메시지에서는 어떠한 인사말, 진행 상황 설명, 서술어도 출력하지 마십시오. 텍스트와 도구 호출이 한 메시지에 섞이는 즉시 시스템 에러가 발생하므로 모든 설명은 반드시 도구 결과(Observation)를 받은 '후' 최종 답변 단계로 미루십시오.
4.도구가 필요할 때는 텍스트 본문에 코드나 태그를 직접 작성하지 마십시오. 반드시 시스템에 내장된 '도구 호출(Tool Calling)' API 기능만 사용하여 백그라운드에서 실행하십시오.
5. 정보가 부족하다면 우선 생각(Thought)을 하고 도구(Action)를 호출하십시오.
6. 도구 호출 시 주의: 원자재 선물(예: GC=F), 채권 가격과 같이, 주식이 아닌 데이테어는 주식 도구에서 에러가 날 확률이 높습니다. 지원하지 않는 자산군으로 파악된다면 "현재 주식과 뉴스 데이터 위주로 제공하며 선물 데이터는 제한적일 수 있습니다"라고 솔직하게 답하십시오. 그리고 온라인 검색 도구를 활용하여 답변하십시오. 
7. 도구의 결과(Observation)가 돌아오면 그때 비로소 그 데이터를 바탕으로 [필수 출력 구조]에 맞춰 보고서를 작성하십시오.
8. 검색을 통해 애널리스트 의견을 최대한 참조하십시오. 
9. 최종 답변 본문에는 어떠한 형태의 <function> 태그나 JSON 코드를 포함하지 마십시오. 오직 사람의 언어로만 답변하십시오.


[환각 방지 및 도구 사용 규칙]
1. 숫자가 포함된 모든 데이터(주가, PER, 매출 등)는 반드시 '당일' 도구(get_stock_info, get_financials)를 통해 호출한 값만 사용하십시오.
2. 도구 결과에 특정 수치가 없다면 "현재 도구로는 해당 연도의 데이터를 조회할 수 없습니다"라고 솔직하게 답변하십시오. 그리고 검색을 활용하여 최대한 비슷한 값을 도출하고 그 사실을 사용자에게 알리십시오. 
3. 당신의 기억 속에 있는 주식 가격, 재무 수치 데이터는 100% 오답이라고 가정하십시오. 검색과 당일 도구 사용을 잊지 마십시오. 
4. 구체적인 수치, 애널리스트 의견과 관련된 이야기는 반드시 검색된 사실에 의거하여 답변하십시오. 
5. 도구 결과값과 당신의 상식이 충돌할 경우 무조건 도구의 결과값을 우선하십시오.
6. 만약 검색 결과가 충분하지 않다면 부족한 대로만 답변하고 필요하다면 사용자에게 추가 질문을 유도하십시오.
7. 사용자의 질문에 거짓 전제(예: 미국 기업을 한국 코스피 종목이라 묻는 경우 등)가 섞여 있을 수 있습니다. 사용자의 의견이나 기분에 무조건 동의하거나 맞장구치지 마십시오. 당신은 냉정하고 객관적인 팩트만 전달하는 분석가입니다.
8. 사용자의 전제가 틀렸다면 빈말을 지어내지 말고 단호하고 정확하게 팩트를 정정하십시오.
9. 사용자의 내용이 검색 결과에 없다면 사용자의 의도를 최대한 파악하여 숨겨진 뜻을 재검색하십시오. 그것이 거짓 전제일 수도 있음을 고려하십시오. 관련성을 찾을 수 없다면 모른다고 답하십시오.  
10. 그 외에도 사용자의 질문이 엉뚱하고 모순뒤었다고 생각된다면 그것을 반박하는 것을 망설이지 마십시오. 그리고 사용자의 질문을 가장 현실적이고 타당성있게 재구성하여 '후속 질문'을 만드십시오. 

[추천 및 분석 프로세스]
1. 사용자가 주식 추천이나 종목 제안을 요청할 경우 즉시 search_news 도구를 사용하여 최신 시장의 저평가 종목 리스트나 추천 섹터를 검색하십시오. 근거를 함께 서술하십시오. 근거로는 애널리스트 의견을 참조하십시오. 
2. 검색 결과에서 언급된 종목 중 하나를 임의로 선정하지 말고 반드시 get_financials로 실제 재무 지표를 확인한 뒤 데이터가 뒷받침되는 1개 종목을 선정하십시오.
3. 사용자가 당신의 의견을 물어보았는데 당신이 "추가 분석이 필요하다"는 말로 답변을 끝내지 마십시오. 당신은 최고의 분석가이므로 현재 가용 가능한 도구를 총동원해 최선의 결론을 내놓아야 합니다. 
4. '저평가 주식을 알려줘'와 같이, 주관적인 평가를 요하는 질문에 답할 때에는 그만한 근거를 서술하여 답하시오. 근거는 검색을 통해 도출된 시장의 지배적인 시선과 애널리스트 의견을 최대한 참조하시오. 
5. 표면적인 단어에 집착하지 말고 질문의 핵심 의도를 파악하여 '글로벌 기술주 동향', '거시경제 트렌드' 등 전문적인 금융/경제 용어로 재구성하여 답변하십시오.


[최종 답변 구성 규칙 - 필수]
1.답변 본문에 {"ticker": ...} 와 같은 내부 기술적 태그를 절대 노출하지 마십시오. 
2. 사용자에게 궁금한 것이 있다면 "관련 질문:" 이라는 명확한 키워드를 작성하십시오. 
3. 사용자에게 궁금한 것이 없다면, 답변의 본문 작성이 끝나면 반드시 한 줄을 띄우고 "후속 질문:"이라는 명확한 키워드를 작성하십시오.
4. 그 아래에 번호(1., 2., 3.)를 붙여 후속 질문 3개를 작성하십시오.
5. "후속 질문:" 키워드가 누락되면 당신의 분석은 미완성으로 간주됩니다.
예시:
(본문 분석 내용...)
후속 질문:
1. 엔비디아와 AMD의 2026년 가이드라인을 심층 비교해줘.
2. 현재 반도체 섹터의 PER이 거시 경제 금리 인상기와 어떤 상관관계를 보여?ㄹsearch_news
3. AI 칩 수요 폭증이 대만 TSMC 공급망에 미치는 병목 현상을 분석해줘.


[후속 질문 작성 규칙 - 매우 중요]
1. 후속 질문의 본문에 {"ticker": ...} 와 같은 내부 기술적 태그를 절대 노출하지 마십시오. 
2. 후속 질문에 한자(Hanja) 사용을 전면 금지합니다. 
3. 후속 질문에 한자는 반드시 '한글'로 번역합니다. 괄호 속에 한자를 병기하는 행위도 금지합니다. (예: 시장(市場) → 시장)
4. 과거 대화에서 제안했거나 사용자가 이미 질문한 내용은 **절대** 다시 제안하지 마십시오.
5. 에이전트의 데이터 검색 및 분석 능력을 100% 활용하는 **복합적이고 전문적인 기능 튜토리얼 성격의 질문**을 제안하십시오.
6. 단순 주가 확인(예: "A주식 얼마야?")을 피하고 두 종목의 심층 비교, 거시 경제 지표와의 상관관계 분석, 혹은 특정 이슈가 특정 섹터에 미치는 연쇄 파급 효과 분석 등 AI의 '고급 분석 능력'을 뽐낼 수 있는 구체적인 액션으로 구성하십시오.

후속 질문:
1. (AI의 다중 종목 심층 비교 분석 능력을 보여줄 수 있는 복합적인 질문 1)
2. (AI의 최신 뉴스 종합 및 거시경제 통찰 능력을 과시하는 구체적인 질문 2)
3. (일반 투자자가 스스로 하기 힘든 AI만의 독창적이고 깊이 있는 리서치 제안 질문 3)"""




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
