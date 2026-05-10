import json
from langchain_ollama import ChatOllama

def analyze_intent(user_message: str, chat_history: list = None) -> dict:
    """
    사용자의 질문을 분석하여 '단순 잡담/인사'인지 '금융/주식 분석'인지 분류합니다.
    버그 발생 시 무조건 'financial'로 처리하여 메인 에이전트가 처리하도록 하는 안전한 구조입니다.
    """
    default_fallback = {"intent": "financial", "chat_response": ""}
    
    try:
        # 가장 빠르고 저렴한 8B 모델 사용 (로컬 GPU 구동)
        router_llm = ChatOllama(model="llama3.1", temperature=0.1)
        
        history_text = ""
        if chat_history:
            history_text = "[최근 대화 내역]\n"
            for msg in chat_history[-4:]: # 최근 4개 메시지 정도만 컨텍스트로 제공
                role = "사용자" if msg.get("role") == "user" else "AI"
                history_text += f"{role}: {msg.get('content')}\n"
            history_text += "\n"
        
        prompt = f"""
        당신은 금융 에이전트의 '수문장(Router)'입니다.
        사용자의 메시지가 다음 중 어디에 속하는지 분류하세요.
        
        {history_text}[분류 기준]
        1. "chat" (단순 잡담/인사/감사): 
           - 예: "안녕", "고마워", "너 이름이 뭐야?", "오늘 날씨 좋네", "수고했어"
           - 금융 데이터 검색, 주가 확인, 뉴스 검색이 전혀 필요 없는 경우.
        2. "financial" (금융/주식/경제 분석): 
           - 예: "테슬라 주가 알려줘", "애플 실적 어때?", "금리가 어떻게 될까?", "지금 살만한 주식 추천해 줘", "내 포트폴리오 어때"
           - 기업명, 종목코드, 거시경제 지표가 포함되어 있거나 분석/추천/검색이 필요한 경우.
        
        [지시사항]
        - 의도가 "chat"인 경우에만, 사용자에게 보낼 짧고 친절한 답변(1~2문장)을 "chat_response"에 작성하세요. (금융 이야기는 하지 마세요)
        - 의도가 "financial"이면 "chat_response"는 빈 문자열("")로 두세요.
        - 반드시 아래의 순수 JSON 포맷으로만 응답해야 합니다. 마크다운(```json 등)이나 다른 설명은 절대 넣지 마세요.
        
        {{
            "intent": "chat" 또는 "financial",
            "chat_response": "chat일 경우의 친절한 답변"
        }}
        
        사용자 메시지: "{user_message}"
        """
        
        response = router_llm.invoke(prompt)
        text = response.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        
        intent = parsed.get("intent", "financial")
        chat_response = parsed.get("chat_response", "")
        
        # 의도 분류가 명확하지 않다면 무조건 financial로 넘겨서 안전하게 처리
        if intent not in ["chat", "financial"]:
            intent = "financial"
            
        return {"intent": intent, "chat_response": chat_response}
        
    except Exception as e:
        print(f"⚠️ 라우터 분석 실패 (기본값 financial 적용): {e}")
        return default_fallback
