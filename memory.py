import os
import json
import threading
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
MEMORY_PATH = os.path.join(DATA_DIR, "entity_memory.json")

def load_entity_memory():
    """안전하게 기존 메모리를 불러옵니다."""
    if not os.path.exists(MEMORY_PATH):
        return {"facts": []}
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) and "facts" in data else {"facts": []}
    except Exception:
        return {"facts": []}

def save_entity_memory(data):
    """안전하게 메모리를 저장합니다."""
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"메모리 저장 실패: {e}")

def _extract_and_update_worker(chat_history):
    """
    별도의 스레드에서 돌아갈 실제 추출 로직.
    안정성 최우선: 어떤 에러가 발생해도 조용히 종료됩니다.
    """
    try:
        # LLM 초기화 (빠르고 저렴한 모델)
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
        
        # 대화 내용을 하나의 텍스트로 합치기
        chat_text = ""
        for msg in chat_history:
            role = "사용자" if msg["role"] == "user" else "AI"
            chat_text += f"[{role}]: {msg['content']}\n"
        
        prompt = f"""
        당신은 사용자의 투자 프로필과 대화를 기억하는 '메모리 추출기'입니다.
        아래의 [최근 대화]를 읽고, 사용자가 언급한 구체적인 사실(Fact)이나 종목(Entity)을 추출하세요.
        
        [최근 대화]
        {chat_text}
        
        [지시사항]
        1. 사용자가 언급한 구체적인 사실(예: "작년에 테슬라로 손해봤다", "배당주를 선호한다", "의료 쪽에 관심이 있다" 등)만 추출하세요.
        2. 특별한 사실이 없다면 빈 배열([])을 반환하세요.
        3. 반드시 아래의 JSON 포맷으로만 응답해야 하며, 마크다운 기호(```json 등)를 절대 쓰지 마세요.
        
        {{
            "new_facts": [
                "사용자는 작년에 테슬라(TSLA)로 손실을 보았음",
                "사용자는 안정적인 배당주를 선호함"
            ]
        }}
        """
        
        response = llm.invoke(prompt)
        text = response.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        
        new_facts = parsed.get("new_facts", [])
        
        if new_facts:
            # 기존 메모리 로드 및 병합
            current_memory = load_entity_memory()
            
            # 너무 커지는 것을 방지 (최대 30개 유지)
            updated_facts = current_memory["facts"] + new_facts
            
            # 중복 제거 및 리스트 유지
            unique_facts = []
            for f in updated_facts:
                if f not in unique_facts:
                    unique_facts.append(f)
                    
            # 최신 팩트 위주로 30개 자르기 (오래된 것 삭제)
            current_memory["facts"] = unique_facts[-30:]
            
            save_entity_memory(current_memory)
            print(f"✅ 백그라운드 메모리 압축 완료: {len(new_facts)}개 팩트 추가됨")
            
    except Exception as e:
        print(f"⚠️ 백그라운드 메모리 압축 중 에러 발생 (무시됨): {e}")

def trigger_background_compression(chat_history):
    """
    메인 앱의 응답 속도에 영향을 주지 않도록 Thread를 사용해 백그라운드에서 실행합니다.
    에러가 발생해도 메인 Streamlit 앱은 전혀 멈추지 않습니다.
    """
    if not chat_history:
        return
        
    # Thread를 사용하여 비동기로 실행
    thread = threading.Thread(target=_extract_and_update_worker, args=(chat_history,))
    thread.daemon = True
    thread.start()
