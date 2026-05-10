import streamlit as st
import requests
from streamlit_lottie import st_lottie
from agent import agent_executor, system_prompt
from memory import trigger_background_compression, load_entity_memory
from router import analyze_intent

# ────────────────────────────────────────────
# 1. 페이지 기본 설정
# ────────────────────────────────────────────
st.set_page_config(page_title="AI 금융 에이전트", layout="wide", initial_sidebar_state="collapsed")

# Lottie 로딩 애니메이션
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_loading = load_lottieurl("https://lottie.host/80d0d407-74be-4b07-9610-d8650dfc3ec9/u9bA5B0B4o.json")

# ────────────────────────────────────────────
# 2. 전역 CSS (모든 스타일 한 곳에서 관리)
# ────────────────────────────────────────────
st.markdown("""
<style>
    /* ── 배경 그라데이션 애니메이션 ── */
    .stApp {
        background: linear-gradient(-45deg, #ffe2e6, #fff0f5, #ffeaf2, #fdfbfb);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }
    @keyframes gradientBG {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── 불필요한 Streamlit 기본 UI 제거 ── */
    header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }

    /* ── 배경 비눗방울 (position:fixed, 뒤쪽 레이어) ── */
    .bg-bubbles {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
    }
    .bg-bubble {
        position: absolute;
        bottom: -150px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%,
            rgba(255,255,255,0.8),
            rgba(255,255,255,0.1) 60%,
            rgba(255,192,203,0.3) 100%);
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: inset 0 0 10px rgba(255,255,255,0.5),
                    0 5px 15px rgba(255,105,180,0.1);
        animation: rise infinite ease-in;
    }
    @keyframes rise {
        0%   { transform: translateY(0) scale(1);   opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { transform: translateY(-1200px) scale(1.5); opacity: 0; }
    }
    .bg-bubble:nth-child(1) { left:10%; width:40px; height:40px; animation-duration:8s;  animation-delay:0s; }
    .bg-bubble:nth-child(2) { left:25%; width:60px; height:60px; animation-duration:12s; animation-delay:2s; }
    .bg-bubble:nth-child(3) { left:40%; width:30px; height:30px; animation-duration:6s;  animation-delay:4s; }
    .bg-bubble:nth-child(4) { left:60%; width:80px; height:80px; animation-duration:15s; animation-delay:1s; }
    .bg-bubble:nth-child(5) { left:75%; width:50px; height:50px; animation-duration:10s; animation-delay:5s; }
    .bg-bubble:nth-child(6) { left:90%; width:45px; height:45px; animation-duration:9s;  animation-delay:3s; }

    /* ── 둥실둥실 떠오르는 keyframe ── */
    @keyframes float_left {
        0%   { transform: translateY(0px) rotate(0deg); }
        50%  { transform: translateY(-18px) rotate(4deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    @keyframes float_right {
        0%   { transform: translateY(0px) rotate(0deg); }
        50%  { transform: translateY(14px) rotate(-4deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }

    /* ── 좌측 종목 비눗방울 (완벽한 원형) ── */
    .stock-bubble-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: center;
        padding-top: 40px;
    }
    .stock-bubble {
        width: 82px;
        height: 82px;           /* 가로 = 세로 → 완벽한 원 */
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-size: 0.74rem;
        font-weight: 800;
        color: #c2185b;
        cursor: pointer;
        background: radial-gradient(
            circle at 30% 28%,
            rgba(255,255,255,0.95) 0%,
            rgba(255,220,230,0.60) 40%,
            rgba(255,182,193,0.35) 70%,
            rgba(255,105,180,0.25) 100%
        );
        border: 1.5px solid rgba(255,255,255,0.85);
        box-shadow:
            inset 0 -4px 12px rgba(255,105,180,0.18),
            inset 0  4px 10px rgba(255,255,255,0.70),
            0 6px 18px rgba(255,105,180,0.22);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        word-break: keep-all;
        line-height: 1.3;
        padding: 6px;
        user-select: none;
        animation: float_left 4s ease-in-out infinite;
    }
    .stock-bubble:nth-child(even) {
        animation: float_right 5s ease-in-out infinite;
    }
    .stock-bubble:hover {
        transform: scale(1.14);
        box-shadow:
            inset 0 -4px 12px rgba(255,105,180,0.28),
            inset 0  4px 10px rgba(255,255,255,0.90),
            0 12px 30px rgba(255,105,180,0.40);
    }

    /* ── 중앙 카드 (오늘의 핵심 화두) ── */
    @keyframes suckIn {
        0%   { transform: scale(0.85) translateY(-40px); opacity: 0; }
        100% { transform: scale(1)    translateY(0);     opacity: 1; }
    }
    .center-main-card {
        animation: suckIn 0.8s cubic-bezier(0.2,0.8,0.2,1) forwards;
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,192,203,0.5);
        border-radius: 28px;
        padding: 38px 36px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 10px 40px rgba(255,182,193,0.3);
    }
    .center-main-card h2 { color: #d81b60; margin-bottom: 8px; }
    .center-main-card p  { color: #6a1b9a; font-size: 1.1rem; font-weight: 600; }

    /* ── 타이틀 ── */
    .title-text {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff4081, #d50000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.2rem;
        margin-bottom: 16px;
    }

    /* ── 우측 질문 버튼 ── */
    div[data-testid="column"]:nth-child(3) button {
        animation: float_right 5s ease-in-out infinite;
        border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
        background: linear-gradient(135deg,rgba(255,204,204,0.6),rgba(255,153,204,0.5));
        border: 1px solid rgba(255,153,204,0.4);
        color: #c2185b;
        height: 110px;
        width: 100%;
        margin-bottom: 16px;
        font-size: 0.9rem;
        font-weight: 700;
        transition: all 0.3s;
        box-shadow: 0 8px 32px rgba(255,153,204,0.15);
    }
    div[data-testid="column"]:nth-child(3) button:hover {
        transform: scale(1.05);
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(255,105,180,0.8);
        box-shadow: 0 0 20px rgba(255,105,180,0.3);
    }

    /* ── 채팅 답변 너비 제한 및 중앙 정렬 ── */
    [data-testid="stChatMessage"] {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* ── 화면 하단 여유 공간 확보 (입력창에 가려지지 않게) ── */
    .block-container {
        padding-bottom: 450px !important;
        overflow-y: auto !important;
    }

    /* ── 하단 채팅 입력창 디자인 (Gemini 스타일 고정) ── */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 100px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 800px !important;
        max-width: 90vw !important; /* 모바일 화면 대응 */
        z-index: 999;
    }
    
    /* ── 입력창 내부 텍스트 영역 높이 고정 ── */
    [data-testid="stChatInputTextArea"] {
        min-height: 140px !important;
    }

    /* ── 추천 후속 질문 프리미엄 필(Pill) 버튼 커스텀 ── */
    div[data-testid="stChatMessage"] div[data-testid="column"] button {
        background-color: rgba(255, 255, 255, 0.85) !important;
        border: 1.2px solid rgba(255, 105, 180, 0.25) !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: #c2185b !important;
        box-shadow: 0 3px 8px rgba(255, 105, 180, 0.05) !important;
        transition: all 0.18s ease !important;
        white-space: normal !important;
        height: auto !important;
        line-height: 1.3 !important;
        cursor: pointer !important;
    }
    div[data-testid="stChatMessage"] div[data-testid="column"] button:hover {
        background-color: #ff4d8f !important;   /* 클릭된다는 느낌이 올 진한 핑크 */
        border-color: #ff4d8f !important;
        color: #ffffff !important;
        transform: translateY(-3px) scale(1.04) !important;
        box-shadow: 0 8px 20px rgba(255, 77, 143, 0.35) !important;
    }
    div[data-testid="stChatMessage"] div[data-testid="column"] button:active {
        transform: translateY(0px) scale(0.97) !important;   /* 누른 시 누려들어가는 햄틱 */
        background-color: #e91e8c !important;
        box-shadow: 0 2px 6px rgba(255, 77, 143, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────
# 3. 배경 비눗방울 HTML 주입
# ────────────────────────────────────────────
st.markdown("""
<div class="bg-bubbles">
    <div class="bg-bubble"></div>
    <div class="bg-bubble"></div>
    <div class="bg-bubble"></div>
    <div class="bg-bubble"></div>
    <div class="bg-bubble"></div>
    <div class="bg-bubble"></div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────
# 4. 상태 관리 & 헬퍼 함수
# ────────────────────────────────────────────
import re

def parse_follow_up(text):
    """답변 하단의 후속 질문 블록을 감지해 본문과 질문 리스트로 쪼갭니다."""
    parts = re.split(r"후속\s*질문[:\s]*", text, flags=re.IGNORECASE)
    main_text = parts[0].strip()
    questions = []
    
    # 💡 방어 코드: 만약 쪼갰는데 앞부분(본문)이 비어있다면?
    if not main_text:
        main_text = "현재 해당 질문에 대한 명확한 분석 데이터를 찾지 못했습니다. 아래 추천 질문을 눌러 다른 관점으로 탐색해 보세요."

    if len(parts) > 1:
        # 1. 2. 3. 혹은 - 로 시작하는 한 줄 질문 추출
        q_matches = re.findall(r"\d+[\.\s\-]+([^\n]+)", parts[1])
        questions = [q.strip() for q in q_matches if q.strip()]
        
    return main_text, questions

if "messages" not in st.session_state:
    st.session_state.messages = []
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = None

# ── 쿼리 파라미터로 버블 클릭 감지 ──
if "stock" in st.query_params:
    clicked = st.query_params["stock"]
    st.session_state.trigger_query = f"{clicked} 최신 주가와 관련 뉴스를 분석해줘."
    st.query_params.clear()

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"관심종목": [], "투자성향": "파악 중", "최근관심사": ""}
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "compressed_turns" not in st.session_state:
    st.session_state.compressed_turns = 0


# ────────────────────────────────────────────
# 5. 데이터 (daily DB 및 LLM 동적 질문)
# ────────────────────────────────────────────
import os, json
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "daily_cache.json")

daily_main_q = "시장이 닫힌 후, 오늘의 주요 기술주 흐름은 어떠한가요?"
right_bubbles = ["최근 나스닥 등락 원인", "금리 인하 수혜주", "지금 핫한 테마", "오늘 주목할 특징주"]

# 캐시 파일이 있으면 LLM이 생성한 'ai_summary' 데이터로 덮어씌움
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
            summary_data = cache.get("ai_summary", {})
            if summary_data:
                daily_main_q = summary_data.get("main_question", daily_main_q)
                raw_subs = summary_data.get("sub_questions", right_bubbles)
                
                # 💡 방어 로직: 중복된 질문과 메인 질문과 똑같은 질문을 강제로 제거합니다.
                unique_subs = []
                for q in raw_subs:
                    if q not in unique_subs and q != daily_main_q:
                        unique_subs.append(q)
                
                # 만약 중복을 다 지웠더니 4개가 안 된다면, 비상용 질문으로 채워 넣습니다.
                fallback_qs = ["어제 장 마감 후 특징주는?", "이번 주 주요 경제 일정은?", "현재 시장 리스크 요인은?", "앞으로 주목해야 할 테마는?"]
                while len(unique_subs) < 4:
                    unique_subs.append(fallback_qs.pop(0))
                    
                right_bubbles = unique_subs[:4]
    except Exception as e:
        import streamlit as st
        st.sidebar.error(f"캐시 로드 실패: {e}")

left_stocks = [
    {"name": "애플",      "ticker": "AAPL"},
    {"name": "MS",        "ticker": "MSFT"},
    {"name": "엔비디아",  "ticker": "NVDA"},
    {"name": "알파벳",    "ticker": "GOOGL"},
    {"name": "아마존",    "ticker": "AMZN"},
    {"name": "메타",      "ticker": "META"},
    {"name": "테슬라",    "ticker": "TSLA"},
    {"name": "일라이\n릴리","ticker": "LLY"},
]


import json
from langchain_ollama import ChatOllama

def update_user_profile_in_background(recent_chats, current_profile):
    """최근 3번의 대화를 분석해 사용자의 프로필을 JSON으로 업데이트합니다."""
    try:
        # 가볍고 빠른 모델 호출 (로컬 GPU 구동)
        profiler_llm = ChatOllama(model="llama3.1", temperature=0)
        
        # 최근 대화를 텍스트로 묶기
        chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_chats])
        
        prompt = f"""
        당신은 사용자의 대화를 분석하는 프로파일러입니다.
        아래의 '최근 대화'를 분석하여, 기존 '사용자 프로필'을 최신화하십시오.
        
        [기존 사용자 프로필]
        {json.dumps(current_profile, ensure_ascii=False)}
        
        [최근 대화]
        {chat_text}
        
        지시사항:
        1. 대화에서 언급된 특정 주식 티커나 기업명은 '관심종목' 리스트에 추가 (최대 5개 유지).
        2. 대화의 맥락을 보고 '투자성향'을 10자 이내로 업데이트 (예: 안전추구, 빅테크 선호 등).
        3. 가장 최근에 물어본 경제 테마를 '최근관심사'에 10자 이내로 요약.
        4. 무조건 아래 JSON 형식으로만 출력할 것. 마크다운 기호 불가.
        
        {{
            "관심종목": ["AAPL", "TSLA"],
            "투자성향": "빅테크 선호",
            "최근관심사": "금리 인하"
        }}
        """
        
        response = profiler_llm.invoke(prompt)
        text = response.content.replace("```json", "").replace("```", "").strip()
        new_profile = json.loads(text)
        return new_profile
    except Exception as e:
        print(f"프로필 업데이트 실패: {e}")
        return current_profile


# ────────────────────────────────────────────
# 6. 3단 레이아웃
# ────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1.5, 6, 1.5], gap="large")

# ── 좌측: 완벽한 원형 비눗방울 (순수 HTML) ──
with col_left:
    # <style>은 전역 블록에 이미 있으므로 여기서는 <div>만 생성
    html_parts = ['<div class="stock-bubble-wrap">']
    for s in left_stocks:
        name = s["name"]
        html_parts.append(
            f'<div class="stock-bubble" onclick="window.location.search=\'?stock={name}\'">'
            f'{name}</div>'
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)

# ── 우측: 서브 질문 버튼 ──
with col_right:
    st.write("")
    st.write("")
    if not st.session_state.messages:
        for q in right_bubbles:
            if st.button(q, key=q):
                st.session_state.trigger_query = q

# ── 중앙: 타이틀 + 챗봇 ──
with col_center:
    st.markdown("<div class='title-text'>AI 금융 에이전트</div>", unsafe_allow_html=True)

    # 채팅창 입력값을 위에서 미리 판별하여 prompt 할당
    chat_input = st.chat_input("종목명이나 뉴스를 물어보세요...")
    prompt = chat_input or st.session_state.trigger_query

    # 대화 기록이 없고, 방금 입력한 프롬프트도 없을 때만 오늘의 화두 카드 표시
    if not st.session_state.messages and not prompt:
        st.markdown(f"""
        <div class='center-main-card'>
            <h2>오늘의 핵심 화두</h2>
            <p>"{daily_main_q}"</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button("👉 위 질문으로 분석 시작하기", use_container_width=True):
                st.session_state.trigger_query = daily_main_q
                st.rerun()

    # 채팅 기록 렌더링
    # → 후속 질문 버튼은 가장 마지막 assistant 메시지에만 표시
    #last_assistant_idx = max(
    #    (i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
    #    default=-1
    #)
    total_msgs = len(st.session_state.messages)
    
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                main_text, q_list = parse_follow_up(message["content"])
                st.markdown(main_text)
                
                # 💡 핵심 방어 로직: 
                # 1. 이 메시지가 전체 대화의 맨 마지막 메시지일 것
                # 2. 그리고 사용자가 방금 새로운 질문(prompt)을 입력하지 않은 대기 상태일 것
                is_last_message = (msg_idx == total_msgs - 1)
                
                if q_list and is_last_message and not prompt:
                    st.markdown("<p style='font-size:0.8rem; font-weight:800; color:#c2185b; margin: 14px 0 6px 0;'>🔮 이 질문은 어떠세요?</p>", unsafe_allow_html=True)
                    cols = st.columns(len(q_list))
                    for idx, q in enumerate(q_list):
                        with cols[idx]:
                            if st.button(q, key=f"fu_{msg_idx}_{idx}"):
                                st.session_state.trigger_query = q
                                st.toast("분석을 시작합니다...", icon="🔍")
                                st.rerun()
            else:
                st.markdown(message["content"])

    # 만약 사용자 입력(prompt)이 있다면 처리 시작
    if prompt:
        st.session_state.trigger_query = None
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            anim_ph = st.empty()
            with anim_ph.container():
                if lottie_loading:
                    st_lottie(lottie_loading, height=140, key="loading_anim")
                else:
                    st.info("데이터를 분석 중입니다...")

            try:
                # 🚦 1. 라우터(수문장) 검사: 가벼운 잡담인지 하드한 질문인지 분류
                route_result = analyze_intent(prompt, st.session_state.messages)
                
                if route_result["intent"] == "chat":
                    # 단순 잡담이면 메인 에이전트를 깨우지 않고 바로 답변
                    final_answer = route_result["chat_response"]
                    
                else:
                    # 'financial' 이면 기존의 무거운 로직(70B 모델 + 도구 + 프로필) 실행
                    
                    # 1. 마스터 브리핑 불러오기
                    master_briefing = ""
                    if os.path.exists(CACHE_PATH):
                        with open(CACHE_PATH, "r", encoding="utf-8") as f:
                            cache_data = json.load(f)
                            master_briefing = cache_data.get("ai_summary", {}).get("master_briefing", "")

                    # 2. 사용자 프로필 문자열화 및 Entity Memory 로드
                    profile_str = json.dumps(st.session_state.user_profile, ensure_ascii=False)
                    entity_memory = load_entity_memory()
                    facts_str = "\n".join([f"- {f}" for f in entity_memory.get("facts", [])])
                    if not facts_str:
                        facts_str = "- 아직 추출된 개별 팩트가 없습니다."

                    # 3. LLM에게 보낼 최종 메시지 배열 생성
                    llm_messages = [("system", system_prompt)]

                    # 4. 슬라이딩 윈도우: 최근 5턴(10개 메시지) 가져오기
                    recent_memory = st.session_state.messages[-10:]
                    for msg in recent_memory:
                        llm_messages.append((msg["role"], msg["content"]))

                    # 💡 핵심: 마지막 user 메시지(방금 한 질문)에 백그라운드 데이터(브리핑 + 프로필 + 팩트)를 몰래 덧붙임
                    if llm_messages:
                        last_role, last_content = llm_messages[-1]
                        if last_role == "user":
                            enriched_prompt = f"""[참고용 백그라운드 데이터]
- 오늘의 시장 요약: {master_briefing}
- 사용자 맞춤형 프로필: {profile_str}

[사용자의 장기 기억 팩트]
{facts_str}

[사용자 질문]
{last_content}"""
                            llm_messages[-1] = ("user", enriched_prompt)

                    # 5. 메인 에이전트(70B) 실행
                    response = agent_executor.invoke({
                        "messages": llm_messages
                    })

                    # 6. 결과물 파싱
                    raw = response["messages"][-1].content
                    if isinstance(raw, list):
                        final_answer = "".join(
                            p.get("text", "") for p in raw
                            if isinstance(p, dict) and "text" in p
                        )
                    else:
                        final_answer = str(raw)

            except Exception as e:
                final_answer = f"오류 발생: {e}"


            anim_ph.empty()

            # 💡 핵심 2: 롤링 압축(Rolling Compression) 트리거
            st.session_state.turn_count += 1
            if st.session_state.turn_count >= 3:
                # 화면 로딩에 방해되지 않도록 기존 프로필 최신화 진행
                latest_chats = st.session_state.messages[-6:] # 질문답변 3세트
                new_profile = update_user_profile_in_background(latest_chats, st.session_state.user_profile)
                st.session_state.user_profile = new_profile
                st.session_state.turn_count = 0 # 카운터 초기화

            # 💡 핵심 3: 사용자가 제안한 '8턴 도달 시 3턴 압축' 로직
            # 총 턴 수 = len(messages) // 2
            total_turns = len(st.session_state.messages) // 2
            uncompressed_turns = total_turns - st.session_state.compressed_turns

            if uncompressed_turns >= 8:
                # 가장 오래된 3턴(6개 메시지) 추출
                start_idx = st.session_state.compressed_turns * 2
                end_idx = start_idx + 6
                chats_to_compress = st.session_state.messages[start_idx:end_idx]
                
                # 백그라운드 스레드로 압축 실행 (UI 멈춤 없음)
                trigger_background_compression(chats_to_compress)
                
                # 압축 카운터 증가
                st.session_state.compressed_turns += 3

            # 신규 답변: 후속질문 분리 렌더링 + toast 피드백
            main_text, q_list = parse_follow_up(final_answer)
            st.markdown(main_text)
            if q_list:
                st.markdown("<p style='font-size:0.8rem; font-weight:800; color:#c2185b; margin: 14px 0 6px 0;'>🔮 이 질문은 어떠세요?</p>", unsafe_allow_html=True)
                cols = st.columns(len(q_list))
                new_msg_idx = len(st.session_state.messages)  # 저장 전 미리 인덱스 계산
                for idx, q in enumerate(q_list):
                    with cols[idx]:
                        if st.button(q, key=f"fu_{new_msg_idx}_{idx}"):
                            st.session_state.trigger_query = q
                            st.toast("분석을 시작합니다...", icon="🔍")
                            st.rerun()

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
