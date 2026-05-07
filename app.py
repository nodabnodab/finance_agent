import streamlit as st
import requests
from streamlit_lottie import st_lottie
from agent import agent_executor, system_prompt

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
# 4. 상태 관리
# ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = None

# ── 쿼리 파라미터로 버블 클릭 감지 ──
if "stock" in st.query_params:
    clicked = st.query_params["stock"]
    st.session_state.trigger_query = f"{clicked} 최신 주가와 관련 뉴스를 분석해줘."
    st.query_params.clear()

# ────────────────────────────────────────────
# 5. 데이터 (daily DB 흉내)
# ────────────────────────────────────────────
daily_main_q = "인텔의 잠재 성장력은 얼마일까요?"

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

right_bubbles = ["어제 나스닥 하락 원인?", "금리 인하 수혜주는?", "지금 가장 핫한 테마"]

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
    for q in right_bubbles:
        if st.button(q, key=q):
            st.session_state.trigger_query = q

# ── 중앙: 타이틀 + 챗봇 ──
with col_center:
    st.markdown("<div class='title-text'>AI 금융 에이전트</div>", unsafe_allow_html=True)

    # 대화 없을 때만 오늘의 화두 카드 표시
    if not st.session_state.messages:
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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 입력창
    chat_input = st.chat_input("종목명이나 뉴스를 물어보세요...")
    prompt = chat_input or st.session_state.trigger_query

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
                response = agent_executor.invoke({
                    "messages": [("system", system_prompt), ("user", prompt)]
                })
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
            st.markdown(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
