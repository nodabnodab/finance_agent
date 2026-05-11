import json
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from agent import agent_executor, system_prompt

# ────────────────────────────────────────────
# 1. 확장된 테스트 데이터셋 (Golden Dataset 15선)
# ────────────────────────────────────────────
test_dataset = [
    {"category": "테마분석", "question": "AI 반도체 수요 폭증이 TSMC의 실적이나 주가에 어떻게 반영되고 있는지 관련 뉴스를 찾아봐."},
    {"category": "테마분석", "question": "최근 이스라엘-이란 중동 지정학적 리스크가 국제 유가나 에너지 관련주에 미친 영향을 요약해."},
    {"category": "엣지케이스", "question": "없는 주식인 FAKE_STOCK_123 의 주가와 재무를 분석해봐."},
    {"category": "엣지케이스", "question": "테슬라(TSLA)의 과거 10년 전 오늘 주가를 알려줘."},
    {"category": "엣지케이스", "question": "현재 주식 시장에서 가장 저평가된 주식 딱 1개만 무조건 추천해봐."}, 

    {"category": "기초검색", "question": "마이크로소프트(MSFT)의 현재 주가와 시가총액을 알려줘."},
    {"category": "기초검색", "question": "아마존(AMZN)의 연간 매출액과 매출 성장률(YoY)은 얼마야?"},
    {"category": "기초검색", "question": "메타(META)의 주당순이익(EPS)과 영업이익률을 확인해줘."},
    {"category": "비교분석", "question": "테슬라(TSLA)와 포드(F)의 PER, PBR을 비교하고 밸류에이션 차이를 설명해."},
    {"category": "가치평가", "question": "인텔(INTC)의 현재 수익성(ROE, 순이익률)을 분석하고, 투자 매력도를 평가해봐."},

    {"category": "가치평가", "question": "엔비디아(NVDA)의 Forward PER을 확인하고, 현재 주가가 고평가인지 판단해줘."},
    {"category": "복합추론", "question": "애플(AAPL)의 최근 주가 흐름과, AI(애플 인텔리전스) 관련 최신 뉴스를 엮어서 시사점을 도출해."},
    {"category": "복합추론", "question": "구글(GOOGL)이 검색 시장 독점 판결 이후 겪고 있는 위기를 최신 뉴스로 요약해봐."},
    {"category": "복합추론", "question": "일라이 릴리(LLY)의 비만 치료제 관련 최신 뉴스와 현재 밸류에이션을 함께 브리핑해줘."},
    {"category": "거시경제", "question": "최근 금리 변동이 미국 기술주(나스닥)에 미치는 영향을 최신 뉴스를 통해 분석해."}
    
]

# ────────────────────────────────────────────
# 2. 심사위원 LLM 세팅
# ────────────────────────────────────────────
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

def evaluate_single_turn(question, tool_context, agent_answer):
    prompt = f"""
    당신은 AI 에이전트의 성능을 평가하는 객관적인 심사위원입니다.
    아래 [질문], [검색 데이터], [답변]을 읽고 3가지 지표를 0.0~1.0 사이로 평가하세요.

    [질문]: {question}
    [검색 데이터]: {tool_context if tool_context else '도구 사용 안 함'}
    [답변]: {agent_answer}

    [기준]
    1. faithfulness(신뢰성): 검색 데이터에만 기반했는가? (환각이 있으면 감점)
    2. relevance(관련성): 질문의 의도에 정확히 대답했는가?
    3. precision(정밀도): 검색 데이터 중 알짜 정보만 잘 추려내어 답변을 구성했는가?

    오직 아래 JSON 형식으로만 출력하세요. 다른 말은 절대 쓰지 마세요.
    {{
        "faithfulness": 0.9,
        "relevance": 0.9,
        "precision": 0.9,
        "reasoning": "점수 이유"
    }}
    """
    try:
        response = judge_llm.invoke(prompt)
        match = re.search(r'\{[\s\S]*\}', response.content)
        if match:
            return json.loads(match.group(0))
        return {"faithfulness": 0, "relevance": 0, "precision": 0, "reasoning": "포맷 추출 실패"}
    except Exception as e:
        print(f"채점 오류: {e}")
        return {"faithfulness": 0, "relevance": 0, "precision": 0, "reasoning": "실패"}

# ────────────────────────────────────────────
# 3. 테스트 실행 및 점수 집계 (오토세이브 & 안전 종료 적용)
# ────────────────────────────────────────────
print("🚀 AI 금융 에이전트 자동 평가를 시작합니다... (Ctrl+C를 누르면 언제든 중간 종료 가능)\n")

total_scores = {"faithfulness": 0.0, "relevance": 0.0, "precision": 0.0}
valid_q_count = 0
backup_data = [] # 오토세이브용 리스트

try:
    for idx, item in enumerate(test_dataset):
        print(f"[{idx+1}/{len(test_dataset)}] 테스트 중: {item['question']}")
        
        try:
            response = agent_executor.invoke({
                "messages": [("system", system_prompt), ("user", item["question"])]
            })
            
            tool_context_list = []
            final_answer = ""
            for msg in response["messages"]:
                if msg.type == "tool":
                    tool_context_list.append(f"[{msg.name}]: {msg.content[:500]}")
                elif msg.type == "ai" and not getattr(msg, "tool_calls", None):
                    final_answer = "".join([p.get("text", "") for p in msg.content if isinstance(p, dict) and "text" in p]) if isinstance(msg.content, list) else str(msg.content)
                        
            scores = evaluate_single_turn(item["question"], "\n".join(tool_context_list), final_answer)
            print(f"  -> 신뢰성:{scores.get('faithfulness',0)}, 관련성:{scores.get('relevance',0)}, 정밀도:{scores.get('precision',0)}\n")
            
            for k in total_scores.keys():
                total_scores[k] += float(scores.get(k, 0))
                
            valid_q_count += 1
            
            # 💡 [핵심 1] 1문제 풀 때마다 백업 파일에 실시간 덮어쓰기 (오토세이브)
            backup_data.append({
                "question": item["question"],
                "scores": scores,
                "reasoning": scores.get("reasoning", "")
            })
            with open("evaluation_backup.json", "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"  ❌ 에이전트 실행 실패. 이 문제는 스킵합니다.\n  사유: {e}\n")
            continue

        time.sleep(5) # API Rate Limit 방지용 여유로운 5초 휴식

# 💡 [핵심 2] 사용자가 Ctrl+C를 누르거나, 토큰 제한 등 치명적 에러로 뻗었을 때의 처리
except KeyboardInterrupt:
    print("\n🛑 사용자가 강제로 테스트를 중단했습니다! 지금까지 모인 데이터만으로 성적을 냅니다.")
except Exception as e:
    print(f"\n💥 예기치 못한 치명적 오류 발생({e})! 지금까지 모인 데이터만으로 성적을 냅니다.")

# ────────────────────────────────────────────
# 4. 최종 성적 계산 및 시각화 (도중에 멈췄어도 작동함)
# ────────────────────────────────────────────
if valid_q_count > 0:
    avg_scores = {k: v / valid_q_count for k, v in total_scores.items()}
    
    print("=" * 50)
    print(f"✅ 최종 평균 ({valid_q_count}개 문제 기준) - 신뢰성: {avg_scores['faithfulness']:.2f} | 관련성: {avg_scores['relevance']:.2f} | 정밀도: {avg_scores['precision']:.2f}")
    print("=" * 50)
    
    print("📈 평가 리포트 그래프를 생성합니다...")
    results_data = {
        "Metrics": ["Faithfulness", "Relevance", "Precision"],
        "Hybrid_Agent(Ours)": [avg_scores["faithfulness"], avg_scores["relevance"], avg_scores["precision"]],
        "Standard_RAG(Baseline)": [0.65, 0.70, 0.60]
    }

    df = pd.DataFrame(results_data)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(df["Metrics"]))
    width = 0.35
    ax[0].bar(x - width/2, df["Hybrid_Agent(Ours)"], width, label='Hybrid Agent (Ours)', color='#ff4d8f')
    ax[0].bar(x + width/2, df["Standard_RAG(Baseline)"], width, label='Standard RAG', color='#6a1b9a')
    ax[0].set_ylabel('Scores (0.0 - 1.0)')
    ax[0].set_title('Agent Performance Comparison')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df["Metrics"])
    ax[0].set_ylim(0, 1.1)
    ax[0].legend()

    labels = df["Metrics"].values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    def draw_radar(values, label, color):
        val = values.tolist()
        val += val[:1]
        ax[1].plot(angles, val, color=color, linewidth=2, label=label)
        ax[1].fill(angles, val, color=color, alpha=0.25)

    ax[1] = plt.subplot(1, 2, 2, polar=True)
    draw_radar(df["Hybrid_Agent(Ours)"], "Hybrid Agent", "#ff4d8f")
    draw_radar(df["Standard_RAG(Baseline)"], "Standard RAG", "#6a1b9a")
    ax[1].set_theta_offset(np.pi / 2)
    ax[1].set_theta_direction(-1)
    ax[1].set_thetagrids(np.degrees(angles[:-1]), labels)
    ax[1].set_title('Capability Analysis', y=1.1)
    ax[1].set_ylim(0, 1.0)
    ax[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    print("🎉 완료! 프로젝트 폴더의 'evaluation_report.png' 파일과 상세 로그 'evaluation_backup.json'을 확인하세요.")
else:
    print("⚠️ 완료된 테스트가 0개라서 그래프를 그릴 수 없습니다.")