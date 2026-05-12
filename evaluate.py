import json
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
# agent.py에서 에이전트와 도구, 프롬프트를 가져옵니다.
from agent import agent_executor, system_prompt, llm 

# ────────────────────────────────────────────
# 1. 대조군(Baseline) 설정: 도구가 없는 순수 70B 모델
# ────────────────────────────────────────────
# 기존 LLM 객체를 그대로 쓰되, 도구 없이 직접 호출할 용도입니다.
baseline_llm = llm 

# ────────────────────────────────────────────
# 2. 심사위원 LLM 및 Side-by-Side 채점 함수
# ────────────────────────────────────────────
judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)



def evaluate_sbs(question, tool_context, answer_hybrid, answer_baseline):
    prompt = f"""
    당신은 아주 냉철한 금융 전문 심사위원입니다. 
    동일한 [질문]에 대해 두 명의 AI 에이전트가 내놓은 답변을 비교 평가하세요.

    [질문]: {question}
    [참고 데이터(Hybrid 전용)]: {tool_context if tool_context else '데이터 없음'}
    
    [답변 A (Hybrid)]: {answer_hybrid}
    [답변 B (Baseline)]: {answer_baseline}

    [평가 기준]
    1. faithfulness: 참고 데이터에 충실한가? (Baseline은 데이터가 없으므로 상식에 의존함)
    2. relevance: 질문의 의도를 정확히 관통하는가?
    3. precision: 불필요한 사족 없이 핵심 금융 정보를 전달하는가?
    4. fact_rigidity: 사용자의 잘못된 전제나 함정 질문을 정확히 지적하고 교정하는가?

    각 지표에 대해 A와 B에게 각각 0.0~1.0 사이의 점수를 부여하세요.
    JSON 형식으로만 출력하십시오.
    {{
        "A": {{"faithfulness": 0.0, "relevance": 0.0, "precision": 0.0, "fact_rigidity": 0.0}},
        "B": {{"faithfulness": 0.0, "relevance": 0.0, "precision": 0.0, "fact_rigidity": 0.0}},
        "winner": "A or B or Tie",
        "reasoning": "승리 이유와 결정적 차이점 요약"
    }}
    """
    try:
        response = judge_llm.invoke(prompt)
        match = re.search(r'\{[\s\S]*\}', response.content)
        if match:
            return json.loads(match.group(0))
        return None
    except Exception as e:
        print(f"채점 오류: {e}")
        return None

# ────────────────────────────────────────────
# 3. 테스트 실행 루프 (실시간 비교 및 오토세이브)
# ────────────────────────────────────────────

# ────────────────────────────────────────────
# 4. 테스트 데이터셋 (Golden Dataset)
# ────────────────────────────────────────────
test_dataset = [
    # 1단계: 단일 도구 검증 (현재가 및 기본 재무)
    {"category": "기초검색", "question": "마이크로소프트(MSFT)의 현재 주가와 시가총액을 알려줘."},
    {"category": "기초검색", "question": "아마존(AMZN)의 연간 매출액과 매출 성장률(YoY)은 얼마야?"},
    {"category": "기초검색", "question": "메타(META)의 주당순이익(EPS)과 영업이익률을 확인해줘."},
    {"category": "기초검색", "question": "마이크론(Mu)은 현재 코스피 시장에서 시가총액 순위가 몇 위야?"},

    # 2단계: 다중 도구 및 심층 분석 (비교 및 가치평가)
    {"category": "비교분석", "question": "테슬라(TSLA)와 포드(F)의 PER, PBR을 비교하고 밸류에이션 차이를 설명해."},
    {"category": "가치평가", "question": "인텔(INTC)의 현재 수익성(ROE, 순이익률)을 분석하고, 투자 매력도를 평가해봐."},
    {"category": "가치평가", "question": "엔비디아(NVDA)의 Forward PER을 확인하고, 현재 주가가 고평가인지 판단해줘."},
    
    # 3단계: 뉴스 결합 및 거시 경제 추론
    {"category": "복합추론", "question": "애플(AAPL)의 최근 주가 흐름과, AI(애플 인텔리전스) 관련 최신 뉴스를 엮어서 시사점을 도출해."},
    {"category": "복합추론", "question": "구글(GOOGL)이 검색 시장 독점 판결 이후 겪고 있는 위기를 최신 뉴스로 요약해봐."},
    {"category": "복합추론", "question": "일라이 릴리(LLY)의 비만 치료제 관련 최신 뉴스와 현재 밸류에이션을 함께 브리핑해줘."},
    
    # 4단계: 섹터 및 테마 분석 (에이전트의 논리력 테스트)
    {"category": "거시경제", "question": "최근 금리 변동이 미국 기술주(나스닥)에 미치는 영향을 최신 뉴스를 통해 분석해."},
    {"category": "테마분석", "question": "AI 반도체 수요 폭증이 TSMC의 실적이나 주가에 어떻게 반영되고 있는지 관련 뉴스를 찾아봐."},
    {"category": "테마분석", "question": "최근 이스라엘-이란 중동 지정학적 리스크가 국제 유가나 에너지 관련주에 미친 영향을 요약해."},
    
    # 5단계: 까다로운 엣지 케이스 (존재하지 않거나 어려운 조건)
    {"category": "엣지케이스", "question": "없는 주식인 FAKE_STOCK_123 의 주가와 재무를 분석해봐."},
    {"category": "엣지케이스", "question": "테슬라(TSLA)의 과거 10년 전 오늘 주가를 알려줘. (현재 도구로 한계가 있는지 테스트)"},
    {"category": "엣지케이스", "question": "현재 주식 시장에서 가장 저평가된 주식 딱 1개만 무조건 추천해봐."},
    {"category": "엣지케이스", "question": "일론 머스크가 어제 삼성전자를 인수했다는 뉴스가 있던데, 향후 주가 전망 분석해줘."},
    {"category": "엣지케이스", "question": "소프트뱅크가 쿠팡(CPNG)을 상장 폐지하고 일본 증시로 이전 상장한다는 공식 발표 내용을 요약해줘."},
    {"category": "엣지케이스", "question": "워런 버핏의 버크셔 해서웨이가 어제 매집한 비트코인을 대량 매도해서 투자자들에게 질타를 받는다던데, 주가에 악영향이 있을까?"}
]



print("🚀 [Side-by-Side] 하이브리드 vs 순수 70B 모델 비교 테스트 시작...\n")

hybrid_total = {"faithfulness": 0.0, "relevance": 0.0, "precision": 0.0, "fact_rigidity": 0.0}
baseline_total = {"faithfulness": 0.0, "relevance": 0.0, "precision": 0.0, "fact_rigidity": 0.0}
valid_count = 0
backup_data = []

try:
    for idx, item in enumerate(test_dataset):
        print(f"[{idx+1}/{len(test_dataset)}] 질문: {item['question']}")
        
        try:
            # 1. Hybrid 에이전트 실행 (도구 사용)
            res_h = agent_executor.invoke({"messages": [("system", system_prompt), ("user", item["question"])]})
            
            tool_context = ""
            ans_h = ""
            for msg in res_h["messages"]:
                if msg.type == "tool": tool_context += f"[{msg.name}]: {msg.content[:300]} "
                elif msg.type == "ai" and not getattr(msg, "tool_calls", None):
                    ans_h = msg.content

            # 2. Baseline 실행 (순수 LLM - 도구 없이 시스템 프롬프트만 전달)
            res_b = baseline_llm.invoke([("system", system_prompt), ("user", item["question"])])
            ans_b = res_b.content

            # 3. 심사위원 채점
            scores = evaluate_sbs(item["question"], tool_context, ans_h, ans_b)
            
            if scores:
                print(f"  🏆 Winner: {scores['winner']} | 사유: {scores['reasoning'][:50]}...")
                for k in hybrid_total.keys():
                    hybrid_total[k] += float(scores["A"][k])
                    baseline_total[k] += float(scores["B"][k])
                valid_count += 1
                
                # 실시간 백업
                backup_data.append({"question": item["question"], "scores": scores})
                with open("evaluation_backup.json", "w", encoding="utf-8") as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"  ❌ 실행 실패: {e}")
            continue
        
        time.sleep(2) # Rate Limit 방지

except KeyboardInterrupt:
    print("\n🛑 중단됨. 현재까지의 데이터를 집계합니다.")

# ────────────────────────────────────────────
# 5. 시각화 (진짜 결과로 막대 그래프 그리기)
# ────────────────────────────────────────────
if valid_count > 0:
    metrics = ["Faithfulness", "Relevance", "Precision", "Fact Rigidity"]
    h_avg = [hybrid_total[k.lower().replace(" ", "_")] / valid_count for k in metrics]
    b_avg = [baseline_total[k.lower().replace(" ", "_")] / valid_count for k in metrics]

    df = pd.DataFrame({
        "Metrics": metrics,
        "Hybrid Agent (Ours)": h_avg,
        "Pure 70B (Baseline)": b_avg
    })

    # 막대 그래프 생성 (기존 스타일 유지)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, df["Hybrid Agent (Ours)"], width, label='Hybrid Agent (Ours)', color='#ff4d8f')
    ax.bar(x + width/2, df["Pure 70B (Baseline)"], width, label='Pure 70B (Baseline)', color='#444444')
    
    ax.set_ylabel('Scores')
    ax.set_title('Hybrid vs Pure 70B Side-by-Side Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    print("\n🎉 평가 완료! 'evaluation_report.png'에서 실제 성능 격차를 확인하세요.")