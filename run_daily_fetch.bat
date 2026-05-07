@echo off
REM ──────────────────────────────────────────────────────────────
REM  AI 금융 에이전트 - 데일리 데이터 수집 자동 실행 스크립트
REM  Windows 작업 스케줄러에 등록하여 매일 오전 5:00 KST 실행
REM ──────────────────────────────────────────────────────────────

REM conda 가상환경 활성화 (finance_agent 환경)
call conda activate finance_agent

REM 스크립트가 있는 폴더로 이동
cd /d C:\Users\arcre\Desktop\finance_agent

REM 데이터 수집 실행
python daily_data_fetcher.py >> data\fetch_log.txt 2>&1

echo [%date% %time%] 데일리 수집 완료 >> data\fetch_log.txt
