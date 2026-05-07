import os
import json
from dotenv import load_dotenv

def update_models_config():
    load_dotenv()
    
    config_path = "api_models_config.json"
    
    # 기존 JSON 파일 읽기
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ api_models_config.json 파일이 없습니다.")
        return

    print("🔍 API 서버와 통신하여 사용 가능한 모델 목록을 동기화합니다...\n")

    # 1. Google Gemini 실제 모델 확인
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            # 텍스트 생성이 가능한 모델만 필터링
            models = [
                m.name.replace("models/", "") 
                for m in genai.list_models() 
                if "generateContent" in m.supported_generation_methods
            ]
            config["providers"]["google"]["available_models"] = models
            print(f"✅ [Google] API 확인 완료! {len(models)}개의 모델 업데이트됨.")
        except ImportError:
            print("⚠️ [Google] google-generativeai 패키지가 설치되지 않아 확인을 건너뜁니다.")
        except Exception as e:
            print(f"❌ [Google] API 통신 실패: {e}")
    else:
        print("ℹ️ [Google] GEMINI_API_KEY가 없어 확인을 건너뜁니다.")

    # 2. OpenAI 실제 모델 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            # OpenAI 서버에 직접 질의하여 gpt 관련 챗 모델만 필터링
            models_data = client.models.list()
            chat_models = sorted([m.id for m in models_data if "gpt" in m.id or "o1" in m.id])
            config["providers"]["openai"]["available_models"] = chat_models
            print(f"✅ [OpenAI] API 확인 완료! {len(chat_models)}개의 모델 업데이트됨.")
        except ImportError:
            print("⚠️ [OpenAI] openai 패키지가 설치되지 않아 확인을 건너뜁니다.")
        except Exception as e:
            print(f"❌ [OpenAI] API 통신 실패: {e}")
    else:
        print("ℹ️ [OpenAI] OPENAI_API_KEY가 없어 확인을 건너뜁니다.")

    # JSON 덮어쓰기
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        
    print("\n🎉 api_models_config.json 파일이 최신 API 상태로 동기화되었습니다!")

if __name__ == "__main__":
    update_models_config()
