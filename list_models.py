import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("사용 가능한 Gemini 모델 목록:")
print("=" * 60)
for model in genai.list_models():
    if "generateContent" in [m for m in model.supported_generation_methods]:
        print(f"  - {model.name}  ({model.display_name})")
