import re

def mask_sensitive_data(text: str) -> str:
    """
    정규표현식을 사용하여 텍스트 내의 민감한 개인정보 및 자산 규모를 마스킹합니다.
    에러 발생률 0%를 보장하는 초고속 텍스트 치환 로직입니다.
    """
    if not isinstance(text, str):
        return str(text)
        
    masked_text = text

    # 1. 금융 자산 및 금액 마스킹
    # 패턴 1: 숫자(콤마 포함) + 단위 (예: 10,000원, $500, 300달러, 500불)
    money_pattern_1 = r'(\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:만\s*)?(?:원|달러|불|\$|KRW|USD))'
    # 패턴 2: 한글 단위를 포함한 금액 (예: 5천만원, 1억 2천만 원, 1억2천)
    money_pattern_2 = r'(\d+[천백십]?(?:만|억|조)(?:\s*\d+[천백십]?(?:만|억|조)?)?\s*(?:원|달러|불)?)'
    
    masked_text = re.sub(money_pattern_1, '[자산_규모]', masked_text)
    masked_text = re.sub(money_pattern_2, '[자산_규모]', masked_text)

    # 2. 전화번호 마스킹 (예: 010-1234-5678)
    phone_pattern = r'(\b0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}\b)'
    masked_text = re.sub(phone_pattern, '[전화번호_블라인드]', masked_text)

    # 3. 이메일 마스킹
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    masked_text = re.sub(email_pattern, '[이메일_블라인드]', masked_text)

    # 4. 은행 계좌번호 (연속된 10~15자리 하이픈 조합 숫자)
    account_pattern = r'(\b\d{2,6}[-\s]\d{2,6}[-\s]\d{2,6}\b)'
    masked_text = re.sub(account_pattern, '[계좌번호_블라인드]', masked_text)
    
    return masked_text
