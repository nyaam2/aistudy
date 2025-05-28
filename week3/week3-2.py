from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 1. 입력 설정
tokenizer.src_lang = "eng_Latn"
text = "what is your name?"
inputs = tokenizer(text, return_tensors="pt")

# 2. 대상 언어 토큰 ID 가져오기
# NLLB 모델은 <lang_code> 형태의 special token 사용
tgt_lang = "kor_Hang"
tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

# 3. 번역 실행
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tgt_lang_id,  # 이게 핵심!
        max_length=40
    )

# 4. 결과 디코딩
translated = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
print(translated)
