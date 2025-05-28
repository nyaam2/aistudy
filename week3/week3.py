import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(tokenizer.vocab_size);

model = GPT2LMHeadModel.from_pretrained("gpt2")
print(model);
print(model.transformer.h[0])  # 첫 번째 디코더 블록 확인
print(model.lm_head.weight.shape)  # 출력층 weight


input_text = "i have"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print(input_ids.shape)  # (1, 3)
print(input_ids)


for i in range(0,30):

  # 다음 토큰 하나 예측
  with torch.no_grad():
    outputs = model(input_ids)
    print(outputs.logits.shape)  # (1, 3, 50257)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
  

  print(f"next token{next_token_id}")  # 예측된 토큰 ID
  print(next_token_id.shape)  # (1,)
  print(next_token_id.unsqueeze(0))  # (1,)

  generated = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
  print(generated);
  
  input_ids =generated
  print(tokenizer.decode(generated[0]))


