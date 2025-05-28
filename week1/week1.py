import torch
import torch.nn as nn
import math

# 1. 입력 임베딩
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# 2. 위치 인코딩 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# 3. 다중 헤드 어텐션
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0  #d_moel차원을 n_heads로 나눌 수 있어야 함
        self.d_k = d_model // n_heads  #각 헤드의 차원
        self.n_heads = n_heads         #헤드 수

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        #다중헤드어텐션을 위한 분할 함수
        def transform(x):
            # [batch, seq_len, d_model] -> [batch, n_heads, seq_len, d_k]
            return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        #print(self.q_linear(q));
        #print("\n");
        #print((self.q_linear(q)).shape);
        q = transform(self.q_linear(q))
       
        print(f"쿼리 : {q}");
        print(q.shape);
        k = transform(self.k_linear(k))
        v = transform(self.v_linear(v))
        #print(f"키 : {k.transpose(-2, -1)}");
        #print(k.transpose(-2, -1).shape);

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        #print(f"스코더 : {scores}");
        #print(scores.shape);

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        #print(f"어텐션 : {attn}");
        #print(attn.shape);
        context = torch.matmul(attn, v)
        #print(f"컨텍스트 : {context}");
        #print(context.shape);

        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_proj(context)

# 4. 피드포워드 네트워크
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# 5. 인코더 레이어
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))

        # Feedforward + residual + norm
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        
        return x

# 6. 전체 인코더 블록
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=8):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        #print(f"임베딩 결과 : {x}")
        x = self.positional_encoding(x)
        #print(f"위치인코딩 결과 : {x}")
        for layer in self.layers:
            x = layer(x, mask)
            #print(f"레이어 통과 결과 : {x}")
            #print(x.shape)
        return x
    

# 1. 마스크드 멀티헤드 어텐션 (디코더 자체 입력에 대해)
class MaskedMultiHeadAttention(MultiHeadAttention):
    def forward(self, q, k, v, mask=None):
        seq_len = q.size(1)  # [batch, seq_len, d_model] → 두 번째 차원
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(q.device)  # [seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]

        if mask is not None:
            # 사용자가 준 mask와 causal mask 둘 다 있을 경우
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            mask = mask & causal_mask
        else:
            mask = causal_mask  # [1, 1, seq_len, seq_len]

        return super().forward(q, k, v, mask)


# 2. 디코더 레이어
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))

        # 2. Cross-Attention (Encoder output과)
        cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))

        # 3. Feed Forward
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

# 3. 전체 디코더
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers, max_len=512):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    # 예제 입력
vocab_size = 10000
d_model = 6
n_heads = 3
d_ff = 2048
num_layers = 6
max_len = 100

encoder = Encoder(vocab_size, d_model, n_heads, d_ff, num_layers, max_len)

# input: [batch_size, seq_len]
input_tensor = torch.randint(0, vocab_size, (2, 30))  # batch 2, 길이 30
output = encoder(input_tensor)

#print(output)  # [2, 30, 512]

decoder = Decoder(vocab_size=10000, d_model=6, n_heads=3, d_ff=2048, num_layers=6)
tgt = torch.randint(0, 10000, (2, 30))  # 예: [batch=2, seq_len=30]

output = decoder(tgt, output)      # enc_output: 인코더 결과


#print(output)  # [2, 30, 512]
#print(tgt);


