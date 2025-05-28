import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from google.colab import files
import pandas as pd
import glob
import os
# import openpyxl # openpyxl is used by pandas, no need to import directly

uploaded = files.upload()  # 파일 업로드 창 뜸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

# 1. Positional Encoding (sinusoidal)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# 2. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
     B, L_q, _ = q.size()
     B, L_k, _ = k.size()
     B, L_v, _ = v.size()

     q = self.W_q(q).reshape(B, L_q, self.n_heads, self.d_k).transpose(1, 2)
     k = self.W_k(k).reshape(B, L_k, self.n_heads, self.d_k).transpose(1, 2)
     v = self.W_v(v).reshape(B, L_v, self.n_heads, self.d_k).transpose(1, 2)

     out, _ = self.attention(q, k, v, mask)

     out = out.transpose(1, 2).contiguous().reshape(B, L_q, self.n_heads * self.d_k)
     return self.fc(out)



# 4. Feed Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

# 5. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x2 = self.mha(x, x, x, mask)
        x = self.norm1(x + x2)
        x2 = self.ffn(x)
        x = self.norm2(x + x2)
        return x

# 6. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x2 = self.self_mha(x, x, x, tgt_mask)
        x = self.norm1(x + x2)
        x2 = self.enc_dec_mha(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + x2)
        x2 = self.ffn(x)
        x = self.norm3(x + x2)
        return x

# 7. Full Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, d_ff=2048, num_layers=6, max_len=100):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_emb(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.pos_enc(self.tgt_emb(tgt))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)




# ==== MODULES (Transformer core: same as before) ====
# [모듈 구현은 이전 내용 유지 – 생략하지 않음. 그대로 두되 이어 붙임.]
# (생략 없이 전체 Transformer 모듈 구현 코드 유지)

# ==== TOY DATASET ====
class ToyDataset(Dataset):
    def __init__(self):
        self.pairs = [
            ("i love you", "나는 너를 사랑해"),
            ("hello", "안녕"),
            ("thank you", "고마워"),
            ("how are you", "어떻게 지내"),
            ("good night", "잘 자")
        ]
        self.src_vocab = {word: idx+2 for idx, word in enumerate(set(w for s, _ in self.pairs for w in s.split()))}
        self.tgt_vocab = {word: idx+2 for idx, word in enumerate(set(w for _, t in self.pairs for w in t.split()))}
        self.src_vocab["<pad>"] = 0
        self.tgt_vocab["<pad>"] = 0
        self.src_vocab["<sos>"] = 1
        self.tgt_vocab["<sos>"] = 1

        self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}

    def encode(self, sentence, vocab):
        return [vocab.get(word, 0) for word in sentence.split()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = torch.tensor(self.encode(src, self.src_vocab), dtype=torch.long)
        tgt_ids = torch.tensor([1] + self.encode(tgt, self.tgt_vocab), dtype=torch.long)  # <sos> + target
        return src_ids, tgt_ids

# ==== TRAIN LOOP ====
def create_mask(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return src_mask, tgt_mask

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def train():
    # Use the RealDataset instance created in __main__
    # dataset = RealDataset(csv_path="merged_kor_eng.csv") # This line is removed
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print(2)
    model = Transformer(
    src_vocab_size=len(dataset.src_vocab),
    tgt_vocab_size=len(dataset.tgt_vocab),
    d_model=128, n_heads=4, d_ff=512, num_layers=2
    ).to(device)
    print(3)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(4)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for src, tgt in loader:
            print(5)
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = create_mask(src, tgt_input)
            # Masks are already on the correct device from create_mask if src/tgt are on device
            # src_mask = src_mask.to(device)
            # tgt_mask = tgt_mask.to(device)
            preds = model(src, tgt_input, src_mask, tgt_mask)
            print(6)

            preds = preds.view(-1, preds.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(preds, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def translate(model, src_sentence, dataset, max_len=20):
    model.eval()

    # 문장 토큰화 및 숫자 변환
    src_ids = [dataset.src_vocab.get(word, 0) for word in src_sentence.split()]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, S), to device
    tgt_tensor = torch.tensor([[1]], dtype=torch.long).to(device)  # <sos> 시작, to device

    for _ in range(max_len):
        src_mask, tgt_mask = create_mask(src_tensor, tgt_tensor)
        # Masks are already on the correct device
        # src_mask = src_mask.to(device)
        # tgt_mask = tgt_mask.to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
        tgt_tensor = torch.cat([tgt_tensor, next_token], dim=1)

        if next_token.item() == 0:  # <pad> 토큰이면 중단
            break

    # 디코딩 (숫자 → 단어)
    # Move tgt_tensor back to cpu for list conversion
    out_tokens = tgt_tensor.squeeze().tolist()[1:]  # <sos> 제외
    translated = [dataset.inv_tgt_vocab.get(tok, "<unk>") for tok in out_tokens]
    return " ".join(translated)


def merge_aihub_xlsx_to_csv(uploaded_files_dict, save_path="merged_kor_eng.csv"):
    import pandas as pd
    import io # Import io module to read bytes as file-like objects

    dfs = []

    # Iterate through the uploaded files dictionary
    for filename, file_content in uploaded_files_dict.items():
        # Only process files ending with .xlsx (case-insensitive)
        if filename.lower().endswith(".xlsx"):
            try:
                # Use io.BytesIO to treat the byte content as a file
                df = pd.read_excel(io.BytesIO(file_content), engine="openpyxl")
                if "원문" in df.columns and "번역문" in df.columns:
                    df = df[["원문", "번역문"]].dropna()
                    dfs.append(df)
                    print(f"✅ 불러옴: {filename} ({len(df)} rows)")
                else:
                    print(f"⚠️ 열 이름 불일치: {filename} → 스킵됨")
            except Exception as e:
                print(f"❌ 읽기 실패: {filename} → {e}")

    if not dfs:
        print("❗ 병합할 수 있는 데이터가 없음.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(save_path, index=False)
    print(f"\n📝 저장 완료: {save_path} ({len(full_df)} 문장)")

class RealDataset(Dataset):
    def __init__(self, csv_path, num_samples=3000, max_len=50):
        import pandas as pd
        df = pd.read_csv(csv_path).dropna().head(num_samples)

        df["원문"] = df["원문"].astype(str)
        df["번역문"] = df["번역문"].astype(str)

        self.max_len = max_len
        self.pairs = list(zip(df["원문"], df["번역문"]))

        # 토큰 단위 단어 수집
        src_words = set(word for s, _ in self.pairs for word in s.split())
        tgt_words = set(word for _, t in self.pairs for word in t.split())

        self.src_vocab = {word: idx + 4 for idx, word in enumerate(src_words)}
        self.tgt_vocab = {word: idx + 4 for idx, word in enumerate(tgt_words)}

        # 특별 토큰
        self.src_vocab["<pad>"] = 0
        self.tgt_vocab["<pad>"] = 0
        self.src_vocab["<sos>"] = 1
        self.tgt_vocab["<sos>"] = 1
        self.src_vocab["<eos>"] = 2
        self.tgt_vocab["<eos>"] = 2
        self.src_vocab["<unk>"] = 3
        self.tgt_vocab["<unk>"] = 3

        self.inv_tgt_vocab = {v: k for k, v in self.tgt_vocab.items()}

    def encode(self, sentence, vocab):
        tokens = sentence.split()
        tokens = tokens[:self.max_len]  # 최대 길이 제한
        return [vocab.get(word, vocab["<unk>"]) for word in tokens]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        src_ids = self.encode(src, self.src_vocab)
        tgt_ids = self.encode(tgt, self.tgt_vocab)

        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        tgt_tensor = torch.tensor([1] + tgt_ids + [2], dtype=torch.long)  # <sos> ~ <eos>

        return src_tensor, tgt_tensor

if __name__ == '__main__':
    # Note: The original code first created a ToyDataset instance,
    # then called merge_aihub_xlsx_to_csv, then created a RealDataset
    # inside the train function.
    # This order seems mixed.
    # Let's ensure the real data is processed and used for training.

    # Process the uploaded real data
    merge_aihub_xlsx_to_csv(uploaded)

    # Create the dataset using the merged CSV
    try:
        dataset = RealDataset(csv_path="merged_kor_eng.csv")
    except FileNotFoundError:
        print("Error: merged_kor_eng.csv not found. Please upload the Excel files and ensure they are processed correctly.")
        exit() # Exit if the dataset cannot be created

    print(f"RealDataset created with {len(dataset)} samples.")
    print(f"Source vocabulary size: {len(dataset.src_vocab)}")
    print(f"Target vocabulary size: {len(dataset.tgt_vocab)}")


    model = Transformer(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        d_model=128, n_heads=4, d_ff=512, num_layers=2
    ).to(device) # Move model to device


    # 저장된 모델이 있다면 load
    # try:
    #     model.load_state_dict(torch.load("transformer.pt"))
    #     print("Loaded saved model state_dict.")
    # except FileNotFoundError:
    #     print("No saved model found. Training from scratch.")

    print(1)

    # 학습 먼저 수행
    if torch.cuda.is_available():
        print(f"Training on: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU.")

    train() # train function will now use the 'dataset' defined here

    # 번역 테스트
    # The translate function uses the same 'dataset' instance for vocab
    test_sentences = [
         "안녕",
        "기분이 어때요?"
        # Add some example sentences from your real dataset if known
        # e.g., "오늘 날씨 어때요?" (if you add Korean input translation)
    ]

    # You might need to adapt the translate function slightly or
    # have a separate inference mode that handles tokenization for new inputs
    # that might not be in the exact training vocabulary.
    # For testing with words *potentially* in the real vocab:
    print("\n--- Translation Test ---")
    for sent in test_sentences:
        # Need to translate from SRC (Korean) to TGT (English) based on the dataset example?
        # The dataset is src=Korean, tgt=English based on "원문", "번역문".
        # The toy examples are "i love you" (English) -> "나는 너를 사랑해" (Korean).
        # Let's assume the RealDataset is English -> Korean based on the original variable names/comments.
        # If "원문" is Korean and "번역문" is English, the model is Kor -> Eng.
        # Let's stick to the toy example structure for the test sentences for now.
        # If your real dataset is Korean -> English, you should test with Korean sentences.
        result = translate(model, sent, dataset)
        print(f"{sent} -> {result}")