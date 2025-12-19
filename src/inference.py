# inference.py
# 추론 로직 (Google Play 리뷰, label 없음)

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from model import SentimentBERT
from inferencedataset import InferenceDataset

def inference_epoch(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())

    return preds


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "sentiment140_bert.pt"
    DATA_PATH = "data/google_play_store_reviews.csv"
    MAX_LEN = 64
    BATCH_SIZE = 32

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    df = pd.read_csv(DATA_PATH)
    df["text"] = df.iloc[:, 4].astype(str)  # Content 컬럼
    texts = df["text"].dropna().tolist()

    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SentimentBERT()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    preds = inference_epoch(model, loader, DEVICE)

    pos_ratio = (pd.Series(preds) == 1).mean()
    print(f"Positive ratio: {pos_ratio:.2f}")

