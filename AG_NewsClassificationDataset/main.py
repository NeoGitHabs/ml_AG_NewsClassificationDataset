import torch
import uvicorn
import torchtext
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchtext.data.utils import get_tokenizer
from contextlib import asynccontextmanager
from pathlib import Path

torchtext.disable_torchtext_deprecation_warning()


# ── Константы ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
VOCAB_PATH  = BASE_DIR / "vocab_CheckNews_AG_NewsClassificationDataset.pth"
MODEL_PATH  = BASE_DIR / "model_CheckNews_AG_NewsClassificationDataset.pth"
CLASSES     = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Архитектура ────────────────────────────────────────────────────────────────
class CheckNews(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, output_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.lin       = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.lin(hidden[-1])


# ── Загрузка модели и vocab ────────────────────────────────────────────────────
def load_artifacts():
    vocab = torch.load(VOCAB_PATH, map_location=DEVICE, weights_only=False)
    vocab.set_default_index(vocab["<unk>"])

    model = CheckNews(len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    return vocab, model


# ── Предобработка текста ───────────────────────────────────────────────────────
tokenizer = get_tokenizer("basic_english")

def preprocess(text: str, vocab, unk_idx: int) -> torch.Tensor:
    ids = [vocab[t] for t in tokenizer(text)] or [unk_idx]
    return torch.tensor([ids], dtype=torch.int64, device=DEVICE)


# ── Lifespan (загружаем один раз при старте) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vocab, app.state.model = load_artifacts()
    app.state.unk_idx = app.state.vocab["<unk>"]
    yield


# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="AG News Classifier", lifespan=lifespan)


class TextSchema(BaseModel):
    text: str


@app.post("/predict")
def predict(item: TextSchema):
    if not item.text.strip():
        raise HTTPException(status_code=422, detail="Текст не может быть пустым")

    x = preprocess(item.text, app.state.vocab, app.state.unk_idx)

    with torch.no_grad():
        label = torch.argmax(app.state.model(x), dim=1).item()

    return {"label": CLASSES[label]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)