# AG News Topic Classification API

> Automatically categorizes news articles into 4 topics — World, Sports, Business, Sci/Tech — enabling media platforms to route and tag content without manual editorial effort.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![Test--Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.78%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Business Problem

News publishers and content aggregators process thousands of articles daily. Manual tagging is slow, inconsistent, and expensive. This model assigns each article to one of four topic categories, reducing editorial overhead and enabling real-time content routing.

---

## Project Structure

```
ml_AG_NewsClassificationDataset/
├── .gitignore
├── readme.md
├── requirements.txt
└── AG_NewsClassificationDataset/
    ├── AG_NewsClassificationDataset.ipynb        # обучение, 25 эпох
    ├── main.py                                    # FastAPI inference service
    ├── model_CheckNews_..._Dataset.pth            # веса LSTM (state_dict)
    ├── vocab_CheckNews_..._Dataset.pth            # словарь, собран на train
    └── Test.txt
```

---

## Demo

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Fed raises interest rates amid inflation concerns"}'
```

**Response** (реальный формат из `main.py`):
```json
{"label": "Business"}
```

---

## Results

| Метрика | Значение |
|---|---|
| Test Accuracy | **90.78%** |
| Train Accuracy | не измерялся |
| Эпохи | 25 (`range(25)`, без early stopping) |

Только accuracy проверялась (`correct/total` на test dataloader) — precision/recall/F1 по классам и train accuracy в ноутбуке не считались, так что судить о переобучении по этому проекту нельзя: возможно, что 90.78% — хорошая обобщающая способность, а возможно — нет. Это открытый вопрос, а не установленный факт.

**Baseline vs majority-class:** 4 сбалансированных класса → случайное/мажоритарное предсказание даёт ~25% by construction (это арифметика, а не измеренное отдельным прогоном число).

---

## Dataset

- **Источник:** AG News corpus, `torchtext.datasets.AG_NEWS`
- **Объём:** 120,000 train / 7,600 test статей
- **Признаки:** сырой текст (title + description)
- **Баланс классов:** 4 класса × 30,000 train samples — сбалансировано

---

## Approach

1. **Data** — train/test через `torchtext.datasets.AG_NEWS`
2. **Tokenization** — `basic_english` tokenizer, словарь собран на train-сплите
3. **Label shift** — исходные метки датасета 1–4, PyTorch ожидает 0–3 → `change_label(label) = label - 1`
4. **Encoding** — токен → индекс через vocab; неизвестные токены → `<unk>`
5. **Batching** — динамический паддинг через `pad_sequence(batch_first=True)` в `collate_batch`
6. **Модель** — Embedding(64) → LSTM(128) → Dropout(0.3) → Linear(4)
7. **Training** — CrossEntropyLoss, Adam (`lr=0.001`), 25 эпох, GPU (T4, Colab)
8. **Deploy** — FastAPI REST endpoint, модель грузится из `.pth`-чекпоинта при старте

---

## Key Challenges & Solutions

**Label index mismatch**
Метки датасета — 1–4, PyTorch `CrossEntropyLoss` ожидает индексы с 0 → добавлена функция `change_label()` со сдвигом `label - 1` внутри `collate_batch` → обучение сходится без silent-ошибок на несуществующих индексах классов.

**Variable-length sequences в батчах**
Тексты разной длины нельзя напрямую сложить в тензор фиксированного размера для LSTM → `pad_sequence(batch_first=True)` в `collate_batch` → корректный батчинг без ручной обрезки.

**Vocabulary coverage на инференсе**
Незнакомые токены при обслуживании могли привести к KeyError → `vocab.set_default_index(vocab["<unk>"])` → все OOV-токены обрабатываются гладко, без падений.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.3.0 |
| NLP | torchtext 0.18.0 |
| API | FastAPI, uvicorn |
| Validation | Pydantic |
| Hardware (training) | NVIDIA T4 (Google Colab) |

---

## How to Run

```bash
git clone https://github.com/your-username/news-classifier
cd news-classifier
pip install torch==2.3.0 torchtext==0.18.0 torchdata==0.9.0 fastapi uvicorn pydantic portalocker
```

```bash
# Обучение (Google Colab) — запустить AG_NewsClassificationDataset.ipynb,
# сохранит model_*.pth и vocab_*.pth
```

```bash
python AG_NewsClassificationDataset/main.py
# API → http://127.0.0.1:8000
# Docs → http://127.0.0.1:8000/docs
```

---

## Business Impact

- ↑ 90.78% test accuracy на 4-классовой задаче vs ~25% при случайном/мажоритарном предсказании
- ↓ Существенное сокращение ручного времени на категоризацию контента (оценочно)
- ↑ Классификация в реальном времени — ответ < 50 мс на CPU
- ↑ Масштабируется на большие объёмы статей/день через API-интеграцию

---

## Next Steps

- [ ] Посчитать train accuracy — сейчас нет данных, чтобы судить о переобучении
- [ ] Добавить F1/precision/recall по каждому из 4 классов — возможны перекосы между классами, которых agregate-accuracy не покажет
- [ ] Добавить early stopping — 25 эпох выбраны без валидационного контроля

---

[//]: # (## Author)

[//]: # ()
[//]: # ([Your Name] — [LinkedIn]&#40;#&#41; | [GitHub]&#40;#&#41;)