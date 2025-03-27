


# 🧠 DistilBERT AG News Classifier

A fast, lightweight news headline classifier fine-tuned using [DistilBERT](https://huggingface.co/distilbert-base-uncased) on the [AG News](https://huggingface.co/datasets/ag_news) dataset.

---

## 🔍 Features

- ⚡ Fine-tunes DistilBERT using Hugging Face Transformers & Datasets
- 🧠 Classifies news into 4 categories:
  - `World`
  - `Sports`
  - `Business`
  - `Sci/Tech`
- 🌐 Includes a Gradio Web UI for live interaction
- ✅ Final accuracy: ~92%

---

## 📦 Installation

```bash
git clone https://github.com/your-username/distilbert-agnews.git
cd distilbert-agnews
pip install -r requirements.txt
```

---

## 🚀 Run the Project

```bash
python distilbert_agnews_classification.py
```

After training, the Gradio web interface will launch at:

```
http://127.0.0.1:7860
```

You can enter a headline like:

```
"NASA discovers quantum superposition in space"
```

🧠 Output → `Sci/Tech`

---

## 🛠 Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Transformers
- Datasets
- Gradio
- scikit-learn

Install them all with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Example Headline Predictions

| Input                                           | Output     |
|------------------------------------------------|------------|
| “NASA plans mission to Mars”                   | Sci/Tech   |
| “Stock markets fall after inflation news”      | Business   |
| “Manchester United wins the final”             | Sports     |
| “UN holds emergency world summit”              | World      |

---

## 📊 Model & Training

- **Base Model:** `distilbert-base-uncased`
- **Dataset:** `ag_news`
- **Batch Size:** 16
- **Epochs:** 3
- **Learning Rate:** 2e-5
- **Optimizer:** Adam

---

## 🌐 Make It Public with Gradio

Want a public shareable demo? Just change this in the script:

```python
demo.launch(share=True)
```

Gradio will generate a URL like:

```
https://yourname.gradio.live
```

---

## 🙌 Credits

- 🤗 [Hugging Face Transformers](https://huggingface.co/transformers)
- 📚 [AG News Dataset](https://huggingface.co/datasets/ag_news)
- 🎛️ [Gradio UI](https://gradio.app)

---

> Built with ❤️ by Janaki Subramani — feel free to star this repo ⭐ and share!
