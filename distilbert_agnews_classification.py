# distilbert_agnews_classification.py
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import gradio as gr

# Load AG News dataset
dataset = load_dataset("ag_news")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Preprocessing
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

# Accuracy function
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(10000)),
    eval_dataset=tokenized_dataset["test"].select(range(1000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print(f"\n✅ Final Accuracy: {metrics['eval_accuracy'] * 100:.2f}%")

def classify_text(text):
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # <-- move inputs to device

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    return label_map[prediction]


# Gradio UI
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline..."),
    outputs=gr.Label(num_top_classes=4),
    title="DistilBERT News Classifier",
    description="Classify news headlines into World, Sports, Business, or Sci/Tech using a fine-tuned DistilBERT model."
)

demo.launch(share=True)
