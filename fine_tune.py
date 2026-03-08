import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

MODEL_NAME = "vinai/phobert-base-v2"

# LOAD DATASET
ds = load_dataset("tridm/UIT-VSMEC")

TEXT_COL = "Sentence"
LABEL_COL = "Emotion"

# LABEL MAP
all_labels = sorted(set(ds["train"][LABEL_COL]))

label2id = {lbl: i for i, lbl in enumerate(all_labels)}
id2label = {i: lbl for lbl, i in label2id.items()}

def encode_label(example):
    example["label"] = label2id[example[LABEL_COL]]
    return example

ds = ds.map(encode_label)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    return tokenizer(
        example[TEXT_COL],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

ds_tok = ds.map(preprocess)

ds_tok = ds_tok.remove_columns(["Sentence", "Emotion"])
ds_tok.set_format("torch")

# MODEL
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
)

# METRICS
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# TRAINING ARGUMENTS
# TRAINING ARGUMENTS
args = TrainingArguments(
    output_dir="vsmec_emotion_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",  # <--- ĐỔI Ở ĐÂY
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_steps=50,
    fp16=True
)

# TRAINER (Sửa ở dòng này)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    processing_class=tokenizer, # <--- Đổi 'tokenizer' thành 'processing_class'
    compute_metrics=compute_metrics,
)

trainer.train()

save_dir = "vsmec_emotion_model/best"

trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print("Saved to", save_dir)