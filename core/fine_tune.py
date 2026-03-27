import os
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

class EmotionModelTrainer:
    def __init__(self, model_name="vinai/phobert-base-v2", dataset_name="tridm/UIT-VSMEC"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.text_col = "Sentence"
        self.label_col = "Emotion"
        
        # Đường dẫn lưu tự động
        self.output_dir = "vsmec_emotion_model"
        self.best_model_dir = os.path.join(self.output_dir, "best")
        
        # Biến chứa dữ liệu nội bộ
        self.tokenizer = None
        self.label2id = {}
        self.id2label = {}
        self.all_labels = []
        
        # Load sẵn metric
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": self.accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": self.f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    def train_and_save(self):
        ds = load_dataset(self.dataset_name)
        
        self.all_labels = sorted(set(ds["train"][self.label_col]))
        self.label2id = {lbl: i for i, lbl in enumerate(self.all_labels)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        def encode_label(example):
            example["label"] = self.label2id[example[self.label_col]]
            return example

        ds = ds.map(encode_label)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def preprocess(example):
            return self.tokenizer(
                example[self.text_col],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        ds_tok = ds.map(preprocess)
        ds_tok = ds_tok.remove_columns([self.text_col, self.label_col])
        ds_tok.set_format("torch")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.all_labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_steps=50,
            fp16=True, # Tối ưu hóa cho GPU có hỗ trợ CUDA
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["validation"],
            processing_class=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

        trainer.save_model(self.best_model_dir)
        self.tokenizer.save_pretrained(self.best_model_dir)
        
        return self.best_model_dir