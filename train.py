from datasets import load_dataset
from transformers import(
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
dataset=load_dataset(
    "csv",
    data_files={
        "train":"data/train.csv",
        "test":"data/test.csv",
    }
)
# print(dataset)
tokenizer=AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_lenght=128
    )
dataset=dataset.map(tokenize)
model=AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
training_args=TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none"
)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.train()
trainer.save_model("model")
print("\n Model Fine-Tuned Successfully")