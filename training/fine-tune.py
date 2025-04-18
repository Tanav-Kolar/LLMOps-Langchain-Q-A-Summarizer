from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

dataset = load_dataset("json", data_files="training/fine_tune_data.json")

def preprocess(example):
    inputs = tokenizer(
        example["question"], example.get("context", ""), truncation=True, padding="max_length", max_length=512
    )
    inputs["start_positions"] = 0
    inputs["end_positions"] = len(example["answer"]) - 1  # dummy
    return inputs

tokenized = dataset["train"].map(preprocess)

trainer = Trainer(
    model=model,
    args=TrainingArguments("./model/distilbert-finetuned", per_device_train_batch_size=4, num_train_epochs=3),
    train_dataset=tokenized
)

trainer.train()
