# Continued Pretraining: This is more like immersing your model in a new
# environment. Instead of jumping straight to the task, you first train
# the model further on domain-specific unlabeled data (e.g., medical papers,
# legal documents). It’s about making the model more familiar with your
# domain before fine-tuning.

from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Load domain-specific text (replace with your own corpus)
dataset = load_dataset("text", data_files={"train": "domain_corpus.txt"})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True)

# Data collator for Masked Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Load pre-trained model
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Training arguments
training_args = TrainingArguments(
    output_dir="./domain_pretrained_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Pretrain the model
trainer.train()