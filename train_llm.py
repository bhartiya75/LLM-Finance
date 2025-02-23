from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# Custom Dataset for financial text
class FinancialDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):  # Increased max_length for more data
        self.texts = texts
        self.labels = labels if labels is not None else [0] * len(texts)  # Default to 0 if no labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Enable Metal GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = pd.read_csv("financial_text.csv")
texts = data['text'].tolist()

# Tokenize data
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
dataset = FinancialDataset(texts, tokenizer=tokenizer)

# Split into train and validation (since we’re fine-tuning for text understanding, no labels needed initially)
train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
train_dataset = FinancialDataset(train_texts, tokenizer=tokenizer)
val_dataset = FinancialDataset(val_texts, tokenizer=tokenizer)

# Load pre-trained model and move to GPU if available
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
model.to(device)

# Training arguments (optimized for GPU, adjusted for more data)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Reduced for memory with all columns
    per_device_eval_batch_size=4,   # Reduced for memory
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    device=device,  # Ensure device is set
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_llm")
tokenizer.save_pretrained("./fine_tuned_llm")