from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# Custom Dataset for financial text (no labels, just text for MLM)
class FinancialDataset(Dataset):
    def __init__(self, texts, tokenizer=None, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encoding.items()}

# Enable Metal GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = pd.read_csv("financial_text.csv")  # Use text-only dataset
texts = data['text'].tolist()

# Tokenize data
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
dataset = FinancialDataset(texts, tokenizer=tokenizer)

# Split into train and validation
train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
train_dataset = FinancialDataset(train_texts, tokenizer=tokenizer)
val_dataset = FinancialDataset(val_texts, tokenizer=tokenizer)

# Load pre-trained model for masked language modeling and move to GPU if available
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
model.to(device)

# Data collator for MLM (handles masking automatically)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training arguments (optimized for GPU, adjusted for more data)
training_args = TrainingArguments(
    output_dir="./results_mlm",  # Different output for masked LM
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Reduced for memory with all columns
    per_device_eval_batch_size=4,   # Reduced for memory
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_mlm',  # Different logs for masked LM
    logging_steps=10,
    eval_strategy="epoch",  # Updated from evaluation_strategy for future compatibility
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer (using data collator for MLM)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,  # Add data collator for masking
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_llm_mlm")  # Different model directory for masked LM
tokenizer.save_pretrained("./fine_tuned_llm_mlm")