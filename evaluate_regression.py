from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import torch

# Enable Metal GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load fine-tuned model and tokenizer for regression
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_llm_regression")
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_llm_regression")
model.to(device)

# Load test data (use a portion of financial_text_regression.csv or new data)
data = pd.read_csv("financial_text_regression.csv")
texts = data['text'].tolist()
true_labels = data['label'].tolist()  # Actual stock prices in millions USD

# Tokenize and predict
predictions = []
for text in texts[:5]:  # Test on first 5 examples (adjust as needed for more or new data)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.logits.squeeze().item()  # Get predicted value (in millions USD)
    predictions.append(prediction)

# Calculate basic metrics (e.g., mean absolute error, in millions USD)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(true_labels[:5], predictions)
print(f"Mean Absolute Error (in millions USD): {mae:.2f}")
print("Predictions vs True Labels (in millions USD):")
for pred, true in zip(predictions, true_labels[:5]):
    print(f"Predicted: {pred:.2f}, True: {true:.2f}")

# Example of predicting a new financial text (in millions USD)
new_text = "Date: 2025-02-23, Revenue: 120M, Net Income: 15M, Month End Stock Price: N/A"  # Adjust to match your text format
inputs = tokenizer(new_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
inputs = {key: val.to(device) for key, val in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
    predicted_price = outputs.logits.squeeze().item()  # Predicted in millions USD
print(f"Predicted Stock Price for new text (in millions USD): {predicted_price:.2f}")