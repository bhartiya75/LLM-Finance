from transformers import DistilBertTokenizer, DistilBertForMaskedLM, pipeline
import torch

# Enable Metal GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load fine-tuned model and tokenizer for masked language modeling
model = DistilBertForMaskedLM.from_pretrained("./fine_tuned_llm_mlm")
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_llm_mlm")
model.to(device)

# Create a pipeline for masked language modeling
mlm_pipeline = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.backends.mps.is_available() else -1  # Use MPS device (0) or CPU (-1)
)

# Example of filling a masked token in financial text (in millions USD)
text = "Date: 2025-02-23, Revenue: 120M, Net Income: [MASK]M, Month End Stock Price: 180"
predictions = mlm_pipeline(text)
print("Masked token predictions (in millions USD):")
for pred in predictions:
    print(f"{pred['sequence']} (Score: {pred['score']:.4f})")

# Example of generating text or understanding (optional, for longer sequences, in millions USD)
text = "Date: 2024-12-31, Revenue: 100M, Net Income: 15M, Total Assets: [MASK]M"
predictions = mlm_pipeline(text)
print("\nMasked token predictions for longer text (in millions USD):")
for pred in predictions:
    print(f"{pred['sequence']} (Score: {pred['score']:.4f})")