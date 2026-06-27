from transformers import pipeline
classifier=pipeline(
    "text-classification",
    model="model",
    tokenizer="model"
)
while True:
    text=input("\nEnter a sentence(or type 'exit):")
    if text.lower()=="exit":
        break
    result=classifier(text)[0]
    label="Positive" if result["label"]=="LABEL_1" else "Negative"
    print(f"\nPrediction :{label}")
    print(f"Confidence :{result["score"]:.4f}")