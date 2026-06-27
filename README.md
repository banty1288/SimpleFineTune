# 🚀 SimpleFineTune - Sentiment Analysis using DistilBERT

A beginner-friendly NLP project demonstrating how to fine-tune a pretrained Transformer model (DistilBERT) for binary sentiment classification using the Hugging Face Transformers library.

This project covers the complete fine-tuning pipeline, from preparing a dataset to training, saving, and performing inference with a custom model.

---

## 📌 Features

* Fine-tune a pretrained DistilBERT model
* Binary sentiment classification (Positive / Negative)
* Custom CSV dataset
* Hugging Face Trainer API
* Save and reload the fine-tuned model
* Interactive command-line inference
* Beginner-friendly project structure

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* Pandas

---

---

## 📊 Dataset

The project uses a simple CSV dataset.

Example:

| Text                       | Label |
| -------------------------- | ----- |
| I love this movie          | 1     |
| Excellent doctor           | 1     |
| Worst experience ever      | 0     |
| This medicine made me sick | 0     |

Label Mapping

```
1 → Positive
0 → Negative
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone <your-repository-url>
cd SimpleFineTune
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run:

```bash
python train.py
```

The script will:

* Load the dataset
* Tokenize text
* Fine-tune DistilBERT
* Evaluate the model
* Save the trained model inside the `model/` directory

---

## 🧪 Inference

Run:

```bash
python inference.py
```

Example

```
Enter a sentence:
This medicine made me feel worse.

Prediction:
Negative

Confidence:
0.5089
```

---

## 🧠 Model

**Base Model**

* DistilBERT (`distilbert-base-uncased`)

**Task**

* Sequence Classification

**Number of Classes**

* 2

---

## 📈 Training Configuration

| Parameter           |                   Value |
| ------------------- | ----------------------: |
| Epochs              |                       5 |
| Batch Size          |                       4 |
| Learning Rate       |                    2e-5 |
| Max Sequence Length |                     128 |
| Optimizer           | AdamW (Trainer Default) |

---

## 📚 Learning Outcomes

This project demonstrates:

* Loading datasets with Hugging Face Datasets
* Text tokenization
* Transfer Learning
* Fine-tuning Transformer models
* Using the Trainer API
* Model evaluation
* Saving and loading custom models
* Running inference on unseen text

---

## 🚀 Future Improvements

* Larger training dataset
* Streamlit web application
* Multi-class sentiment analysis
* Healthcare sentiment classification
* Emotion detection
* LoRA (Parameter-Efficient Fine-Tuning)
* Model deployment on Hugging Face Spaces

---

## 👨‍💻 Author

**Banty Kumar**

B.Tech Electrical Engineering
National Institute of Technology Patna

---

## 📄 License

This project is released under the MIT License.
