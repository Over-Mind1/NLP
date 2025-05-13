
# 🧠 Sentiment Analyzer – Amazon Reviews with LSTM

This project is a complete pipeline for **Sentiment Analysis** using **Deep Learning (LSTM)** on **Amazon Reviews** dataset. It covers data preprocessing, visualization, model training, evaluation, and deployment using **Streamlit** on **Hugging Face Spaces**.

## 🔗 Live Demo

👉 Try the sentiment analyzer live here: [Hugging Face Spaces – OverMind0](https://huggingface.co/spaces/OverMind0/sentiment_analysis)

---

## 📂 Project Structure

| File/Folder                          | Description |
|-------------------------------------|-------------|
| `sentiment-analyzer-part1.ipynb`    | Data loading, cleaning, preprocessing, and visualization |
| `sentiment-analyzer.ipynb`          | LSTM model building, training, evaluation, and saving |
| `app.py`                            | Streamlit app for user interaction |
| `model.h5`                          | Trained LSTM model |
| `tokenizer.pickle`                  | Saved tokenizer for text preprocessing |
| `requirements.txt`                  | Dependencies needed to run the project |

---

## 🔍 Overview

- **Dataset**: Amazon Reviews (binary sentiment classification)
- **Model**: Keras-based BILSTM
- **Deployment**: Streamlit + Hugging Face Spaces
- **Goal**: Classify reviews as Positive or Negative

---

## 🚀 How to Run Locally

1. **Clone the repo**
```bash
git clone https://github.com/Over-Mind1/NLP/Sentiment Analyzer.git
cd sentiment-analyzer
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 🧠 Model Details

* **Embedding Layer**: Converts text to dense vectors
* **BILSTM Layer**: Captures temporal dependencies in review sequences
* **Dense Output**: Sigmoid activation for binary classification
* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam

---

## 📊 Performance

* Accuracy: \~93% on validation data
---

## 📌 Technologies Used

* Python
* Pandas, NumPy, Matplotlib, Seaborn
* Keras / TensorFlow
* Streamlit
* Hugging Face Spaces

---

## 🙋‍♂️ Author

**Mohamed Sabry Hussien**
[LinkedIn](https://www.linkedin.com/in/mo7amedsabry) | [Kaggle](https://www.kaggle.com/mo7amedsabry)

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify.

```


---
````
