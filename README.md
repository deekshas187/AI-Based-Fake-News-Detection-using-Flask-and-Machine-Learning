# Fake News Detection using Flask and Machine Learning

A web-based Fake News Detection application developed using Flask, Machine Learning, Natural Language Processing (NLP), and Scikit-learn.

The system analyzes news content and classifies whether the news is Real or Fake using NLP preprocessing and machine learning classification techniques.

---

# Features

- Fake News Classification
- NLP Text Preprocessing
- TF-IDF Vectorization
- Machine Learning Prediction
- Real-time News Analysis
- Confidence Score Display
- Responsive User Interface
- Flask Web Application

---

# Technologies Used

## Frontend
- HTML
- CSS

## Backend
- Flask
- Python

## Machine Learning
- Scikit-learn
- Pandas
- Natural Language Processing (NLP)
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

# Project Structure

```bash
fake-news-detector/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

# How It Works

1. User enters a news article or headline.
2. NLP preprocessing cleans the text.
3. TF-IDF converts text into numerical vectors.
4. Machine learning model analyzes the content.
5. System predicts:
   - Real News
   - Fake News

---

# Machine Learning Workflow

- Data Collection
- Data Cleaning
- Text Preprocessing
- Feature Extraction using TF-IDF
- Model Training
- Prediction
- Web Integration

---

# Installation and Setup

## Clone Repository

```bash
git clone https://github.com/deekshas187/fake-news-detector.git
```

## Move into Project Folder

```bash
cd fake-news-detector
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train.py
```

## Run Application

```bash
python app.py
```

---

# Open in Browser

```bash
http://127.0.0.1:5000
```

---

# Future Improvements

- BERT Transformer Integration
- Real-time Fact Checking APIs
- News Source Verification
- Multi-language Detection
- Dark Mode Interface
- Browser Extension
- Deep Learning Models

---

# Concepts Used

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Text Classification
- Web Development

---

# Author

Deeksha S

---

# License

This project is developed for educational and learning purposes.
