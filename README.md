# Fake News Detection

A machine learning project that uses Natural Language Processing (NLP) and Logistic Regression to classify news articles based on patterns learned from a labeled dataset.

## Current Status

The current version contains the basic machine learning pipeline:

- Data quality checking
- Text preprocessing
- Train-test split
- TF-IDF feature extraction
- Logistic Regression
- Model prediction
- Model evaluation

The project is currently implemented in a single Python file. It will be improved and refactored as development continues.

## Tech Stack

- Python
- Pandas
- NumPy
- NLTK
- Regex
- Scikit-learn
- UV

## Machine Learning Pipeline

News Dataset
↓
Data Quality Check
↓
Text Preprocessing
↓
Train-Test Split
↓
TF-IDF Vectorization
↓
Logistic Regression
↓
Prediction
↓
Evaluation

## Text Preprocessing

The current preprocessing pipeline includes:

- Removing HTML tags
- Removing URLs
- Converting text to lowercase
- Removing punctuation
- Removing numbers
- Removing extra whitespace
- Removing English stopwords

The project also calculates word count, character count, and frequently occurring words.

## Model

The current machine learning model is Logistic Regression.

TF-IDF is used to convert the cleaned text into numerical features.

The dataset is divided into:

- 80% training data
- 20% testing data

The train/test split uses stratification and a fixed random state for reproducibility.

## Dataset

The dataset contains news text and corresponding labels.

The dataset is not included in this repository because the CSV file is approximately 125 MB.

To run the project locally, place the dataset at:

data/news.csv

## How to Run

### 1. Clone the repository

git clone https://github.com/Rehan-7899/fake-news-detection.git

cd fake-news-detection

### 2. Install dependencies

This project uses UV for dependency management.

uv sync

### 3. Add the dataset

Place the dataset at:

data/news.csv

### 4. Run the project

uv run python src/fake_news_detection/main.py

The program will train the model and display the accuracy and classification report.

## Project Structure

fake-news-detection/
│
├── data/
│   └── news.csv
│
├── src/
│   └── fake_news_detection/
│       └── main.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock

## Current Progress

- [x] Data loading
- [x] Data quality checking
- [x] Text preprocessing
- [x] Train-test split
- [x] TF-IDF feature extraction
- [x] Logistic Regression
- [x] Model prediction
- [x] Model evaluation
- [ ] Improve model
- [ ] Compare different models
- [ ] Save trained model
- [ ] Create reusable prediction pipeline
- [ ] Refactor project into multiple files
- [ ] Build FastAPI backend
- [ ] Create prediction API
- [ ] Add user interface
- [ ] Deployment

## Future Plans

The project will eventually be extended with:

- Model comparison and improvement
- Model and vectorizer persistence
- Prediction of new user-provided news
- Modular project structure
- FastAPI backend
- Prediction API
- User interface
- Deployment

## Author

Rehan Shaikh