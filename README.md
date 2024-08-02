# Fake News Classification Using Natural Language Processing
-[Link].(https://drive.google.com/file/d/1uxkhEZUXWLu2SSKg9qcEUEG4QO_MvYUy/view?usp=drive_link/)

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Algorithms Used](#algorithms-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributions](#contributions)
- [License](#license)

## Introduction
This project aims to detect fake news articles using machine learning and natural language processing (NLP) techniques. With the rise of social media, the spread of fake news has become a significant issue, making it essential to develop robust methods for automatic fake news detection.

## Project Structure

├── data
│ ├── fake.csv
│ ├── real.csv
│ └── preprocessed_data.csv
├── notebooks
│ ├── EDA.ipynb
│ ├── Logistic_Regression.ipynb
│ ├── Naive_Bayes.ipynb
│ └── SVM.ipynb
├── models
│ ├── logistic_regression_model.pkl
│ ├── naive_bayes_model.pkl
│ └── svm_model.pkl
├── src
│ ├── data_preprocessing.py
│ ├── train.py
│ └── evaluate.py
├── README.md
└── requirements.txt


## Datasets
The dataset contains two types of news articles: genuine and fake. Each dataset contains around 35,000 entries. The datasets are divided into the following files:
- `fake.csv`: Contains fake news articles.
- `real.csv`: Contains real news articles.

## Methodology
1. **Data Collection**: Gathered a diverse dataset of news articles labeled as either fake or real.
2. **Text Preprocessing**: Cleaned and preprocessed data to remove noise and stop-words.
3. **TF-IDF Vectorization**: Converted text data into numerical vectors using TF-IDF.
4. **Training and Validation**: Split the dataset into training and validation sets and trained various machine learning algorithms.
5. **Evaluation**: Generated classification reports and confusion matrices to evaluate model performance.

## Algorithms Used
- **Logistic Regression**
- **Naive Bayes Classifier**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Trees**
- **Random Forest**
- **Principal Components Analysis (PCA)**
- **Passive-Aggressive Classifier**

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
### Data Preprocessing
To preprocess the data, run:

```bash
Copy code
python src/data_preprocessing.py
```

### Training Models
To train the models, run:

```bash
Copy code
python src/train.py
```

### Evaluating Models
To evaluate the models, run:

```bash
Copy code
python src/evaluate.py
```

### Results
The performance of different classifiers was evaluated based on metrics such as accuracy, precision, recall, and F1 score. The best-performing model achieved an accuracy of XX% (replace XX with the actual result).

### Future Scope
Larger Datasets: Obtaining larger labeled datasets to improve model accuracy.
Multi-Modal Analysis: Combining text analysis with images and videos.
Robustness: Enhancing model robustness to handle diverse and evolving fake news strategies.

### Contributions
Data Collection and Preprocessing: 
Model Training and Evaluation:
Documentation and Reporting: 

License
This project is licensed under the MIT License. See the LICENSE file for details.
