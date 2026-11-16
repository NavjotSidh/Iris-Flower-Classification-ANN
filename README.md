#  Iris Flower Classification using Artificial Neural Networks

An interactive Machine Learning application that classifies Iris flowers into three species using an Artificial Neural Network (ANN) built with TensorFlow/Keras and deployed using Streamlit.

## Project Overview

This project implements an Artificial Neural Network to classify Iris flowers based on four physical measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model predicts one of three Iris species:

1. Iris Setosa
2. Iris Versicolor
3. Iris Virginica

The application provides an interactive interface where users can enter flower measurements and receive predictions along with class probability distributions.

## 🖥️ Application Preview

### Home Dashboard
![Home Dashboard](screenshots/01_Home_Dashboard.png)

### Flower Prediction
![Flower Prediction](screenshots/02_Flower_Prediction.png)

### Model Performance
![Model Performance](screenshots/03_Model_Accuracy_Loss.png)

### Confusion Matrix and Classification Report
![Confusion Matrix](screenshots/04_Confusion_Matrix.png)

### ANN Architecture
![ANN Architecture](screenshots/05_ANN_Architecture.png)

## 🧠 ANN Architecture

The Artificial Neural Network consists of the following layers:

| Layer | Neurons | Activation |
|---|---:|---|
| Input Layer | 4 | - |
| Hidden Layer 1 | 16 | ReLU |
| Hidden Layer 2 | 8 | ReLU |
| Output Layer | 3 | Softmax |

### Model Configuration

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 8
- Epochs: 100

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| NumPy | Numerical Computing |
| Pandas | Data Manipulation |
| Scikit-learn | Data Preprocessing and Evaluation |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network Implementation |
| Matplotlib | Data Visualization |
| Seaborn | Dataset Loading and Visualization |
| Streamlit | Interactive Web Application |

## 📊 Model Performance

The model achieved the following results in the evaluated run:

- Test Accuracy: 84.21%
- Test Loss: 0.3423

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Training and validation accuracy/loss curves are also included to monitor model learning and generalization.

*Note: Results may vary slightly between training runs.*

## ✨ Application Features

- Interactive flower classification.
- User-friendly input sliders.
- ANN-based species prediction.
- Prediction confidence score.
- Class probability visualization.
- Training and validation accuracy graphs.
- Training and validation loss graphs.
- Confusion matrix.
- Classification report.
- ANN architecture overview.

## 📂 Project Structure

```text
Iris-Flower-Classification-ANN/
│
├── app.py
├── requirements.txt
├── README.md
│
└── screenshots/
    ├── 01_Home_Dashboard.png
    ├── 02_Flower_Prediction.png
    ├── 03_Model_Accuracy_Loss.png
    ├── 04_Confusion_Matrix.png
    ├── 05_ANN_Architecture.png
    └── 06_Technologies_Used.png
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Iris-Flower-Classification-ANN.git
```

### 2. Navigate to the project directory

```bash
cd Iris-Flower-Classification-ANN
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## 🎯 Learning Outcomes

Through this project, I explored:

- Data preprocessing and label encoding.
- Feature scaling using StandardScaler.
- Training and testing data splitting.
- Artificial Neural Network architecture.
- Activation functions: ReLU and Softmax.
- Model optimization using Adam.
- Categorical Crossentropy loss.
- Model evaluation and performance visualization.
- Building interactive ML applications using Streamlit.

## 👨‍💻 Author

**Navjot Sidhu**

B.Tech Computer Science and Engineering | Data Science

Interested in Machine Learning, Deep Learning and Data Science.

---

⭐ If you find this project interesting, consider giving the repository a star!