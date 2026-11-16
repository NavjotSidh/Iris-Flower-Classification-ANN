import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Iris ANN Classifier",
    page_icon="🌸",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f8f9fc;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
}

h1 {
    color: #253858;
}

h2, h3 {
    color: #344563;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TRAIN ANN MODEL
# ==========================================

@st.cache_resource
def train_model():

    # Load dataset
    df = sns.load_dataset('iris')

    X = df.drop(columns=['species'])
    Y = df['species']

    # Label Encoding
    encoder = LabelEncoder()
    Y_int = encoder.fit_transform(Y)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y_int,
        test_size=0.25,
        random_state=42,
        stratify=Y_int
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One Hot Encoding
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    # ANN Architecture
    model = Sequential([
        Dense(16, input_shape=(4,), activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train_scaled,
        y_train_cat,
        epochs=100,
        batch_size=8,
        validation_split=0.2,
        verbose=0
    )

    # Evaluate
    loss, accuracy = model.evaluate(
        X_test_scaled,
        y_test_cat,
        verbose=0
    )

    # Predictions
    y_pred = np.argmax(
        model.predict(X_test_scaled, verbose=0),
        axis=1
    )

    return (
        df, model, scaler, encoder,
        history, accuracy, loss,
        y_test, y_pred, X_test
    )


# Load model
with st.spinner("Training ANN model... Please wait"):
    (
        df, model, scaler, encoder,
        history, accuracy, loss,
        y_test, y_pred, X_test
    ) = train_model()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌸 Iris Classifier")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Make Prediction",
        "📊 Model Performance",
        "🧠 About ANN"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Artificial Neural Network project "
    "using TensorFlow and Streamlit."
)

st.sidebar.markdown("### Model Information")
st.sidebar.write("Algorithm: ANN")
st.sidebar.write("Optimizer: Adam")
st.sidebar.write("Activation: ReLU + Softmax")
st.sidebar.write("Epochs: 100")


# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.title("🌸 Iris Flower Classification")

    st.markdown(
        "### Artificial Neural Network Based Classification System"
    )

    st.write(
        """
        This application uses an Artificial Neural Network (ANN)
        built with TensorFlow/Keras to classify Iris flowers into
        three different species based on their physical measurements.
        """
    )

    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Samples", len(df))

    with col2:
        st.metric("Features", 4)

    with col3:
        st.metric("Flower Classes", 3)

    with col4:
        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    st.markdown("---")

    st.subheader("📋 Dataset Overview")

    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("🌺 Flower Species")

    col1, col2, col3 = st.columns(3)

    species = encoder.classes_

    with col1:
        st.info("**Setosa**")
        st.write("Class 0")

    with col2:
        st.success("**Versicolor**")
        st.write("Class 1")

    with col3:
        st.warning("**Virginica**")
        st.write("Class 2")


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "🔍 Make Prediction":

    st.title("🔍 Predict Iris Flower Species")

    st.write(
        "Enter the flower measurements below to predict its species."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sepal Measurements")

        sepal_length = st.slider(
            "Sepal Length (cm)",
            4.0, 8.0, 5.1, 0.1
        )

        sepal_width = st.slider(
            "Sepal Width (cm)",
            2.0, 4.5, 3.5, 0.1
        )

    with col2:

        st.subheader("Petal Measurements")

        petal_length = st.slider(
            "Petal Length (cm)",
            1.0, 7.0, 1.4, 0.1
        )

        petal_width = st.slider(
            "Petal Width (cm)",
            0.1, 2.5, 0.2, 0.1
        )

    st.markdown("---")

    if st.button("🌸 Predict Flower Species", use_container_width=True):

        # Input Data
        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=[
                'sepal_length',
                'sepal_width',
                'petal_length',
                'petal_width'
            ]
        )

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediction
        probabilities = model.predict(input_scaled, verbose=0)[0]

        predicted_class = np.argmax(probabilities)

        predicted_species = encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = probabilities[predicted_class] * 100

        # Display Result
        st.markdown("---")

        st.subheader("Prediction Result")

        st.success(
            f"🌺 Predicted Species: {predicted_species.upper()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.markdown("### Class Probabilities")

        probability_df = pd.DataFrame({
            "Species": encoder.classes_,
            "Probability": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Species")
        )


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance Dashboard")

    st.write("Evaluation results of the trained ANN model.")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Test Loss",
            f"{loss:.4f}"
        )

    with col3:
        st.metric(
            "Test Samples",
            len(y_test)
        )

    st.markdown("---")

    # Accuracy Graph
    st.subheader("Training vs Validation Accuracy")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        history.history['accuracy'],
        label='Training Accuracy'
    )

    ax.plot(
        history.history['val_accuracy'],
        label='Validation Accuracy'
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # Loss Graph
    st.subheader("Training vs Validation Loss")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        history.history['loss'],
        label='Training Loss'
    )

    ax.plot(
        history.history['val_loss'],
        label='Validation Loss'
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # Confusion Matrix
    st.markdown("---")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)
    plt.close(fig)

    # Classification Report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )


# ==========================================
# ABOUT ANN PAGE
# ==========================================

elif page == "🧠 About ANN":

    st.title("🧠 About Artificial Neural Network")

    st.write(
        """
        An Artificial Neural Network is a machine learning model
        inspired by the structure and functioning of biological
        neural networks.
        """
    )

    st.markdown("---")

    st.subheader("ANN Architecture")

    st.code("""
        Input Layer
        -----------
        4 Neurons

             |
             v

        Hidden Layer 1
        --------------
        16 Neurons
        Activation: ReLU

             |
             v

        Hidden Layer 2
        --------------
        8 Neurons
        Activation: ReLU

             |
             v

        Output Layer
        -------------
        3 Neurons
        Activation: Softmax

             |
             v

        Flower Classification
    """)

    st.markdown("---")

    st.subheader("Technologies Used")

    technologies = {
        "Python": "Programming Language",
        "NumPy": "Numerical Computing",
        "Pandas": "Data Manipulation",
        "Scikit-learn": "Preprocessing and Evaluation",
        "TensorFlow": "Deep Learning Framework",
        "Keras": "Neural Network API",
        "Streamlit": "Frontend and Deployment"
    }

    for tech, description in technologies.items():

        st.write(f"**{tech}** — {description}")

    st.markdown("---")

    st.subheader("Model Configuration")

    config = pd.DataFrame({
        "Parameter": [
            "Optimizer",
            "Loss Function",
            "Batch Size",
            "Epochs",
            "Hidden Layers",
            "Output Activation"
        ],
        "Value": [
            "Adam",
            "Categorical Crossentropy",
            "8",
            "100",
            "16, 8",
            "Softmax"
        ]
    })

    st.dataframe(config, use_container_width=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "<center>Built with ❤️ using Python, TensorFlow & Streamlit</center>",
    unsafe_allow_html=True
)