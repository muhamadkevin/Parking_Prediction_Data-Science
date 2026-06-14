# 🚗 Parking Density Prediction System

A Streamlit-based web application to predict and analyze parking density status (Low, Medium, High) based on location coordinates, time, total capacity, queue length, and nearby traffic conditions.

This application uses multiple Machine Learning models (such as SVM, XGBoost, Ridge, Naive Bayes, and KNN) trained on the *Parking Stream* dataset.

---

## ✨ Key Features

1. **Exploratory Data Analysis (EDA)**:
   - **Basic Statistics**: General overview of the dataset and distributions of capacity, occupancy, and queue length.
   - **Temporal Patterns**: Analysis of parking density by hour (rush hour vs. non-rush hour), weekday vs. weekend, and interactive heatmap analysis.
   - **Distribution & Correlation**: Pearson correlation matrix and feature correlations with respect to the `OccupancyRate`.
   - **Categorical Analysis**: Target class distribution (*Sepi* / Low, *Sedang* / Medium, *Padat* / High).

2. **Density Prediction**:
   - Interactive input form to enter real-time parking conditions.
   - Prediction of density status along with a visual representation of probability distributions.
   - Detailed, automated reasoning explaining the factors behind the model's decision.

---

## 🛠️ Tech Stack & Libraries

- **Core**: Python 3.x
- **Web Framework**: Streamlit
- **Data Analysis & Visualization**: Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn, Joblib, XGBoost

---

## 🚀 Running Locally

1. **Clone this repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/repo-name.git
   cd repo-name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment to Streamlit Community Cloud

This application is ready to be deployed directly to **Streamlit Community Cloud** with the following file structure:
- `app.py`: The main application source code.
- `requirements.txt`: Python package requirements.
- `.gitignore`: Prevents pushing cache files and Jupyter notebook checkpoints.
- `.pkl` & `.json` & `.csv`: Necessary model files and dataset for the app to function.
