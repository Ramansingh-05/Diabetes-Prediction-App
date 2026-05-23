# 🩺 Diabetes Prediction App

A Machine Learning web application built with Streamlit that predicts whether a person is likely to have diabetes based on medical input data.

The app uses a **Random Forest Classifier** trained on the diabetes dataset and provides:

- Diabetes prediction
- Prediction probability
- Feature importance visualization
- Dataset analysis
- Classification report
- Confusion matrix
- BMI category analysis

---

# 🚀 Features

✅ Interactive Streamlit UI  
✅ Machine Learning prediction system  
✅ Dataset preview and summary  
✅ Real-time patient input form  
✅ Probability-based prediction  
✅ Feature importance graph  
✅ Classification metrics  
✅ Error handling for dataset loading  
✅ Responsive wide layout interface

---

# 🧠 Machine Learning Model

This project uses:

- **Random Forest Classifier**
- Train/Test Split
- Scikit-learn metrics for evaluation

---

# 📂 Project Structure

```bash
project-folder/
│
├── app.py                # Main Streamlit application
├── Diabetes.csv          # Dataset file
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

# 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 🔧 Setup Instructions

## 1️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

---

## 2️⃣ Create Virtual Environment

### Linux / Mac

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

---

### Windows

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

# 📊 Dataset Information

The dataset contains medical diagnostic measurements used for diabetes prediction.

### Features Used

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Target Variable

- `0` → No Diabetes
- `1` → Diabetes

---

# 📈 Model Performance

The application displays:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Feature Importance Chart

---

# ⚠️ Important Notes

- Ensure the dataset file name matches exactly:

```bash
Diabetes.csv
```

Linux systems are case-sensitive.

- This application is for educational purposes only.
- It should not replace professional medical diagnosis.

---

# 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# 🧪 Example Run

After starting the app, open:

```bash
http://localhost:8501
```

in your browser.

---

# 📌 Future Improvements

- Add model saving/loading
- Deploy to Streamlit Cloud
- Add multiple ML models
- Improve UI design
- Add authentication system
- Add real medical datasets
- Add downloadable reports

---

# 👨‍💻 Author

Developed by Raman 

---

# 📜 License

This project is open-source and available under the MIT License.
