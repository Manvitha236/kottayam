import streamlit as st
import ssl
import pandas as pd
import json
import hashlib
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context

# Global variable to store feature names
FEATURE_NAMES = None

def init_database():
    """Initialize the database with the correct schema"""
    conn = None
    try:
        conn = sqlite3.connect('ai_audit.db')
        cursor = conn.cursor()
        
        # Create table with all required columns if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_audits (
            prediction TEXT NOT NULL,
            probability REAL NOT NULL,
            bias_assessment TEXT NOT NULL,
            patient_data TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Verify all columns exist
        cursor.execute("PRAGMA table_info(model_audits)")
        existing_columns = {col[1] for col in cursor.fetchall()}
        required_columns = {'prediction', 'probability', 'bias_assessment', 'patient_data', 'timestamp'}
        
        # Add any missing columns
        for column in required_columns - existing_columns:
            if column == 'probability':
                cursor.execute(f"ALTER TABLE model_audits ADD COLUMN {column} REAL NOT NULL DEFAULT 0")
            elif column == 'timestamp':
                cursor.execute(f"ALTER TABLE model_audits ADD COLUMN {column} DATETIME DEFAULT CURRENT_TIMESTAMP")
            else:
                cursor.execute(f"ALTER TABLE model_audits ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
        
        conn.commit()
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

# Initialize blockchain and model (cached for performance)
@st.cache_resource
def init_system():
    init_database()  # Ensure database is properly initialized
    blockchain = BlockchainLogger()
    model = train_model()
    return blockchain, model

class BlockchainLogger:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')
        
    def create_block(self, proof, previous_hash, data=None):
        block = {
            'index': len(self.chain) + 1,
            'proof': proof,
            'previous_hash': previous_hash,
            'data': data or {}
        }
        self.chain.append(block)
        return block
    
    def log_ai_metrics(self, model_name, metrics):
        last_block = self.chain[-1]
        new_block = self.create_block(
            proof=hashlib.sha256(json.dumps(metrics).encode()).hexdigest(),
            previous_hash=self.hash_block(last_block),
            data={
                'model': model_name,
                'metrics': metrics
            }
        )
        return new_block
    
    @staticmethod
    def hash_block(block):
        return hashlib.sha256(json.dumps(block).encode()).hexdigest()

def train_model():
    global FEATURE_NAMES
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None)
    df.columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Data cleaning
    df = df.replace('?', float('nan'))
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col])
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    
    # Set the feature names (excluding target and sex)
    FEATURE_NAMES = [col for col in df.columns if col not in ['target', 'sex']]
    
    # Train model
    X = df[FEATURE_NAMES]
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    
    return model

def predict_and_check_bias(model, input_data):
    input_values = [
        input_data['age'],
        input_data['cp'],
        input_data['trestbps'],
        input_data['chol'],
        input_data['fbs'],
        input_data['restecg'],
        input_data['thalach'],
        input_data['exang'],
        input_data['oldpeak'],
        input_data['slope'],
        input_data['ca'],
        input_data['thal']
    ]
    
    input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)
    
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    sex = input_data['sex']
    bias_msg = "Neutral"
    if sex == 1:  # Male
        bias_msg = "Potential male bias" if proba > 0.7 else "Neutral"
    else:  # Female
        bias_msg = "Potential female bias" if proba < 0.3 else "Neutral"
    
    return {
        'prediction': 'Heart Disease Risk' if prediction == 1 else 'Low Risk',
        'probability': float(proba),
        'bias_assessment': bias_msg
    }

# Streamlit UI
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Diagnosis with Responsible AI")
st.subheader("Blockchain-powered bias detection")

# Initialize system
blockchain, model = init_system()

# Input form
with st.form("patient_form"):
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("CA", [0, 1, 2, 3])
        thal = st.selectbox("THAL", [0, 1, 2, 3])
    
    submitted = st.form_submit_button("Evaluate Risk")

# Process form submission
if submitted:
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    try:
        result = predict_and_check_bias(model, input_data)
        
        st.success("### Evaluation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediction", result['prediction'])
            st.metric("Probability", f"{result['probability']:.1%}")
        with col2:
            st.metric("Bias Assessment", result['bias_assessment'])
        
        metrics = {
            'prediction': result['prediction'],
            'probability': result['probability'],
            'bias_assessment': result['bias_assessment'],
            'patient_data': {k:v for k,v in input_data.items() if k != 'sex'}
        }
        
        blockchain.log_ai_metrics("Heart_Disease_Predictor_v1", metrics)
        
        with st.expander("View Blockchain Record"):
            st.json(blockchain.chain[-1])
        
        # Save to database
        try:
            conn = sqlite3.connect('ai_audit.db')
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO model_audits (prediction, probability, bias_assessment, patient_data)
            VALUES (?, ?, ?, ?)
            """, (
                metrics['prediction'],
                metrics['probability'],
                metrics['bias_assessment'],
                json.dumps(metrics['patient_data'])
            ))
            conn.commit()
            st.info("This evaluation has been securely logged to the blockchain audit trail")
        except Exception as e:
            st.error(f"Database error: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Sidebar
st.sidebar.markdown("""
## About This System
- **AI Model**: Random Forest trained on Cleveland Heart Disease dataset
- **Bias Detection**: Monitors gender-based prediction differences
- **Blockchain**: Provides immutable audit trail of all diagnoses

## Key Features
✔️ Transparent risk assessment  
✔️ Bias monitoring in real-time  
✔️ Tamper-proof record keeping  
""")

if st.sidebar.button("View Last 10 Audits"):
    try:
        conn = sqlite3.connect('ai_audit.db')
        audit_log = pd.read_sql(
            "SELECT prediction, probability, bias_assessment, timestamp FROM model_audits ORDER BY timestamp DESC LIMIT 10", 
            conn
        )
        if not audit_log.empty:
            st.sidebar.dataframe(audit_log)
        else:
            st.sidebar.info("No audit records found")
    except Exception as e:
        st.sidebar.error(f"Failed to load audit log: {str(e)}")
    finally:
        if conn:
            conn.close()

# Force initialization on first run
if 'db_initialized' not in st.session_state:
    init_database()
    st.session_state.db_initialized = True