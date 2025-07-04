import pandas as pd
import ssl
import json
import hashlib
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ssl._create_default_https_context = ssl._create_unverified_context

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

def train_and_evaluate():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None)
    df.columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    df = df.replace('?', float('nan'))
    
    for col in df.columns[:-1]:
        df[col] = pd.to_numeric(df[col])
    
    df = df.dropna()
    
    df['target'] = (df['target'] > 0).astype(int)
    
    X = df.drop(['target', 'sex'], axis=1)
    y = df['target']
    
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    
    test_data = X_test.copy()
    test_data['sex'] = df.loc[X_test.index, 'sex']
    test_data['prediction'] = model.predict(X_test)
    
    male_acc = test_data[test_data['sex'] == 1]['prediction'].mean()
    female_acc = test_data[test_data['sex'] == 0]['prediction'].mean()
    
    return {
        'overall_accuracy': model.score(X_test, y_test),
        'male_accuracy': male_acc,
        'female_accuracy': female_acc,
        'bias_score': male_acc - female_acc
    }

if __name__ == "__main__":
    blockchain = BlockchainLogger()
    model_name = "Heart_Disease_Predictor_v1"
    metrics = train_and_evaluate()
    
    print("\n=== AI Model Metrics ===")
    print(json.dumps(metrics, indent=2))
    
    blockchain.log_ai_metrics(model_name, metrics)
    print("\n=== Block Added to Chain ===")
    print(json.dumps(blockchain.chain[-1], indent=2))
    
    conn = sqlite3.connect('ai_audit.db')
    pd.DataFrame([metrics]).to_sql('model_audits', conn, if_exists='append')
    conn.close()