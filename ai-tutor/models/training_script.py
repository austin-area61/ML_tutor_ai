import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(data_file, model_file):
    # Load the data
    data = pd.read_csv(data_file)
    
    # Split the data into features and labels
    X = data[['question_text_encoded', 'incorrect_answer_encoded']]
    y = data['correct_answer']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model('data/processed/processed_data.csv', 'models/tutor_model.pkl')
