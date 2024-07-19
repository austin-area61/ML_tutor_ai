import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

def train_model(input_file):
    # Print the path for debugging
    print(f"Loading data from: {input_file}")
    
    # Load the processed data
    data = pd.read_csv(input_file)
    
    # Separate features and labels
    X = data[['question_text_encoded', 'incorrect_answer_encoded']]
    y = data['correct_answer']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    return model

if __name__ == "__main__":
    # Define the absolute path to the processed data
    input_file = 'C:/Users/AUSTIN/ML_tutor_ai/data/processed/processed_data.csv'
    
    # Print the absolute path for debugging
    print(f"Absolute path to the processed data: {input_file}")
    
    # Train the model
    train_model(input_file)
