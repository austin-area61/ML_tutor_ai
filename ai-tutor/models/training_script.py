import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pickle
from src.utils.openai import generate_feedback, get_student_history, generate_personalized_feedback

def train_model(input_file):
    data = pd.read_csv(input_file)
    # Prepare data for training
    X = data.drop(columns=['student_id', 'response', 'expected_answer', 'feedback'])
    y = data['feedback']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')
    model_path = os.path.join(os.getcwd(), 'models', 'feedback_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Generate feedback using OpenAI API for a sample student response
    sample_response = "This is a sample student response."
    expected_answer = "This is the expected answer."
    feedback = generate_feedback(sample_response, expected_answer)
    print(f'Generated Feedback: {feedback}')

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), 'data', 'processed', 'processed_data.csv')
    train_model(input_file)
