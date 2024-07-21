import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'ai-tutor'))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from src.utils.openai import generate_feedback, get_student_history, generate_personalized_feedback

def train_model(input_file):
    data = pd.read_csv(input_file)
    
    # Prepare data for training
    X = data[['question', 'response', 'expected_response']]
    y = data['feedback']
    
    # You may need to vectorize the text data (using TF-IDF, CountVectorizer, etc.)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X['response'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')
    
    model_path = os.path.join(os.getcwd(), 'ai-tutor', 'models', 'feedback_model.pkl')
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
