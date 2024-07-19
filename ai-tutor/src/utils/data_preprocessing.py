import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_file, output_file, threshold=5):
    # Load the data
    data = pd.read_csv(input_file)
    
    # Filter for poorly answered questions based on threshold
    poorly_answered = data[data['incorrect_count'] > threshold]
    
    # Select relevant columns
    features = poorly_answered[['question_text', 'incorrect_answer']]
    labels = poorly_answered['correct_answer']
    
    # Encode text data
    le_question = LabelEncoder()
    le_answer = LabelEncoder()
    features['question_text_encoded'] = le_question.fit_transform(features['question_text'])
    features['incorrect_answer_encoded'] = le_answer.fit_transform(features['incorrect_answer'])
    
    # Combine encoded features and labels
    processed_data = pd.concat([features[['question_text_encoded', 'incorrect_answer_encoded']], labels], axis=1)
    
    # Save the processed data
    processed_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Define absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, '../../data/raw/example_data.csv')
    output_file = os.path.join(base_dir, '../../data/processed/processed_data.csv')
    
    # Run preprocessing with a threshold value
    preprocess_data(input_file, output_file, threshold=5)
