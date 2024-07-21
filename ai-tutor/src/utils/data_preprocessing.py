import pandas as pd
import os

def preprocess_data(input_file, output_file, threshold):
    data = pd.read_csv(input_file)
    
    # Filter for poorly answered questions based on 'incorrect_count'
    poorly_answered = data[data['incorrect_count'] > threshold]
    
    # Save the preprocessed data
    poorly_answered.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), 'data', 'raw', 'example_data.csv')
    output_file = os.path.join(os.getcwd(), 'data', 'processed', 'processed_data.csv')
    preprocess_data(input_file, output_file, threshold=5)
