import pandas as pd
import os

def preprocess_data(input_file, output_file, threshold=5):
    data = pd.read_csv(input_file)
    # Add data cleaning and preprocessing steps here
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), 'data', 'raw', 'example_data.csv')
    output_file = os.path.join(os.getcwd(), 'data', 'processed', 'processed_data.csv')
    preprocess_data(input_file, output_file, threshold=5)
