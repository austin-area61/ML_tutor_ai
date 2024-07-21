import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file, threshold):
    # Verify if the file exists
    if not os.path.isfile(input_file):
        print(f"Error: File not found at {input_file}")
        return
    
    data = pd.read_csv(input_file)
    
    # Ensure the 'incorrect_count' column exists
    if 'incorrect_count' not in data.columns:
        print("Error: 'incorrect_count' column not found in the input file.")
        return
    
    # Filter questions based on incorrect_count threshold
    poorly_answered = data[data['incorrect_count'] > threshold]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(poorly_answered.select_dtypes(include=['float64', 'int64']))
    data_imputed_df = pd.DataFrame(data_imputed, columns=poorly_answered.select_dtypes(include=['float64', 'int64']).columns)
    
    # Standardize numeric features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed_df)
    data_scaled_df = pd.DataFrame(data_scaled, columns=data_imputed_df.columns)
    
    # Combine with non-numeric features
    non_numeric = poorly_answered.select_dtypes(exclude=['float64', 'int64']).reset_index(drop=True)
    final_data = pd.concat([non_numeric, data_scaled_df], axis=1)
    
    # Save the preprocessed data
    final_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = os.path.join(os.getcwd(), 'data', 'raw', 'example_data.csv')
    output_file = os.path.join(os.getcwd(), 'data', 'processed', 'processed_data.csv')
    preprocess_data(input_file, output_file, threshold=5)
