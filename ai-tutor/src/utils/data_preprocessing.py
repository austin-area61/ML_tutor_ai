import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main():
    # Load data
    data = pd.read_csv('data/raw/example_data.csv')

    # Display the first few rows of the dataset
    print(data.head())

    # Convert categorical data to numerical data
    label_encoder = LabelEncoder()

    data['topic'] = label_encoder.fit_transform(data['topic'])
    data['correct_answer'] = label_encoder.fit_transform(data['correct_answer'])
    data['student_answer'] = label_encoder.fit_transform(data['student_answer'])

    # Convert 'answer_options' to a numerical format (e.g., one-hot encoding)
    answer_options = data['answer_options'].str.get_dummies(sep=', ')
    data = pd.concat([data, answer_options], axis=1).drop(columns=['answer_options'])
    
    # Save preprocessed data to a new CSV file
    data.to_csv('data/processed/processed_data.csv', index=False)

    print("Data preprocessing complete. Processed data saved to 'data/processed/processed_data.csv'.")

if __name__ == "__main__":
    main()
