import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    # Load the preprocessed data
    data = pd.read_csv('ai-tutor/data/processed/processed_data.csv')

    # Define feature columns and target column
    feature_columns = ['student_id', 'topic', 'answer_options']
    target_column = 'correct'

    # Split data into features and target
    X = data[feature_columns]
    y = data[target_column]

    # Further split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Save the trained model
    if not os.path.exists('ai-tutor/models'):
        os.makedirs('ai-tutor/models')
    joblib.dump(model, 'ai-tutor/models/tutor_model.joblib')

if __name__ == "__main__":
    main()
