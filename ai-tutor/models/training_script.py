import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load preprocessed data
    data = pd.read_csv('data/processed/processed_data.csv')

    # Define features and target variable
    X = data.drop(columns=['student_id', 'student_answer', 'score'])
    y = data['student_answer']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the trained model
    joblib.dump(model, 'models/random_forest_model.pkl')

    print("Model training complete. Model saved to 'models/random_forest_model.pkl'.")

if __name__ == "__main__":
    main()
