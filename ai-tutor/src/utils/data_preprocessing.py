import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, '../../data/raw/example_data.csv')
    
    # Load data
    data = pd.read_csv(raw_data_path)
    
    # Separate features into numeric and categorical
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns
    
    # Define preprocessing for numeric columns (impute missing values and scale)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical columns (impute missing values and one-hot encode)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply transformations
    data_processed = preprocessor.fit_transform(data)
    
    # Convert to DataFrame (optional)
    data_processed_df = pd.DataFrame(data_processed, columns=numeric_features.tolist() + preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out().tolist())
    
    # Save processed data
    processed_data_path = os.path.join(script_dir, '../../data/processed/processed_data.csv')
    data_processed_df.to_csv(processed_data_path, index=False)

if __name__ == "__main__":
    main()
