"""
Data preprocessing utilities for metadata prediction.
"""
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import Config


class DataPreprocessor:
    """Handles encoding and scaling of metadata."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
    
    def fit_transform(self, df):
        """
        Fit and transform the data.
        
        Args:
            df: pandas DataFrame containing the data
            
        Returns:
            Transformed DataFrame with encoded and scaled columns
        """
        df = df.copy()
        
        # Encode categorical variables
        for col in Config.CATEGORICAL_TARGETS:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Scale numeric variables
        for col in Config.NUMERIC_TARGETS:
            self.scalers[col] = StandardScaler()
            df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
        
        return df
    
    def transform(self, df):
        """
        Transform new data using fitted encoders/scalers.
        
        Args:
            df: pandas DataFrame containing the data
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        for col in Config.CATEGORICAL_TARGETS:
            df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        for col in Config.NUMERIC_TARGETS:
            df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def get_num_classes(self):
        """
        Get number of classes for each categorical variable.
        
        Returns:
            Dictionary mapping category names to number of classes
        """
        return {
            col: len(encoder.classes_)
            for col, encoder in self.label_encoders.items()
        }
