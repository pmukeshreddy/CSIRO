"""
Configuration settings for the Metadata Predictor model.
"""
import torch


class Config:
    """Configuration class for model training and inference."""
    
    # Paths
    TRAIN_CSV = 'train.csv'
    TRAIN_IMG_DIR = 'train'
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Reduce to 8 if GPU memory issues
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Metadata targets
    NUMERIC_TARGETS = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    CATEGORICAL_TARGETS = ['State', 'Species']
    
    # Training
    VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    SEED = 42
    
    # Model checkpoint
    MODEL_PATH = 'metadata_predictor_stage1.pth'
    PREPROCESSOR_PATH = 'preprocessor_stage1.pkl'
    
    @classmethod
    def display(cls):
        """Display configuration settings."""
        print(f"Device: {cls.DEVICE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Image size: {cls.IMG_SIZE}×{cls.IMG_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.EPOCHS}")
