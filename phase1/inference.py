"""
Inference script for making predictions on new images.
"""
import os
import pickle
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from config import Config
from model import MetadataPredictor


class MetadataInference:
    """Class for making metadata predictions on new images."""
    
    def __init__(self, model_path, preprocessor_path, device=None):
        """
        Initialize inference.
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Path to saved preprocessor
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load preprocessor
        print(f"Loading preprocessor from {preprocessor_path}...")
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Initialize model
        num_classes = checkpoint.get('num_classes') or self.preprocessor.get_num_classes()
        self.model = MetadataPredictor(
            num_states=num_classes['State'],
            num_species=num_classes['Species']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded on {self.device}")
    
    def predict_single(self, image_path):
        """
        Predict metadata for a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary containing predictions
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Extract predictions
        ndvi_scaled = outputs['ndvi'].item()
        height_scaled = outputs['height'].item()
        state_idx = torch.argmax(outputs['state_logits'], dim=1).item()
        species_idx = torch.argmax(outputs['species_logits'], dim=1).item()
        
        # Inverse transform numeric predictions
        ndvi = self.preprocessor.scalers['Pre_GSHH_NDVI'].inverse_transform([[ndvi_scaled]])[0][0]
        height = self.preprocessor.scalers['Height_Ave_cm'].inverse_transform([[height_scaled]])[0][0]
        
        # Decode categorical predictions
        state = self.preprocessor.label_encoders['State'].inverse_transform([state_idx])[0]
        species = self.preprocessor.label_encoders['Species'].inverse_transform([species_idx])[0]
        
        return {
            'ndvi': ndvi,
            'height_cm': height,
            'state': state,
            'species': species,
            'ndvi_scaled': ndvi_scaled,
            'height_scaled': height_scaled
        }
    
    def predict_batch(self, image_paths):
        """
        Predict metadata for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for img_path in tqdm(image_paths, desc='Predicting'):
            try:
                pred = self.predict_single(img_path)
                pred['image_path'] = img_path
                predictions.append(pred)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return predictions
    
    def predict_to_dataframe(self, image_paths):
        """
        Predict and return results as DataFrame.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            pandas DataFrame with predictions
        """
        predictions = self.predict_batch(image_paths)
        return pd.DataFrame(predictions)


def main():
    """Example usage."""
    # Initialize inference
    inference = MetadataInference(
        model_path=Config.MODEL_PATH,
        preprocessor_path=Config.PREPROCESSOR_PATH
    )
    
    # Example: Predict on test images
    test_dir = 'test'  # Replace with your test directory
    if os.path.exists(test_dir):
        image_paths = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ][:10]  # Process first 10 images as example
        
        print(f"\nPredicting on {len(image_paths)} images...")
        results_df = inference.predict_to_dataframe(image_paths)
        
        print("\nPredictions:")
        print(results_df[['image_path', 'ndvi', 'height_cm', 'state', 'species']].head())
        
        # Save results
        results_df.to_csv('predictions.csv', index=False)
        print("\n✓ Predictions saved to 'predictions.csv'")
    else:
        print(f"Test directory '{test_dir}' not found")
        print("\nUsage example:")
        print("  from inference import MetadataInference")
        print("  inference = MetadataInference('metadata_predictor_stage1.pth', 'preprocessor_stage1.pkl')")
        print("  result = inference.predict_single('path/to/image.jpg')")
        print("  print(result)")


if __name__ == '__main__':
    main()
