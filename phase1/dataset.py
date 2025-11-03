"""
PyTorch Dataset for metadata prediction.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MetadataDataset(Dataset):
    """Dataset for training metadata predictor."""
    
    def __init__(self, df, img_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            df: pandas DataFrame with image paths and metadata
            img_dir: Directory containing images
            transform: Optional torchvision transforms
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing image and targets
        """
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, row['image_path'].split('/')[-1])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get targets (scaled/encoded)
        ndvi = row['Pre_GSHH_NDVI_scaled']
        height = row['Height_Ave_cm_scaled']
        state = row['State_encoded']
        species = row['Species_encoded']
        
        return {
            'image': image,
            'ndvi': torch.tensor(ndvi, dtype=torch.float32),
            'height': torch.tensor(height, dtype=torch.float32),
            'state': torch.tensor(state, dtype=torch.long),
            'species': torch.tensor(species, dtype=torch.long)
        }
