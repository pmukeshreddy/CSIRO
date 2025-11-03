"""
Main training script for Stage 1: Metadata Predictor.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import Config
from data_preprocessing import DataPreprocessor
from dataset import MetadataDataset
from model import MetadataPredictor
from train import Trainer
from evaluate import Evaluator
from utils import (
    visualize_sample_images,
    plot_categorical_distributions,
    plot_training_curves,
    plot_regression_results,
    print_summary
)

warnings.filterwarnings('ignore')


def setup_transforms(config):
    """Setup data transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def main():
    """Main training function."""
    # Initialize config
    config = Config()
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    print("="*80)
    print("🌾 Stage 1: Metadata Predictor Training")
    print("="*80)
    config.display()
    print("="*80)
    
    # Load data
    print("\n📁 Loading data...")
    df = pd.read_csv(config.TRAIN_CSV)
    print(f"Dataset shape: {df.shape}")
    
    # Get unique images
    df_unique = df.drop_duplicates(subset=['sample_id']).reset_index(drop=True)
    print(f"Unique images: {len(df_unique)}")
    
    # Optional: Visualize data (comment out if not needed)
    # print("\n📊 Visualizing data...")
    # visualize_sample_images(df_unique, config.TRAIN_IMG_DIR)
    # plot_categorical_distributions(df_unique)
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df_unique)
    
    num_classes = preprocessor.get_num_classes()
    print(f"Number of states: {num_classes['State']}")
    print(f"Number of species: {num_classes['Species']}")
    
    # Split data
    train_df, val_df = train_test_split(
        df_processed,
        test_size=config.VAL_SPLIT,
        random_state=config.SEED,
        stratify=df_processed['State']
    )
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Setup transforms
    train_transform, val_transform = setup_transforms(config)
    
    # Create datasets
    train_dataset = MetadataDataset(train_df, config.TRAIN_IMG_DIR, train_transform)
    val_dataset = MetadataDataset(val_df, config.TRAIN_IMG_DIR, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = MetadataPredictor(
        num_states=num_classes['State'],
        num_species=num_classes['Species']
    ).to(config.DEVICE)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Initialize trainer
    trainer = Trainer(model, optimizer, config.DEVICE)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'state_acc': [],
        'species_acc': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 80)
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update scheduler
        scheduler.step(val_metrics['total_loss'])
        
        # Save history
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['total_loss'])
        history['state_acc'].append(val_metrics['state_accuracy'])
        history['species_acc'].append(val_metrics['species_accuracy'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
        print(f"  NDVI: {train_metrics['ndvi_loss']:.4f}, Height: {train_metrics['height_loss']:.4f}")
        print(f"  State: {train_metrics['state_loss']:.4f}, Species: {train_metrics['species_loss']:.4f}")
        
        print(f"\nVal Loss: {val_metrics['total_loss']:.4f}")
        print(f"  NDVI: {val_metrics['ndvi_loss']:.4f}, Height: {val_metrics['height_loss']:.4f}")
        print(f"  State: {val_metrics['state_loss']:.4f} (Acc: {val_metrics['state_accuracy']:.4f})")
        print(f"  Species: {val_metrics['species_loss']:.4f} (Acc: {val_metrics['species_accuracy']:.4f})")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'preprocessor': preprocessor,
                'num_classes': num_classes
            }, config.MODEL_PATH)
            print(f"\n✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    # Plot training curves
    print("\n📈 Generating visualizations...")
    plot_training_curves(history, 'training_curves.png')
    
    # Evaluate on validation set
    print("\n🎯 Evaluating on validation set...")
    checkpoint = torch.load(config.MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = Evaluator(model, config.DEVICE)
    predictions, targets = evaluator.get_predictions(val_loader)
    metrics = evaluator.calculate_metrics(predictions, targets)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Plot regression results
    plot_regression_results(predictions, targets, 'regression_results.png')
    
    # Save preprocessor
    print("\n💾 Saving preprocessor...")
    with open(config.PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"✓ Preprocessor saved to '{config.PREPROCESSOR_PATH}'")
    
    # Print summary
    print_summary(best_val_loss, metrics, config)


if __name__ == '__main__':
    main()
