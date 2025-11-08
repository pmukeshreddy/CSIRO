class BiomassDataset(Dataset):
    """Dataset with advanced preprocessing"""
    
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
        # Metadata features
        self.metadata_features = [
            'Pre_GSHH_NDVI_scaled',
            'Height_Ave_cm_scaled',
            'NDVI_Height_interaction_scaled',
            'NDVI_squared_scaled',
            'Height_log_scaled',
            'State_encoded',
            'Species_encoded',
            'NDVI_category',
            'Height_category'
        ]
        
        # Add additional features if they exist
        if 'NDVI_species_mean_scaled' in df.columns:
            self.metadata_features.extend([
                'NDVI_species_mean_scaled',
                'Height_species_mean_scaled',
                'NDVI_state_mean_scaled'
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_filename = row['image_path'].split('/')[-1]
        img_path = os.path.join(self.img_dir, img_filename)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Metadata
        metadata = torch.tensor(
            row[self.metadata_features].values.astype(np.float32),
            dtype=torch.float32
        )
        
        # Prepare output
        output = {
            'image': image,
            'metadata': metadata,
            'image_id': row['image_id']
        }
        
        # Add targets if not test
        if not self.is_test:
            targets = torch.tensor(
                row[config.BIOMASS_TARGETS].values.astype(np.float32),
                dtype=torch.float32
            )
            output['targets'] = targets
        
        return output
