class AdvancedPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}

    def engineer_features(self,df):
        df = df.copy()

        df["NDVI_category"] = pd.cut(df["Pre_GSHH_NDVI"],bins=[-1, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 1, 2, 3, 4]).astype(int)

        df["Height_category"] = pd.cut(
            df['Height_Ave_cm'],
            bins=[0, 10, 20, 40, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)


        df['NDVI_Height_interaction'] = df['Pre_GSHH_NDVI'] * df['Height_Ave_cm']
        df['NDVI_squared'] = df['Pre_GSHH_NDVI'] ** 2
        df['Height_log'] = np.log1p(df['Height_Ave_cm'])

        if len(df) > 100:
            df['NDVI_species_mean'] = df.groupby('Species')['Pre_GSHH_NDVI'].transform('mean')
            df['Height_species_mean'] = df.groupby('Species')['Height_Ave_cm'].transform('mean')
            df['NDVI_state_mean'] = df.groupby('State')['Pre_GSHH_NDVI'].transform('mean')
        return df

    def detect_outliers(self,df,targets):
        df = df.copy()

        for target in targets:
            if target in df.columns:
                Q1 = df[target].quantile(0.25)
                Q2 = df[target].quantile(0.75)

                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Mark outliers but don't remove (model can learn from them)
                df[f'{target}_is_outlier'] = (
                    (df[target] < lower_bound) | (df[target] > upper_bound)
                ).astype(int)
        return df

    def fit_transform(self,df,targets):
        df = df.copy()

        df = self.engineer_features(df)

        df = self.detect_outliers(df,targets)

        for col in ["State","Species"]:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])

            numeric_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'NDVI_Height_interaction',
                       'NDVI_squared', 'Height_log']

        if 'NDVI_species_mean' in df.columns:
            numeric_cols.extend(['NDVI_species_mean', 'Height_species_mean', 'NDVI_state_mean'])
        
        for col in numeric_cols:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
            else:
                df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])

        self.feature_stats['num_species'] = len(self.label_encoders['Species'].classes_)
        self.feature_stats['num_states'] = len(self.label_encoders['State'].classes_)
        return df
    def transform(self, df):
        """Transform test data using fitted encoders/scalers"""
        df = df.copy()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categorical
        for col in ['State', 'Species']:
            # Handle unseen categories
            df[f'{col}_encoded'] = df[col].map(
                lambda x: self.label_encoders[col].transform([x])[0]
                if x in self.label_encoders[col].classes_
                else -1
            )
        
        # Scale numeric
        numeric_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'NDVI_Height_interaction',
                       'NDVI_squared', 'Height_log']
        
        if 'NDVI_species_mean' in df.columns:
            numeric_cols.extend(['NDVI_species_mean', 'Height_species_mean', 'NDVI_state_mean'])
        
        for col in numeric_cols:
            if col in self.scalers:
                df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
        
        return df
