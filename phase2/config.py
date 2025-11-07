class Config:
    TRAIN_CSV = "/kaggle/input/csiro-biomass/train.csv"
    TRAIN_IMG_DIR = "/kaggle/input/csiro-biomass/train"
    TEST_CSV = "/kaggle/input/csiro-biomass/test.csv"
    TEST_IMG_DIR = "/kaggle/input/csiro-biomass/test"


    IMG_SIZE = 384
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4

    EPOCHS = 45
    WARMUP_EPOCHS = 5
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    #augmentations
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    
    NUM_FOLDS = 3
    BACKBONES = ['tf_efficientnetv2_m',
        'convnext_tiny',
        'resnet50d']

    USE_PSEUDO_LABELING = True
    PSEUDO_THRESHOLD = 0.8
    PSEUDO_WEIGHT = 0.3


    USE_TTA = True
    TTA_AUGMENTATIONS = 5

    BIOMASS_TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    TARGET_WEIGHTS = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5
    }

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 2


config = Config()
