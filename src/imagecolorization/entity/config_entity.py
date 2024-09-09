from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_dir : Path
    local_data_file: Path
    unzip_dir : Path
    
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    data_path_black : Path
    data_path_grey : Path
    BATCH_SIZE : int
    IMAGE_SIZE : list
    DATA_RANGE: int
    
    
@dataclass(frozen=True)
class ModelBuildingConfig:
    root_dir: Path
    KERNEL_SIZE_RES: int
    PADDING: int
    STRIDE: int
    BIAS: bool
    SCALE_FACTOR: int
    DIM: int
    DROPOUT_RATE: float
    KERNEL_SIZE_GENERATOR: int
    INPUT_CHANNELS: int
    OUTPUT_CHANNELS: int
    IN_CHANNELS: int
    
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    test_data_path: Path
    train_data_path: Path
    LEARNING_RATE: float
    LAMBDA_RECON: int
    DISPLAY_STEP: int
    IMAGE_SIZE: list
    INPUT_CHANNELS: int
    OUTPUT_CHANNELS: int
    EPOCH: int
    BATCH_SIZE : int
    
    
@dataclass(frozen=True)
class ModelEvalutaionConfig:
    test_data : Path
    generator_model : Path
    critic_model : Path
    all_params: dict