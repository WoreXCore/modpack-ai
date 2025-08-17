from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class DataCollectionConfig(BaseSettings):
    """Configuration for data collection from Modrinth."""

    base_url: str = "https://api.modrinth.com/v2"
    rate_limit_delay: float = 0.5
    max_concurrent_requests: int = 10
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("data/raw")

    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=".env")


class ModelConfig(BaseSettings):
    """Configuration for LLaMA 3 model training."""
    model_name: str = "meta-llama/Llama-3.1-8B"
    output_dir: Path = Path("models/fine_tuned")
    training_data_path: Path = Path("data/processed/training_data.json")

    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_length: int = 2048

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    model_config = SettingsConfigDict(env_prefix="MODEL_", env_file=".env")


class AppConfig(BaseSettings):
    """Main application configuration."""
    data_config: DataCollectionConfig = DataCollectionConfig()
    model_llm_config: ModelConfig = ModelConfig()

    supported_versions: List[str] = ["1.20.1", "1.19.4", "1.19.2", "1.18.2", "1.16.5"]
    supported_loaders: List[str] = ["forge", "fabric", "quilt", "neoforge"]
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")
