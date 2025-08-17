# Modpack AI with LLaMA 3 Fine-tuning

Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

## Features

- 🕷️ **Async Modrinth Scraper**: Efficiently scrapes mod data from Modrinth API
- 🧠 **LLaMA Fine-tuning**: Uses LoRA for efficient fine-tuning without full model retraining
- 🎯 **Smart Modpack Generation**: Generates themed modpacks with compatibility checking
- 📊 **Scalable Architecture**: Clean, maintainable code structure following best practices
- 🔧 **CLI Interface**: Easy-to-use command-line interface with interactive mode
 
## Installation

```bash
# Clone the repository
git clone https://github.com/WoreXCore/modpack-ai.git
cd modpack-ai

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry env activate
```

### Step-by-Step Usage

1. **Scrape mod data from Modrinth:**
```bash
poetry run python -m modpack.main scrape-mods --max-mods 3000
```

2. **Create training dataset:**
```bash
poetry run python -m modpack.main create-dataset --mods-file data/raw/mods_data.json --num-examples 1000
```

3. **Train the model:**
```bash
poetry run -m modpack.main train-model
```

4. **Generate a modpack:**
```bash
python -m modpack.main generate \
    --minecraft-version 1.20.1 \
    --mod-loader forge \
    --theme "technology and automation" \
    --max-mods 30
```

5. **Interactive mode:**
```bash
python -m modpack.cli.main interactive
```

## Configuration

Create a `.env` file to customize settings:

```bash
# Data Collection
DATA_BASE_URL=https://api.modrinth.com/v2
DATA_RATE_LIMIT_DELAY=0.5
DATA_MAX_CONCURRENT_REQUESTS=10

# Model Training
MODEL_NAME=meta-llama/Llama-3.1-8B
MODEL_BATCH_SIZE=4
MODEL_LEARNING_RATE=2e-4
MODEL_NUM_EPOCHS=3
MODEL_LORA_R=16
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4060 Ti or better)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models and data