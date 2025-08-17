"""Fine-tuning script for LLaMA model."""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger

from ..config.settings import ModelConfig


class LLaMATrainer:
    """Trainer class for fine-tuning LLaMA model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model.config, "pretraining_tp"):
            self.model.config.pretraining_tp = 1

        logger.info("Model and tokenizer loaded successfully")

    def setup_lora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA setup completed")

    def prepare_dataset(self, data_path: str):
        """Prepare dataset for training."""
        logger.info(f"Loading training data from {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        formatted_data = [
            {"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"}
            for item in data
        ]

        dataset = Dataset.from_list(formatted_data)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def train(self, dataset):
        """Train the model."""
        logger.info("Starting training")

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            bf16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=None
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.config.output_dir))

        logger.info(f"Training completed. Model saved to {self.config.output_dir}")

    def run_full_training(self):
        """Run the complete training pipeline."""
        self.load_model_and_tokenizer()
        self.setup_lora()
        dataset = self.prepare_dataset(str(self.config.training_data_path))
        self.train(dataset)
