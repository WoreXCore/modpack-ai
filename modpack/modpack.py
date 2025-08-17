"""Modpack AI using fine-tuned LLaMA model."""

import json
import re
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger

from modpack.config.settings import ModelConfig
from modpack.models.schemas import ModpackRequest, GeneratedModpack


class ModpackAI:
    """Generator for creating modpacks using fine-tuned LLaMA model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the fine-tuned model."""
        logger.info("Loading fine-tuned model")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load fine-tuned weights
        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.config.output_dir),
            torch_dtype=torch.float16
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.output_dir),
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded successfully")

    def generate_modpack(self, request: ModpackRequest) -> Optional[GeneratedModpack]:
        """Generate a modpack based on user request."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Create prompt from request
        prompt = self._create_prompt(request)

        # Generate response
        response = self._generate_response(prompt)

        # Parse response to extract modpack data
        modpack_data = self._parse_response(response)

        if modpack_data:
            return GeneratedModpack(**modpack_data)
        else:
            logger.error("Failed to parse generated modpack")
            return None

    def _create_prompt(self, request: ModpackRequest) -> str:
        """Create a prompt from the user request."""
        theme_parts = []

        if request.theme:
            theme_parts.append(request.theme)

        if request.categories:
            categories_str = ", ".join([cat.value for cat in request.categories])
            theme_parts.append(f"focusing on {categories_str}")

        if request.performance_focus:
            theme_parts.append("with good performance")

        if request.lightweight:
            theme_parts.append("that's lightweight")

        theme = " ".join(theme_parts) if theme_parts else "general gameplay"

        prompt = f"Create a modpack for Minecraft {request.minecraft_version} using {request.mod_loader.value} with a focus on {theme}"

        if request.max_mods != 50:
            prompt += f" with around {request.max_mods} mods"

        if request.additional_requirements:
            prompt += f". Additional requirements: {request.additional_requirements}"

        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        return formatted_prompt

    def _generate_response(self, prompt: str) -> str:
        """Generate response using the fine-tuned model."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=2048,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        response = response.replace(prompt, "").strip()

        return response

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the generated response to extract modpack data."""
        try:
            # Look for JSON data between <modpack> tags
            json_match = re.search(r'<modpack>\s*(.*?)\s*</modpack>', response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
                modpack_data = json.loads(json_str)
                return modpack_data
            else:
                logger.warning("No modpack JSON found in response")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
