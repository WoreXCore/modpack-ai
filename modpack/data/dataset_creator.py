"""Create training dataset from scraped mod data."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
import pandas as pd

from ..models.schemas import ModInfo, ModLoader, ModCategory
from ..config.settings import AppConfig


class DatasetCreator:
    """Creates training dataset for LLaMA fine-tuning."""

    def __init__(self, config: AppConfig):
        self.config = config

    def load_mods_data(self, filepath: Path) -> List[ModInfo]:
        """Load mod data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [ModInfo(**mod_data) for mod_data in data]

    def generate_modpack_examples(self, mods: List[ModInfo], num_examples: int = 1000) -> List[Dict[str, str]]:
        """Generate training examples for modpack creation."""
        examples = []

        # Group mods by categories and compatibility
        mods_by_version = {}
        mods_by_loader = {}

        for mod in mods:
            for version in mod.versions:
                if version not in mods_by_version:
                    mods_by_version[version] = []
                mods_by_version[version].append(mod)

            for loader in mod.loaders:
                if loader not in mods_by_loader:
                    mods_by_loader[loader] = []
                mods_by_loader[loader].append(mod)

        logger.info(f"Generating {num_examples} training examples")

        for _ in range(num_examples):
            # Randomly select minecraft version and loader
            version = random.choice(self.config.supported_versions)
            loader = random.choice(self.config.supported_loaders)

            # Filter compatible mods
            compatible_mods = []
            for mod in mods:
                if version in mod.versions and loader in mod.loaders:
                    compatible_mods.append(mod)

            if len(compatible_mods) < 5:  # Need minimum mods for a pack
                continue

            # Generate a themed modpack
            theme_templates = [
                "technology and automation",
                "magic and adventure",
                "exploration and building",
                "performance optimization",
                "kitchen sink (all-in-one)",
                "lightweight gameplay",
                "hardcore survival",
                "creative building"
            ]

            theme = random.choice(theme_templates)
            num_mods = random.randint(10, min(50, len(compatible_mods)))

            # Select mods based on theme and popularity
            selected_mods = self._select_mods_for_theme(compatible_mods, theme, num_mods)

            # Create training example
            user_prompt = self._generate_user_prompt(version, loader, theme)
            assistant_response = self._generate_assistant_response(selected_mods, version, loader, theme)

            examples.append({
                "instruction": user_prompt,
                "response": assistant_response
            })

        logger.info(f"Generated {len(examples)} training examples")
        return examples

    @staticmethod
    def _select_mods_for_theme(mods: List[ModInfo], theme: str, num_mods: int) -> List[ModInfo]:
        """Select mods that fit the given theme."""
        # Define theme keywords
        theme_keywords = {
            "technology": ["tech", "machine", "automation", "industrial", "energy", "power"],
            "magic": ["magic", "spell", "wizard", "arcane", "mystical", "enchant"],
            "exploration": ["biome", "dimension", "world", "generation", "explore"],
            "building": ["decoration", "furniture", "architecture", "build", "construct"],
            "performance": ["optimization", "fps", "performance", "boost", "fast"],
            "survival": ["hardcore", "difficult", "challenge", "realistic", "tough"]
        }

        # Score mods based on theme relevance
        scored_mods = []
        for mod in mods:
            score = mod.downloads + mod.follows  # Base popularity score

            # Add theme bonus
            theme_score = 0
            for keyword_category, keywords in theme_keywords.items():
                if keyword_category in theme.lower():
                    for keyword in keywords:
                        if keyword in mod.title.lower() or keyword in mod.description.lower():
                            theme_score += 100
                        if keyword in mod.categories:
                            theme_score += 200

            scored_mods.append((mod, score + theme_score))

        # Sort by score and select top mods
        scored_mods.sort(key=lambda x: x[1], reverse=True)
        selected = [mod for mod, _ in scored_mods[:num_mods]]

        # Ensure we have essential mods (libraries, optimization)
        essential_keywords = ["library", "api", "core", "optimization"]
        essential_mods = []

        for mod in mods:
            if any(keyword in mod.title.lower() or keyword in mod.categories for keyword in essential_keywords):
                if mod not in selected and len(essential_mods) < 5:
                    essential_mods.append(mod)

        # Replace some random mods with essential ones
        if essential_mods:
            replace_count = min(len(essential_mods), len(selected) // 4)
            for i in range(replace_count):
                selected[-(i+1)] = essential_mods[i]

        return selected

    @staticmethod
    def _generate_user_prompt(version: str, loader: str, theme: str) -> str:
        """Generate a user prompt for the training example."""
        prompts = [
            f"Create a modpack for Minecraft {version} using {loader} with a focus on {theme}",
            f"I want a {theme} modpack for Minecraft {version} ({loader})",
            f"Generate a modpack for {theme} gameplay on Minecraft {version} with {loader}",
            f"Help me create a {loader} modpack for Minecraft {version} focused on {theme}"
        ]

        base_prompt = random.choice(prompts)

        # Add additional requirements sometimes
        if random.random() < 0.3:
            extras = [
                " with good performance",
                " that's lightweight",
                " with popular mods",
                " for multiplayer",
                " for single player"
            ]
            base_prompt += random.choice(extras)

        return base_prompt

    @staticmethod
    def _generate_assistant_response(mods: List[ModInfo], version: str, loader: str, theme: str) -> str:
        """Generate assistant response in the desired format."""
        # Create modpack JSON
        modpack_data = {
            "name": f"{theme.title()} Pack",
            "description": f"A carefully curated {theme} modpack for Minecraft {version}",
            "minecraft_version": version,
            "mod_loader": loader,
            "mods": [
                {
                    "id": mod.id,
                    "name": mod.title,
                    "slug": mod.slug,
                    "description": mod.description[:200] + "..." if len(mod.description) > 200 else mod.description,
                    "categories": mod.categories,
                    "downloads": mod.downloads
                }
                for mod in mods
            ],
            "total_mods": len(mods),
            "estimated_performance_impact": random.choice(["Low", "Medium", "High"]),
            "compatibility_notes": "All mods have been tested for compatibility with the specified Minecraft version and mod loader."
        }

        response = f"I'll create a {theme} modpack for you! Here are the details:\n\n"
        response += f"**Modpack Name:** {modpack_data['name']}\n"
        response += f"**Minecraft Version:** {version}\n"
        response += f"**Mod Loader:** {loader.title()}\n"
        response += f"**Total Mods:** {len(mods)}\n\n"

        response += "<modpack>\n"
        response += json.dumps(modpack_data, indent=2)
        response += "\n</modpack>"

        return response

    def create_training_dataset(self, mods_file: Path, output_file: Path, num_examples: int = 1000):
        """Create complete training dataset."""
        # Load mods data
        mods = self.load_mods_data(mods_file)
        logger.info(f"Loaded {len(mods)} mods")

        # Generate training examples
        examples = self.generate_modpack_examples(mods, num_examples)

        # Format for training (Alpaca format)
        training_data = []
        for example in examples:
            training_data.append({
                "instruction": example["instruction"],
                "input": "",
                "output": example["response"]
            })

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(training_data)} training examples to {output_file}")
