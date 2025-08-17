"""Command-line interface for the Modpack AI."""

import click
import asyncio
from pathlib import Path
from loguru import logger

from modpack.config.settings import AppConfig
from modpack.data.modrinth_scraper import ModrinthScraper
from modpack.data.dataset_creator import DatasetCreator
from modpack.training.llama_trainer import LLaMATrainer
from modpack.modpack import ModpackAI
from modpack.models.schemas import ModpackRequest, ModLoader, ModCategory


@click.group()
def cli():
    """Minecraft Modpack AI with LLaMA 3 Fine-tuning."""
    pass


@cli.command()
@click.option('--max-mods', default=5000, help='Maximum number of mods to scrape')
@click.option('--output-file', default='mods_data.json', help='Output file name')
def scrape_mods(max_mods: int, output_file: str):
    """Scrape mod data from Modrinth."""
    config = AppConfig()

    async def run_scraper():
        async with ModrinthScraper(config.data_config) as scraper:
            mods = await scraper.scrape_all_mods(max_mods=max_mods)
            scraper.save_mods_to_file(mods, output_file)

    asyncio.run(run_scraper())
    logger.info(f"Scraping completed. Data saved to {output_file}")


@cli.command()
@click.option('--mods-file', required=True, help='Path to scraped mods JSON file')
@click.option('--num-examples', default=1000, help='Number of training examples to generate')
@click.option('--output-file', default='training_data.json', help='Output training data file')
def create_dataset(mods_file: str, num_examples: int, output_file: str):
    """Create training dataset from scraped mod data."""
    config = AppConfig()
    creator = DatasetCreator(config)

    mods_path = Path(mods_file)
    output_path = Path("data/processed") / output_file

    creator.create_training_dataset(mods_path, output_path, num_examples)
    logger.info(f"Dataset created: {output_path}")


@cli.command()
@click.option('--data-file', help='Path to training data file')
def train_model(data_file: str):
    """Train the LLaMA model on modpack data."""
    config = AppConfig()

    if data_file:
        config.model_config.training_data_path = Path(data_file)

    # Pass ModelConfig directly
    trainer = LLaMATrainer(config.model_llm_config)
    trainer.run_full_training()
    logger.info("Training completed")


@cli.command()
@click.option('--minecraft-version', required=True, help='Minecraft version (e.g., 1.20.1)')
@click.option('--mod-loader', required=True, type=click.Choice(['forge', 'fabric', 'quilt', 'neoforge']), help='Mod loader')
@click.option('--theme', help='Modpack theme/focus')
@click.option('--categories', help='Comma-separated categories')
@click.option('--performance-focus', is_flag=True, help='Focus on performance')
@click.option('--lightweight', is_flag=True, help='Lightweight modpack')
@click.option('--max-mods', default=50, help='Maximum number of mods')
@click.option('--requirements', help='Additional requirements')
def generate(minecraft_version: str, mod_loader: str, theme: str, categories: str,
             performance_focus: bool, lightweight: bool, max_mods: int, requirements: str):
    """Generate a modpack using the trained model."""
    config = AppConfig()
    generator = ModpackAI(config.model_config)

    # Parse categories
    category_list = []
    if categories:
        for cat in categories.split(','):
            cat = cat.strip().lower()
            try:
                category_list.append(ModCategory(cat))
            except ValueError:
                logger.warning(f"Unknown category: {cat}")

    request = ModpackRequest(
        minecraft_version=minecraft_version,
        mod_loader=ModLoader(mod_loader),
        theme=theme,
        categories=category_list,
        performance_focus=performance_focus,
        lightweight=lightweight,
        max_mods=max_mods,
        additional_requirements=requirements
    )

    modpack = generator.generate_modpack(request)

    if modpack:
        click.echo("Generated Modpack:")
        click.echo("=" * 50)
        click.echo(f"Name: {modpack.name}")
        click.echo(f"Description: {modpack.description}")
        click.echo(f"Minecraft Version: {modpack.minecraft_version}")
        click.echo(f"Mod Loader: {modpack.mod_loader}")
        click.echo(f"Total Mods: {modpack.total_mods}")
        click.echo(f"Performance Impact: {modpack.estimated_performance_impact}")

        if modpack.installation_notes:
            click.echo(f"Installation Notes: {modpack.installation_notes}")

        if modpack.compatibility_warnings:
            click.echo("Compatibility Warnings:")
            for warning in modpack.compatibility_warnings:
                click.echo(f"  - {warning}")

        output_file = f"generated_modpack_{modpack.minecraft_version}_{modpack.mod_loader}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modpack.model_dump_json(indent=2))

        click.echo(f"\nFull modpack data saved to: {output_file}")
    else:
        click.echo("Failed to generate modpack")


@cli.command()
def interactive():
    """Interactive modpack generation."""
    config = AppConfig()
    generator = ModpackAI(config.model_config)

    click.echo("Welcome to the Minecraft Modpack Generator!")
    click.echo("=" * 50)

    # Get user preferences
    minecraft_version = click.prompt(
        "Minecraft version",
        type=click.Choice(config.supported_versions),
        default="1.20.1"
    )

    mod_loader = click.prompt(
        "Mod loader",
        type=click.Choice(config.supported_loaders),
        default="forge"
    )

    theme = click.prompt("Modpack theme/focus (optional)", default="", show_default=False)
    if not theme:
        theme = None

    performance_focus = click.confirm("Focus on performance?", default=False)
    lightweight = click.confirm("Lightweight modpack?", default=False)
    max_mods = click.prompt("Maximum number of mods", default=50, type=int)

    requirements = click.prompt("Additional requirements (optional)", default="", show_default=False)
    if not requirements:
        requirements = None

    # Create request
    request = ModpackRequest(
        minecraft_version=minecraft_version,
        mod_loader=ModLoader(mod_loader),
        theme=theme,
        performance_focus=performance_focus,
        lightweight=lightweight,
        max_mods=max_mods,
        additional_requirements=requirements
    )

    click.echo("\nGenerating modpack...")
    modpack = generator.generate_modpack(request)

    if modpack:
        click.echo("\n" + "=" * 60)
        click.echo("GENERATED MODPACK")
        click.echo("=" * 60)
        click.echo(f"🎮 Name: {modpack.name}")
        click.echo(f"📝 Description: {modpack.description}")
        click.echo(f"⚙️  Minecraft Version: {modpack.minecraft_version}")
        click.echo(f"🔧 Mod Loader: {modpack.mod_loader.title()}")
        click.echo(f"📦 Total Mods: {modpack.total_mods}")
        click.echo(f"⚡ Performance Impact: {modpack.estimated_performance_impact}")

        click.echo(f"\n📋 MODS LIST:")
        for i, mod in enumerate(modpack.mods[:10], 1):
            click.echo(f"  {i:2d}. {mod['name']}")

        if len(modpack.mods) > 10:
            click.echo(f"     ... and {len(modpack.mods) - 10} more mods")

        # Save to file
        timestamp = click.DateTime().now().strftime("%Y%m%d_%H%M%S")
        output_file = f"modpack_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modpack.model_dump_json(indent=2))

        click.echo(f"\n💾 Full modpack saved to: {output_file}")

        if click.confirm("\nWould you like to generate another modpack?"):
            interactive()
    else:
        click.echo("❌ Failed to generate modpack. Please try again.")


if __name__ == "__main__":
    cli()
