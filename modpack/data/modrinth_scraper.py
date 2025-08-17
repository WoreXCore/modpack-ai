"""Scraper for collecting mod data from Modrinth API."""

import asyncio
import json
from typing import List, Dict, Any, Optional

import aiohttp
from loguru import logger
from tqdm.asyncio import tqdm

from ..config.settings import DataCollectionConfig
from ..models.schemas import ModInfo


class ModrinthScraper:
    """Asynchronous scraper for Modrinth mod data."""

    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

    async def __aenter__(self):
        """Async context manager entry."""
        headers = {
            "User-Agent": "worexcore/modpack-ai/0.1.0 (worexgriefofficial@gmail.com)"
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    @staticmethod
    async def _handle_ratelimit(response) -> bool:
        """Check ratelimit headers and sleep if needed. Returns True if sleep was performed."""
        ratelimit_remaining = response.headers.get("X-Ratelimit-Remaining")
        ratelimit_reset = response.headers.get("X-Ratelimit-Reset")
        if ratelimit_remaining is not None and ratelimit_reset is not None:
            if int(ratelimit_remaining) == 0:
                logger.warning(f"Ratelimit reached, sleeping for {ratelimit_reset}s")
                await asyncio.sleep(float(ratelimit_reset))
                return True
        return False

    async def fetch_mod_list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch list of mods from Modrinth with proactive ratelimit handling and retry on 429."""
        url = f"{self.config.base_url}/search"
        params = {
            "limit": limit,
            "offset": offset,
            "facets": '[ ["project_type:mod"] ]'
        }
        max_retries = 5
        base_delay = 1
        attempt = 0

        async with self.semaphore:
            while attempt < max_retries:
                try:
                    async with self.session.get(url, params=params) as response:
                        if await self._handle_ratelimit(response):
                            continue
                        if response.status == 200:
                            data = await response.json()
                            return data.get("hits", [])
                        elif response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            ratelimit_reset = response.headers.get("X-Ratelimit-Reset")
                            if retry_after:
                                delay = float(retry_after)
                            elif ratelimit_reset:
                                delay = float(ratelimit_reset)
                            else:
                                delay = base_delay * (2 ** attempt)
                            logger.warning(f"Rate limited on mod list, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            error_text = await response.text()
                            logger.error(f"Failed to fetch mod list: {response.status}")
                            logger.error(error_text)
                            return []
                except Exception as e:
                    logger.error(f"Error fetching mod list: {e}")
                    return []
            logger.error("Exceeded max retries for mod list fetch")
            return []

    async def fetch_mod_details(self, mod_id: str) -> Optional[ModInfo]:
        """Fetch detailed information about a specific mod with proactive ratelimit handling and retry on 429."""
        url = f"{self.config.base_url}/project/{mod_id}"
        max_retries = 5
        base_delay = 1  # seconds
        attempt = 0

        async with self.semaphore:
            while attempt < max_retries:
                try:
                    async with self.session.get(url) as response:
                        if await self._handle_ratelimit(response):
                            continue
                        if response.status == 200:
                            data = await response.json()
                            return ModInfo(
                                id=data["id"],
                                slug=data["slug"],
                                title=data["title"],
                                description=data["description"],
                                categories=data.get("categories", []),
                                versions=data.get("game_versions", []),
                                loaders=data.get("loaders", []),
                                downloads=data.get("downloads", 0),
                                follows=data.get("followers", 0),
                                created=data.get("published", ""),
                                updated=data.get("updated", ""),
                                license=data.get("license", {}).get("id") if data.get("license") else None,
                                source_url=data.get("source_url"),
                                wiki_url=data.get("wiki_url"),
                                discord_url=data.get("discord_url")
                            )
                        elif response.status == 429:
                            retry_after = response.headers.get("Retry-After")
                            ratelimit_reset = response.headers.get("X-Ratelimit-Reset")
                            if retry_after:
                                delay = float(retry_after)
                            elif ratelimit_reset:
                                delay = float(ratelimit_reset)
                            else:
                                delay = base_delay * (2 ** attempt)
                            logger.warning(f"Rate limited on mod {mod_id}, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            logger.warning(f"Failed to fetch mod {mod_id}: {response.status}")
                            return None
                except Exception as e:
                    logger.error(f"Error fetching mod {mod_id}: {e}")
                    return None
            logger.error(f"Exceeded max retries for mod {mod_id}")
            return None


    async def scrape_all_mods(self, max_mods: int = 10000) -> List[ModInfo]:
        """Scrape all available mods up to max_mods limit."""
        logger.info(f"Starting to scrape up to {max_mods} mods from Modrinth")

        # First, get the list of all mods
        all_mod_ids = []
        offset = 0
        limit = 100

        while len(all_mod_ids) < max_mods:
            mod_list = await self.fetch_mod_list(limit=limit, offset=offset)
            if not mod_list:
                break

            mod_ids = [mod["project_id"] for mod in mod_list]
            all_mod_ids.extend(mod_ids)

            offset += limit
            logger.info(f"Collected {len(all_mod_ids)} mod IDs so far")

            if len(mod_list) < limit:  # No more results
                break

        # Limit to max_mods
        all_mod_ids = all_mod_ids[:max_mods]
        logger.info(f"Fetching details for {len(all_mod_ids)} mods")

        # Fetch detailed information for each mod
        tasks = [self.fetch_mod_details(mod_id) for mod_id in all_mod_ids]
        results = await tqdm.gather(*tasks, desc="Fetching mod details")

        # Filter out None results
        mods = [mod for mod in results if mod is not None]
        logger.info(f"Successfully scraped {len(mods)} mods")

        return mods

    def save_mods_to_file(self, mods: List[ModInfo], filename: str = "mods_data.json"):
        """Save scraped mods to JSON file."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.config.output_dir / filename

        # Convert to dict format for JSON serialization
        mods_data = [mod.model_dump() for mod in mods]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mods_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(mods)} mods to {filepath}")