"""Pydantic models for data structures."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ModLoader(str, Enum):
    """Supported mod loaders."""
    FORGE = "forge"
    FABRIC = "fabric"
    QUILT = "quilt"
    NEOFORGE = "neoforge"


class ModCategory(str, Enum):
    """Mod categories."""
    TECHNOLOGY = "technology"
    ADVENTURE = "adventure"
    MAGIC = "magic"
    DECORATION = "decoration"
    UTILITY = "utility"
    OPTIMIZATION = "optimization"
    LIBRARY = "library"
    FOOD = "food"
    STORAGE = "storage"
    TRANSPORTATION = "transportation"


class ModInfo(BaseModel):
    """Information about a single mod."""

    id: str
    slug: str
    title: str
    description: str
    categories: List[str]
    versions: List[str]
    loaders: List[str]
    downloads: int
    follows: int
    created: str
    updated: str
    license: Optional[str] = None
    source_url: Optional[str] = None
    wiki_url: Optional[str] = None
    discord_url: Optional[str] = None


class ModpackRequest(BaseModel):
    """User request for modpack generation."""

    minecraft_version: str
    mod_loader: ModLoader
    theme: Optional[str] = None
    categories: List[ModCategory] = Field(default_factory=list)
    performance_focus: bool = False
    lightweight: bool = False
    max_mods: int = Field(default=50, ge=10, le=200)
    additional_requirements: Optional[str] = None


class GeneratedModpack(BaseModel):
    """Generated modpack response."""

    name: str
    description: str
    minecraft_version: str
    mod_loader: str
    mods: List[Dict[str, Any]]
    total_mods: int
    estimated_performance_impact: str
    installation_notes: Optional[str] = None
    compatibility_warnings: List[str] = Field(default_factory=list)
