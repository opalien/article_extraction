from __future__ import annotations

# Re-export key tables for convenience
from .base import Base
from .epoch_table import EpochTable
from .country_table import CountryTable
from .hardware_table import HardwareTable
from .paper_document_table import PaperDocumentTable
from .paper_text import PaperTextTable

__all__ = [
    "Base",
    "EpochTable",
    "CountryTable",
    "HardwareTable",
    "PaperDocumentTable",
    "PaperTextTable",
]



