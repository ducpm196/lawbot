#!/usr/bin/env python3
"""
LawBot Pages Package
====================

Page modules for LawBot application.
"""

__version__ = "8.3"
__description__ = "Page modules for LawBot application"

# Import page modules
try:
    from . import search
    from . import analysis
    from . import system
except ImportError:
    # Allow partial imports for development
    pass

__all__ = [
    "search",
    "analysis", 
    "system"
]
