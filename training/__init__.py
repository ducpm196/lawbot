"""
Training Module - LawBot v8.2
=============================

Training components for LawBot including:
- Bi-encoder training for retrieval
- Light ranking training with hard negative mining
- Cross-encoder training for final reranking
- Hyperparameter optimization
- Unified training engine
"""

# Import only what exists to avoid import errors
try:
    from .base_script import BaseTrainingScript
except ImportError:
    BaseTrainingScript = None

try:
    from .engine import TrainingEngine
except ImportError:
    TrainingEngine = None

try:
    from .hpo import HyperparameterOptimizer
except ImportError:
    HyperparameterOptimizer = None

try:
    from .run_light_ranking import LightRankingTrainer
except ImportError:
    LightRankingTrainer = None

try:
    from .run_reranker import CrossEncoderTrainer
except ImportError:
    CrossEncoderTrainer = None

try:
    from .run_create_faiss_index import create_faiss_index
except ImportError:
    create_faiss_index = None

# Build __all__ dynamically based on what was successfully imported
__all__ = []

if BaseTrainingScript:
    __all__.append("BaseTrainingScript")
if TrainingEngine:
    __all__.append("TrainingEngine")
if HyperparameterOptimizer:
    __all__.append("HyperparameterOptimizer")
if LightRankingTrainer:
    __all__.append("LightRankingTrainer")
if CrossEncoderTrainer:
    __all__.append("CrossEncoderTrainer")
if create_faiss_index:
    __all__.append("create_faiss_index")
