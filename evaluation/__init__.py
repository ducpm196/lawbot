"""
Evaluation Module - LawBot v8.1
===============================

Comprehensive evaluation components for LawBot including:
- Comprehensive metrics calculation for all 3 tiers
- Batch evaluation for multiple queries
- Performance reporting and analysis
- Multi-tier pipeline evaluation with detailed reporting
"""

from .metrics import MetricsCalculator, BatchEvaluator, EvaluationReporter
from .run_evaluation import generate_comprehensive_evaluation

__all__ = [
    # Core metrics
    "MetricsCalculator",
    "BatchEvaluator",
    "EvaluationReporter",
    # Evaluation functions
    "generate_comprehensive_evaluation",
]
