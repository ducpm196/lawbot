from typing import Dict, List, Any, Optional
from pathlib import Path

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field

"""
Pydantic Schemas for LawBot Configuration
=========================================

Defines the data models for validating the application's configuration.
"""


class PathsConfig(BaseModel):
    """Schema for core project paths."""

    root_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    model_dir: Path
    reports_dir: Path
    log_dir: Path
    feature_dir: Path
    train_data_path: Path
    test_data_path: Path
    legal_corpus_path: Path
    faiss_index_path: Path
    content_map_path: Path
    index_to_aid_path: Path


class ContrastiveLearningConfig(BaseModel):
    """Schema for Contrastive Learning enhancement settings."""

    enabled: bool
    loss_type: str
    max_length: PositiveInt
    batch_size: PositiveInt
    epochs: PositiveInt
    margin: float
    distance_metric: str


class ADAPTConfig(BaseModel):
    """Schema for ADAPT domain adaptation settings."""

    enabled: bool
    domain_data_ratio: float
    adaptation_steps: PositiveInt
    learning_rate: PositiveFloat
    warmup_steps: int


class BiEncoderConfig(BaseModel):
    """Schema for Bi-Encoder model settings with Contrastive Learning + ADAPT."""

    model_name: str
    enhancement_techniques: Dict[str, Any]  # Contrastive Learning + ADAPT configs
    training_params: Dict[str, Any]  # Standard training parameters


class CombinedRerankerModelConfig(BaseModel):
    """Schema for a single model in combined reranker."""

    enabled: bool
    weight: float
    model_name: str
    max_length: PositiveInt
    enhancement: str
    purpose: str
    status: str


class CombinedRerankerConfig(BaseModel):
    """Schema for combined reranker ensemble."""

    enabled: bool
    ensemble_method: str
    enhancement_strategy: str
    models: Dict[str, CombinedRerankerModelConfig]
    top_k: PositiveInt
    performance_balance: str
    ensemble_strategy: str


class RerankerPipelineConfig(BaseModel):
    """Schema for the multi-stage reranking pipeline."""

    combined_reranker: CombinedRerankerConfig
    cross_encoder: Optional[Dict[str, Any]] = None  # Legacy support
    light_reranker: Optional[Dict[str, Any]] = None  # Legacy support


class HPOConfig(BaseModel):
    """Schema for hyperparameter optimization settings."""

    enabled: bool
    framework: str
    bi_encoder: Dict[str, Any]
    light_ranking: Dict[str, Any]
    combined_reranker: Dict[str, Any]


class AppConfig(BaseModel):
    """Schema for application and search settings."""

    top_k_retrieval: PositiveInt
    top_k_light: PositiveInt  # ✅ Thêm top_k_light
    top_k_final: PositiveInt
    cache_questions_ttl_seconds: int
    cache_pipeline_ttl_seconds: int
    reranker_weights: Dict[str, float]
    
    # HOÀN TRẢ về ban đầu: KHÔNG có threshold parameters
    # retrieval_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    # light_reranker_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    # cross_encoder_threshold: float = Field(default=0.05, ge=0.0, le=1.0)


class PipelineTierConfig(BaseModel):
    """Schema for pipeline tier configuration."""

    name: str
    purpose: str
    techniques: List[str]
    performance: str
    enhancement: Optional[str] = None
    combination: Optional[str] = None
    strategy: Optional[str] = None


class PipelineConfig(BaseModel):
    """Schema for pipeline configuration."""

    version: str
    architecture: str
    enhancement_strategy: str
    tiers: Dict[str, PipelineTierConfig]


class LoggingConfig(BaseModel):
    """Schema for logging configuration."""

    level: str
    format: str
    max_size_mb: PositiveInt
    backup_count: PositiveInt
    console_level: str
    file_level: str


class Config(BaseModel):
    """Root schema for the entire application configuration."""

    paths: PathsConfig
    bi_encoder: BiEncoderConfig
    reranker_pipeline: RerankerPipelineConfig
    hyperparameter_optimization: HPOConfig
    app: AppConfig
    logging: LoggingConfig
    pipeline: PipelineConfig
