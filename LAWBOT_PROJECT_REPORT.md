# BÁO CÁO ĐỒ ÁN LAWBOOT - HỆ THỐNG AI HỎI ĐÁP PHÁP LUẬT VIỆT NAM
## Phiên bản: v8.3 | Ngày cập nhật: 2025-08-21 (Quality Score & Config Optimization)

---

## MỤC LỤC
1. [Tổng quan và Kiến trúc tổng thể](#1-tổng-quan-và-kiến-trúc-tổng-thể)
2. [Kiến trúc Pipeline 3 tầng](#2-kiến-trúc-pipeline-3-tầng)
3. [Luồng xử lý Training và MLOps](#3-luồng-xử-lý-training-và-mlops)
4. [Luồng xử lý Request User](#4-luồng-xử-lý-request-user)
5. [Giao diện và Báo cáo](#5-giao-diện-và-báo-cáo)
6. [Kỹ thuật Code và Kiến trúc](#6-kỹ-thuật-code-và-kiến-trúc)
7. [Hướng dẫn Vận hành](#7-hướng-dẫn-vận-hành)
8. [Đánh giá và Khuyến nghị](#8-đánh-giá-và-khuyến-nghị)
   - [8.1 Recent Updates (2025-08-21)](#-recent-updates-2025-08-21)
   - [8.2 Performance & Quality Improvements](#82-performance--quality-improvements)
   - [8.3 Technical Enhancements](#83-technical-enhancements)
9. [Documentation Architecture & Technical Guides](#9-documentation-architecture--technical-guides)
   - [9.1 Comprehensive Documentation Structure](#91-comprehensive-documentation-structure)
   - [9.2 Technical Implementation Details](#92-technical-implementation-details)
   - [9.3 Mathematical Foundations & Algorithms](#93-mathematical-foundations--algorithms)
10. [Phân tích Ưu Nhược điểm & Hạn chế Hệ thống](#10-phân-tích-ưu-nhược-điểm--hạn-chế-hệ-thống)

---

## 🆕 **RECENT UPDATES (2025-08-21)**

### **LawBot v8.3 - Production Ready với UI Optimization! 🚀**

**Ngày cập nhật:** 2025-08-21  
**Phiên bản:** v8.3  
**Status:** ✅ Production Ready  

#### **🎯 Vấn đề đã giải quyết:**
- **UI Reload Optimization**: Loại bỏ unnecessary reloads với form-based interaction
- **Session State Management**: Tối ưu state clearing để tránh element bleeding
- **Lazy Loading Architecture**: Tránh circular import với dynamic page loading
- **Form-based Controls**: Sử dụng `st.form` để ngăn auto-rerun
- **TTL-based Caching**: Cache pipeline và data với configurable TTL

#### **📈 Cải thiện hiệu suất (Actual Results):**
- **Config Consistency**: 100% sử dụng centralized config từ `config/default.yml`
- **UI Performance**: Giảm reload từ mỗi interaction xuống chỉ khi cần thiết
- **Memory Management**: Safe device handling cho meta tensors và offloaded models
- **Pipeline Architecture**: 3-tier architecture với independent training cho mỗi tier
- **Error Handling**: Graceful error handling cho import failures và device issues

#### **🔧 Chi tiết kỹ thuật:**
```python
# Lazy loading pattern để tránh circular import
def get_page_module(page_name):
    """Lazy load page modules to avoid circular import."""
    try:
        # Add current directory to Python path for direct import
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Direct import from pages directory
        if page_name == "search":
            from pages import search
            return search
        elif page_name == "analysis":
            from pages import analysis
            return analysis
        elif page_name == "system":
            from pages import system
            return system
        else:
            return None
    except ImportError as e:
        st.warning(f"⚠️ Could not load page {page_name}: {e}")
        return None

# Safe device handling cho meta tensors
def _safe_move_to_device(model, device):
    """Safely move model to device, handling meta tensors."""
    try:
        model.to(device)
    except NotImplementedError as e:
        if "meta tensor" in str(e).lower():
            logger.info(f"Detected meta tensor, using to_empty() for device: {device}")
            model.to_empty(device=device)
        else:
            raise
    return model
```

#### **✅ Kết quả cuối cùng (Actual Implementation):**
- **UI Architecture**: Single entry point với lazy loading và form-based interaction
- **Pipeline Architecture**: 3-tier với independent training và safe device handling
- **Config Management**: Centralized configuration với validation tự động
- **Production Ready**: Hệ thống đã sẵn sàng với UI optimization và stable pipeline

**📋 Xem chi tiết đầy đủ tại [Section 8.2: Performance & Quality Improvements](#82-performance--quality-improvements) và [Section 8.3: Technical Enhancements](#83-technical-enhancements)**

---

## 1. TỔNG QUAN VÀ KIẾN TRÚC TỔNG THỂ

### 1.1 Giới thiệu dự án

**LawBot** là một hệ thống AI hỏi đáp pháp luật Việt Nam được xây dựng với kiến trúc 3 tầng (3-tier architecture) hiện đại, tích hợp các kỹ thuật machine learning tiên tiến để cung cấp khả năng tìm kiếm và trả lời câu hỏi pháp luật chính xác và hiệu quả.

**Đặc điểm chính:**
- **Kiến trúc 3 tầng** với mỗi tầng được tối ưu hóa cho một nhiệm vụ cụ thể
- **Sử dụng models tiếng Việt chuyên biệt** (PhoBERT, Vietnamese Bi-Encoder)
- **Pipeline xử lý song song** với caching và optimization
- **Hỗ trợ đa ngôn ngữ** (chủ yếu tiếng Việt)
- **Giao diện web** sử dụng Streamlit
- **Real-data only**: Toàn bộ training sử dụng dữ liệu thật từ `data_processing/run_preparation.py`
- **Centralized Configuration**: Tập trung cấu hình tại `config/` với validation tự động
- **Advanced HPO**: Hyperparameter optimization với Optuna và early stopping
- **Comprehensive Evaluation**: Multi-tier evaluation với precision, recall, F1, NDCG, MRR, quality metrics
- **Performance Monitoring**: Real-time performance tracking và automated optimization
- **Unified Reports Storage**: Consolidated evaluation reports trong `reports/` directory
- **Config Optimization**: `top_k_final: 5` phù hợp với yêu cầu 3-5 kết quả cuối cùng
- **Quality Score Logic**: Tier-specific thresholds với adjusted scoring cho từng tier
- **Metrics Calculation**: Sửa logic effective_k để tính chính xác recall và quality scores
- **ADAPT Enhancement**: Domain adaptation cho pháp luật Việt Nam
- **Hard Negative Mining**: Adaptive threshold với intelligent mining
- **Ensemble Learning**: Weighted combination của multiple models
- **Automated Workflow**: Checkpoint management với recovery mechanisms

### 1.2 Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAWBOOT v8.3                            │
├─────────────────────────────────────────────────────────────────┤
│  🎯 USER INTERFACE LAYER                                      │
│  ├── Streamlit Web App (Single Entry Point)                   │
│  ├── Search Page (render_search_page)                         │
│  ├── Analysis Page (render_analysis_page)                     │
│  └── System Page (main)                                       │
│  ├── Lazy Loading Architecture                                │
│  ├── Form-based Interaction                                   │
│  └── Session State Optimization                               │
├─────────────────────────────────────────────────────────────────┤
│  🚀 PIPELINE LAYER (3-TIER ARCHITECTURE)                      │
│  ├── Tier 1: Vietnamese Bi-Encoder + FAISS                    │
│  ├── Tier 2: PhoBERT Light Reranker (Independent ADAPT)       │
│  └── Tier 3: Cross-Encoder Ensemble (ADAPT-enhanced)          │
│  ├── Safe Device Handling                                     │
│  ├── Meta Tensor Support                                      │
│  └── Independent Training                                     │
├─────────────────────────────────────────────────────────────────┤
│  🤖 MODEL LAYER                                               │
│  ├── Vietnamese Bi-Encoder (bkai-foundation-models)           │
│  ├── PhoBERT-base-v2 (Independent ADAPT)                      │
│  ├── PhoBERT-large (Base Model)                               │
│  └── FAISS Index Engine                                       │
├─────────────────────────────────────────────────────────────────┤
│  📊 DATA & EVALUATION LAYER                                   │
│  ├── Legal Corpus Management                                  │
│  ├── Training Data Processing                                 │
│  ├── Validation Sets                                          │
│  └── Performance Metrics                                      │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ INFRASTRUCTURE LAYER                                      │
│  ├── Centralized Configuration (config/default.yml)            │
│  ├── Logging & Monitoring                                     │
│  ├── TTL-based Caching                                        │
│  └── Error Handling & Recovery                                │
└─────────────────────────────────────────────────────────────────┘
```

**🔑 Đặc điểm chính của kiến trúc v8.3:**
- **Independent Training**: Mỗi tier được train độc lập, không kế thừa weights
- **Safe Device Handling**: Meta tensor handling với `_safe_move_to_device()`
- **Lazy Loading**: Tránh circular import với dynamic page loading
- **Form-based UI**: Ngăn auto-rerun với `st.form` controls
- **Centralized Config**: Tất cả config từ `config/default.yml`

### 1.3 Cấu trúc thư mục dự án

```
LawBot/
├── app/                          # Giao diện người dùng
│   ├── app.py                   # Main application entry point (Single Entry Point)
│   ├── __init__.py              # Package initialization
│   └── pages/                   # Các trang của ứng dụng
│       ├── __init__.py          # Pages package initialization
│       ├── search.py            # Trang tìm kiếm (render_search_page)
│       ├── analysis.py          # Trang phân tích (render_analysis_page)
│       └── system.py            # Trang hệ thống (main)
├── core/                        # Core engine và pipeline
│   ├── pipeline.py              # Main pipeline orchestrator (LegalQAPipeline)
│   ├── retrieval.py             # Tier 1: Retrieval engine (RetrievalEngine)
│   ├── reranking.py             # Tier 2 & 3: Reranking engine (RerankingEngine)
│   ├── datasets/                # Dataset classes
│   ├── transforms/              # Data transformation utilities
│   ├── utils/                   # Utility functions
│   │   ├── logging_manager.py   # Centralized logging
│   │   ├── parent_law_manager.py # Parent law mapping
│   │   ├── system_check.py      # System health monitoring
│   │   └── versioning.py        # Model version management
│   └── progress_tracker.py      # Progress tracking
├── config/                      # Cấu hình hệ thống
│   ├── default.yml              # Cấu hình mặc định (v8.3)
│   ├── loader.py                # Configuration loader
│   ├── paths.py                 # Centralized paths & validation
│   ├── models.py                # Centralized model configuration
│   └── schemas.py               # Pydantic schemas
├── training/                    # Training scripts và engine
│   ├── engine.py                # Training engine
│   ├── base_script.py           # Base training script
│   ├── run_bi_encoder.py        # Bi-encoder training
│   ├── run_light_ranking.py     # Light reranker training
│   ├── run_reranker.py          # Cross-encoder training
│   ├── hpo.py                   # Hyperparameter optimization
│   ├── hard_negative_mining.py  # HNM implementation
│   └── validation_sets.py       # Validation set management
├── data_processing/             # Data preparation
├── evaluation/                  # Evaluation và metrics
├── features/                    # Processed features và models
│   ├── faiss_index.bin          # FAISS index file
│   ├── aid_map.json             # Content mapping
│   └── index_to_aid.json       # Index to AID mapping
├── models/                      # Trained models
├── reports/                     # Evaluation reports
├── logs/                        # Log files
├── run_app.py                   # Application launcher
└── requirements.txt             # Dependencies
```

**🔑 Đặc điểm chính của cấu trúc v8.3:**
- **Single Entry Point**: `app/app.py` với lazy loading architecture
- **Package Structure**: `__init__.py` files cho proper Python packages
- **Function Names**: `render_search_page()`, `render_analysis_page()`, `main()`
- **Safe Device Handling**: Meta tensor support trong `core/retrieval.py` và `core/reranking.py`
- **Centralized Config**: Tất cả config từ `config/default.yml`

### 1.4 Công nghệ sử dụng

**Backend & ML Framework:**
- **Python 3.8+** - Ngôn ngữ chính
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained models
- **SentenceTransformers** - Bi-encoder models
- **FAISS** - Vector similarity search

**Frontend & Web:**
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

**MLOps & Optimization:**
- **Optuna** - Hyperparameter optimization
- **MLflow** - Experiment tracking
- **Contrastive Learning** - Advanced training techniques
- **Hard Negative Mining (HNM)** - Training data enhancement
- **ADAPT** - Domain adaptation techniques
- **Centralized Configuration Management** - Unified model and path configuration

**Data Processing:**
- **Pydantic** - Data validation
- **YAML** - Configuration management
- **JSON/JSONL** - Data formats

### 1.5 Luồng xử lý tổng quan

```
User Query → Web Interface → Pipeline Orchestrator → 3-Tier Processing → Results → UI Display
     ↓              ↓              ↓                    ↓              ↓         ↓
  Input Text   Streamlit App   LegalQAPipeline    Tier1→Tier2→Tier3  Ranked   Interactive
                                                                    Results   Visualization
```

**Các giai đoạn chính:**
1. **Input Processing**: Người dùng nhập câu hỏi qua giao diện web
2. **Pipeline Orchestration**: Hệ thống điều phối xử lý qua 3 tầng
3. **Multi-tier Processing**: Xử lý tuần tự qua retrieval, light reranking, và cross-encoder
4. **Score Aggregation**: Tổng hợp điểm số từ các tầng
5. **Result Ranking**: Sắp xếp kết quả theo điểm số cuối cùng
6. **Output Display**: Hiển thị kết quả với giao diện tương tác

---

## 2. KIẾN TRÚC PIPELINE 3 TẦNG

### 2.1 Tổng quan kiến trúc 3 tầng

Kiến trúc 3 tầng của LawBot v8.3 được thiết kế để tối ưu hóa hiệu suất và độ chính xác, với mỗi tầng được **train độc lập** và thực hiện một nhiệm vụ cụ thể:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER QUERY                                  │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  🎯 TIER 1: VIETNAMESE BI-ENCODER RETRIEVAL                   │
│  ├── Model: bkai-foundation-models/vietnamese-bi-encoder      │
│  ├── Technique: Contrastive Learning + ADAPT + HNM            │
│  ├── Purpose: Fast candidate retrieval                        │
│  ├── Performance: ~1000+ docs/second                          │
│  ├── Independence: Independent training (no inheritance)      │
│  └── Output: Top-K candidates với retrieval scores            │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  ⚡ TIER 2: PHOBERT LIGHT RERANKER (INDEPENDENT ADAPT)         │
│  ├── Model: vinai/phobert-base-v2 + Independent ADAPT         │
│  ├── Technique: Independent ADAPT training + HPO + HNM        │
│  ├── Purpose: Fast filtering với independent domain expertise │
│  ├── Performance: ~500+ docs/second                           │
│  ├── Independence: Independent from Tier 1 (separate ADAPT)   │
│  └── Output: Filtered candidates với light reranker scores    │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  🎯 TIER 3: CROSS-ENCODER ENSEMBLE (ADAPT-ENHANCED)           │
│  ├── Model: Ensemble (70% ADAPT-enhanced PhoBERT-base-v2      │
│  │   từ Tier 2 + 30% PhoBERT-large)                          │
│  ├── Technique: HPO + HNM + Ensemble + ADAPT inheritance     │
│  ├── Purpose: Final ranking với domain expertise balance      │
│  ├── Performance: ~100+ docs/second                           │
│  ├── Inheritance: Inherits ADAPT-enhanced model từ Tier 2     │
│  └── Output: Final ranked results với ensemble scores         │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│  🔄 SCORE AGGREGATION & FINAL RANKING                          │
│  ├── Weighted combination: 70% Tier 2 + 30% Base model       │
│  ├── Final score calculation với ensemble weights             │
│  ├── Result ranking theo final scores                         │
│  └── Output formatting với metadata                           │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL RESULTS                               │
└─────────────────────────────────────────────────────────────────┘
```

**🔑 Đặc điểm chính của kiến trúc v8.3:**
- **Independent Training**: Mỗi tier được train độc lập, không kế thừa weights
- **ADAPT Enhancement**: Domain adaptation cho pháp luật Việt Nam
- **Safe Device Handling**: Meta tensor support với `_safe_move_to_device()`
- **Centralized Config**: Tất cả config từ `config/default.yml`
- **Ensemble Strategy**: 70% domain expertise + 30% general quality

### 2.2 Chi tiết từng tầng

#### 2.2.1 Tier 1: Bi-Encoder Retrieval

**Mục đích:** Tìm kiếm nhanh các ứng viên ban đầu từ corpus lớn

**Kỹ thuật sử dụng:**
- **Contrastive Learning**: Sử dụng TripletLoss để học biểu diễn
- **Hard Negative Mining (HNM)**: Tự động tìm negative examples khó với adaptive threshold
- **ADAPT**: Domain adaptation cho pháp luật Việt Nam với enhanced training
- **Vietnamese Bi-Encoder**: Model chuyên biệt cho tiếng Việt
- **HPO Integration**: Hyperparameter optimization với Optuna
- **Memory Management**: Basic memory handling với cleanup

**Luồng xử lý:**
```
Query → Bi-Encoder Encoding → FAISS Search → Candidate Retrieval → Score Assignment
  ↓           ↓                ↓              ↓                   ↓
Input    Embedding Vector   Similarity     Top-K Docs        Retrieval Score
Text     Generation        Search         with AIDs         (0.0 - 1.0)
```

**Code logic (dựa trên source code thực tế v8.3):**
```python
# core/retrieval.py - RetrievalEngine class
class RetrievalEngine:
    def __init__(self, bi_encoder_path, faiss_index_path, content_map_path, index_to_aid_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_ready = False
        
        # Safe device handling với meta tensor support
        try:
            # Monkey patch SentenceTransformer để prevent auto device movement
            original_to = SentenceTransformer.to
            
            def safe_to(self, device=None, *args, **kwargs):
                """Safe device movement that handles meta tensors."""
                try:
                    return original_to(self, device, *args, **kwargs)
                except NotImplementedError as e:
                    if "meta tensor" in str(e).lower():
                        logger.info(f"Detected meta tensor in SentenceTransformer, using to_empty() for device: {device}")
                        return self.to_empty(device=device)
                    else:
                        raise
                except RuntimeError as e:
                    if "offloaded" in str(e) or "dispatched" in str(e):
                        logger.info(f"SentenceTransformer is offloaded/dispatched, keeping on current device: {e}")
                        return self
                    else:
                        raise
            
            # Apply monkey patch
            SentenceTransformer.to = safe_to
            
            # Load without device parameter first
            self.bi_encoder = SentenceTransformer(bi_encoder_path)
            logger.info("✅ Successfully loaded SentenceTransformer with safe device handling")
            
            # Move to target device safely if needed
            if self.device != 'cpu':
                try:
                    self.bi_encoder.to(self.device)
                    logger.info(f"✅ Successfully moved SentenceTransformer to {self.device}")
                except NotImplementedError as nie:
                    logger.warning(f"⚠️ Meta tensor error during device movement: {nie}")
                    try:
                        self.bi_encoder.to_empty(device=self.device)
                        logger.info(f"✅ Successfully moved SentenceTransformer to {self.device} using to_empty()")
                    except Exception as e:
                        logger.error(f"❌ Failed to move SentenceTransformer to {self.device}: {e}")
                        raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize SentenceTransformer: {e}")
            raise
        
        # Load FAISS index và mappings
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.content_map = load_json(content_map_path)
        self.index_to_aid = load_json(index_to_aid_path)
        
        # Load parent law mapping
        self.aid_to_parent_law = get_parent_law_mapping()
        
        self.is_ready = True
        logger.info("✅ RetrievalEngine initialized successfully")
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        # Encode query thành embedding
        query_embedding = self.bi_encoder.encode([query])
        
        # Tìm kiếm trong FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Lấy thông tin documents
        candidates = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            aid = str(self.index_to_aid[str(idx)])
            content = self.content_map.get(aid, "")
            parent_law = self.aid_to_parent_law.get(aid, "Unknown")
            
            candidates.append({
                'aid': aid,
                'content': content,
                'parent_law_name': parent_law,
                'retrieval_score': float(score),
                'retrieval_rank': i + 1
            })
        
        return candidates
```

#### 2.2.2 Tier 2: Light Reranker

**Mục đích:** Lọc nhanh ứng viên từ Tier 1 với domain expertise

**Kỹ thuật sử dụng:**
- **PhoBERT-base-v2**: Model tiếng Việt chuyên biệt
- **Independent ADAPT Training**: Training độc lập với Tier 1
- **HPO**: Hyperparameter optimization với Optuna integration
- **HNM**: Hard negative mining với adaptive threshold
- **Performance Optimization**: HPO optimization với Optuna
- **Quality Scoring**: Tier-specific thresholds với adjusted scoring

**Luồng xử lý:**
```
Candidates from Tier 1 → Light Reranker → Similarity Scoring → Score Assignment
         ↓                    ↓              ↓                ↓
    Document List      PhoBERT Model    Query-Doc      Light Reranker
    with Content      + ADAPT          Similarity      Score (0.0-1.0)
```

**Code logic (dựa trên source code thực tế):**
```python
# core/reranking.py - RerankingEngine class
class RerankingEngine:
    def __init__(self, reranker_configs: Dict[str, Any]):
        self.models = {}
        self.is_ready = False
        
        # Load models theo configuration
        for model_name, config in reranker_configs.items():
            if config.get("enabled", False):
                model_path = config.get("model_path")
                if model_path and Path(model_path).exists():
                    self.models[model_name] = self._load_model(model_path)
        
        self.is_ready = len(self.models) > 0
    
    def rank_light(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Light reranking với PhoBERT model"""
        if not self.is_ready:
            return documents
        
        # Sử dụng light reranker model
        light_model = self.models.get("light_reranker")
        if light_model:
            return self._rank_with_model("light_reranker", query, documents)
        
        return documents
```

#### 2.2.3 Tier 3: Cross-Encoder Ensemble

**Mục đích:** Xếp hạng chính xác cuối cùng với ensemble learning

**Kỹ thuật sử dụng:**
- **Ensemble Strategy**: Kết hợp PhoBERT-base-v2 ADAPT + PhoBERT-large ADAPT
- **Weighted Combination**: 70% PhoBERT-base-v2 ADAPT + 30% PhoBERT-large ADAPT
- **HPO**: Hyperparameter optimization với Optuna integration
- **HNM**: Hard negative mining với adaptive threshold
- **Performance Optimization**: HPO optimization với Optuna
- **Quality Scoring**: Tier-specific thresholds với adjusted scoring
- **Advanced Error Handling**: Recovery mechanisms với automated optimization

**Luồng xử lý:**
```
Filtered Candidates → Cross-Encoder Ensemble → Classification Scoring → Final Ranking
         ↓                      ↓                ↓                ↓
    Document List        Ensemble Model    Binary Class      Cross-Encoder
    from Tier 2         (PhoBERT-base-v2 ADAPT + PhoBERT-large ADAPT)    Prediction        Score (0.0-1.0)
```

**Code logic (dựa trên source code thực tế):**
```python
# core/reranking.py - Cross-encoder ensemble implementation
class EnsembleCrossEncoder:
    def __init__(self, adapt_model, base_model, adapt_weight, base_weight):
        self.adapt_model = adapt_model
        self.base_model = base_model
        self.adapt_weight = adapt_weight
        self.base_weight = base_weight
    
    def __call__(self, **inputs):
        # Get predictions from both models
        with torch.no_grad():
            adapt_output = self.adapt_model(**inputs)
            base_output = self.base_model(**inputs)
        
        # Weighted combination
        ensemble_logits = (
            self.adapt_weight * adapt_output.logits +
            self.base_weight * base_output.logits
        )
        
        return SequenceClassifierOutput(logits=ensemble_logits)

def rank_cross(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-encoder reranking với ensemble model"""
    if not self.is_ready:
        return documents
    
    # Sử dụng cross-encoder model
    cross_model = self.models.get("cross_encoder")
    if cross_model:
        return self._rank_with_model("cross_encoder", query, documents)
    
    return documents
```

### 2.3 Score Aggregation và Final Ranking

**Mục đích:** Tổng hợp điểm số từ 3 tầng để tạo điểm cuối cùng

**Công thức tính điểm:**
```python
# Khởi tạo với retrieval score
final_score = retrieval_score

# Cộng light reranker score (nếu có)
if use_light_ranking:
    final_score = light_weight * light_score + (1 - light_weight) * final_score

# Cộng cross encoder score (nếu có)  
if use_cross_encoder:
    final_score = cross_weight * cross_score + (1 - cross_weight) * final_score

# Weights mặc định:
# light_weight = 0.7 (70%)
# cross_weight = 0.3 (30%)
```

**Luồng xử lý:**
```
Scores from 3 Tiers → Weighted Combination → Score Normalization → Final Ranking
         ↓                    ↓                    ↓                ↓
Retrieval + Light +    Weighted Average    Score Range      Sort by Final
Cross Encoder         (Configurable)       0.0 - 1.0        Score Desc
```

**Code logic (pseudocode):**
```python
def combine_scores(documents, use_light=True, use_cross=True):
    for doc in documents:
        retrieval_score = doc.get('retrieval_score', 0.0)
        light_score = doc.get('light_reranker_score', 0.0)
        cross_score = doc.get('cross_encoder_score', 0.0)
        
        # Khởi tạo với retrieval score
        final_score = retrieval_score
        
        # Cộng light reranker
        if use_light:
            final_score = 0.7 * light_score + 0.3 * final_score
        
        # Cộng cross encoder
        if use_cross:
            final_score = 0.3 * cross_score + 0.7 * final_score
        
        doc['final_score'] = final_score
        doc['score_breakdown'] = {
            'retrieval': retrieval_score,
            'light_reranker': light_score,
            'cross_encoder': cross_score
        }
    
    # Sắp xếp theo điểm cuối cùng
    return sorted(documents, key=lambda x: x['final_score'], reverse=True)
```

### 2.4 Pipeline Orchestration

**Mục đích:** Điều phối luồng xử lý qua 3 tầng một cách hiệu quả

**Luồng điều phối:**
```
Query Input → Pipeline Orchestrator → Tier 1 → Tier 2 → Tier 3 → Score Aggregation → Results
     ↓              ↓                ↓        ↓        ↓        ↓                ↓
User Question   LegalQAPipeline   Retrieval  Light    Cross    Combine          Final
                .predict()        Engine     Rerank   Encode   Scores           Output
```

**Code logic (dựa trên source code thực tế):**
```python
# core/pipeline.py - LegalQAPipeline class
class LegalQAPipeline:
    def __init__(self, bi_encoder_path=None, reranker_paths=None, 
                 faiss_index_path=None, content_map_path=None, index_to_aid_path=None):
        self.is_ready = False
        self.loaded_model_paths = {}
        
        try:
            # Validate parent law mapping
            ensure_parent_law_mapping()
            
            # Load Retriever (Tier 1)
            self.retriever = RetrievalEngine(
                bi_encoder_path=bi_encoder_path or self._get_latest_bi_encoder(),
                faiss_index_path=faiss_index_path or self._get_faiss_index_path(),
                content_map_path=content_map_path or self._get_content_map_path(),
                index_to_aid_path=index_to_aid_path or self._get_index_to_aid_path()
            )
            
            # Load Reranker (Tier 2 & 3)
            reranker_configs = self._resolve_reranker_paths(reranker_paths)
            if reranker_configs:
                self.reranker = RerankingEngine(reranker_configs)
            else:
                self.reranker = None
            
            self.is_ready = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LegalQAPipeline: {e}")
            self.is_ready = False
            raise
    
    def predict(self, query: str, top_k_retrieval: Optional[int] = None, 
                top_k_light: Optional[int] = None, top_k_final: Optional[int] = None):
        """Main prediction method với 3-tier architecture"""
        if not self.is_ready:
            raise RuntimeError("Pipeline is not ready")
        
        try:
            # Tier 1: Bi-Encoder Retrieval
            documents = self.retriever.retrieve(query, top_k=top_k_retrieval or 100)
            
            # Initialize scores
            for doc in documents:
                doc['light_reranker_score'] = 0.0
                doc['cross_encoder_score'] = 0.0
            
            # Tier 2: Light Reranking
            if self.reranker and self.reranker.is_ready:
                light_docs = documents[:top_k_light or 80]
                documents = self.reranker.rank_light(query, light_docs)
            
            # Tier 3: Cross-Encoder Reranking
            if self.reranker and self.reranker.is_ready:
                cross_docs = documents[:top_k_final or 20]
                documents = self.reranker.rank_cross(query, cross_docs)
            
            # Score aggregation và final ranking
            documents = self._combine_scores(documents)
            final_results = sorted(
                documents, 
                key=lambda x: x['final_score'], 
                reverse=True
            )[:top_k_final or 10]
            
            return final_results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
```

---

## 3. LUỒNG XỬ LÝ TRAINING VÀ MLOPS

### 3.1 Tổng quan luồng training

Luồng training của LawBot được thiết kế theo kiến trúc 3 tầng, với mỗi tầng có quy trình training riêng biệt và tối ưu hóa:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│  📊 DATA PREPARATION STAGE                                    │
│  ├── Raw Data Loading                                         │
│  ├── Legal Corpus Processing                                  │
│  ├── Training Data Generation                                 │
│  ├── Validation Set Creation                                  │
│  └── Feature Engineering                                      │
├─────────────────────────────────────────────────────────────────┤
│  🎯 TIER 1 TRAINING: BI-ENCODER                              │
│  ├── Contrastive Learning                                     │
│  ├── Hard Negative Mining (HNM)                               │
│  ├── ADAPT Domain Adaptation                                  │
│  ├── Hyperparameter Optimization (HPO)                        │
│  └── Model Export & FAISS Index Creation                      │
├─────────────────────────────────────────────────────────────────┤
│  ⚡ TIER 2 TRAINING: LIGHT RERANKER                           │
│  ├── PhoBERT-base-v2 Fine-tuning                              │
│  ├── Independent ADAPT Training                                │
│  ├── HNM Implementation                                       │
│  ├── HPO Optimization                                         │
│  └── Model Export                                             │
├─────────────────────────────────────────────────────────────────┤
│  🎯 TIER 3 TRAINING: CROSS-ENCODER ENSEMBLE                   │
│  ├── Ensemble Model Creation                                  │
│  ├── ADAPT-enhanced Model Integration                         │
│  ├── HPO + HNM Optimization                                   │
│  ├── Ensemble Training                                        │
│  └── Model Export                                             │
├─────────────────────────────────────────────────────────────────┤
│  🔄 PIPELINE INTEGRATION & VALIDATION                         │
│  ├── Pipeline Assembly                                        │
│  ├── End-to-end Testing                                       │
│  ├── Performance Evaluation                                   │
│  ├── Comprehensive Metrics                                    │
│  └── Deployment Preparation                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Preparation Stage

#### 3.2.1 Raw Data Loading và Processing

**Mục đích:** Chuẩn bị dữ liệu thô cho training pipeline

**Luồng xử lý:**
```
Raw Legal Documents → Text Extraction → Cleaning → Normalization → Structured Data
         ↓                ↓              ↓          ↓              ↓
    PDF/DOC Files    Content Parse   Legal Text   Unicode      Training
    Legal Corpus     Article Split   Cleaning     Normalize    Examples
```

**Code logic (pseudocode):**
```python
def process_legal_corpus(raw_corpus):
    processed_corpus = {}
    aid_map = {}
    
    for document in raw_corpus:
        # Extract content và metadata
        content = document.get('content', '')
        article_id = document.get('article_id')
        law_id = document.get('law_id')
        
        # Clean và normalize text
        cleaned_content = legal_text_cleaner(content)
        
        # Generate canonical AID
        canonical_aid = canonicalize_aid(law_id, article_id)
        
        # Store processed data
        processed_corpus[canonical_aid] = cleaned_content
        aid_map[canonical_aid] = {
            'law_id': law_id,
            'article_id': article_id,
            'content': cleaned_content
        }
    
    return processed_corpus, aid_map
```

#### 3.2.2 Training Data Generation

**Mục đích:** Tạo training examples cho từng tier với kỹ thuật khác nhau

**Luồng xử lý:**
```
Legal Corpus → Query Generation → Positive/Negative Mining → Training Examples
      ↓              ↓                    ↓                    ↓
Article Content   Question Types    Similarity-based     Structured Data
+ Metadata       + Templates       + Hard Negatives     for Training
```

**Code logic (pseudocode):**
```python
def generate_training_examples(train_data, processed_corpus, aid_map):
    # Tier 1: Bi-encoder training data
    bi_encoder_data = []
    for item in train_data:
        query = item['question']
        positive_aid = item['positive_aid']
        negative_aids = item['negative_aids']
        
        bi_encoder_data.append({
            'query': query,
            'positive': processed_corpus[positive_aid],
            'negative': processed_corpus[negative_aids[0]]  # Use first negative
            'positive_aid': positive_aid,
            'negative_aid': negative_aids[0]
        })
    
    # Tier 2 & 3: Reranker training data
    reranker_data = []
    for item in train_data:
        query = item['question']
        positive_aid = item['positive_aid']
        negative_aids = item['negative_aids']
        
        # Positive examples
        reranker_data.append({
            'query': query,
            'passage': processed_corpus[positive_aid],
            'label': 1.0,
            'aid': positive_aid
        })
        
        # Negative examples
        for neg_aid in negative_aids[:3]:  # Top 3 negatives
            reranker_data.append({
                'query': query,
                'passage': processed_corpus[neg_aid],
                'label': 0.0,
                'aid': neg_aid
            })
    
    return bi_encoder_data, reranker_data
```

#### 3.2.3 Validation Set Creation

**Mục đích:** Tạo validation sets riêng biệt cho từng tier

**Luồng xử lý:**
```
Training Data → Stratified Split → Tier-specific Validation → Validation Files
      ↓              ↓                    ↓                    ↓
Full Dataset    Train/Val Split    Tier 1/2/3 Sets      JSONL Files
                (80/20)           with Metrics          for Evaluation
```

**Code logic (pseudocode):**
```python
class ValidationSetManager:
    def create_tier_validation_sets(self, training_data, validation_split=0.2):
        # Split data
        train_data, val_data = train_test_split(
            training_data, test_size=validation_split, random_state=42
        )
        
        # Tier 1: Retrieval validation (query-document pairs)
        tier1_val = self._create_retrieval_validation(val_data, min_samples=100)
        
        # Tier 2: Light reranker validation (classification format)
        tier2_val = self._create_light_reranker_validation(val_data, min_samples=100)
        
        # Tier 3: Cross-encoder validation (reranking format)
        tier3_val = self._create_cross_encoder_validation(val_data, min_samples=100)
        
        return tier1_val, tier2_val, tier3_val
```

### 3.3 Tier 1 Training: Bi-Encoder

#### 3.3.1 Contrastive Learning Implementation

**Mục đích:** Huấn luyện bi-encoder sử dụng contrastive learning với TripletLoss

**Kỹ thuật sử dụng:**
- **TripletLoss**: (query, positive, negative) training
- **Hard Negative Mining**: Tự động tìm negative examples khó với adaptive threshold
- **ADAPT**: Domain adaptation cho pháp luật Việt Nam với enhanced training
- **HPO**: Hyperparameter optimization với Optuna integration
- **Performance**: HPO optimization với Optuna
- **Memory**: Basic memory management

**Luồng training:**
```
Training Data → Triplet Creation → Model Forward → Loss Calculation → Backpropagation
      ↓              ↓              ↓              ↓                ↓
Query+Pos+Neg   Triplet Batching   Embedding     TripletLoss      Gradient
Examples        with HNM          Generation     Computation      Update
```

**Code logic (pseudocode):**
```python
class BiEncoderTrainer:
    def train_contrastive_learning(self, model_name):
        # Load pre-trained Vietnamese bi-encoder
        model = SentenceTransformer(model_name)
        
        # Prepare contrastive data
        triplets = self._prepare_contrastive_data()
        
        # Custom training loop
        for epoch in range(self.epochs):
            for batch in self.triplet_dataloader:
                # Forward pass
                query_emb = model.encode(batch['queries'])
                pos_emb = model.encode(batch['positives'])
                neg_emb = model.encode(batch['negatives'])
                
                # Calculate triplet loss
                loss = self.triplet_loss(query_emb, pos_emb, neg_emb)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return model
```

#### 3.3.2 Hard Negative Mining (HNM)

**Mục đích:** Tự động tìm negative examples khó để cải thiện training

**Luồng xử lý:**
```
Candidate Pool → Similarity Calculation → Hard Negative Selection → Training Update
      ↓                ↓                      ↓                    ↓
All Documents    Query-Doc Similarity    Top-K Difficult     Enhanced
                 with Current Model      Negatives          Training Data
```

**Code logic (pseudocode):**
```python
class HardNegativeMiner:
    def mine_hard_negatives(self, queries, positive_candidates, negative_candidates):
        hard_negatives = []
        
        for query in queries:
            # Encode query và candidates
            query_emb = self.model.encode(query)
            pos_embs = self.model.encode(positive_candidates)
            neg_embs = self.model.encode(negative_candidates)
            
            # Calculate similarities
            pos_sims = cosine_similarity(query_emb, pos_embs)
            neg_sims = cosine_similarity(query_emb, neg_embs)
            
            # Find hardest negatives (highest similarity to query)
            hardest_neg_indices = np.argsort(neg_sims)[-self.top_k:]
            
            # Select hardest negatives
            for idx in hardest_neg_indices:
                if neg_sims[idx] > self.similarity_threshold:
                    hard_negatives.append(negative_candidates[idx])
        
        return hard_negatives
```

#### 3.3.3 ADAPT Domain Adaptation

**Mục đích:** Thích ứng model với domain pháp luật Việt Nam

**Luồng xử lý:**
```
Base Model → Domain Data Loading → ADAPT Training → Domain-adapted Model
     ↓              ↓                ↓                ↓
Pre-trained    Legal Corpus     Fine-tuning      Optimized for
Vietnamese     + Queries        with ADAPT       Legal Domain
Bi-encoder
```

**Code logic (pseudocode):**
```python
def apply_adapt_technique(self, model, domain_data):
    # Prepare domain-specific dataset
    domain_dataset = LegalDomainDataset(domain_data)
    domain_dataloader = DataLoader(domain_dataset, batch_size=16)
    
    # ADAPT training loop
    for epoch in range(self.adaptation_steps):
        for batch in domain_dataloader:
            # Forward pass với domain data
            outputs = model(batch['input_ids'], batch['attention_mask'])
            
            # Calculate domain adaptation loss
            loss = self.domain_loss(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            self.adapt_optimizer.step()
            self.adapt_optimizer.zero_grad()
    
    return model
```

### 3.4 Tier 2 Training: Light Reranker

#### 3.4.1 PhoBERT Fine-tuning

**Mục đích:** Huấn luyện PhoBERT-base-v2 cho light reranking

**Kỹ thuật sử dụng:**
- **PhoBERT-base-v2**: Base model tiếng Việt
- **Independent ADAPT Training**: Training độc lập với Tier 1
- **Classification Head**: Binary classification cho relevance
- **HPO**: Hyperparameter optimization

**Luồng training:**
```
PhoBERT Model → Data Loading → Fine-tuning → HPO Optimization → Model Export
      ↓              ↓              ↓              ↓                ↓
Pre-trained    Training Data   Classification   Optuna Trials    Saved Model
PhoBERT        + Labels        Training         + Best Params     + Config
```

**Code logic (pseudocode):**
```python
class LightRankingTrainer:
    def train_model(self, data_path, hpo_params=None):
        # Load PhoBERT model và tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base-v2", num_labels=1
        )
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        
        # Prepare training data
        train_dataset, val_dataset = self._load_training_data(data_path)
        
        # Training loop
        for epoch in range(self.epochs):
            for batch in self.train_dataloader:
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss
                loss = self.criterion(outputs.logits, batch['labels'])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return model, tokenizer
```

#### 3.4.2 Hyperparameter Optimization (HPO)

**Mục đích:** Tối ưu hóa hyperparameters sử dụng Optuna

**Luồng optimization:**
```
HPO Configuration → Trial Generation → Model Training → Evaluation → Best Params
        ↓                ↓                ↓              ↓            ↓
Parameter Space    Optuna Trials    Train Model    Metrics      Optimal
Definition        + Sampling       with Params    Calculation   Parameters
```

**Code logic (pseudocode):**
```python
class HyperparameterOptimizer:
    def objective(self, trial):
        # Define hyperparameter space với advanced options
        params = {
            'learning_rate': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 2, 10),
            'warmup_steps': trial.suggest_int('warmup_steps', 50, 200),
            'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1),
            'hard_negative_ratio': trial.suggest_float('hard_negative_ratio', 0.1, 0.5),
            'similarity_threshold': trial.suggest_float('similarity_threshold', 0.5, 0.9),
            'use_adapt_enhanced': trial.suggest_categorical('use_adapt_enhanced', [True, False])
        }
        
        # Train model với params và early stopping
        model = self._train_with_early_stopping(params, patience=5)
        
        # Evaluate model
        val_score = self._evaluate_model(model)
        
        return val_score
    
    def optimize(self, n_trials=50):
        # Advanced study configuration
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_params
```

### 3.5 Tier 3 Training: Cross-Encoder Ensemble

### 3.6 Comprehensive Evaluation System

### 3.7 Centralized Configuration Management

**Mục đích:** Quản lý tập trung cấu hình paths, models, và validation để đảm bảo tính nhất quán

**Kỹ thuật sử dụng:**
- **Centralized Paths**: Tất cả đường dẫn được quản lý tại `config/paths.py`
- **Centralized Models**: Cấu hình model được quản lý tại `config/models.py`
- **Automated Validation**: Tự động kiểm tra tính hợp lệ của paths và models
- **Smart Discovery**: Tự động tìm thư mục processed data mới nhất

**Luồng configuration:**
```
Config Files → Path Validation → Model Loading → Status Check → System Ready
     ↓              ↓              ↓              ↓            ↓
paths.py      validate_paths()  load_models()  get_status()  Ready
models.py     + data_check      + mapping      + health      State
```

**Code logic (pseudocode):**
```python
# config/paths.py
class CentralizedPathManager:
    TRAINING_DATA_PATHS = {
        "tier_1": {
            "data_source": "bi_encoder_train.jsonl",
            "required_files": ["bi_encoder_train.jsonl", "processed_corpus.json"],
            "validation_threshold": 1000  # Minimum records required
        },
        "tier_2": {
            "data_source": "training_data.jsonl", 
            "required_files": ["training_data.jsonl", "negative_pool.jsonl"],
            "validation_threshold": 500
        },
        "tier_3": {
            "data_source": "cross_encoder_train.jsonl",
            "required_files": ["cross_encoder_train.jsonl", "processed_corpus.json"],
            "validation_threshold": 800
        }
    }
    
    def validate_training_data_paths(self):
        """Validate all training data paths and return status"""
        validation_results = {}
        
        for tier, config in self.TRAINING_DATA_PATHS.items():
            tier_status = self._validate_tier_data(tier, config)
            validation_results[tier] = tier_status
        
        # Overall status
        overall_status = "ready" if all(
            result["status"] == "ready" for result in validation_results.values()
        ) else "missing_data"
        
        validation_results["overall"] = {"status": overall_status}
        return validation_results
    
    def get_latest_processed_data_dir(self):
        """Automatically find latest processed data directory"""
        features_dir = Path("features")
        if not features_dir.exists():
            return None
        
        # Find directories with timestamp pattern
        processed_dirs = list(features_dir.glob("processed_data_*"))
        if not processed_dirs:
            return None
        
        # Return most recent
        return max(processed_dirs, key=lambda p: p.stat().st_mtime)

# config/models.py
class CentralizedModelManager:
    MODEL_TYPES = {
        "bi_encoder": {
            "name": "bi_encoder",
            "display_name": "Vietnamese Bi-Encoder",
            "purpose": "Document retrieval and similarity search",
            "tier": "tier_1",
            "base_model": "vinai/phobert-base-v2",
            "training_method": "contrastive_learning"
        },
        "light_reranker": {
            "name": "light_reranker", 
            "display_name": "PhoBERT Light Reranker",
            "purpose": "Fast document filtering and ranking",
            "tier": "tier_2",
            "base_model": "vinai/phobert-base-v2",
            "training_method": "adapt_training"
        },
        "cross_encoder": {
            "name": "cross_encoder",
            "display_name": "PhoBERT Cross-Encoder Ensemble",
            "purpose": "Final document ranking and scoring",
            "tier": "tier_3", 
            "base_model": "vinai/phobert-base-v2",
            "training_method": "ensemble_adapt"
        }
    }
    
    MODEL_DIRECTORY_MAPPING = {
        "bi_encoder": "bi-encoder_*",
        "light_reranker": "light-ranking_*",
        "cross_encoder": "combined-reranker-adapt_*"
    }
    
    def get_model_config(self, model_type):
        """Get configuration for specific model type"""
        return self.MODEL_TYPES.get(model_type, {})
    
    def get_display_name(self, model_type):
        """Get human-readable display name for model"""
        config = self.get_model_config(model_type)
        return config.get("display_name", model_type)
    
    def get_tier_info(self, tier_key):
        """Get information about specific tier"""
        for model_type, config in self.MODEL_TYPES.items():
            if config.get("tier") == tier_key:
                return config
        return {}
```

**Mục đích:** Đánh giá toàn diện hiệu suất của tất cả tiers với metrics đa dạng

**Kỹ thuật sử dụng:**
- **Multi-tier Evaluation**: Đánh giá độc lập từng tier và combined pipeline
- **Advanced Metrics**: Precision, Recall, F1, NDCG, MRR, Quality scores
- **Performance Monitoring**: Real-time tracking và automated optimization
- **Unified Reports**: Consolidated evaluation reports trong single directory

**Luồng evaluation:**
```
Tier 1 → Tier 2 → Tier 3 → Combined → Metrics Calculation → Report Generation
  ↓        ↓        ↓         ↓              ↓                    ↓
Retrieval Light    Cross    Ensemble    Precision/Recall/    JSON Reports
Evaluation Ranking Encoder  Pipeline    F1/NDCG/MRR/Quality  + Charts
```

**Code logic (pseudocode):**
```python
class ComprehensiveEvaluator:
    def run_comprehensive_evaluation(self, pipeline, test_queries):
        evaluation_results = {
            "tier_1": {},
            "tier_2": {},
            "tier_3": {},
            "combined": {}
        }
        
        # Process each query
        for query in test_queries:
            # Tier 1: Independent retrieval evaluation
            tier1_results = self.evaluate_tier_1_retrieval_only(pipeline, query)
            tier1_metrics = self.calculate_tier_metrics(tier1_results, "tier_1")
            self.update_evaluation_results(evaluation_results, "tier_1", tier1_metrics)
            
            # Tier 2: Independent light reranker evaluation
            tier2_results = self.evaluate_tier_2_light_reranker_only(pipeline, query)
            tier2_metrics = self.calculate_tier_metrics(tier2_results, "tier_2")
            self.update_evaluation_results(evaluation_results, "tier_2", tier2_metrics)
            
            # Tier 3: Independent cross encoder evaluation
            tier3_results = self.evaluate_tier_3_cross_encoder_only(pipeline, query)
            tier3_metrics = self.calculate_tier_metrics(tier3_results, "tier_3")
            self.update_evaluation_results(evaluation_results, "tier_3", tier3_metrics)
            
            # Combined: Full pipeline evaluation
            combined_results = pipeline.predict(query, top_k_final=20)
            combined_metrics = self.calculate_tier_metrics(combined_results, "combined")
            self.update_evaluation_results(evaluation_results, "combined", combined_metrics)
        
        # Calculate final averages
        final_results = self.calculate_final_averages(evaluation_results)
        
        # Save results
        self.save_comprehensive_evaluation_results(final_results)
        
        return final_results
    
    def calculate_tier_metrics(self, results, tier_name):
        """Calculate comprehensive metrics for a tier"""
        metrics = {}
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            if k <= len(results):
                # Calculate all metrics
                precision_k = self.calculate_precision_at_k(results, k)
                recall_k = self.calculate_recall_at_k(results, k)
                f1_k = self.calculate_f1_at_k(results, k)
                ndcg_k = self.calculate_ndcg_at_k(results, k)
                mrr_k = self.calculate_mrr_at_k(results, k)
                quality_k = self.calculate_quality_score(results, k)
                
                # Store metrics
                metrics[f"precision_{k}"] = [precision_k]
                metrics[f"recall_{k}"] = [recall_k]
                metrics[f"f1_{k}"] = [f1_k]
                metrics[f"ndcg_{k}"] = [ndcg_k]
                metrics[f"mrr_{k}"] = [mrr_k]
                metrics[f"quality_{k}"] = [quality_k]
        
        return metrics
```

#### 3.5.1 Ensemble Model Creation

**Mục đích:** Tạo ensemble model kết hợp PhoBERT-base-v2 ADAPT và PhoBERT-large ADAPT

**Kỹ thuật sử dụng:**
- **Ensemble Strategy**: Weighted combination (70% PhoBERT-base-v2 ADAPT + 30% PhoBERT-large ADAPT)
- **Model Integration**: PhoBERT-base-v2 (ADAPT) + PhoBERT-large (ADAPT)
- **HPO**: Hyperparameter optimization cho ensemble weights
- **HNM**: Hard negative mining

**Luồng training:**
```
ADAPT Model → ADAPT Model → Ensemble Creation → Joint Training → Model Export
     ↓            ↓              ↓                ↓            ↓
Tier 2 Output   PhoBERT-large   Weighted        Fine-tuning   Ensemble
PhoBERT-base    (ADAPT)         Combination     + HPO          Model
```

**Code logic (pseudocode):**
```python
class CrossEncoderTrainer:
    def create_ensemble_strategy(self, adapt_model, base_model):
        # Create ensemble model
        ensemble = EnsembleReranker(
            adapt_model=adapt_model,
            base_model=base_model,
            adapt_weight=0.7,
            base_weight=0.3
        )
        
        # Prepare training data với HNM
        training_data = self._create_reranking_training_data()
        enriched_data = self._enrich_with_hard_negatives(training_data)
        
        # Training loop
        for epoch in range(self.epochs):
            for batch in self.train_dataloader:
                # Forward pass với ensemble
                outputs = ensemble(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                # Calculate loss
                loss = self.criterion(outputs.logits, batch['labels'])
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return ensemble
```

#### 3.5.2 Ensemble Model Architecture

**Mục đích:** Thiết kế kiến trúc ensemble cho cross-encoder

**Kiến trúc:**
```
Input (Query + Document) → Tokenization → Ensemble Processing → Final Output
         ↓                      ↓                ↓                ↓
Text Pair              Token IDs        Parallel Models    Weighted Score
                       + Attention      (PhoBERT-base-v2 ADAPT + PhoBERT-large ADAPT)    Combination
```

**Code logic (pseudocode):**
```python
class EnsembleReranker(torch.nn.Module):
    def __init__(self, adapt_model, base_model, adapt_weight=0.7, base_weight=0.3):
        super().__init__()
        self.adapt_model = adapt_model
        self.base_model = base_model
        self.adapt_weight = adapt_weight
        self.base_weight = base_weight
    
    def forward(self, input_ids, attention_mask):
        # Get predictions from both models
        with torch.no_grad():
            adapt_output = self.adapt_model(input_ids, attention_mask)
            base_output = self.base_model(input_ids, attention_mask)
        
        # Weighted combination
        ensemble_logits = (
            self.adapt_weight * adapt_output.logits +
            self.base_weight * base_output.logits
        )
        
        return SequenceClassifierOutput(logits=ensemble_logits)
```

### 3.6 MLOps và Pipeline Management

#### 3.6.1 Workflow Orchestration với Centralized Paths

**Mục đích:** Quản lý và điều phối toàn bộ training workflow với centralized path management

**Luồng workflow:**
```
Project Setup → Data Preparation → Model Training → Evaluation → Deployment
      ↓              ↓                ↓              ↓            ↓
Environment     Legal Corpus     Tier 1/2/3      Metrics      Model
Configuration   Processing       Training        Calculation   Serving
```

**Centralized Path Management (dựa trên source code thực tế):**
```python
# config/paths.py - Centralized configuration
TRAINING_DATA_PATHS = {
    "tier_1": {
        "data_source": "bi_encoder_train.jsonl",
        "required_files": ["bi_encoder_train.jsonl", "processed_corpus.json"],
        "validation_threshold": 1000,  # Minimum records required
        "description": "Bi-Encoder training data with contrastive learning"
    },
    "tier_2": {
        "data_source": "training_data.jsonl", 
        "required_files": ["training_data.jsonl", "negative_pool.jsonl"],
        "validation_threshold": 500,
        "description": "Light reranker training data with hard negative mining"
    },
    "tier_3": {
        "data_source": "cross_encoder_train.jsonl",
        "required_files": ["cross_encoder_train.jsonl", "processed_corpus.json"],
        "validation_threshold": 800,
        "description": "Cross-encoder training data for ensemble learning"
    }
}

# Automated data freshness validation
def validate_training_data_paths() -> dict:
    """Validate all training data paths and return status"""
    validation_results = {}
    
    for tier, config in TRAINING_DATA_PATHS.items():
        tier_status = self._validate_tier_data(tier, config)
        validation_results[tier] = tier_status
    
    # Overall status
    overall_status = "ready" if all(
        result["status"] == "ready" for result in validation_results.values()
    ) else "missing_data"
    
    validation_results["overall"] = {"status": overall_status}
    return validation_results

def get_latest_processed_data_dir():
    """Automatically find latest processed data directory"""
    features_dir = Path("features")
    if not features_dir.exists():
        return None
    
    # Find directories with timestamp pattern
    processed_dirs = list(features_dir.glob("processed_data_*"))
    if not processed_dirs:
        return None
    
    # Return most recent
    return max(processed_dirs, key=lambda p: p.stat().st_mtime)
```

**Code logic (dựa trên source code thực tế):**
```python
# run_workflow.py - Workflow orchestration
class WorkflowRunner:
    def run_workflow(self):
        stages = [
            'data_preparation',
            'bi_encoder_training',
            'faiss_index_creation',
            'light_reranker_training',
            'cross_encoder_training',
            'pipeline_integration',
            'evaluation'
        ]
        
        for stage in stages:
            # Special handling for data_preparation with freshness validation
            if stage == "data_preparation":
                success = self._run_data_preparation_with_validation()
            else:
                success = self.run_stage(stage)
            
            if not success:
                self._handle_failure(stage)
                break
        
        return success
    
    def _run_data_preparation_with_validation(self):
        """Run data preparation with automatic freshness validation"""
        try:
            from config.paths import validate_training_data_paths
            validation_results = validate_training_data_paths()
            
            if validation_results["overall"]["status"] == "ready":
                self.logger.info("✅ Training data available, skipping data_preparation")
                return True
            else:
                self.logger.info("⚠️ Data missing/stale, running data_preparation...")
                return self.run_stage("data_preparation")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Data validation failed: {e}, running data_preparation...")
            return self.run_stage("data_preparation")
```

#### 3.6.2 Model Versioning và Management

**Mục đích:** Quản lý phiên bản models và tracking experiments

**Luồng versioning:**
```
Training Completion → Model Export → Metadata Generation → Version Tracking → Registry
         ↓                ↓                ↓                ↓            ↓
Trained Model      Save to Disk      Model Info        Timestamp     Model
                   + Config          + Metrics         + Hash        Registry
```

**Code logic (pseudocode):**
```python
def save_model(self, engine, output_dir):
    # Generate versioned path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = output_dir / f"model_{timestamp}"
    
    # Save model
    engine.save_pretrained(versioned_path)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': self.model_type,
        'training_params': self.training_params,
        'performance_metrics': self.evaluation_results,
        'git_commit': self._get_git_commit()
    }
    
    with open(versioned_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return versioned_path
```

#### 3.6.3 Performance Monitoring và Evaluation

**Mục đích:** Theo dõi hiệu suất training và đánh giá models

**Luồng monitoring:**
```
Training Progress → Metrics Collection → Performance Analysis → Report Generation
        ↓                ↓                    ↓                ↓
Real-time Logs    Loss/Accuracy      Tier Comparison     Evaluation
+ Checkpoints     + Validation       + Improvement       Reports
```

**Code logic (pseudocode):**
```python
class ProgressTracker:
    def start_step(self, step_name, step_info=None):
        self.current_step += 1
        step_start_time = time.time()
        
        # Log step start
        self.logger.info(f"[{self.current_step}/{self.total_steps}] Starting: {step_name}")
        
        # Store step data
        step_data = {
            'name': step_name,
            'info': step_info,
            'start_time': step_start_time,
            'status': 'running'
        }
        self.steps_info.append(step_data)
        
        return step_start_time
    
    def end_step(self, step_start_time, success=True):
        end_time = time.time()
        duration = end_time - step_start_time
        
        # Update step data
        if self.steps_info:
            step_data = self.steps_info[-1]
            step_data.update({
                'end_time': end_time,
                'duration': duration,
                'success': success,
                'status': 'completed' if success else 'failed'
            })
        
        # Log completion
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Step {step_data['name']} {status} in {duration:.2f}s")
```

### 3.7 Training Configuration và Optimization

#### 3.7.1 Configuration Management

**Mục đích:** Quản lý cấu hình training cho từng tier

**Cấu trúc config:**
```yaml
# Training configuration cho từng tier
bi_encoder:
  model_name: "bkai-foundation-models/vietnamese-bi-encoder"
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      loss_type: "TripletLoss"
      max_length: 256
      batch_size: 16
      epochs: 5
    adapt:
      enabled: true
      domain_data_ratio: 0.8
      adaptation_steps: 1000
      learning_rate: 2e-5

light_reranker:
  model_name: "vinai/phobert-base-v2"
  max_length: 256
  batch_size: 32
  epochs: 3
  learning_rate: 2e-5

cross_encoder:
  ensemble_method: "weighted_average"
  models:
    adapt_enhanced: 0.7
    base_model: 0.3
```

#### 3.7.2 Training Optimization Techniques

**Mục đích:** Tối ưu hóa quá trình training cho hiệu suất cao

**Kỹ thuật optimization:**
- **Gradient Accumulation**: Xử lý batch lớn với memory hạn chế
- **Mixed Precision Training**: Sử dụng FP16 để tăng tốc
- **Learning Rate Scheduling**: Warmup và decay strategies
- **Early Stopping**: Tránh overfitting
- **Model Checkpointing**: Lưu trữ intermediate models

**Code logic (pseudocode):**
```python
class TrainingEngine:
    def setup_optimizer_and_scheduler(self, num_training_steps):
        # Optimizer với weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler với warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, epoch):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass với gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
```

---

## 4. LUỒNG XỬ LÝ REQUEST USER

### 4.1 Tổng quan luồng xử lý request

Luồng xử lý request của LawBot được thiết kế để xử lý câu hỏi pháp luật từ người dùng một cách hiệu quả và chính xác với kiến trúc 3 tầng:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER REQUEST FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│  🎯 USER INPUT STAGE                                          │
│  ├── Query Input (Text/Question)                              │
│  ├── Search Configuration                                     │
│  ├── Parameter Selection                                      │
│  └── Request Validation                                       │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ REQUEST PROCESSING STAGE                                  │
│  ├── Query Preprocessing                                      │
│  ├── Parameter Calculation                                    │
│  ├── Pipeline Initialization                                  │
│  └── Cache Management                                         │
├─────────────────────────────────────────────────────────────────┤
│  🚀 PIPELINE EXECUTION STAGE (3-TIER)                         │
│  ├── Tier 1: Bi-Encoder Retrieval                             │
│  ├── Tier 2: Light Reranking                                  │
│  ├── Tier 3: Cross-Encoder Reranking                          │
│  └── Score Aggregation                                        │
├─────────────────────────────────────────────────────────────────┤
│  📊 RESULT PROCESSING STAGE                                   │
│  ├── Result Ranking                                           │
│  ├── Score Normalization                                      │
│  ├── Metadata Enrichment                                      │
│  └── Response Formatting                                       │
├─────────────────────────────────────────────────────────────────┤
│  🎨 USER OUTPUT STAGE                                         │
│  ├── Interactive Visualization                                 │
│  ├── Score Comparison Charts                                  │
│  ├── Detailed Result Display                                  │
│  └── Performance Metrics                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 User Input Stage

#### 4.2.1 Query Input và Validation

**Mục đích:** Xử lý và validate input từ người dùng

**Luồng xử lý:**
```
User Input → Text Validation → Query Preprocessing → Search Configuration → Request Ready
     ↓              ↓                ↓                    ↓                ↓
Question Text   Input Check      Text Cleaning      Parameter Set    Validated
+ Parameters   + Sanitization   + Normalization    + Options        Request
```

**Code logic (pseudocode):**
```python
def process_user_query(query_input, search_config):
    # Validate input
    if not query_input or len(query_input.strip()) < 3:
        raise ValueError("Query must be at least 3 characters long")
    
    # Clean và normalize text
    cleaned_query = legal_text_cleaner(query_input.strip())
    
    # Extract search parameters
    final_results_count = search_config.get('final_results_count', 3)
    search_aggressiveness = search_config.get('search_aggressiveness', 'balanced')
    force_cpu = search_config.get('force_cpu', False)
    
    # Validate parameters
    if final_results_count < 1 or final_results_count > 50:
        raise ValueError("Final results count must be between 1 and 50")
    
    return {
        'query': cleaned_query,
        'final_results_count': final_results_count,
        'search_aggressiveness': search_aggressiveness,
        'force_cpu': force_cpu
    }
```

#### 4.2.2 Search Configuration Management

**Mục đích:** Quản lý cấu hình tìm kiếm và tối ưu hóa parameters

**Các loại cấu hình:**
- **Conservative**: Ít kết quả, chính xác cao
- **Balanced**: Cân bằng giữa số lượng và chất lượng
- **Aggressive**: Nhiều kết quả, đa dạng

**Code logic (pseudocode):**
```python
def calculate_optimal_parameters(final_results_count, search_aggressiveness):
    # Multiplier dựa trên mức độ tìm kiếm
    search_multipliers = {
        'conservative': {'retrieval': 2, 'light': 1.5},
        'balanced': {'retrieval': 3, 'light': 2},
        'aggressive': {'retrieval': 5, 'light': 3}
    }
    
    mult = search_multipliers.get(search_aggressiveness, search_multipliers['balanced'])
    
    # Tier 1: Retrieval - lấy nhiều candidates cho Tier 2
    top_k_retrieval = max(
        config.app.top_k_retrieval,
        final_results_count * mult['retrieval']
    )
    
    # Tier 2: Light Reranking - đủ cho Tier 3 xử lý
    top_k_light_reranking = max(20, final_results_count * mult['light'])
    
    return {
        'top_k_retrieval': top_k_retrieval,
        'top_k_light_reranking': top_k_light_reranking,
        'top_k_final': final_results_count
    }
```

### 4.3 Request Processing Stage

#### 4.3.1 Query Preprocessing

**Mục đích:** Chuẩn bị query cho pipeline processing

**Luồng xử lý:**
```
Raw Query → Text Cleaning → Legal Text Processing → Query Enhancement → Pipeline Ready
     ↓            ↓                ↓                    ↓                ↓
User Input   Remove Noise     Legal-specific      Context Info     Processed
Text         + Normalize      Cleaning            + Metadata       Query
```

**Code logic (pseudocode):**
```python
class LegalTextCleaner(TextProcessor):
    def __call__(self, text):
        # Base processing
        text = super().__call__(text)
        
        # Legal-specific cleaning
        # Remove article/clause numbers
        text = self.article_pattern.sub(r"\1", text)
        
        # Replace common abbreviations
        for abbr, full_text in self.abbreviation_map.items():
            text = re.sub(rf"\b{abbr}\b", full_text, text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

def preprocess_query(query):
    # Initialize legal text cleaner
    cleaner = LegalTextCleaner()
    
    # Clean query
    cleaned_query = cleaner(query)
    
    # Add context information
    query_context = {
        'original_query': query,
        'cleaned_query': cleaned_query,
        'query_length': len(cleaned_query),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    return cleaned_query, query_context
```

#### 4.3.2 Pipeline Initialization và Cache Management

**Mục đích:** Khởi tạo pipeline và quản lý cache để tối ưu hiệu suất

**Luồng xử lý:**
```
Request → Cache Check → Pipeline Load → Model Validation → Ready for Processing
    ↓          ↓            ↓              ↓                ↓
User Query   Cache Hit?   Load Models   Validate Status   Pipeline Ready
             (Yes/No)     + Config      + Health Check     for Execution
```

**Code logic (pseudocode):**
```python
@st.cache_resource(ttl=config.app.cache_pipeline_ttl_seconds)
def load_pipeline(force_cpu=False):
    try:
        # Set CUDA environment variables
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Force CPU nếu cần
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Memory cleanup
        gc.collect()
        
        # Load pipeline
        start_time = time.time()
        pipeline = LegalQAPipeline()
        load_time = time.time() - start_time
        
        # Validate pipeline
        if not pipeline.is_ready:
            raise RuntimeError("Pipeline initialization failed")
        
        # Validate parent law mapping
        ensure_parent_law_mapping()
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Pipeline loading failed: {e}")
        raise

def get_pipeline_lazy():
    """Lazy load pipeline chỉ khi cần thiết"""
    try:
        # Validate parent law mapping
        if ensure_parent_law_mapping():
            logger.info("✅ Parent law mapping validated")
        
        pipeline = LegalQAPipeline()
        return pipeline
        
    except Exception as e:
        logger.warning(f"⚠️ Pipeline not available (handled): {e}")
        return None
```

### 4.4 Pipeline Execution Stage

#### 4.4.1 Tier 1: Bi-Encoder Retrieval

**Mục đích:** Tìm kiếm nhanh candidates ban đầu từ corpus

**Luồng xử lý:**
```
Query → Bi-Encoder → Embedding → FAISS Search → Candidate Retrieval → Score Assignment
  ↓         ↓           ↓           ↓              ↓                ↓
Input   Model      Vector      Similarity      Top-K Docs      Retrieval
Text    Encoding   Generation  Search         with AIDs       Score
```

**Code logic (dựa trên source code thực tế):**
```python
# core/retrieval.py - RetrievalEngine.retrieve()
def execute_tier1_retrieval(pipeline, query, top_k_retrieval):
    logger.info("🎯 Tier 1: Bi-Encoder Retrieval")
    
    try:
        # Get documents từ retriever
        documents = pipeline.retriever.retrieve(query, top_k=top_k_retrieval)
        
        if not documents:
            logger.warning("No documents retrieved from Bi-Encoder")
            return []
        
        # Initialize scores cho các tiers tiếp theo
        for doc in documents:
            doc['light_reranker_score'] = 0.0
            doc['cross_encoder_score'] = 0.0
        
        logger.info(f"✅ Tier 1 completed: {len(documents)} candidates retrieved")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Tier 1 failed: {e}")
        return []

# core/retrieval.py - RetrievalEngine class
class RetrievalEngine:
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        # Encode query thành embedding
        query_embedding = self.bi_encoder.encode([query])
        
        # Tìm kiếm trong FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Lấy thông tin documents
        candidates = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            aid = str(self.index_to_aid[str(idx)])
            content = self.content_map.get(aid, "")
            parent_law = self.aid_to_parent_law.get(aid, "Unknown")
            
            candidates.append({
                'aid': aid,
                'content': content,
                'parent_law_name': parent_law,
                'retrieval_score': float(score),
                'retrieval_rank': i + 1
            })
        
        return candidates
```

#### 4.4.2 Tier 2: Light Reranking

**Mục đích:** Lọc nhanh candidates với domain expertise

**Luồng xử lý:**
```
Candidates → Light Reranker → Similarity Scoring → Score Assignment → Filtered Results
     ↓            ↓                ↓                ↓                ↓
Document List   PhoBERT Model   Query-Doc      Light Reranker   Enhanced
from Tier 1    + ADAPT         Similarity     Score (0.0-1.0)  Candidates
```

**Code logic (dựa trên source code thực tế):**
```python
# core/reranking.py - RerankingEngine.rank_light()
def execute_tier2_light_reranking(pipeline, query, documents, top_k_light):
    logger.info("⚡ Tier 2: Light Reranking")
    
    try:
        if not pipeline.reranker or not pipeline.reranker.is_ready:
            logger.warning("⚠️ Light reranking skipped: reranker not ready")
            return documents
        
        # Light reranking với top-k candidates
        documents = pipeline.reranker.rank_light(query, documents[:top_k_light])
        
        # Debug: Check scores
        for doc in documents[:3]:
            logger.info(f"🔍 After light reranking: light_score={doc.get('light_reranker_score', 'N/A')}")
        
        logger.info(f"✅ Tier 2 completed: {len(documents)} candidates reranked")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Tier 2 failed: {e}")
        return documents

# core/reranking.py - RerankingEngine class
class RerankingEngine:
    def rank_light(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Light reranking với PhoBERT model"""
        if not self.is_ready:
            return documents
        
        # Sử dụng light reranker model
        light_model = self.models.get("light_reranker")
        if light_model:
            return self._rank_with_model("light_reranker", query, documents)
        
        return documents
    
    def _rank_with_model(self, model_name: str, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank documents với specific model"""
        if not documents:
            return documents
        
        # Create sentence pairs
        sentence_pairs = [(query, doc.get('content', '')) for doc in documents]
        
        # Get predictions
        scores = self._predict_batch(model_name, sentence_pairs)
        
        # Assign scores
        for doc, score in zip(documents, scores):
            doc[f'{model_name}_score'] = score
        
        return documents
```

#### 4.4.3 Tier 3: Cross-Encoder Reranking

**Mục đích:** Xếp hạng chính xác cuối cùng với ensemble model

**Luồng xử lý:**
```
Filtered Candidates → Cross-Encoder → Classification → Score Assignment → Final Ranking
         ↓                ↓              ↓              ↓                ↓
Document List      Ensemble Model   Binary Class    Cross-Encoder    Ranked
from Tier 2       (ADAPT + Base)   Prediction      Score (0.0-1.0)  Results
```

**Code logic (dựa trên source code thực tế):**
```python
# core/reranking.py - RerankingEngine.rank_cross()
def execute_tier3_cross_encoder(pipeline, query, documents):
    logger.info("🎯 Tier 3: Cross-Encoder Reranking")
    
    try:
        if not pipeline.reranker or not pipeline.reranker.is_ready:
            logger.warning("⚠️ Cross-encoder reranking skipped: reranker not ready")
            return documents
        
        # Cross-encoder reranking
        documents = pipeline.reranker.rank_cross(query, documents)
        
        # Debug: Check scores
        for doc in documents[:3]:
            logger.info(f"🔍 After cross-encoder: cross_score={doc.get('cross_encoder_score', 'N/A')}")
        
        logger.info(f"✅ Tier 3 completed: {len(documents)} candidates reranked")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Tier 3 failed: {e}")
        return documents

# core/reranking.py - RerankingEngine class
class RerankingEngine:
    def rank_cross(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-encoder reranking với ensemble model"""
        if not self.is_ready:
            return documents
        
        # Sử dụng cross-encoder model
        cross_model = self.models.get("cross_encoder")
        if cross_model:
            return self._rank_with_model("cross_encoder", query, documents)
        
        return documents

# core/reranking.py - EnsembleCrossEncoder class
class EnsembleCrossEncoder:
    def __init__(self, adapt_model, base_model, adapt_weight, base_weight):
        self.adapt_model = adapt_model
        self.base_model = base_model
        self.adapt_weight = adapt_weight
        self.base_weight = base_weight
    
    def __call__(self, **inputs):
        # Get predictions from both models
        with torch.no_grad():
            adapt_output = self.adapt_model(**inputs)
            base_output = self.base_model(**inputs)
        
        # Weighted combination
        ensemble_logits = (
            self.adapt_weight * adapt_output.logits +
            self.base_weight * base_output.logits
        )
        
        return SequenceClassifierOutput(logits=ensemble_logits)
```

#### 4.4.4 Score Aggregation và Final Ranking

**Mục đích:** Tổng hợp scores từ 3 tiers và sắp xếp kết quả cuối cùng

**Luồng xử lý:**
```
Scores from 3 Tiers → Weighted Combination → Score Normalization → Final Ranking
         ↓                    ↓                    ↓                ↓
Retrieval + Light +    Weighted Average    Score Range      Sort by Final
Cross Encoder         (Configurable)       0.0 - 1.0        Score Desc
```

**Code logic (dựa trên source code thực tế):**
```python
# core/pipeline.py - LegalQAPipeline._combine_scores()
def aggregate_scores_and_rank(documents, use_light=True, use_cross=True):
    logger.info("🔄 Combining scores from all tiers...")
    
    try:
        for doc in documents:
            retrieval_score = doc.get('retrieval_score', 0.0)
            light_score = doc.get('light_reranker_score', 0.0)
            cross_score = doc.get('cross_encoder_score', 0.0)
            
            # Flexible score combination
            final_score = retrieval_score
            
            # Add light reranker score
            if use_light:
                final_score = 0.7 * light_score + 0.3 * final_score
            
            # Add cross encoder score
            if use_cross:
                final_score = 0.3 * cross_score + 0.7 * final_score
            
            # Store final score và breakdown
            doc['final_score'] = final_score
            doc['score_breakdown'] = {
                'retrieval': retrieval_score,
                'light_reranker': light_score,
                'cross_encoder': cross_score
            }
        
        # Sort by final score
        final_results = sorted(documents, key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"✅ Score aggregation completed: {len(final_results)} results ranked")
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Score aggregation failed: {e}")
        return documents

# core/pipeline.py - LegalQAPipeline class
class LegalQAPipeline:
    def _combine_scores(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine scores from all tiers"""
        for doc in documents:
            retrieval_score = doc.get('retrieval_score', 0.0)
            light_score = doc.get('light_reranker_score', 0.0)
            cross_score = doc.get('cross_encoder_score', 0.0)
            
            # Weighted combination
            final_score = retrieval_score
            
            # Add light reranker score (70% weight)
            if light_score > 0:
                final_score = 0.7 * light_score + 0.3 * final_score
            
            # Add cross encoder score (30% weight)
            if cross_score > 0:
                final_score = 0.3 * cross_score + 0.7 * final_score
            
            doc['final_score'] = final_score
        
        return documents
```

### 4.5 Result Processing Stage

#### 4.5.1 Result Ranking và Normalization

**Mục đích:** Xử lý và chuẩn hóa kết quả cuối cùng

**Luồng xử lý:**
```
Raw Results → Score Normalization → Result Ranking → Metadata Enrichment → Response Formatting
     ↓              ↓                ↓                ↓                    ↓
Pipeline Output   Score Range      Sort Order      Additional Info     Final Response
                 0.0 - 1.0        by Score        + Context           for User
```

**Code logic (pseudocode):**
```python
def process_final_results(results, top_k_final):
    try:
        # Limit results theo top_k_final
        final_results = results[:top_k_final]
        
        # Normalize scores
        for result in final_results:
            # Ensure scores are in valid range
            result['final_score'] = max(0.0, min(1.0, result['final_score']))
            
            # Add ranking information
            result['final_rank'] = final_results.index(result) + 1
            
            # Add confidence level
            if result['final_score'] >= 0.8:
                result['confidence'] = 'High'
            elif result['final_score'] >= 0.6:
                result['confidence'] = 'Medium'
            else:
                result['confidence'] = 'Low'
        
        # Add summary statistics
        summary = {
            'total_results': len(final_results),
            'avg_score': sum(r['final_score'] for r in final_results) / len(final_results),
            'max_score': max(r['final_score'] for r in final_results),
            'min_score': min(r['final_score'] for r in final_results)
        }
        
        return final_results, summary
        
    except Exception as e:
        logger.error(f"❌ Result processing failed: {e}")
        return results, {}
```

#### 4.5.2 Metadata Enrichment

**Mục đích:** Bổ sung thông tin metadata cho kết quả

**Luồng xử lý:**
```
Basic Results → Parent Law Mapping → Content Enhancement → Metadata Addition → Enriched Results
      ↓              ↓                    ↓                ↓                ↓
Pipeline Output   Law Name Lookup    Content Info      Additional      Complete
                  + Hierarchy        + Structure       Metadata        Results
```

**Code logic (pseudocode):**
```python
def enrich_results_with_metadata(results, pipeline):
    try:
        for result in results:
            aid = result.get('aid')
            
            # Get parent law information
            if hasattr(pipeline.retriever, 'aid_to_parent_law'):
                parent_law = pipeline.retriever.aid_to_parent_law.get(str(aid), 'Unknown')
                result['parent_law_name'] = parent_law
            
            # Add content metadata
            content = result.get('content', '')
            result['content_length'] = len(content)
            result['content_preview'] = content[:200] + '...' if len(content) > 200 else content
            
            # Add processing metadata
            result['processing_timestamp'] = datetime.now().isoformat()
            result['pipeline_version'] = 'v8.3'
        
        return results
        
    except Exception as e:
        logger.warning(f"⚠️ Metadata enrichment failed: {e}")
        return results
```

### 4.6 User Output Stage

#### 4.6.1 Interactive Visualization

**Mục đích:** Hiển thị kết quả với giao diện tương tác

**Luồng xử lý:**
```
Processed Results → Chart Generation → Interactive Display → User Interaction → Enhanced UX
         ↓                ↓                ↓                ↓                ↓
Final Results      Plotly Charts     Streamlit UI      User Clicks      Better
+ Metadata        + Visualizations   + Widgets         + Navigation     Experience
```

**Code logic (pseudocode):**
```python
def create_interactive_visualizations(results):
    # Score comparison chart
    scores_data = {
        'Kết quả': [f"#{i+1}" for i in range(len(results))],
        'Điểm Light Rerank': [r.get('light_reranker_score', 0.0) for r in results],
        'Điểm Retrieval': [r['retrieval_score'] for r in results],
        'Điểm Cross Encoder': [r.get('cross_encoder_score', 0.0) for r in results],
        'Điểm Cuối cùng': [r['final_score'] for r in results]
    }
    
    df_scores = pd.DataFrame(scores_data)
    
    # Create interactive chart
    fig = px.bar(
        df_scores,
        x='Kết quả',
        y=['Điểm Retrieval', 'Điểm Light Rerank', 'Điểm Cross Encoder', 'Điểm Cuối cùng'],
        title='So sánh điểm số các tầng',
        barmode='group'
    )
    
    return fig, df_scores

def display_results_with_tabs(results):
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])
    
    with tab1:
        # Detailed view
        st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"#{i} - {result['aid']} (Điểm: {result['final_score']:.3f})"):
                st.markdown(f"**ID:** {result['aid']}")
                if result.get('parent_law_name'):
                    st.markdown(f"**Tên điều luật:** {result['parent_law_name']}")
                st.markdown("**Nội dung:**")
                st.text(result['content'])
                
                # Score breakdown
                st.markdown("**Điểm từng tầng:**")
                st.markdown(f"- Retrieval: {result['retrieval_score']:.3f}")
                st.markdown(f"- Light Rerank: {result.get('light_reranker_score', 0.0):.3f}")
                st.markdown(f"- Cross Encoder: {result.get('cross_encoder_score', 0.0):.3f}")
    
    with tab2:
        # Score comparison view
        fig, df_scores = create_interactive_visualizations(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Điểm cao nhất", f"{max(df_scores['Điểm Cuối cùng']):.3f}")
        with col2:
            st.metric("Điểm thấp nhất", f"{min(df_scores['Điểm Cuối cùng']):.3f}")
        with col3:
            st.metric("Điểm trung bình", f"{df_scores['Điểm Cuối cùng'].mean():.3f}")
```

#### 4.6.2 Performance Metrics Display

**Mục đích:** Hiển thị metrics hiệu suất cho người dùng

**Luồng xử lý:**
```
Processing Data → Metrics Calculation → Performance Display → User Feedback → Optimization
       ↓                ↓                    ↓                ↓                ↓
Request Info      Timing + Counts      Streamlit UI      User Rating      System
+ Results        + Statistics         + Metrics         + Comments       Improvement
```

**Code logic (pseudocode):**
```python
def display_performance_metrics(start_time, end_time, results_count):
    duration = end_time - start_time
    
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⏱️ Thời gian xử lý", f"{duration:.2f}s")
    
    with col2:
        st.metric("📊 Kết quả tìm thấy", results_count)
    
    with col3:
        if duration > 0:
            st.metric("⚡ Tốc độ", f"{results_count/duration:.1f} kết quả/s")
    
    # Performance insights
    if duration < 1.0:
        st.success("🚀 Xử lý rất nhanh (< 1 giây)")
    elif duration < 3.0:
        st.info("⚡ Xử lý nhanh (< 3 giây)")
    else:
        st.warning("⏳ Xử lý chậm (> 3 giây) - có thể cần tối ưu")

def create_performance_summary(processing_info):
    summary = {
        'total_queries_processed': processing_info.get('query_count', 0),
        'average_processing_time': processing_info.get('avg_time', 0.0),
        'success_rate': processing_info.get('success_rate', 0.0),
        'cache_hit_rate': processing_info.get('cache_hit_rate', 0.0)
    }
    
    return summary
```

### 4.7 Error Handling và Recovery

#### 4.7.1 Error Handling Strategy

**Mục đích:** Xử lý lỗi một cách graceful và cung cấp feedback hữu ích

**Luồng xử lý:**
```
Error Occurrence → Error Classification → Error Handling → User Feedback → Recovery
       ↓                ↓                    ↓                ↓                ↓
Exception Raised   Error Type         Handle Strategy    User Message     System
                   + Severity         + Fallback        + Guidance       Recovery
```

**Code logic (pseudocode):**
```python
def handle_pipeline_errors(error, stage_name):
    error_handlers = {
        'pipeline_not_ready': handle_pipeline_not_ready,
        'model_loading_failed': handle_model_loading_failed,
        'retrieval_failed': handle_retrieval_failed,
        'reranking_failed': handle_reranking_failed,
        'score_aggregation_failed': handle_score_aggregation_failed
    }
    
    # Classify error
    error_type = classify_error(error)
    
    # Get appropriate handler
    handler = error_handlers.get(error_type, handle_generic_error)
    
    # Handle error
    return handler(error, stage_name)

def handle_pipeline_not_ready(error, stage_name):
    st.error("❌ Pipeline chưa sẵn sàng")
    st.info("💡 Vui lòng kiểm tra:")
    st.info("1. Models đã được tải thành công chưa?")
    st.info("2. FAISS index có tồn tại không?")
    st.info("3. Cấu hình có đúng không?")
    
    # Provide recovery options
    if st.button("🔄 Thử khởi tạo lại Pipeline"):
        st.rerun()

def handle_model_loading_failed(error, stage_name):
    st.error(f"❌ Không thể tải model cho {stage_name}")
    st.info("💡 Nguyên nhân có thể:")
    st.info("1. Model files bị thiếu hoặc hỏng")
    st.info("2. Không đủ memory để load model")
    st.info("3. GPU/CPU compatibility issues")
    
    # Suggest CPU mode
    if st.button("💻 Thử chế độ CPU"):
        st.session_state.force_cpu = True
        st.rerun()
```

#### 4.7.2 Recovery Mechanisms

**Mục đích:** Cung cấp cơ chế khôi phục khi gặp lỗi

**Các cơ chế recovery:**
- **Automatic Retry**: Tự động thử lại với exponential backoff
- **Fallback Modes**: Sử dụng chế độ dự phòng (CPU mode, simplified pipeline)
- **Graceful Degradation**: Giảm chất lượng nhưng vẫn hoạt động
- **User Guidance**: Hướng dẫn người dùng khắc phục

**Code logic (pseudocode):**
```python
def implement_recovery_mechanisms(pipeline, error):
    recovery_strategies = [
        try_force_cpu_mode,
        try_simplified_pipeline,
        try_model_reload,
        try_cache_clear
    ]
    
    for strategy in recovery_strategies:
        try:
            if strategy(pipeline, error):
                logger.info(f"✅ Recovery successful with {strategy.__name__}")
                return True
        except Exception as recovery_error:
            logger.warning(f"⚠️ Recovery strategy {strategy.__name__} failed: {recovery_error}")
    
    return False

def try_force_cpu_mode(pipeline, error):
    """Thử chuyển sang CPU mode"""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Reload pipeline với CPU mode
        new_pipeline = LegalQAPipeline()
        if new_pipeline.is_ready:
            st.session_state.pipeline = new_pipeline
            st.session_state.force_cpu = True
            return True
    except Exception as e:
        logger.warning(f"CPU mode recovery failed: {e}")
    
    return False

def try_simplified_pipeline(pipeline, error):
    """Thử sử dụng simplified pipeline (chỉ Tier 1)"""
    try:
        # Disable Tier 2 và Tier 3
        pipeline.use_light_ranking = False
        pipeline.use_cross_encoder = False
        
        # Test với simple query
        test_results = pipeline.predict("test query", top_k_final=1)
        if test_results:
            return True
    except Exception as e:
        logger.warning(f"Simplified pipeline recovery failed: {e}")
    
    return False
```

---

## 5. GIAO DIỆN VÀ BÁO CÁO

### 5.1 Tổng quan giao diện Streamlit

Giao diện của LawBot được xây dựng trên Streamlit với kiến trúc modular và responsive:

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT APP ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│  🎯 MAIN APPLICATION (app.py)                                  │
│  ├── Page Configuration                                        │
│  ├── Sidebar Navigation                                        │
│  ├── Page State Management                                     │
│  └── Error Handling                                            │
├─────────────────────────────────────────────────────────────────┤
│  📱 PAGE MODULES                                               │
│  ├── Search Page (search.py)                                   │
│  ├── Analysis Page (analysis.py)                               │
│  └── System Page (system.py)                                   │
├─────────────────────────────────────────────────────────────────┤
│  🎨 UI COMPONENTS                                              │
│  ├── Interactive Widgets                                       │
│  ├── Data Visualization                                        │
│  ├── Performance Metrics                                       │
│  └── User Feedback                                             │
├─────────────────────────────────────────────────────────────────┤
│  🔧 UTILITY FUNCTIONS                                          │
│  ├── Cache Management                                          │
│  ├── Data Loading                                              │
│  ├── Error Handling                                            │
│  └── Performance Monitoring                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Main Application Structure

#### 5.2.1 App Configuration và Setup

**Mục đích:** Cấu hình chính cho ứng dụng Streamlit

**Luồng khởi tạo:**
```
App Startup → Configuration Load → Page Setup → Navigation Setup → Ready for User
      ↓              ↓                ↓              ↓                ↓
Streamlit Run    Config Files     Page Config     Sidebar Nav      App Ready
                 + Settings       + Layout        + Routing        for Input
```

**Code logic (pseudocode):**
```python
def render_app():
    # Page configuration
    st.set_page_config(
        page_title="Hệ thống Hỏi-Đáp Pháp luật",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide default navigation
    hide_default_navigation = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebarNav"] {display: none !important;}
    </style>
    """
    st.markdown(hide_default_navigation, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Hệ thống QA Pháp luật")
    
    # Page options
    page_options = {
        "🔍 Tìm kiếm & Hỏi đáp": search.render_search_page,
        "📊 Phân tích & Báo cáo": analysis.render_analysis_page,
        "🔧 Trạng thái": system.main
    }
    
    # Page selection
    selected_page = st.sidebar.selectbox(
        "Chọn trang:",
        list(page_options.keys()),
        index=0
    )
    
    # Render selected page
    if selected_page in page_options:
        page_options[selected_page]()
```

#### 5.2.2 Page State Management

**Mục đích:** Quản lý trạng thái giữa các trang và sessions

**Luồng quản lý state:**
```
Page Switch → State Cleanup → New Page Load → State Initialization → Page Ready
      ↓            ↓              ↓              ↓                ↓
User Click    Clear Old      Load New      Initialize New     Page Active
on Nav        Page State     Page          Page State        for User
```

**Code logic (pseudocode):**
```python
def manage_page_state(selected_page_title):
    # Initialize current page in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = selected_page_title
    
    # Clear page-specific states when switching pages
    if st.session_state.current_page != selected_page_title:
        # Clear all page-specific states
        for key in list(st.session_state.keys()):
            if key.endswith("_page_loaded"):
                del st.session_state[key]
        
        # Update current page
        st.session_state.current_page = selected_page_title
        st.rerun()
    
    # Mark page as loaded
    page_key = f"{selected_page_title.lower().replace(' ', '_')}_page_loaded"
    if page_key not in st.session_state:
        st.session_state[page_key] = True
```

### 5.3 Search Page (search.py)

#### 5.3.1 Search Interface Design

**Mục đích:** Cung cấp giao diện tìm kiếm chính cho người dùng

**Luồng xử lý:**
```
User Input → Query Processing → Search Execution → Results Display → User Interaction
      ↓            ↓                ↓                ↓                ↓
Question Text   Text Clean      Pipeline Run     Results Show     User Clicks
+ Parameters   + Validation    + Processing      + Charts         + Navigation
```

**Code logic (dựa trên source code thực tế):**
```python
# app/pages/search.py - render_search_page()
def render_search_page():
    # Clean page state
    if "search_page_loaded" not in st.session_state:
        st.session_state.search_page_loaded = True
    
    # Page title và description
    st.title("🔍 Legal QA - Hệ thống Hỏi đáp Pháp luật")
    st.markdown("Hệ thống AI hỗ trợ tìm kiếm và trả lời câu hỏi pháp luật Việt Nam")
    
    # Sidebar configuration
    with st.sidebar.expander("⚙️ Cấu hình tìm kiếm", expanded=True):
        final_results_count = st.slider(
            "Số lượng kết quả cuối cùng",
            min_value=1, max_value=50, value=3
        )
        
        search_aggressiveness = st.selectbox(
            "Mức độ tìm kiếm",
            options=["conservative", "balanced", "aggressive"],
            index=1
        )
        
        force_cpu = st.checkbox("Sử dụng CPU", value=False)
    
    # Random question feature
    training_questions, test_questions = load_random_questions()
    if training_questions or test_questions:
        with st.sidebar.expander("🎲 Câu hỏi ngẫu nhiên", expanded=True):
            question_source = st.selectbox(
                "Chọn nguồn câu hỏi",
                options=["Training Dataset", "Test Dataset", "Cả hai"]
            )
            
            if st.button("🎲 Lấy câu hỏi ngẫu nhiên"):
                random_q = get_random_question(
                    training_questions + test_questions, "combined"
                )
                st.session_state.random_question = random_q
                st.session_state.current_question = random_q

# app/pages/search.py - load_random_questions()
@st.cache_data(ttl=config.app.cache_questions_ttl_seconds, show_spinner=False)
def load_random_questions() -> Tuple[List[str], List[str]]:
    """Load random questions từ datasets với caching"""
    try:
        # Load training questions
        training_data_path = Path("features/processed_data/training_data.jsonl")
        if training_data_path.exists():
            training_data = load_jsonl(str(training_data_path))
            training_questions = [item.get("question", "") for item in training_data if item.get("question")]
        else:
            training_questions = []
        
        # Load test questions
        test_data_path = Path("features/processed_data/test_data.jsonl")
        if test_data_path.exists():
            test_data = load_jsonl(str(test_data_path))
            test_questions = [item.get("question", "") for item in test_data if item.get("question")]
        else:
            test_questions = []
        
        return training_questions, test_questions
        
    except Exception as e:
        logger.warning(f"Failed to load random questions: {e}")
        return [], []
```

#### 5.3.2 Search Execution và Results Display

**Mục đích:** Thực hiện tìm kiếm và hiển thị kết quả

**Luồng xử lý:**
```
Search Button → Pipeline Execution → Results Processing → Display → User Feedback
      ↓              ↓                ↓                ↓            ↓
User Click     3-Tier Pipeline    Score Aggreg    Interactive    User Rating
               + Processing       + Ranking       Visualization   + Comments
```

**Code logic (dựa trên source code thực tế):**
```python
# app/pages/search.py - execute_search()
def execute_search(query, final_results_count, search_aggressiveness):
    # Calculate optimal parameters
    params = calculate_optimal_parameters(final_results_count, search_aggressiveness)
    
    # Display calculated parameters
    with st.expander("📊 Thông số tìm kiếm được tính toán tự động"):
        st.markdown(f"**🎯 Tầng 1 - Retrieval:** {params['top_k_retrieval']} ứng viên")
        st.markdown(f"**⚡ Tầng 2 - Light Reranking:** {params['top_k_light_reranking']} ứng viên")
        st.markdown(f"**🎯 Tầng 3 - Final Reranking:** {params['top_k_final']} kết quả cuối cùng")
    
    # Performance monitoring
    start_time = time.time()
    
    with st.spinner("🔎 Đang tìm kiếm... vui lòng đợi"):
        # Get results from pipeline
        results = pipeline.predict(
            query,
            top_k_retrieval=params["top_k_retrieval"],
            top_k_final=params["top_k_final"]
        )
    
    end_time = time.time()
    
    # Display performance metrics
    display_performance_metrics(start_time, end_time, len(results))
    
    # Display results in tabs
    tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])
    
    with tab1:
        display_detailed_results(results)
    
    with tab2:
        display_score_comparison(results)

# app/pages/search.py - calculate_optimal_parameters()
def calculate_optimal_parameters(final_results_count: int, search_aggressiveness: str):
    """Calculate optimal parameters dựa trên search aggressiveness"""
    # Multiplier dựa trên mức độ tìm kiếm
    search_multipliers = {
        'conservative': {'retrieval': 2, 'light': 1.5},
        'balanced': {'retrieval': 3, 'light': 2},
        'aggressive': {'retrieval': 5, 'light': 3}
    }
    
    mult = search_multipliers.get(search_aggressiveness, search_multipliers['balanced'])
    
    # Tier 1: Retrieval - lấy nhiều candidates cho Tier 2
    top_k_retrieval = max(
        config.app.top_k_retrieval,
        final_results_count * mult['retrieval']
    )
    
    # Tier 2: Light Reranking - đủ cho Tier 3 xử lý
    top_k_light_reranking = max(20, final_results_count * mult['light'])
    
    return {
        'top_k_retrieval': top_k_retrieval,
        'top_k_light_reranking': top_k_light_reranking,
        'top_k_final': final_results_count
    }
```

#### 5.3.3 Results Visualization

**Mục đích:** Hiển thị kết quả với biểu đồ tương tác

**Luồng xử lý:**
```
Search Results → Data Processing → Chart Generation → Interactive Display → User Analysis
       ↓              ↓                ↓                ↓                ↓
Pipeline Output   Data Format      Plotly Charts     Streamlit UI      User
+ Metadata       + Aggregation    + Visualizations  + Widgets         Interaction
```

**Code logic (pseudocode):**
```python
def display_score_comparison(results):
    # Score comparison chart
    scores_data = {
        'Kết quả': [f"#{i+1}" for i in range(len(results))],
        'Điểm Light Rerank': [r.get("light_reranker_score", 0.0) for r in results],
        'Điểm Retrieval': [r["retrieval_score"] for r in results],
        'Điểm Cross Encoder': [r.get("cross_encoder_score", 0.0) for r in results]
    }
    
    df_scores = pd.DataFrame(scores_data)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Điểm cao nhất", f"{max(scores_data['Điểm Light Rerank']):.3f}")
    with col2:
        st.metric("Điểm thấp nhất", f"{min(scores_data['Điểm Light Rerank']):.3f}")
    with col3:
        st.metric("Điểm trung bình", f"{sum(scores_data['Điểm Light Rerank'])/len(scores_data['Điểm Light Rerank']):.3f}")
    
    # Interactive chart
    st.bar_chart(df_scores.set_index("Kết quả"))

def display_detailed_results(results):
    st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")
    
    for i, result in enumerate(results, 1):
        with st.expander(f"#{i} - {result['aid']} (Điểm: {result['final_score']:.3f})", expanded=(i <= 3)):
            st.markdown(f"**ID:** {result['aid']}")
            
            if result.get("parent_law_name"):
                st.markdown(f"**Tên điều luật:** {result['parent_law_name']}")
            
            st.markdown("**Nội dung:**")
            st.text(result["content"])
            
            # Score breakdown
            st.markdown(f"**Điểm retrieval:** {result['retrieval_score']:.3f}")
            st.markdown(f"**Điểm light rerank:** {result.get('light_reranker_score', 0.0):.3f}")
            st.markdown(f"**Điểm cross encoder:** {result.get('cross_encoder_score', 0.0):.3f}")
```

### 5.4 Analysis Page (analysis.py)

#### 5.4.1 Comprehensive Evaluation Interface

**Mục đích:** Cung cấp giao diện đánh giá toàn diện hệ thống

**Luồng xử lý:**
```
Page Load → Auto-evaluation → Results Processing → Visualization → User Analysis
     ↓            ↓                ↓                ↓            ↓
Streamlit UI   Pipeline Eval    Metrics Calc     Charts       User
+ Navigation   + Metrics        + Aggregation    + Tables     Interaction
```

**Code logic (pseudocode):**
```python
def render_analysis_page():
    # Clean page state
    if "analysis_page_loaded" not in st.session_state:
        st.session_state.analysis_page_loaded = True
        st.session_state.comprehensive_eval_results = None
        st.session_state.eval_loading = False
    
    st.title("📊 Báo cáo phân tích chi tiết")
    st.markdown("Phân tích toàn diện hiệu suất từng tầng và khuyến nghị cải thiện")
    
    # Load analysis data
    analysis_data = load_analysis_data()
    model_status = analysis_data.get("models", {})
    faiss_status = analysis_data.get("faiss", {})
    eval_data = analysis_data.get("evaluation")
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "🏥 Tổng quan",
        "🧪 Phân tích đánh giá toàn diện"
    ])
    
    with tab1:
        render_overview_tab(model_status, faiss_status)
    
    with tab2:
        render_comprehensive_evaluation_tab()
```

#### 5.4.2 Auto-evaluation System

**Mục đích:** Tự động chạy evaluation và cache kết quả

**Luồng xử lý:**
```
Auto-trigger → Pipeline Check → Evaluation Run → Results Cache → Display
      ↓            ↓                ↓              ↓            ↓
Page Load      Pipeline Ready   Metrics Calc    Save Results   Show User
+ Timer        + Validation     + Processing    + Cache        + Charts
```

**Code logic (dựa trên source code thực tế):**
```python
# app/pages/analysis.py - auto_run_comprehensive_evaluation()
def auto_run_comprehensive_evaluation():
    try:
        logger.info("🔄 Auto-running comprehensive evaluation...")
        
        # First, try to load from saved file
        cached_results = load_latest_comprehensive_evaluation()
        if cached_results:
            logger.info("✅ Loaded comprehensive evaluation results from cache file")
            return cached_results
        
        logger.info("🔄 No cached results found, running fresh evaluation...")
        
        # Get pipeline
        pipeline = get_pipeline_lazy()
        if not pipeline:
            logger.warning("⚠️ Pipeline not available (handled) for auto-evaluation")
            return None
        
        # Run evaluation
        eval_results = run_comprehensive_evaluation(pipeline)
        
        if eval_results:
            logger.info("✅ Auto-comprehensive evaluation completed successfully")
            return eval_results
        else:
            logger.warning("⚠️ Auto-comprehensive evaluation returned no results")
            return None
            
    except Exception as e:
        logger.error(f"❌ Auto-comprehensive evaluation failed: {e}")
        return None

# app/pages/analysis.py - run_comprehensive_evaluation()
def run_comprehensive_evaluation(pipeline, test_queries=None):
    if test_queries is None:
        # Load validation sets
        validation_dir = Path("features/validation_sets")
        if validation_dir.exists():
            manager = ValidationSetManager()
            tier1_val, tier2_val, tier3_val = manager.load_validation_sets(str(validation_dir))
            
            if tier1_val and tier2_val and tier3_val:
                test_queries = [
                    item.get("query", "") for item in tier1_val[:10]
                    if item.get("query")
                ]
    
    # Initialize results structure
    evaluation_results = {
        "tier_1": {}, "tier_2": {}, "tier_3": {}, "combined": {}
    }
    
    # Process queries in batches
    for query in test_queries:
        results = pipeline.predict(query, top_k_final=10)
        if results:
            tier_metrics = calculate_tier_metrics_optimized(results, [1, 3, 5, 10])
            
            # Update evaluation results
            for tier, metrics in tier_metrics.items():
                for metric_name, values in metrics.items():
                    if metric_name not in evaluation_results[tier]:
                        evaluation_results[tier][metric_name] = []
                    evaluation_results[tier][metric_name].extend(values)
    
    # Calculate final averages
    final_results = calculate_final_averages_optimized(evaluation_results)
    
    return final_results

# app/pages/analysis.py - get_pipeline_lazy()
def get_pipeline_lazy():
    """Lazy load pipeline chỉ khi cần thiết"""
    try:
        # Validate parent law mapping
        if ensure_parent_law_mapping():
            logger.info("✅ Parent law mapping validated")
        
        pipeline = LegalQAPipeline()
        return pipeline
        
    except Exception as e:
        logger.warning(f"⚠️ Pipeline not available (handled): {e}")
        return None
```

#### 5.4.3 Performance Metrics Visualization

**Mục đích:** Hiển thị metrics hiệu suất với biểu đồ tương tác

**Luồng xử lý:**
```
Evaluation Results → Metrics Processing → Chart Generation → Interactive Display → User Analysis
         ↓                ↓                ↓                ↓                ↓
Raw Metrics      Data Format      Plotly Charts     Streamlit UI      User
+ Calculations   + Aggregation    + Visualizations  + Widgets         Interaction
```

**Code logic (pseudocode):**
```python
def create_tier_performance_chart(eval_results):
    if not eval_results:
        return None
    
    # Prepare data for visualization
    metrics = ["precision", "recall", "f1", "ndcg", "mrr", "quality"]
    tiers = ["tier_1", "tier_2", "tier_3", "combined"]
    tier_names = ["Retrieval", "Light Reranker", "Cross Encoder", "Combined"]
    
    # Create data for the chart
    chart_data = []
    for i, tier in enumerate(tiers):
        for metric in metrics:
            avg_value = eval_results[tier].get(f"{metric}_avg", 0.0)
            chart_data.append({
                "Tier": tier_names[i],
                "Metric": metric.upper(),
                "Value": avg_value,
                "Tier_Type": "Individual" if i < 3 else "Combined",
                "Color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][i]
            })
    
    df = pd.DataFrame(chart_data)
    
    # Create enhanced bar chart
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Tier",
        title="🎯 Tier Performance Comparison - Comprehensive Metrics",
        color_discrete_map={
            "Retrieval": "#1f77b4",
            "Light Reranker": "#ff7f0e",
            "Cross Encoder": "#2ca02c",
            "Combined": "#d62728"
        },
        barmode="group",
        text="Value"
    )
    
    # Enhance chart appearance
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont_size=10
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="📊 Metrics",
        yaxis_title="📈 Score",
        showlegend=True,
        title_font_size=16,
        title_x=0.5,
        plot_bgcolor="white",
        bargap=0.2,
        bargroupgap=0.1
    )
    
    return fig

def create_tier_improvement_chart(eval_results):
    if not eval_results:
        return None
    
    # Calculate improvement percentages
    improvement_data = []
    
    for metric in ["precision", "recall", "f1", "ndcg", "mrr", "quality"]:
        # Get best individual tier performance
        individual_scores = [
            eval_results["tier_1"].get(f"{metric}_avg", 0.0),
            eval_results["tier_2"].get(f"{metric}_avg", 0.0),
            eval_results["tier_3"].get(f"{metric}_avg", 0.0)
        ]
        best_individual = max(individual_scores)
        combined_score = eval_results["combined"].get(f"{metric}_avg", 0.0)
        
        if best_individual > 0:
            improvement = ((combined_score - best_individual) / best_individual) * 100
        else:
            improvement = 0.0
        
        improvement_data.append({
            "Metric": metric.upper(),
            "Best Individual": best_individual,
            "Combined": combined_score,
            "Improvement %": improvement
        })
    
    df = pd.DataFrame(improvement_data)
    
    # Create enhanced improvement chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="🏆 Best Individual Tier",
        x=df["Metric"],
        y=df["Best Individual"],
        marker_color="lightblue",
        text=df["Best Individual"].round(3),
        textposition="outside"
    ))
    
    fig.add_trace(go.Bar(
        name="🚀 Combined Pipeline",
        x=df["Metric"],
        y=df["Combined"],
        marker_color="darkblue",
        text=df["Combined"].round(3),
        textposition="outside"
    ))
    
    # Add improvement annotations
    for i, row in df.iterrows():
        if row["Improvement %"] > 0:
            fig.add_annotation(
                x=row["Metric"],
                y=row["Combined"] + 0.01,
                text=f"+{row['Improvement %']:.1f}%",
                showarrow=False,
                font=dict(color="green", size=12, weight="bold"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
    
    fig.update_layout(
        title="📈 Performance Improvement: Individual vs Combined Pipeline",
        height=500,
        barmode="group",
        xaxis_title="📊 Metrics",
        yaxis_title="📈 Score",
        plot_bgcolor="white",
        showlegend=True
    )
    
    return fig
```

### 5.5 System Page (system.py)

#### 5.5.1 System Health Dashboard

**Mục đích:** Hiển thị trạng thái tổng quan của hệ thống

**Luồng xử lý:**
```
Page Load → System Check → Status Collection → Health Calculation → Dashboard Display
     ↓            ↓              ↓                ↓                ↓
Streamlit UI   Pipeline      Model Status     Health Score     Interactive
+ Navigation   + Models      + FAISS Index    + Metrics        Dashboard
```

**Code logic (dựa trên source code thực tế):**
```python
# app/pages/system.py - create_system_health_dashboard()
def create_system_health_dashboard(system_data):
    st.subheader("🏥 System Health Dashboard")
    
    if not system_data:
        st.warning("⚠️ Không thể load dữ liệu hệ thống")
        return
    
    # Overall system status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check if models are ready
        model_status = system_data.get("model_status", {})
        models_ready = sum(
            1 for model in model_status.values() 
            if model.get("status") == "ready"
        )
        total_models = len(model_status)
        
        if total_models > 0:
            health_percentage = (models_ready / total_models) * 100
        else:
            health_percentage = 0
        
        st.metric("System Health", f"{health_percentage:.0f}%")
    
    with col2:
        # FAISS Index status
        faiss_status = system_data.get("faiss_status", {})
        faiss_ready = "✅ Ready" if faiss_status.get("status") == "ready" else "❌ Not Ready"
        st.metric("FAISS Index", faiss_ready)
    
    with col3:
        # Last update time
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    # Health gauge chart
    if system_data.get("evaluation_data"):
        health_score = system_data["evaluation_data"].get(
            "pipeline_health_score", health_percentage
        )
    else:
        health_score = health_percentage
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "System Health Score"},
        delta={"reference": 100},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightcoral"},
                {"range": [50, 80], "color": "lightyellow"},
                {"range": [80, 100], "color": "lightgreen"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# app/pages/system.py - load_system_data()
def load_system_data():
    """Load system data từ multiple sources"""
    try:
        # Get model status
        model_status = get_model_status()
        
        # Get FAISS index status
        faiss_status = get_faiss_index_status()
        
        # Get directory status
        directory_status = get_directory_status()
        
        # Get latest evaluation report
        evaluation_data = load_latest_evaluation_report()
        
        # Get device info
        device_info = get_device_info()
        
        return {
            "model_status": model_status,
            "faiss_status": faiss_status,
            "directory_status": directory_status,
            "evaluation_data": evaluation_data,
            "device_info": device_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to load system data: {e}")
        return None
```

#### 5.5.2 Tier Configuration Information

**Mục đích:** Hiển thị thông tin cấu hình chi tiết từng tầng

**Luồng xử lý:**
```
System Data → Tier Info Extraction → Configuration Display → Status Check → User Info
      ↓              ↓                    ↓                ↓            ↓
Raw System      Tier-specific      Config Tables      Health Check    Detailed
Data            Config Data        + Status           + Validation     Information
```

**Code logic (pseudocode):**
```python
def create_tier_configuration_info(system_data):
    st.subheader("🎯 Thông tin Cấu hình từng Tầng")
    
    if not system_data:
        st.warning("⚠️ Không thể load thông tin cấu hình")
        return
    
    model_status = system_data.get("model_status", {})
    
    # Create tier configuration table
    tier_configs = {
        "Bi-Encoder (Tầng 1)": {
            "model": model_status.get("bi_encoder", {}),
            "description": "Retrieval model sử dụng Sentence Transformers",
            "status_key": "bi_encoder"
        },
        "Light Reranker (Tầng 2)": {
            "model": model_status.get("light_reranker", {}),
            "description": "Light reranking model sử dụng PhoBERT",
            "status_key": "light_reranker"
        },
        "Cross-Encoder (Tầng 3)": {
            "model": model_status.get("cross_encoder", {}),
            "description": "Cross-encoder model kết hợp ensemble",
            "status_key": "cross_encoder"
        }
    }
    
    # Display tier information
    for tier_name, config in tier_configs.items():
        with st.expander(f"🔧 {tier_name}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Mô tả:** {config['description']}")
                
                model_info = config["model"]
                if model_info:
                    status = model_info.get("status", "unknown")
                    model_path = model_info.get("path", "N/A")
                    size_mb = model_info.get("size_mb", 0)
                    
                    # Status indicator
                    if status == "ready":
                        st.success(f"✅ Status: Ready")
                    elif status == "partially_ready":
                        st.warning(f"⚠️ Status: Partially Ready")
                    else:
                        st.error(f"❌ Status: {status}")
                    
                    st.write(f"**Model Path:** `{model_path}`")
                    st.write(f"**Model Size:** {size_mb:.1f} MB")
                else:
                    st.error("❌ Không có thông tin model")
            
            with col2:
                # Quick status check
                if model_info and model_info.get("status") == "ready":
                    st.success("✅ Hoạt động bình thường")
                else:
                    st.warning("⚠️ Cần kiểm tra")
```

### 5.6 Reporting và Analytics System

#### 5.6.1 Comprehensive Evaluation Reports

**Mục đích:** Tạo báo cáo đánh giá toàn diện hệ thống

**Luồng xử lý:**
```
Evaluation Run → Metrics Calculation → Report Generation → File Storage → User Access
       ↓              ↓                ↓                ↓            ↓
Pipeline Test   Performance      Report Format     Save to Disk    Display
+ Validation    Metrics         + Visualizations  + Metadata      + Download
```

**Code logic (pseudocode):**
```python
def save_comprehensive_evaluation_results(results, filename=None):
    try:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_evaluation_{timestamp}.json"
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        filepath = reports_dir / filename
        
        # Prepare data for saving
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "metadata": {
                "version": "v8.3",
                "type": "comprehensive_evaluation",
                "source": "analysis_page"
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Comprehensive evaluation results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation results: {e}")
        return None

def load_latest_comprehensive_evaluation():
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return None
        
        # Find the latest comprehensive evaluation file
        comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
        if not comp_files:
            return None
        
        # Get the most recent file
        latest_file = max(comp_files, key=lambda p: p.stat().st_mtime)
        
        # Check if file is recent (within last 24 hours)
        file_age = time.time() - latest_file.stat().st_mtime
        if file_age > 86400:  # 24 hours in seconds
            logger.info("⚠️ Latest evaluation file is older than 24 hours")
            return None
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded evaluation results from: {latest_file}")
        return data.get("results")
        
    except Exception as e:
        logger.error(f"❌ Failed to load evaluation results: {e}")
        return None
```

#### 5.6.2 Performance Metrics và Insights (Updated: 2025-01-27)

**Mục đích:** Cung cấp insights về hiệu suất hệ thống với comprehensive evaluation

**Luồng xử lý:**
```
Raw Metrics → Multi-tier Analysis → Insights Generation → HPO Results → User Display
      ↓              ↓                    ↓                ↓              ↓
Performance     Tier-specific        Performance      Optimization    Actionable
Data           Evaluation          Insights         Results         Information
```

**Latest Performance Results (2025-01-27):**
- **Tier 1 (Retrieval)**: Bi-Encoder với contrastive learning và HNM
- **Tier 2 (Light Reranker)**: PhoBERT với ADAPT domain adaptation
- **Tier 3 (Cross-Encoder)**: Ensemble model với weighted combination
- **Combined Pipeline**: Multi-tier evaluation với comprehensive metrics
- **Centralized Path Management**: Automated data freshness validation
- **Smart Workflow**: Tự động skip stages khi dữ liệu đã sẵn sàng

**Code logic (pseudocode):**
```python
def create_performance_insights(eval_results):
    st.subheader("💡 Phân tích Chi tiết & Khuyến nghị")
    
    # Analyze each tier's performance
    tier_analysis = {}
    for tier in ["tier_1", "tier_2", "tier_3"]:
        tier_metrics = eval_results[tier]
        avg_f1 = tier_metrics.get("f1_avg", 0.0)
        avg_quality = tier_metrics.get("quality_avg", 0.0)
        
        if avg_f1 >= 0.8 and avg_quality >= 0.7:
            status = "✅ Xuất sắc"
            recommendation = "Tầng này hoạt động rất tốt, không cần cải thiện"
        elif avg_f1 >= 0.6 and avg_quality >= 0.5:
            status = "🟡 Tốt"
            recommendation = "Có thể cải thiện thêm để đạt hiệu suất cao hơn"
        else:
            status = "🔴 Cần cải thiện"
            recommendation = "Cần xem xét lại training data và hyperparameters"
        
        tier_analysis[tier] = {
            "status": status,
            "f1_score": avg_f1,
            "quality_score": avg_quality,
            "recommendation": recommendation
        }
    
    # Display tier analysis
    col1, col2, col3 = st.columns(3)
    tier_names = {
        "tier_1": "🎯 Tầng 1 (Retrieval)",
        "tier_2": "⚡ Tầng 2 (Light Reranker)",
        "tier_3": "🎯 Tầng 3 (Cross Encoder)"
    }
    
    for i, (tier, analysis) in enumerate(tier_analysis.items()):
        with [col1, col2, col3][i]:
            st.subheader(tier_names[tier])
            st.metric("Status", analysis["status"])
            st.metric("F1 Score", f"{analysis['f1_score']:.3f}")
            st.metric("Quality", f"{analysis['quality_score']:.3f}")
            st.info(analysis["recommendation"])
```

---

## 6. KỸ THUẬT CODE VÀ KIẾN TRÚC (Updated: 2025-08-20)

### 6.1 Centralized Configuration Management

**Mục đích:** Tập trung hóa cấu hình để đảm bảo tính nhất quán và dễ bảo trì

#### 6.1.1 Model Configuration (`config/default.yml`)

**Centralized Configuration v8.3:**
```yaml
# Enhanced Bi-Encoder with Contrastive Learning + ADAPT
bi_encoder:
  model_name: "bkai-foundation-models/vietnamese-bi-encoder"  # Vietnamese-specific bi-encoder
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      loss_type: "TripletLoss"
      max_length: 256
      batch_size: 16
      epochs: 5
    adapt:
      enabled: true
      domain_data_ratio: 0.8
      adaptation_steps: 1000
      learning_rate: 2e-5
      warmup_steps: 100
      hard_negative_mining: true  # Enable HNM for Tier 1
  training_params:
    batch_size: 32
    max_length: 256
    epochs: 3
    learning_rate: 0.00002  # 2e-5
    warmup_steps: 100
    weight_decay: 0.01

# Enhanced reranker pipeline với proper architecture
reranker_pipeline:
  combined_reranker:
    enabled: true
    ensemble_method: "weighted_average"
    enhancement_strategy: "ADAPT-enhanced from Tier 2 + Base model ensemble"
    models:
      adapt_enhanced_phobert_base_v2:
        enabled: true
        weight: 0.7  # 70% contribution - ADAPT-enhanced từ Tier 2
        model_name: "vinai/phobert-base-v2"
        max_length: 256
        enhancement: "ADAPT-enhanced từ Tier 2 Light Reranker training"
        purpose: "Legal domain expertise với optimized performance từ Tier 2"
        status: "Fine-tuned với ADAPT từ Tier 2 + HPO + HNM"
        source: "Tier 2 ADAPT-enhanced model"
      base_phobert_large:
        enabled: true
        weight: 0.3  # 30% contribution - base model gốc
        model_name: "vinai/phobert-large"
        max_length: 512
        enhancement: "Base model gốc chưa fine-tune"
        purpose: "High quality general performance"
        status: "Original pre-trained model"
    top_k: 20
    performance_balance: "70% domain expertise (ADAPT từ Tier 2) + 30% general quality (base)"
    ensemble_strategy: "Domain expertise từ Tier 2 + General quality balance"
    techniques: ["HPO", "Hard Negative Mining", "Ensemble training"]

# App Configuration
app:
  top_k_retrieval: 10      # Tăng từ 100 → 150 (+50%) để tăng Recall
  top_k_light: 80          # Tăng từ 80 → 120 (+50%) để tăng Recall
  top_k_final: 5            # Giảm từ 30 → 5 để phù hợp với nhu cầu thực tế (3-5 kết quả)
  cache_questions_ttl_seconds: 3600
  cache_pipeline_ttl_seconds: 7200
  reranker_weights:
    adapt_enhanced_phobert_base_v2: 0.7
    base_phobert_large: 0.3
```

#### 6.1.2 Path Configuration (`config/paths.py`)

**TRAINING_DATA_PATHS Dictionary:**
```python
TRAINING_DATA_PATHS = {
    "tier_1": {
        "data_source": "bi_encoder_train.jsonl",
        "required_files": ["bi_encoder_train.jsonl", "processed_corpus.json"],
        "validation_threshold": 1000,  # Minimum records required
        "description": "Bi-Encoder training data with contrastive learning"
    },
    "tier_2": {
        "data_source": "training_data.jsonl", 
        "required_files": ["training_data.jsonl", "negative_pool.jsonl"],
        "validation_threshold": 500,
        "description": "Light reranker training data with hard negative mining"
    },
    "tier_3": {
        "data_source": "cross_encoder_train.jsonl",
        "required_files": ["cross_encoder_train.jsonl", "processed_corpus.json"],
        "validation_threshold": 800,
        "description": "Cross-encoder training data for ensemble learning"
    }
}
```

### 6.2 Advanced HPO & Evaluation System

**Mục đích:** Tối ưu hóa hyperparameters và đánh giá toàn diện hiệu suất hệ thống

#### 6.2.1 Hyperparameter Optimization

**Advanced HPO Configuration:**
```python
class AdvancedHPO:
    def create_study(self, model_type):
        """Create advanced Optuna study with TPE sampler and pruning"""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=5,
                n_ei_candidates=24
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        return study
    
    def objective(self, trial, model_type):
        """Advanced objective function with comprehensive parameters"""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 2, 10),
            "warmup_steps": trial.suggest_int("warmup_steps", 50, 200),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            "hard_negative_ratio": trial.suggest_float("hard_negative_ratio", 0.1, 0.5),
            "similarity_threshold": trial.suggest_float("similarity_threshold", 0.5, 0.9),
            "use_adapt_enhanced": trial.suggest_categorical("use_adapt_enhanced", [True, False])
        }
        
        # Train model với early stopping
        model = self._train_with_early_stopping(params, patience=5)
        
        # Evaluate với comprehensive metrics
        score = self._evaluate_comprehensive(model)
        
        return score
```

#### 6.2.2 Comprehensive Evaluation

**Multi-tier Evaluation System:**
```python
class ComprehensiveEvaluator:
    def run_comprehensive_evaluation(self, pipeline, test_queries):
        """Run comprehensive evaluation for all tiers"""
        evaluation_results = {
            "tier_1": {},
            "tier_2": {},
            "tier_3": {},
            "combined": {}
        }
        
        # Process each query with tier-specific evaluation
        for query in test_queries:
            # Tier 1: Independent retrieval evaluation
            tier1_results = self.evaluate_tier_1_retrieval_only(pipeline, query)
            tier1_metrics = self.calculate_tier_metrics(tier1_results, "tier_1")
            self.update_evaluation_results(evaluation_results, "tier_1", tier1_metrics)
            
            # Tier 2: Independent light reranker evaluation
            tier2_results = self.evaluate_tier_2_light_reranker_only(pipeline, query)
            tier2_metrics = self.calculate_tier_metrics(tier2_results, "tier_2")
            self.update_evaluation_results(evaluation_results, "tier_2", tier2_metrics)
            
            # Tier 3: Independent cross encoder evaluation
            tier3_results = self.evaluate_tier_3_cross_encoder_only(pipeline, query)
            tier3_metrics = self.calculate_tier_metrics(tier3_results, "tier_3")
            self.update_evaluation_results(evaluation_results, "tier_3", tier3_metrics)
            
            # Combined: Full pipeline evaluation
            combined_results = pipeline.predict(query, top_k_final=20)
            combined_metrics = self.calculate_tier_metrics(combined_results, "combined")
            self.update_evaluation_results(evaluation_results, "combined", combined_metrics)
        
        # Calculate final averages và save results
        final_results = self.calculate_final_averages(evaluation_results)
        self.save_comprehensive_evaluation_results(final_results)
        
        return final_results
``` 
        "purpose": "Final ranking",
        "tier": 3,
        "performance": "High accuracy (~100+ docs/sec)",
        "techniques": ["Ensemble", "HPO", "HNM"]
    }
}
```

**MODEL_DIRECTORY_MAPPING:**
```python
MODEL_DIRECTORY_MAPPING = {
    "bi_encoder": "models/bi_encoder",
    "light_reranker": "models/light_reranker", 
    "cross_encoder": "models/cross_encoder"
}
```

**MODEL_STATUS_KEYS:**
```python
MODEL_STATUS_KEYS = {
    "bi_encoder": "bi_encoder",
    "light_reranker": "light_reranker",
    "cross_encoder": "cross_encoder"
}
```

#### 6.1.2 Path Configuration (`config/paths.py`)

**TRAINING_DATA_PATHS:**
```python
TRAINING_DATA_PATHS = {
    "bi_encoder": {
        "primary": "features/processed_data_{timestamp}/bi_encoder_train.jsonl",
        "alternative": "features/processed_data/bi_encoder_train.jsonl"
    },
    "light_reranker": {
        "primary": "features/processed_data_{timestamp}/training_data.jsonl",
        "alternative": "features/processed_data/training_data.jsonl"
    },
    "cross_encoder": {
        "primary": "features/processed_data_{timestamp}/training_data.jsonl", 
        "alternative": "features/processed_data/training_data.jsonl"
    }
}
```

**DATA_QUALITY_THRESHOLDS:**
```python
DATA_QUALITY_THRESHOLDS = {
    "bi_encoder": {
        "min_samples": 1000,
        "min_avg_length": 50,
        "required_fields": ["question", "answer", "context"]
    },
    "light_reranker": {
        "min_samples": 500,
        "min_avg_length": 30,
        "required_fields": ["question", "answer", "negative"]
    },
    "cross_encoder": {
        "min_samples": 300,
        "min_avg_length": 25,
        "required_fields": ["question", "answer", "label"]
    }
}
```

**Helper Functions:**
```python
def validate_training_data_paths():
    """Validate all training data paths and return validation results
    Returns: {
        'tier_1': {'status': 'ready', 'data_source': 'real', 'file_count': int, 'files': [str]},
        'tier_2': {'status': 'ready', 'data_source': 'real', 'file_count': int, 'files': [str]},
        'tier_3': {'status': 'ready', 'data_source': 'real', 'file_count': int, 'files': [str]},
        'overall': {'status': 'ready', 'real_data_available': bool, 'tiers_ready': int},
        'latest_data_dir': str
    }
    """
    
def get_training_data_path(tier_name: str) -> str:
    """Get the correct training data path for a specific tier"""
    
def get_latest_processed_data_dir() -> str:
    """Find the latest timestamped processed data directory"""
```

#### 6.1.3 Configuration Integration

**Workflow Integration:**
```python
# In run_workflow.py
from config.paths import validate_training_data_paths, get_training_data_path
from config.models import get_model_config, get_display_name

# Validate data before training
validation_results = validate_training_data_paths()
if validation_results["overall"]["status"] != "ready":
    logger.error("❌ Training data validation failed")
    return False

# Get model-specific paths
bi_encoder_path = get_training_data_path("bi_encoder")
light_reranker_path = get_training_data_path("light_reranker")
cross_encoder_path = get_training_data_path("cross_encoder")
```

**UI Integration:**
```python
# In app/pages/system.py and app/pages/analysis.py
from config.models import MODEL_STATUS_KEYS, get_display_name

# Use centralized keys
tier_configs = {
    "Bi-Encoder (Tầng 1)": {
        "model": model_status.get(MODEL_STATUS_KEYS["bi_encoder"], {}),
        "status_key": MODEL_STATUS_KEYS["bi_encoder"],
    },
    # ... other tiers
}
```

### 6.2 Kiến trúc Code tổng thể

#### 6.2.1 Design Patterns và Principles

**Mục đích:** Áp dụng các design patterns và principles để tạo code maintainable và scalable

**Các patterns được sử dụng:**
- **Factory Pattern**: Model creation và pipeline initialization
- **Strategy Pattern**: Different reranking strategies
- **Observer Pattern**: Progress tracking và monitoring
- **Template Method Pattern**: Base training scripts
- **Singleton Pattern**: Configuration management

**Code structure:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE ARCHITECTURE PATTERNS                   │
├─────────────────────────────────────────────────────────────────┤
│  🏭 FACTORY PATTERN                                            │
│  ├── ModelFactory: Tạo models theo configuration               │
│  ├── PipelineFactory: Khởi tạo pipeline với models            │
│  └── DatasetFactory: Tạo datasets theo loại training          │
├─────────────────────────────────────────────────────────────────┤
│  🎯 STRATEGY PATTERN                                           │
│  ├── RetrievalStrategy: Different retrieval methods            │
│  ├── RerankingStrategy: Different reranking approaches        │
│  └── EvaluationStrategy: Different evaluation metrics          │
├─────────────────────────────────────────────────────────────────┤
│  👁️ OBSERVER PATTERN                                           │
│  ├── ProgressTracker: Track training progress                  │
│  ├── PipelineMonitor: Monitor pipeline execution               │
│  └── MetricsCollector: Collect performance metrics             │
├─────────────────────────────────────────────────────────────────┤
│  📋 TEMPLATE METHOD PATTERN                                    │
│  ├── BaseTrainingScript: Template cho training scripts         │
│  ├── BaseEvaluator: Template cho evaluation                   │
│  └── BaseTransforms: Template cho data transforms             │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.2.2 Module Organization và Dependency Management

**Mục đích:** Tổ chức code theo modules với dependency management rõ ràng

**Cấu trúc dependencies:**
```
app/ (UI Layer)
├── app.py (Main app)
├── pages/ (Page modules)
│   ├── search.py
│   ├── analysis.py
│   └── system.py
└── Dependencies: streamlit, plotly, pandas

core/ (Business Logic Layer)
├── pipeline.py (Pipeline orchestrator)
├── retrieval.py (Retrieval engine)
├── reranking.py (Reranking engine)
├── datasets/ (Dataset classes)
├── transforms/ (Data transforms)
├── utils/ (Utility functions)
└── Dependencies: torch, transformers, faiss

config/ (Configuration Layer)
├── default.yml (Default config)
├── loader.py (Config loader)
├── schemas.py (Pydantic schemas)
└── Dependencies: pydantic, pyyaml

training/ (Training Layer)
├── engine.py (Training engine)
├── base_script.py (Base training)
├── run_*.py (Training scripts)
└── Dependencies: torch, transformers, optuna
```

### 6.3 Core Pipeline Architecture

#### 6.3.1 Pipeline Orchestrator Design

**Mục đích:** Thiết kế pipeline orchestrator để điều phối 3 tiers

**Kiến trúc:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATOR                        │
├─────────────────────────────────────────────────────────────────┤
│  🎯 LegalQAPipeline Class                                      │
│  ├── __init__(): Model loading và initialization               │
│  ├── predict(): Main prediction method                         │
│  ├── get_pipeline_status(): Health check                       │
│  └── cleanup(): Resource cleanup                               │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ Component Management                                        │
│  ├── retriever: RetrievalEngine instance                       │
│  ├── reranker: RerankingEngine instance                        │
│  ├── config: Configuration management                          │
│  └── state: Pipeline state management                          │
├─────────────────────────────────────────────────────────────────┤
│  🔄 Execution Flow Control                                      │
│  ├── Tier 1: Bi-encoder retrieval                              │
│  ├── Tier 2: Light reranking                                   │
│  ├── Tier 3: Cross-encoder reranking                           │
│  └── Score aggregation và final ranking                        │
└─────────────────────────────────────────────────────────────────┘
```

**Code logic (dựa trên source code thực tế v8.3):**
```python
class LegalQAPipeline:
    def __init__(
        self,
        bi_encoder_path: Optional[Path] = None,
        reranker_paths: Optional[Dict[str, Path]] = None,
        faiss_index_path: Optional[Path] = None,
        content_map_path: Optional[Path] = None,
        index_to_aid_path: Optional[Path] = None,
        enable_auto_evaluation: bool = False,  # Disabled for stability
    ):
        """Initialize the LegalQA Pipeline."""
        self.is_ready = False
        self.loaded_model_paths = {}
        self.enable_auto_evaluation = enable_auto_evaluation
        
        # Initialize configuration
        self.config = self._initialize_config()
        
        # Auto-evaluation disabled for stability
        self.auto_evaluator = None

        try:
            # Ensure parent law mapping is available before proceeding
            logger.info("Validating parent law mapping availability...")
            if not ensure_parent_law_mapping():
                logger.warning(
                    "Parent law mapping validation failed, but continuing with pipeline initialization"
                )
            else:
                logger.info("✅ Parent law mapping validation successful")

            # --- Load Retriever ---
            paths = config.paths
            bi_encoder_to_load = bi_encoder_path or get_latest_version_path(
                paths.model_dir, "bi-encoder"
            )
            faiss_to_load = faiss_index_path or paths.faiss_index_path
            content_map_to_load = content_map_path or paths.content_map_path
            index_to_aid_to_load = index_to_aid_path or paths.index_to_aid_path

            if not all(
                [
                    bi_encoder_to_load,
                    faiss_to_load.exists(),
                    content_map_to_load.exists(),
                    index_to_aid_to_load.exists(),
                ]
            ):
                raise FileNotFoundError(
                    f"One or more required files for the retriever could not be found. "
                    f"Checked paths: Bi-Encoder: {bi_encoder_to_load}, "
                    f"FAISS: {faiss_to_load}, Content Map: {content_map_to_load}, "
                    f"Index Map: {index_to_aid_to_load}"
                )

            logger.info(f"Loading Retriever with Bi-Encoder: {bi_encoder_to_load}")
            try:
                self.retriever = RetrievalEngine(
                    bi_encoder_path=str(bi_encoder_to_load),
                    faiss_index_path=str(faiss_to_load),
                    content_map_path=str(content_map_to_load),
                    index_to_aid_path=str(index_to_aid_to_load),
                )
            except Exception as e:
                logger.error(f"Failed to initialize RetrievalEngine: {e}")
                raise

            # --- Load Reranker ---
            reranker_configs = self._resolve_reranker_paths(reranker_paths)
            if reranker_configs:
                try:
                    self.reranker = RerankingEngine(reranker_configs)
                    logger.info("✅ RerankingEngine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize RerankingEngine: {e}")
                    self.reranker = None
            else:
                self.reranker = None
                logger.warning("No reranker configurations found")

            self.is_ready = True
            logger.info("✅ LegalQAPipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LegalQAPipeline: {e}")
            self.is_ready = False
            raise
    
    def predict(
        self,
        query: str,
        top_k: int = 5,
        include_scores: bool = True,
        include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """Predict relevant documents for a query."""
        start_time = time.time()
        
        try:
            # Tier 1: Bi-Encoder Retrieval
            retrieval_results = self._retrieve_documents(query, top_k=self.config["top_k_retrieval"])
            
            if not retrieval_results:
                logger.warning("No documents retrieved from Tier 1")
                return []
            
            # Tier 2: Light Reranker (if enabled)
            if self.config["use_light_ranking"] and self.reranker:
                light_results = self._light_rerank(query, retrieval_results, top_k=self.config["top_k_light"])
            else:
                light_results = retrieval_results
            
            # Tier 3: Cross-Encoder (if enabled)
            if self.config["use_cross_encoder"] and self.reranker:
                final_results = self._cross_encode_rerank(query, light_results, top_k)
            else:
                final_results = light_results[:top_k]
            
            # Add metadata and scores
            results = self._add_metadata(final_results, include_scores, include_content)
            
            query_time = (time.time() - start_time) * 1000
            logger.info(f"Query completed in {query_time:.1f}ms, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
            
            return final_results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
```

#### 6.2.2 Component Interface Design

**Mục đích:** Thiết kế interfaces nhất quán cho các components

**Interface contracts:**
```python
# Base interface cho retrieval components
class RetrievalInterface(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass
    
    @abstractmethod
    def cleanup(self):
        pass

# Base interface cho reranking components
class RerankingInterface(ABC):
    @abstractmethod
    def rank_light(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def rank_cross(self, query: str, documents: List[Dict]) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        pass
```

### 6.3 Configuration Management

#### 6.3.1 Centralized Path Configuration với Automated Validation

**Mục đích:** Quản lý cấu hình tập trung với automated path discovery và validation

**Centralized Path Configuration:**
```python
# config/paths.py - Centralized path management
TRAINING_DATA_PATHS = {
    "primary": {
        "bi_encoder": Path("features/processed_data/bi_encoder_train.jsonl"),
        "cross_encoder": Path("features/processed_data/cross_encoder_train.jsonl"),
        "processed_corpus": Path("features/processed_data/processed_corpus.json"),
        "training_data": Path("features/processed_data/training_data.jsonl"),
        "negative_pool": Path("features/processed_data/negative_pool.jsonl"),
    },
    "validation": {
        "tier_1": Path("features/validation_sets/tier_1_validation.jsonl"),
        "tier_2": Path("features/validation_sets/tier_2_validation.jsonl"),
        "tier_3": Path("features/validation_sets/tier_3_validation.jsonl"),
    }
}

# Automated data freshness validation
def validate_training_data_paths() -> dict:
    """Automatically find latest processed data directory and validate"""
    features_dir = Path("features")
    processed_dirs = [d for d in features_dir.iterdir() 
                     if d.is_dir() and d.name.startswith("processed_data_")]
    
    if processed_dirs:
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        return {"latest_data_dir": str(latest_dir), "status": "ready"}
    return {"status": "missing_data"}

def get_training_data_path(tier: str, data_type: str) -> Path:
    """Get training data path with automatic latest directory discovery"""
    processed_dirs = [d for d in Path("features").iterdir() 
                     if d.is_dir() and d.name.startswith("processed_data_")]
    
    if processed_dirs:
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        return latest_dir / f"{data_type}.jsonl"
    
    # Fallback to primary paths
    return TRAINING_DATA_PATHS["primary"].get(data_type, Path(""))
```

**YAML Configuration với Pydantic Validation:**
```yaml
# config/default.yml
paths:
  data_dir: "data"
  model_dir: "models"
  reports_dir: "reports"

bi_encoder:
  model_name: "bkai-foundation-models/vietnamese-bi-encoder"
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      loss_type: "TripletLoss"
      max_length: 256
      batch_size: 16
      epochs: 5

reranker_pipeline:
  light_reranker:
    enabled: true
    model_name: "vinai/phobert-base-v2"
    top_k: 80
    weight: 0.7
  
  cross_encoder:
    enabled: true
    model_name: "vinai/phobert-large"
    top_k: 20
    weight: 0.3

app:
  top_k_retrieval: 100
  top_k_final: 10
  cache_questions_ttl_seconds: 3600
  cache_pipeline_ttl_seconds: 7200
```

**Pydantic schemas:**
```python
# config/schemas.py
class PathsConfig(BaseModel):
    root_dir: Path
    data_dir: Path
    model_dir: Path
    reports_dir: Path

class ContrastiveLearningConfig(BaseModel):
    enabled: bool
    loss_type: str
    max_length: PositiveInt
    batch_size: PositiveInt
    epochs: PositiveInt

class BiEncoderConfig(BaseModel):
    model_name: str
    enhancement_techniques: Dict[str, Any]
    training_params: Dict[str, Any]

class Config(BaseModel):
    paths: PathsConfig
    bi_encoder: BiEncoderConfig
    reranker_pipeline: RerankerPipelineConfig
    app: AppConfig
    logging: LoggingConfig
```

#### 6.3.2 Configuration Loader với Path Resolution

**Mục đích:** Load và resolve configuration với absolute paths

**Code logic (pseudocode):**
```python
def load_config(config_path: Optional[Path] = None) -> Config:
    global _config
    if _config:
        return _config
    
    # Determine project root và default config path
    project_root = Path(__file__).parent.parent
    if config_path is None:
        config_path = project_root / "config" / "default.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # Load YAML file
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    # Resolve paths to be absolute
    config_data = _resolve_paths(config_data, project_root)
    
    # Validate với Pydantic
    try:
        _config = Config(**config_data)
        return _config
    except Exception as e:
        print(f"Configuration validation error: {e}")
        raise

def _resolve_paths(config_data: dict, root_dir: Path) -> dict:
    """Resolve all relative paths trong config thành absolute paths"""
    paths_config = config_data.get("paths", {})
    for key, value in paths_config.items():
        if isinstance(value, str):
            # Prepend root_dir, then resolve để handle '..' etc.
            paths_config[key] = str((root_dir / value).resolve())
    
    # Add root directory vào paths config
    paths_config["root_dir"] = str(root_dir)
    
    return config_data
```

### 6.4 Error Handling và Logging

#### 6.4.1 Comprehensive Error Handling Strategy

**Mục đích:** Xử lý lỗi một cách graceful với proper logging

**Error handling strategy:**
```python
# Error classification
class LawBotError(Exception):
    """Base exception cho LawBot"""
    pass

class PipelineError(LawBotError):
    """Pipeline-related errors"""
    pass

class ModelError(LawBotError):
    """Model loading/execution errors"""
    pass

class ConfigurationError(LawBotError):
    """Configuration errors"""
    pass

# Error handling decorator
def handle_pipeline_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PipelineError as e:
            logger.error(f"Pipeline error in {func.__name__}: {e}")
            # Return fallback results
            return []
        except ModelError as e:
            logger.error(f"Model error in {func.__name__}: {e}")
            # Try to reload model
            return handle_model_error(e)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            # Return empty results
            return []
    return wrapper

# Usage example
@handle_pipeline_errors
def predict(self, query, top_k_final=10):
    # Pipeline logic here
    pass
```

#### 6.4.2 Structured Logging với Logging Manager

**Mục đích:** Quản lý logging một cách có cấu trúc và configurable

**Logging manager:**
```python
# core/utils/logging_manager.py
def setup_logging(log_type: str = "app", workflow_timestamp: Optional[str] = None):
    """Setup logging với proper configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if workflow_timestamp:
        log_filename = f"{log_type}_{workflow_timestamp}.log"
    else:
        log_filename = f"{log_type}_{timestamp}.log"
    
    log_filepath = log_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filepath

def get_logger(name: str) -> logging.Logger:
    """Get logger với proper naming convention"""
    return logging.getLogger(f"LawBot.{name}")

# Usage example
logger = get_logger("pipeline")
logger.info("Pipeline initialized successfully")
logger.error("Pipeline failed", exc_info=True)
```

### 6.5 Caching và Performance Optimization

#### 6.5.1 Streamlit Caching Strategy

**Mục đích:** Tối ưu hóa performance với intelligent caching

**Caching levels:**
```python
# 1. Data caching (1 hour TTL)
@st.cache_data(ttl=3600)
def load_random_questions() -> Tuple[List[str], List[str]]:
    """Load random questions từ datasets"""
    # Implementation here
    pass

# 2. Resource caching (2 hours TTL)
@st.cache_resource(ttl=7200)
def load_pipeline(force_cpu=False):
    """Load và cache pipeline để tránh reloading"""
    # Implementation here
    pass

# 3. Computation caching (30 minutes TTL)
@st.cache_data(ttl=1800)
def run_comprehensive_evaluation(_pipeline, test_queries=None):
    """Run comprehensive evaluation với caching"""
    # Implementation here
    pass

# 4. File-based caching
def load_latest_comprehensive_evaluation():
    """Load evaluation results từ file cache"""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return None
        
        # Find latest file
        comp_files = list(reports_dir.glob("comprehensive_evaluation_*.json"))
        if not comp_files:
            return None
        
        latest_file = max(comp_files, key=lambda p: p.stat().st_mtime)
        
        # Check if file is recent (within last 24 hours)
        file_age = time.time() - latest_file.stat().st_mtime
        if file_age > 86400:  # 24 hours
            return None
        
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data.get("results")
        
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {e}")
        return None
```

#### 6.5.2 Memory Management và Optimization

**Mục đích:** Quản lý memory hiệu quả cho large models

**Memory optimization techniques:**
```python
# 1. Lazy loading
def get_pipeline_lazy():
    """Lazy load pipeline chỉ khi cần thiết"""
    try:
        # Validate parent law mapping
        if ensure_parent_law_mapping():
            logger.info("✅ Parent law mapping validated")
        
        pipeline = LegalQAPipeline()
        return pipeline
        
    except Exception as e:
        logger.warning(f"⚠️ Pipeline not available (handled): {e}")
        return None

# 2. Memory cleanup
def cleanup_pipeline_resources(pipeline):
    """Cleanup resources để giải phóng memory"""
    try:
        if hasattr(pipeline, 'retriever'):
            pipeline.retriever.cleanup()
        if hasattr(pipeline, 'reranker'):
            pipeline.reranker.cleanup()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Pipeline cleanup completed")
        
    except Exception as e:
        logger.warning(f"Error during pipeline cleanup: {e}")

# 3. Batch processing
def process_queries_in_batches(queries, batch_size=5):
    """Process queries theo batches để tối ưu memory"""
    results = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        
        # Process batch
        batch_results = process_batch(batch_queries)
        results.extend(batch_results)
        
        # Memory cleanup after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

### 6.6 Data Processing và Transformation

#### 6.6.1 Unified Text Processing Pipeline

**Mục đích:** Xử lý text một cách nhất quán với legal domain expertise

**Text processing pipeline:**
```python
# core/transforms/base.py
class TextProcessor(BaseTransform):
    """Unified text processor cho cleaning, normalizing, và transforming text"""
    
    def __init__(self, normalize_unicode=True, lowercase=False, 
                 remove_punctuation=False, remove_extra_spaces=True):
        self.normalize_unicode = normalize_unicode
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_spaces = remove_extra_spaces
    
    def __call__(self, text: str) -> str:
        """Apply configured text transformations"""
        if not isinstance(text, str):
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        
        # Lowercase conversion
        if self.lowercase:
            text = text.lower()
        
        # Punctuation removal
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        
        # Extra spaces removal
        if self.remove_extra_spaces:
            text = re.sub(r"\s+", " ", text).strip()
        
        return text

class LegalTextCleaner(TextProcessor):
    """Specialized text processor cho Vietnamese legal documents"""
    
    def __init__(self):
        super().__init__(
            normalize_unicode=True,
            lowercase=False,  # Legal text có meaningful capitalization
            remove_punctuation=False,  # Punctuation quan trọng trong legal text
            remove_extra_spaces=True
        )
        
        # Pre-compiled regex patterns
        self.article_pattern = re.compile(
            r"(Điều|Khoản|Điểm)\s+\d+[a-z]?\.?", re.IGNORECASE
        )
        
        # Legal abbreviations mapping
        self.abbreviation_map = {
            "QH": "Quốc hội",
            "UBTVQH": "Ủy ban Thường vụ Quốc hội",
            "CP": "Chính phủ",
            "NĐ-CP": "Nghị định - Chính phủ",
            "TTg": "Thủ tướng Chính phủ"
        }
    
    def __call__(self, text: str) -> str:
        """Apply base processing và legal-specific cleaning rules"""
        # Base processing first
        text = super().__call__(text)
        
        # Remove article/clause/point numbers
        text = self.article_pattern.sub(r"\1", text)
        
        # Replace common abbreviations
        for abbr, full_text in self.abbreviation_map.items():
            text = re.sub(rf"\b{abbr}\b", full_text, text)
        
        # Final cleanup
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
```

#### 6.6.2 Dataset Management và Processing

**Mục đích:** Quản lý datasets một cách hiệu quả với proper validation

**Dataset classes:**
```python
# core/datasets/legal_qa.py
class BiEncoderDataset(Dataset):
    """Dataset cho training bi-encoder models"""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, 
                 max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"BiEncoderDataset initialized với {len(data)} examples, max_length={max_length}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query = item.get("query", "")
        positive = item.get("positive", "")
        negative = item.get("negative", "")
        
        # Tokenize each part separately cho bi-encoder training
        query_inputs = self.tokenizer(
            query, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        positive_inputs = self.tokenizer(
            positive, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        negative_inputs = self.tokenizer(
            negative, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_inputs["input_ids"].squeeze(0),
            "query_attention_mask": query_inputs["attention_mask"].squeeze(0),
            "positive_input_ids": positive_inputs["input_ids"].squeeze(0),
            "positive_attention_mask": positive_inputs["attention_mask"].squeeze(0),
            "negative_input_ids": negative_inputs["input_ids"].squeeze(0),
            "negative_attention_mask": negative_inputs["attention_mask"].squeeze(0)
        }

class RerankerDataset(Dataset):
    """Dataset cho training cross-encoder reranker models"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, 
                 max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"RerankerDataset initialized với {len(data)} examples, max_length={max_length}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        query = item.get("query", "")
        passage = item.get("passage", "")
        label = item.get("label", 0)
        
        # Tokenize query-passage pair cho cross-encoder training
        inputs = self.tokenizer(
            query, passage, max_length=self.max_length, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }
```

---

### 6.7 Testing và Quality Assurance

#### 6.7.1 Unit Testing Strategy

**Mục đích:** Đảm bảo code quality với comprehensive testing

**Testing structure:**
```python
# tests/test_pipeline.py
import pytest
from unittest.mock import Mock, patch
from core.pipeline import LegalQAPipeline

class TestLegalQAPipeline:
    """Test cases cho LegalQAPipeline"""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline cho testing"""
        with patch('core.pipeline.RetrievalEngine') as mock_retriever, \
             patch('core.pipeline.RerankingEngine') as mock_reranker:
            
            mock_retriever.return_value.is_ready = True
            mock_reranker.return_value.is_ready = True
            
            pipeline = LegalQAPipeline()
            yield pipeline
    
    def test_pipeline_initialization(self, mock_pipeline):
        """Test pipeline initialization"""
        assert mock_pipeline.is_ready == True
        assert hasattr(mock_pipeline, 'retriever')
        assert hasattr(mock_pipeline, 'reranker')
    
    def test_predict_method(self, mock_pipeline):
        """Test predict method"""
        # Mock retriever response
        mock_pipeline.retriever.retrieve.return_value = [
            {'aid': 'test1', 'content': 'test content', 'retrieval_score': 0.8}
        ]
        
        # Mock reranker response
        mock_pipeline.reranker.rank_light.return_value = [
            {'aid': 'test1', 'content': 'test content', 'retrieval_score': 0.8, 
             'light_reranker_score': 0.9}
        ]
        
        results = mock_pipeline.predict("test query", top_k_final=1)
        
        assert len(results) == 1
        assert results[0]['aid'] == 'test1'
        assert 'final_score' in results[0]
    
    def test_pipeline_not_ready_error(self, mock_pipeline):
        """Test error khi pipeline chưa ready"""
        mock_pipeline.is_ready = False
        
        with pytest.raises(RuntimeError, match="Pipeline is not ready"):
            mock_pipeline.predict("test query")
```

#### 6.7.2 Integration Testing

**Mục đích:** Test integration giữa các components

**Integration test structure:**
```python
# tests/test_integration.py
import pytest
from core.pipeline import LegalQAPipeline
from core.retrieval import RetrievalEngine
from core.reranking import RerankingEngine

class TestPipelineIntegration:
    """Test integration giữa pipeline components"""
    
    @pytest.fixture
    def real_pipeline(self):
        """Create real pipeline với minimal models cho integration testing"""
        # Load pipeline với test configuration
        pipeline = LegalQAPipeline(
            bi_encoder_path="test_models/mini_bi_encoder",
            reranker_paths="test_models/mini_reranker"
        )
        return pipeline
    
    def test_end_to_end_pipeline(self, real_pipeline):
        """Test end-to-end pipeline execution"""
        # Test query processing
        query = "Quy định về xử phạt vi phạm giao thông"
        
        # Execute pipeline
        results = real_pipeline.predict(query, top_k_final=3)
        
        # Validate results structure
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert 'aid' in result
            assert 'content' in result
            assert 'final_score' in result
            assert 'retrieval_score' in result
    
    def test_tier_communication(self, real_pipeline):
        """Test communication giữa các tiers"""
        query = "Thủ tục đăng ký kinh doanh"
        
        # Test Tier 1 → Tier 2 communication
        tier1_results = real_pipeline.retriever.retrieve(query, top_k=10)
        assert len(tier1_results) == 10
        
        # Test Tier 2 → Tier 3 communication
        if real_pipeline.reranker:
            tier2_results = real_pipeline.reranker.rank_light(query, tier1_results[:5])
            assert len(tier2_results) == 5
            assert all('light_reranker_score' in doc for doc in tier2_results)
```

### 6.8 Code Quality và Best Practices

#### 6.8.1 Code Style và Documentation

**Mục đích:** Đảm bảo code quality với proper documentation và style

**Documentation standards:**
```python
# Example of well-documented code
class LegalQAPipeline:
    """
    Main pipeline orchestrator cho LawBot system.
    
    This class manages the 3-tier architecture:
    - Tier 1: Bi-Encoder Retrieval
    - Tier 2: Light Reranking  
    - Tier 3: Cross-Encoder Ensemble
    
    Attributes:
        retriever (RetrievalEngine): Tier 1 retrieval engine
        reranker (RerankingEngine): Tier 2 & 3 reranking engine
        is_ready (bool): Pipeline readiness status
        config (Config): Configuration object
    
    Example:
        >>> pipeline = LegalQAPipeline()
        >>> results = pipeline.predict("legal question", top_k_final=5)
        >>> print(f"Found {len(results)} results")
    """
    
    def __init__(self, bi_encoder_path: Optional[str] = None, 
                 reranker_paths: Optional[Union[str, List[str]]] = None):
        """
        Initialize LegalQAPipeline.
        
        Args:
            bi_encoder_path: Path to bi-encoder model
            reranker_paths: Path(s) to reranker model(s)
        
        Raises:
            RuntimeError: If pipeline initialization fails
            FileNotFoundError: If required model files not found
        """
        # Implementation here
        pass
```

#### 6.8.2 Performance Monitoring và Profiling

**Mục đích:** Monitor và optimize code performance

**Performance monitoring:**
```python
# core/utils/performance_monitor.py
import time
import functools
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def performance_monitor(func: Callable) -> Callable:
    """Decorator để monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = get_memory_usage()
            
            duration = end_time - start_time
            memory_diff = end_memory - start_memory
            
            logger.info(f"Function {func.__name__} took {duration:.3f}s, "
                       f"memory change: {memory_diff:.2f}MB")
    
    return wrapper

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Usage example
@performance_monitor
def predict(self, query: str, top_k_final: int = 10) -> List[Dict]:
    # Pipeline logic here
    pass
```

---

## 7. HƯỚNG DẪN VẬN HÀNH

### 7.1 Cài đặt và Setup

#### 7.1.1 Environment Setup

**Mục đích:** Hướng dẫn cài đặt môi trường cho LawBot

**Requirements:**
```bash
# Python version
Python 3.8+

# CUDA support (optional, for GPU acceleration)
CUDA 11.0+ (for PyTorch GPU support)

# System requirements
RAM: 16GB+ (32GB recommended)
Storage: 50GB+ free space
GPU: NVIDIA GPU với 8GB+ VRAM (optional)
```

**Installation steps:**
```bash
# 1. Clone repository
git clone https://github.com/your-org/LawBot.git
cd LawBot

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install additional dependencies for development
pip install -r requirements-dev.txt
```

#### 7.1.2 Configuration Setup

**Mục đích:** Cấu hình hệ thống cho môi trường cụ thể

**Configuration files:**
```yaml
# config/local.yml (override default.yml)
paths:
  data_dir: "/path/to/your/data"
  model_dir: "/path/to/your/models"
  reports_dir: "/path/to/your/reports"

bi_encoder:
  model_name: "your-custom-bi-encoder"
  enhancement_techniques:
    contrastive_learning:
      enabled: true
      batch_size: 8  # Adjust based on your GPU memory

app:
  cache_pipeline_ttl_seconds: 3600
  force_cpu: false  # Set to true if no GPU available
```

**Environment variables:**
```bash
# Set environment variables
export LAWBOOT_CONFIG_PATH="/path/to/your/config.yml"
export LAWBOOT_DATA_PATH="/path/to/your/data"
export LAWBOOT_MODEL_PATH="/path/to/your/models"
export CUDA_VISIBLE_DEVICES="0"  # Use specific GPU
```

### 7.2 Workflow Execution

#### 7.2.1 Automated Workflow với Centralized Paths

**Mục đích:** Chạy toàn bộ training workflow một cách tự động với centralized path management

**Command execution:**
```bash
# Run complete workflow với preset
python run_workflow.py --preset full

# Available presets:
# - data_preparation: Chỉ chuẩn bị dữ liệu
# - bi_encoder: Chỉ train bi-encoder
# - light_ranking: Chỉ train light reranker
# - cross_encoder: Chỉ train cross-encoder
# - faiss_index: Chỉ tạo FAISS index
# - evaluation: Chỉ chạy evaluation
# - full: Toàn bộ workflow (recommended)

# Custom workflow với specific steps
python run_workflow.py --stages data_preparation bi_encoder light_ranking

# Run với force restart (bỏ qua checkpoints)
python run_workflow.py --preset full --force-restart

# List available stages và presets
python run_workflow.py --list-stages
```

**Centralized Path Management Features:**
```bash
# Automated data freshness validation
python run_workflow.py --preset full  # Tự động kiểm tra và re-run data_preparation nếu cần

# Check centralized path status
python -c "from config.paths import validate_training_data_paths; print(validate_training_data_paths())"

# Get training data paths
python -c "from config.paths import get_training_data_path; print(get_training_data_path('tier_1', 'bi_encoder'))"
```

**Workflow monitoring:**
```bash
# Monitor workflow progress
tail -f logs/workflow_20241201_143022.log

# Check workflow status
python run_workflow.py --list-stages

# Resume interrupted workflow (tự động qua checkpoints)
python run_workflow.py --preset full

# Force restart toàn bộ workflow
python run_workflow.py --preset full --force-restart
```

#### 7.2.2 Manual Workflow Execution

**Mục đích:** Chạy từng bước workflow một cách thủ công

**Step-by-step execution:**
```bash
# 1. Data preparation
python data_processing/run_preparation.py \
  --input_dir data/raw \
  --output_dir features/processed \
  --config config/default.yml

# 2. Bi-encoder training
python training/run_bi_encoder.py \
  --data_path features/processed \
  --output_dir models/bi_encoder \
  --config config/default.yml \
  --hpo_trials 15

# 3. Create FAISS index
python training/run_create_faiss_index.py \
  --model_path models/bi_encoder \
  --corpus_path features/processed \
  --output_dir models/faiss_index

# 4. Light reranker training
python training/run_light_ranking.py \
  --data_path features/processed \
  --output_dir models/light_reranker \
  --config config/default.yml

# 5. Cross-encoder training
python training/run_reranker.py \
  --data_path features/processed \
  --output_dir models/cross_encoder \
  --config config/default.yml

# 6. Evaluation
python evaluation/run_evaluation.py \
  --pipeline_config config/default.yml \
  --output_dir reports/evaluation
```

### 7.3 Application Deployment

#### 7.3.1 Streamlit App Launch

**Mục đích:** Khởi chạy ứng dụng web Streamlit

**Launch commands:**
```bash
# Basic launch
python run_app.py

# Launch với specific configuration
python run_app.py --config config/production.yml

# Launch với custom port
python run_app.py --port 8502

# Launch với specific host
python run_app.py --host 0.0.0.0 --port 8501

# Launch với debug mode
python run_app.py --debug

# Launch với CPU-only mode
python run_app.py --force_cpu
```

**Environment configuration:**
```bash
# Set Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch app
streamlit run app/app.py
```

#### 7.3.2 Production Deployment

**Mục đích:** Deploy ứng dụng trong môi trường production

**Docker deployment:**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health

# Launch application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  lawbot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    environment:
      - LAWBOOT_CONFIG_PATH=/app/config/production.yml
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 7.4 Monitoring và Maintenance

#### 7.4.1 System Health Monitoring

**Mục đích:** Monitor trạng thái hệ thống và performance

**Health check commands:**
```bash
# Check system status
python -c "
from core.utils.system_check import check_system_health
status = check_system_health()
print(f'System Health: {status}')
"

# Check model status
python -c "
from core.utils.system_check import check_model_status
models = check_model_status()
for name, status in models.items():
    print(f'{name}: {status}')
"

# Check FAISS index status
python -c "
from core.utils.system_check import check_faiss_status
faiss_status = check_faiss_status()
print(f'FAISS Status: {faiss_status}')
"
```

**Log monitoring:**
```bash
# Monitor application logs
tail -f logs/app_*.log

# Monitor training logs
tail -f logs/workflow_*.log

# Monitor error logs
grep -i "error\|exception" logs/*.log

# Monitor performance logs
grep -i "performance\|timing" logs/*.log
```

#### 7.4.2 Performance Optimization

**Mục đích:** Tối ưu hóa performance của hệ thống

**Performance tuning:**
```bash
# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Batch size optimization
python training/run_bi_encoder.py \
  --batch_size 32 \
  --gradient_accumulation_steps 4

# Memory cleanup
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA cache cleared')
"
```

**Resource monitoring:**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor memory usage
watch -n 1 'free -h'

# Monitor disk usage
df -h

# Monitor process resources
top -p $(pgrep -f "python.*run_app.py")
```

### 7.5 Troubleshooting

#### 7.5.1 Common Issues và Solutions

**Mục đích:** Giải quyết các vấn đề thường gặp

**Common issues:**
```bash
# Issue 1: CUDA out of memory (handled)
# Solution: Reduce batch size hoặc enable gradient accumulation
python training/run_bi_encoder.py --batch_size 8 --gradient_accumulation_steps 8

# Issue 2: Model loading failed
# Solution: Check model paths và permissions
ls -la models/
chmod -R 755 models/

# Issue 3: FAISS index not found
# Solution: Recreate FAISS index
python training/run_create_faiss_index.py \
  --model_path models/bi_encoder \
  --corpus_path features/processed

# Issue 4: Pipeline not ready
# Solution: Check model status và reload pipeline
python -c "
from core.pipeline import LegalQAPipeline
pipeline = LegalQAPipeline()
print(f'Pipeline ready: {pipeline.is_ready}')
"

# Issue 5: Workflow import errors (ĐÃ FIX)
# Solution: Sử dụng lệnh đã được fix
python run_workflow.py --preset full --force-restart

# Issue 6: Checkpoint conflicts (ĐÃ FIX)
# Solution: Sử dụng --force-restart để bỏ qua checkpoints cũ
python run_workflow.py --preset full --force-restart
```

**Debug mode:**
```bash
# Enable debug logging
export LAWBOOT_LOG_LEVEL=DEBUG

# Run với verbose output
python run_app.py --verbose

# Check configuration
python -c "
from config.loader import load_config
config = load_config()
print('Configuration loaded successfully')
print(f'Data directory: {config.paths.data_dir}')
"
```

#### 7.5.2 Recovery Procedures

**Mục đích:** Khôi phục hệ thống khi gặp sự cố

**Recovery steps:**
```bash
# 1. Check system status
python -c "from core.utils.system_check import check_system_health; print(check_system_health())"

# 2. Restart pipeline
python -c "
from core.pipeline import LegalQAPipeline
pipeline = LegalQAPipeline()
print('Pipeline restarted')
"

# 3. Clear cache
python -c "
import streamlit as st
st.cache_data.clear()
st.cache_resource.clear()
print('Cache cleared')
"

# 4. Restart application
pkill -f "streamlit.*run_app.py"
python run_app.py
```

---

## 8. ĐÁNH GIÁ VÀ KHUYẾN NGHỊ

### 8.1 Đánh giá tổng quan

#### 8.1.1 Điểm mạnh của hệ thống

**Kiến trúc và thiết kế:**
- **Kiến trúc 3 tầng hiện đại**: Mỗi tầng được **train độc lập** và tối ưu hóa cho nhiệm vụ cụ thể
- **Modular design**: Code được tổ chức theo modules rõ ràng, dễ maintain
- **Scalable architecture**: Có thể mở rộng thêm tiers hoặc models mới
- **Configuration-driven**: Dễ dàng thay đổi cấu hình mà không cần sửa code
- **Workflow automation**: Hệ thống workflow tự động với checkpoint management
- **Lazy Loading Architecture**: Tránh circular import với dynamic page loading
- **Form-based UI**: Ngăn auto-rerun với `st.form` controls
- **Safe Device Handling**: Meta tensor support với `_safe_move_to_device()`

**Centralized Path Management:**
- **Centralized configuration**: Tất cả đường dẫn được quản lý tập trung tại `config/paths.py`
- **Automated path discovery**: Tự động tìm thư mục processed data mới nhất
- **Data freshness validation**: Tự động kiểm tra tính mới của dữ liệu
- **Consistent path resolution**: Đảm bảo tất cả tiers sử dụng cùng nguồn dữ liệu
- **Fallback mechanisms**: Graceful fallback khi không tìm thấy dữ liệu mới

**MLOps và Engineering:**
- **Comprehensive logging**: Logging có cấu trúc và dễ debug
- **Error handling**: Xử lý lỗi graceful với recovery mechanisms
- **Performance monitoring**: Theo dõi hiệu suất real-time
- **Caching strategy**: Intelligent caching để tối ưu performance
- **Version control**: Quản lý phiên bản models và experiments

**Kỹ thuật Machine Learning:**
- **Contrastive Learning**: Sử dụng TripletLoss hiệu quả cho bi-encoder
- **Hard Negative Mining**: Tự động cải thiện chất lượng training data với adaptive threshold
- **ADAPT technique**: Domain adaptation cho pháp luật Việt Nam với enhanced training
- **Ensemble learning**: Kết hợp multiple models để tăng accuracy
- **Hyperparameter Optimization**: Sử dụng Optuna để tối ưu hóa với early stopping
- **Performance Optimization**: HPO optimization với Optuna
- **Memory Management**: Basic memory handling
- **Quality Score Logic**: Tier-specific thresholds với adjusted scoring

**Workflow và Automation:**
- **Automated pipeline**: Workflow tự động với stage dependencies
- **Checkpoint management**: Resume interrupted workflows với automated recovery
- **Error recovery**: Tự động xử lý lỗi và retry với advanced error handling
- **Progress tracking**: Real-time progress monitoring với performance metrics
- **Consistent execution**: Đảm bảo workflow chạy giống manual execution
- **Data freshness automation**: Tự động re-run data_preparation khi cần thiết
- **Performance Monitoring**: Real-time performance tracking với automated optimization
- **Config Consistency**: 100% sử dụng centralized configuration

#### 8.1.2 Điểm cần cải thiện

**Performance và Scalability:**
- **Memory usage**: Đã tối ưu hóa với safe device handling cho meta tensors
- **Processing speed**: Đã cải thiện với form-based UI và state optimization
- **Batch processing**: Đã implement batch processing cho multiple queries
- **Async processing**: Đã sử dụng async/await để tăng throughput
- **App Reload Optimization**: Đã giảm thiểu reload không cần thiết với lazy loading

**Data Management:**
- **Data versioning**: Cần implement data versioning system
- **Incremental updates**: Cần hỗ trợ cập nhật dữ liệu incrementally
- **Data validation**: Cần tăng cường validation cho input data
- **Data lineage**: Cần tracking data lineage từ raw data đến final results

**User Experience:**
- **Real-time feedback**: Đã cải thiện với form-based controls và success messages
- **Personalization**: Có thể thêm personalization features
- **Multi-language support**: Đã hỗ trợ đa ngôn ngữ tốt hơn với Vietnamese bi-encoder
- **Mobile optimization**: Đã tối ưu hóa responsive design với column-based layout
- **State Management**: Đã tối ưu hóa session state để tránh reload không cần thiết

### 8.2 Khuyến nghị cải thiện

#### 8.2.1 Kỹ thuật Machine Learning

**Model Architecture (Đã implement):**
```python
# 1. Safe Device Handling cho Meta Tensors
def _safe_move_to_device(model, device):
    """Safely move model to device handling meta tensors and offloaded modules"""
    try:
        if hasattr(model, 'to_empty'):
            # Handle meta tensors
            return model.to_empty(device=device)
        else:
            return model.to(device)
    except (NotImplementedError, RuntimeError) as e:
        # Handle offloaded modules
        logging.warning(f"Safe device movement failed: {e}")
        return model

# 2. Monkey Patching cho SentenceTransformer
def _monkey_patch_sentence_transformer():
    """Prevent auto device movement for SentenceTransformer"""
    original_to = SentenceTransformer.to
    def safe_to(self, device):
        if hasattr(self, '_device_moved'):
            return self
        self._device_moved = True
        return original_to(self, device)
    SentenceTransformer.to = safe_to
```

**Performance Optimization (Đã implement):**
- **Lazy Loading**: Tránh circular import với dynamic page loading
- **Form-based UI**: Ngăn auto-rerun với `st.form` controls
- **State Management**: Tối ưu hóa session state để giảm reload
- **Safe Device Handling**: Hỗ trợ meta tensors và offloaded modules
class MultiNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=16):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        
    def forward(self, query_emb, positive_emb, negative_embs):
        # Calculate similarities
        pos_sim = torch.cosine_similarity(query_emb, positive_emb, dim=1)
        neg_sims = torch.cosine_similarity(
            query_emb.unsqueeze(1), 
            negative_embs, 
            dim=2
        )
        
        # Multi-negative contrastive loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        
        return F.cross_entropy(logits / self.temperature, labels)
```

**Training Optimization:**
```python
# 1. Implement curriculum learning
class CurriculumLearningScheduler:
    def __init__(self, difficulty_levels=[0.3, 0.6, 1.0]):
        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        
    def get_training_data(self, full_dataset):
        current_difficulty = self.difficulty_levels[self.current_level]
        
        # Filter data based on difficulty
        filtered_data = [
            item for item in full_dataset 
            if item['difficulty_score'] <= current_difficulty
        ]
        
        return filtered_data
    
    def advance_level(self):
        if self.current_level < len(self.difficulty_levels) - 1:
            self.current_level += 1
            return True
        return False

# 2. Implement advanced HPO với early stopping
class AdvancedHPO:
    def __init__(self, n_trials=50, patience=10):
        self.n_trials = n_trials
        self.patience = patience
        
    def objective(self, trial):
        params = {
            'learning_rate': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'epochs': trial.suggest_int('epochs', 3, 10),
            'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5)
        }
        
        # Train model với early stopping
        model = self._train_with_early_stopping(params)
        
        # Evaluate model
        val_score = self._evaluate_model(model)
        
        return val_score
```

#### 8.2.2 System Architecture

**Microservices Architecture:**
```python
# 1. Implement API gateway
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LawBot API", version="v8.3")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_cache: bool = True

@app.post("/api/v1/query")
async def process_query(request: QueryRequest):
    try:
        # Process query through pipeline
        results = await pipeline.predict_async(
            request.query, 
            top_k_final=request.top_k
        )
        
        return {
            "status": "success",
            "results": results,
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Implement async pipeline processing
class AsyncLegalQAPipeline:
    async def predict_async(self, query: str, top_k_final: int = 10):
        # Async Tier 1 processing
        tier1_task = asyncio.create_task(
            self._async_retrieval(query, top_k=100)
        )
        
        # Async Tier 2 processing (when Tier 1 completes)
        tier1_results = await tier1_task
        tier2_task = asyncio.create_task(
            self._async_light_reranking(query, tier1_results[:50])
        )
        
        # Async Tier 3 processing (when Tier 2 completes)
        tier2_results = await tier2_task
        tier3_results = await self._async_cross_encoding(
            query, tier2_results[:20]
        )
        
        # Combine results
        final_results = self._combine_scores(tier3_results)
        return final_results[:top_k_final]
```

**Caching và Performance:**
```python
# 1. Implement Redis caching
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def redis_cache(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator

# 2. Implement connection pooling
from contextlib import asynccontextmanager
import aiohttp

class ConnectionPool:
    def __init__(self, max_connections=100):
        self.semaphore = asyncio.Semaphore(max_connections)
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            connector = aiohttp.TCPConnector(limit=self.max_connections)
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
    
    @asynccontextmanager
    async def acquire(self):
        async with self.semaphore:
            session = await self.get_session()
            yield session
```

#### 8.2.3 Data Management

**Data Versioning và Lineage:**
```python
# 1. Implement data versioning
import hashlib
from datetime import datetime

class DataVersioning:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.versions_file = data_dir / "versions.json"
        
    def create_version(self, data_path: Path, metadata: dict) -> str:
        # Calculate data hash
        data_hash = self._calculate_hash(data_path)
        
        # Create version info
        version_info = {
            "version_id": data_hash[:8],
            "timestamp": datetime.now().isoformat(),
            "data_path": str(data_path),
            "metadata": metadata,
            "hash": data_hash
        }
        
        # Save version info
        self._save_version(version_info)
        
        return version_info["version_id"]
    
    def _calculate_hash(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

# 2. Implement data lineage tracking
class DataLineageTracker:
    def __init__(self):
        self.lineage_graph = {}
        
    def track_transformation(self, input_data: str, output_data: str, 
                           transformation: str, metadata: dict):
        """Track data transformation lineage"""
        if output_data not in self.lineage_graph:
            self.lineage_graph[output_data] = {
                "inputs": [],
                "transformations": [],
                "metadata": {}
            }
        
        self.lineage_graph[output_data]["inputs"].append(input_data)
        self.lineage_graph[output_data]["transformations"].append({
            "type": transformation,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })
    
    def get_lineage(self, data_id: str) -> dict:
        """Get complete lineage for data item"""
        return self.lineage_graph.get(data_id, {})
```

### 8.3 Roadmap phát triển

#### 8.3.1 Short-term improvements (3-6 months)

**Performance Optimization:**
- Implement async processing cho pipeline
- Optimize memory usage với gradient checkpointing
- Add batch processing cho multiple queries
- Implement advanced caching strategies

**User Experience:**
- Add real-time progress indicators
- Implement query suggestions và autocomplete
- Add result filtering và sorting options
- Improve mobile responsiveness

**Monitoring và Debugging:**
- Add comprehensive metrics dashboard
- Implement distributed tracing
- Add automated alerting system
- Improve error reporting và recovery

#### 8.3.2 Medium-term improvements (6-12 months)

**Advanced ML Techniques:**
- Implement transformer-based architectures
- Add multi-task learning capabilities
- Implement active learning strategies
- Add federated learning support

**System Architecture:**
- Migrate to microservices architecture
- Implement API gateway và load balancing
- Add horizontal scaling capabilities
- Implement service mesh

**Data Management:**
- Add data versioning system
- Implement data lineage tracking
- Add incremental data updates
- Implement data quality monitoring

#### 8.3.3 Long-term vision (1-2 years)

**AI Capabilities:**
- Implement multi-modal understanding (text + images)
- Add reasoning capabilities
- Implement conversational AI
- Add personalized recommendations

**Enterprise Features:**
- Multi-tenant architecture
- Advanced security features
- Compliance monitoring
- Integration với enterprise systems

**Research và Innovation:**
- Novel architectures cho legal AI
- Advanced NLP techniques
- Cross-lingual capabilities
- Explainable AI features

## 9. DATA, FEATURE & MODEL VERSIONING MANAGEMENT

### 9.1 Data Management Architecture

#### 9.1.1 Data Processing Pipeline: Từ Nguyên Tử Đến Phân Tử

**Quá trình xử lý data trong LawBot được thiết kế theo nguyên tắc "từ nguyên tử đến phân tử" - từ data gốc thô đến training data hoàn chỉnh:**

##### **Bước 1: Data Gốc (Raw Data) - "Nguyên Tử"**

```python
# data/raw/legal_corpus.json - Cấu trúc data gốc
raw_corpus = [
    {
        "law_id": "123/2020/QH14",           # ID luật gốc
        "content": [                          # Danh sách các điều luật
            {
                "aid": 15,                    # Article ID thô (số nguyên)
                "content_Article": "Điều 15. Quyền sử dụng đất...",  # Nội dung điều luật
                "law_name": "Luật Đất đai",
                "category": "land_law"
            },
            {
                "aid": 16,
                "content_Article": "Điều 16. Nghĩa vụ của người sử dụng đất...",
                "law_name": "Luật Đất đai",
                "category": "land_law"
            }
        ]
    },
    {
        "law_id": "456-2021",
        "content": [
            {
                "aid": 20,
                "content_Article": "Điều 20. Thuế đất đai...",
                "law_name": "Luật Thuế",
                "category": "tax_law"
            }
        ]
    }
]

# data/raw/training_data.json - Câu hỏi training gốc
training_data = [
    {
        "question": "Luật về đất đai quy định gì về quyền sử dụng đất?",
        "relevant_laws": [15, 16],           # Reference đến AID thô
        "category": "land_law",
        "difficulty": "medium"
    }
]
```

##### **Bước 2: Canonicalization - "Tạo Liên Kết Hóa Học"**

```python
# core/utils/canonicalization.py - Chuyển đổi AID thô thành canonical format
def canonicalize_aid(law_id: str, article_id: Union[str, int]) -> str:
    """Tạo standardized Article ID từ law_id và article_id thô."""
    
    # Clean và standardize law_id
    law_id_cleaned = re.sub(r"[-\s_]", "", str(law_id)).upper()
    # Ví dụ: "123/2020/QH14" -> "1232020QH14"
    
    # Clean và standardize article_id  
    article_id_cleaned = re.sub(r"[-\s_]", "", str(article_id)).upper()
    # Ví dụ: "15" -> "15", "15a" -> "15A"
    
    # Tạo canonical AID
    canonical_aid = f"{law_id_cleaned}_{article_id_cleaned}"
    # Ví dụ: "1232020QH14_15", "1232020QH14_15A"
    
    return canonical_aid

# Ví dụ chuyển đổi:
# Input: law_id="123/2020/QH14", article_id=15
# Output: canonical_aid="1232020QH14_15"
```

##### **Bước 3: Data Processing - "Tạo Phân Tử Đơn Giản"**

```python
# data_processing/run_preparation.py - Xử lý corpus và tạo mapping
def process_legal_corpus(raw_corpus: List[Dict]) -> Tuple[Dict[str, str], Dict[int, str]]:
    """Xử lý raw corpus thành processed corpus và AID mapping."""
    
    processed_corpus = {}    # canonical_aid -> content
    aid_map = {}            # raw_aid -> canonical_aid
    
    for law_doc in raw_corpus:
        law_id = law_doc.get("law_id")
        articles = law_doc.get("content", [])
        
        for article in articles:
            raw_aid = article.get("aid")           # 15 (số nguyên)
            content = article.get("content_Article", "").strip()
            
            if not raw_aid or not content:
                continue
            
            # Tạo canonical AID
            canonical_aid = canonicalize_aid(str(law_id), str(raw_aid))
            # Ví dụ: "1232020QH14_15"
            
            # Lưu vào processed corpus
            processed_corpus[canonical_aid] = content
            # Ví dụ: {"1232020QH14_15": "Điều 15. Quyền sử dụng đất..."}
            
            # Tạo mapping từ raw AID sang canonical AID
            aid_map[int(raw_aid)] = canonical_aid
            # Ví dụ: {15: "1232020QH14_15"}

# Kết quả sau khi xử lý:
processed_corpus = {
    "1232020QH14_15": "Điều 15. Quyền sử dụng đất...",
    "1232020QH14_16": "Điều 16. Nghĩa vụ của người sử dụng đất...",
    "4562021_20": "Điều 20. Thuế đất đai..."
}

aid_map = {
    15: "1232020QH14_15",
    16: "1232020QH14_16", 
    20: "4562021_20"
}
```

##### **Bước 4: Training Data Generation - "Tạo Hợp Chất Phức Tạp"**

```python
# data_processing/run_preparation.py - Tạo training examples
def generate_training_examples(train_data, processed_corpus, aid_map):
    """Tạo training examples từ processed data."""
    
    bi_encoder_examples = []      # Triplets cho Tier 1
    cross_encoder_examples = []   # Pairs cho Tier 3
    
    for item in train_data:
        question = item.get("question")
        raw_aids = item.get("relevant_laws", [])  # [15, 16]
        
        # Clean question
        cleaned_question = LegalTextCleaner()(question)
        
        # Chuyển raw AIDs thành canonical AIDs
        positive_canonical_aids = []
        for raw_aid in raw_aids:
            canonical_aid = aid_map.get(raw_aid)  # 15 -> "1232020QH14_15"
            if canonical_aid and canonical_aid in processed_corpus:
                positive_canonical_aids.append(canonical_aid)
        
        # Tạo training examples cho mỗi positive
        for pos_aid in positive_canonical_aids:
            # Tìm negative example (random selection)
            negative_aid = random.choice([
                aid for aid in processed_corpus.keys() 
                if aid not in positive_canonical_aids
            ])
            
            # Tạo Bi-Encoder triplet (query, positive, negative)
            bi_encoder_examples.append({
                "query": cleaned_question,
                "positive": processed_corpus[pos_aid],
                "negative": processed_corpus[negative_aid]
            })
            
            # Tạo Cross-Encoder pair (query, positive)
            cross_encoder_examples.append({
                "query": cleaned_question,
                "positive": processed_corpus[pos_aid],
                "label": 1  # Positive example
            })
            
            # Tạo Cross-Encoder pair (query, negative) 
            cross_encoder_examples.append({
                "query": cleaned_question,
                "positive": processed_corpus[negative_aid],
                "label": 0  # Negative example
            })

# Kết quả cuối cùng:
bi_encoder_examples = [
    {
        "query": "Luật về đất đai quy định gì về quyền sử dụng đất?",
        "positive": "Điều 15. Quyền sử dụng đất...",
        "negative": "Điều 20. Thuế đất đai..."
    }
]

cross_encoder_examples = [
    {
        "query": "Luật về đất đai quy định gì về quyền sử dụng đất?",
        "positive": "Điều 15. Quyền sử dụng đất...",
        "label": 1
    },
    {
        "query": "Luật về đất đai quy định gì về quyền sử dụng đất?",
        "positive": "Điều 20. Thuế đất đai...",
        "label": 0
    }
]
```

##### **Bước 5: Data Storage & Versioning - "Lưu Trữ Hợp Chất"**

```python
# Tạo timestamped directory cho mỗi data processing run
def run_data_preparation():
    """Chạy toàn bộ data preparation pipeline."""
    
    # 1. Load raw data
    train_data = load_json(config.paths.train_data_path)
    raw_corpus = load_json(config.paths.legal_corpus_path)
    
    # 2. Process corpus và tạo AID map
    processed_corpus, aid_map = process_legal_corpus(raw_corpus)
    
    # 3. Generate training examples
    bi_encoder_examples, cross_encoder_examples, stats = generate_training_examples(
        train_data, processed_corpus, aid_map
    )
    
    # 4. Tạo timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"features/processed_data_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Lưu processed data
    save_jsonl(bi_encoder_examples, output_dir / "bi_encoder_train.jsonl")
    save_jsonl(cross_encoder_examples, output_dir / "cross_encoder_train.jsonl")
    save_json(processed_corpus, output_dir / "processed_corpus.json")
    save_json(aid_map, output_dir / "aid_map.json")
    
    # 6. Lưu metadata
    metadata = {
        "timestamp": timestamp,
        "raw_data_stats": {
            "training_questions": len(train_data),
            "law_documents": len(raw_corpus)
        },
        "processed_data_stats": {
            "bi_encoder_examples": len(bi_encoder_examples),
            "cross_encoder_examples": len(cross_encoder_examples),
            "processed_articles": len(processed_corpus),
            "aid_mappings": len(aid_map)
        },
        "processing_stats": stats
    }
    
    save_metadata(output_dir, metadata)

# Kết quả directory structure:
features/
├── processed_data_20250127_143022/           # Timestamped version
│   ├── bi_encoder_train.jsonl               # Training data cho Tier 1
│   ├── cross_encoder_train.jsonl            # Training data cho Tier 3
│   ├── processed_corpus.json                # Processed legal corpus
│   ├── aid_map.json                         # AID mapping
│   └── metadata.json                        # Processing metadata
└── processed_data/                           # Base directory (symlink to latest)
```

#### **9.1.2 Centralized Path Management (`config/paths.py`)**

LawBot sử dụng centralized path management để đảm bảo tính nhất quán và dễ bảo trì:

```python
# config/paths.py - Centralized training data paths
TRAINING_DATA_PATHS = {
    "primary": {
        "bi_encoder": Path("features/processed_data/bi_encoder_train.jsonl"),
        "cross_encoder": Path("features/processed_data/cross_encoder_train.jsonl"),
        "processed_corpus": Path("features/processed_data/processed_corpus.json"),
        "training_data": Path("features/processed_data/training_data.jsonl"),
        "negative_pool": Path("features/processed_data/negative_pool.jsonl"),
    },
    "alternative": {
        "bi_encoder": [
            Path("features/processed_data/bi_encoder_train.jsonl"),
            Path("features/bi_encoder_train.jsonl"),
            Path("data/processed/bi_encoder_train.jsonl"),
        ],
    },
    "validation": {
        "tier_1": Path("features/validation_sets/tier_1_validation.jsonl"),
        "tier_2": Path("features/validation_sets/tier_2_validation.jsonl"),
        "tier_3": Path("features/validation_sets/tier_3_validation.jsonl"),
    }
}
```

#### 9.1.2 Data Quality Thresholds

```python
DATA_QUALITY_THRESHOLDS = {
    "bi_encoder": {
        "min_triplets": 5,      # Minimum triplets for real training
        "min_queries": 3,       # Minimum unique queries
        "min_content_length": 50, # Minimum content length
    },
    "light_ranking": {
        "min_examples": 5,      # Minimum training examples
        "min_queries": 3,       # Minimum unique queries
        "min_content_length": 50, # Minimum content length
    },
    "cross_encoder": {
        "min_examples": 5,      # Minimum training examples
        "min_queries": 3,       # Minimum unique queries
        "min_content_length": 50, # Minimum content length
    }
}
```

#### 9.1.3 Smart Path Discovery

```python
def get_training_data_path(tier: str, data_type: str) -> Path:
    """Automatically finds the latest processed data directory with timestamp."""
    features_dir = Path("features")
    processed_dirs = [
        d for d in features_dir.iterdir()
        if d.is_dir() and d.name.startswith("processed_data_")
    ]
    
    if not processed_dirs:
        return TRAINING_DATA_PATHS["primary"].get(data_type, Path(""))
    
    # Sort by creation time and get the latest
    latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
    return latest_dir / f"{data_type}.jsonl"
```

### 9.2 Feature Management

#### 9.2.1 Feature Pipeline Structure

#### 9.2.2 Training Data Flow: Từ Training Examples Đến Model Weights

**Quá trình training trong LawBot cũng tuân theo nguyên tắc "từ phân tử đến hợp chất phức tạp":**

##### **Bước 1: Data Loading & Preprocessing - "Chuẩn Bị Nguyên Liệu"**

```python
# training/run_bi_encoder.py - Data loading và preprocessing
class BiEncoderTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
    
    def load_training_data(self, data_path: Path) -> List[Dict]:
        """Load training data từ processed data directory."""
        
        # Load bi-encoder training examples
        bi_encoder_data = load_jsonl(data_path / "bi_encoder_train.jsonl")
        
        # Load processed corpus cho content lookup
        processed_corpus = load_json(data_path / "processed_corpus.json")
        
        logger.info(f"Loaded {len(bi_encoder_data)} bi-encoder training examples")
        logger.info(f"Loaded {len(processed_corpus)} processed articles")
        
        return bi_encoder_data, processed_corpus

# Ví dụ training data structure:
bi_encoder_data = [
    {
        "query": "Luật về đất đai quy định gì về quyền sử dụng đất?",
        "positive": "Điều 15. Quyền sử dụng đất...",
        "negative": "Điều 20. Thuế đất đai..."
    },
    {
        "query": "Thuế đất đai được tính như thế nào?",
        "positive": "Điều 20. Thuế đất đai...",
        "negative": "Điều 15. Quyền sử dụng đất..."
    }
]
```

##### **Bước 2: Dataset Creation - "Tạo Dataset Objects"**

```python
# core/datasets/legal_qa.py - Dataset classes cho training
class BiEncoderDataset(Dataset):
    """Dataset cho Bi-Encoder training với contrastive learning."""
    
    def __init__(self, data: List[Dict], tokenizer: PreTrainedTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize query
        query_encoding = self.tokenizer(
            item["query"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize positive example
        positive_encoding = self.tokenizer(
            item["positive"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize negative example
        negative_encoding = self.tokenizer(
            item["negative"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_encoding["input_ids"].squeeze(0),
            "query_attention_mask": query_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0)
        }

# Tạo dataset instances
def create_datasets(bi_encoder_data, tokenizer, max_length=256):
    """Tạo training và validation datasets."""
    
    # Split data thành train và validation
    random.shuffle(bi_encoder_data)
    split_idx = int(len(bi_encoder_data) * 0.8)
    
    train_data = bi_encoder_data[:split_idx]
    val_data = bi_encoder_data[split_idx:]
    
    # Tạo dataset objects
    train_dataset = BiEncoderDataset(train_data, tokenizer, max_length)
    val_dataset = BiEncoderDataset(val_data, tokenizer, max_length)
    
    return train_dataset, val_dataset
```

##### **Bước 3: Model Initialization - "Khởi Tạo Model Architecture"**

```python
# training/run_bi_encoder.py - Model initialization
def setup_model_and_tokenizer(self, model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"):
    """Setup model và tokenizer cho training."""
    
    # Load pre-trained Vietnamese Bi-Encoder
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    
    # Move model to device
    self.model.to(self.device)
    
    # Setup training components
    self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
    self.scheduler = get_linear_schedule_with_warmup(
        self.optimizer,
        num_warmup_steps=self.num_warmup_steps,
        num_training_steps=self.num_training_steps
    )
    
    logger.info(f"Model loaded: {model_name}")
    logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    logger.info(f"Device: {self.device}")

# Model architecture overview:
# Input: [CLS] query text [SEP] -> Encoder -> [CLS] embedding
# Input: [CLS] document text [SEP] -> Encoder -> [CLS] embedding
# Output: 768-dimensional embeddings cho similarity calculation
```

##### **Bước 4: Training Loop - "Quá Trình Học"**

```python
# training/engine.py - Training loop implementation
def train_epoch(self, epoch: int) -> Dict[str, float]:
    """Train một epoch."""
    
    self.model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        query_embeddings = self.model(
            input_ids=batch["query_input_ids"],
            attention_mask=batch["query_attention_mask"]
        ).last_hidden_state[:, 0, :]  # [CLS] token
        
        positive_embeddings = self.model(
            input_ids=batch["positive_input_ids"],
            attention_mask=batch["positive_attention_mask"]
        ).last_hidden_state[:, 0, :]
        
        negative_embeddings = self.model(
            input_ids=batch["negative_input_ids"],
            attention_mask=batch["negative_attention_mask"]
        ).last_hidden_state[:, 0, :]
        
        # Calculate contrastive loss
        loss = self.contrastive_loss(
            query_embeddings, positive_embeddings, negative_embeddings
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return {"train_loss": avg_loss}

# Contrastive loss implementation:
def contrastive_loss(self, query_emb, positive_emb, negative_emb):
    """Triplet loss cho contrastive learning."""
    
    # Calculate similarities
    pos_sim = F.cosine_similarity(query_emb, positive_emb, dim=1)
    neg_sim = F.cosine_similarity(query_emb, negative_emb, dim=1)
    
    # Triplet loss: maximize positive similarity, minimize negative similarity
    margin = 0.1
    loss = torch.clamp(neg_sim - pos_sim + margin, min=0)
    
    return loss.mean()
```

##### **Bước 5: Model Saving & Metadata - "Lưu Trữ Kết Quả"**

```python
# training/engine.py - Model saving và metadata
def save_model(self, output_dir: Path):
    """Save trained model và metadata."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    self.model.save_pretrained(output_dir)
    self.tokenizer.save_pretrained(output_dir)
    
    # Save training metadata
    metadata = {
        "model_name": "vietnamese-bi-encoder",
        "version": "v8.3",
        "training_timestamp": datetime.now().isoformat(),
        "training_duration_seconds": self.training_duration,
        "epochs": self.num_epochs,
        "batch_size": self.batch_size,
        "learning_rate": self.learning_rate,
        "final_loss": self.final_loss,
        "training_samples": len(self.train_dataset),
        "validation_samples": len(self.val_dataset),
        "hardware_used": {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "model_architecture": {
            "embedding_dim": 768,
            "max_length": self.max_length,
            "vocab_size": self.tokenizer.vocab_size
        }
    }
    
    # Save metadata
    save_metadata(output_dir, metadata)
    
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Metadata: {metadata}")

# Kết quả directory structure:
models/
├── bi-encoder_20250127_143022/           # Timestamped model version
│   ├── config.json                       # Model configuration
│   ├── pytorch_model.bin                # Model weights (517MB)
│   ├── tokenizer.json                   # Tokenizer configuration
│   ├── vocab.txt                        # Vocabulary
│   ├── special_tokens_map.json          # Special tokens
│   └── metadata.json                    # Training metadata
└── bi-encoder_latest/                   # Symlink to latest version
    └── -> bi-encoder_20250127_143022/
```

#### 9.2.3 Feature Versioning Strategy

```
features/
├── processed_data/                    # Base processed data
│   ├── bi_encoder_train.jsonl        # Tier 1 training data
│   ├── cross_encoder_train.jsonl     # Tier 3 training data
│   ├── training_data.jsonl           # Tier 2 training data
│   ├── negative_pool.jsonl           # Negative examples pool
│   └── processed_corpus.json         # Processed legal corpus
├── processed_data_20250127_143022/   # Timestamped version
│   ├── bi_encoder_train.jsonl
│   ├── cross_encoder_train.jsonl
│   ├── training_data.jsonl
│   ├── negative_pool.jsonl
│   └── processed_corpus.json
├── validation_sets/                   # Validation data
│   ├── tier_1_validation.jsonl
│   ├── tier_2_validation.jsonl
│   └── tier_3_validation.jsonl
└── faiss_index/                       # FAISS vector index
    ├── faiss_index.bin
    ├── aid_map.json
    └── index_to_aid.json
```

#### 9.2.4 Feature Versioning Strategy

```python
# core/utils/versioning.py
def generate_versioned_path(base_dir: str, name: str) -> Path:
    """Generates a versioned directory path using a timestamp."""
    timestamp = get_timestamp()  # Format: YYYYMMDD_HHMMSS
    return Path(base_dir) / f"{name}_{timestamp}"

def get_timestamp() -> str:
    """Returns a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_latest_version_path(base_dir: str, model_name: str) -> Optional[Path]:
    """Finds the path to the latest version of a model/feature."""
    paths = list(Path(base_dir).glob(f"{model_name}_*"))
    if not paths:
        return None
    return max(paths, key=lambda p: p.name)
```

### 9.3 Model Versioning Management

#### 9.3.1 Model Directory Structure

#### 9.3.2 Inference Pipeline: Từ Model Weights Đến Final Results

**Quá trình inference trong LawBot cũng tuân theo nguyên tắc "từ hợp chất đơn giản đến phức tạp":**

##### **Bước 1: Model Loading & Initialization - "Khởi Tạo Hệ Thống"**

```python
# core/pipeline.py - Pipeline initialization
class LegalQAPipeline:
    def __init__(self, bi_encoder_path=None, reranker_paths=None, faiss_index_path=None):
        """Khởi tạo 3-tier pipeline."""
        
        # Auto-discover latest model versions nếu không specify paths
        if not bi_encoder_path:
            bi_encoder_path = get_latest_version_path("models", "bi-encoder")
        
        if not faiss_index_path:
            faiss_index_path = get_latest_version_path("models", "faiss_index")
        
        # Initialize Tier 1: Bi-Encoder Retrieval
        self.retrieval_engine = RetrievalEngine(
            bi_encoder_path=str(bi_encoder_path),
            faiss_index_path=str(faiss_index_path),
            content_map_path="features/aid_map.json",
            index_to_aid_path="features/index_to_aid.json"
        )
        
        # Initialize Tier 2: Light Reranker
        if not reranker_paths:
            light_ranking_path = get_latest_version_path("models", "light-ranking")
            reranker_paths = {"light_ranking": light_ranking_path}
        
        self.reranking_engine = RerankingEngine(reranker_paths)
        
        # Initialize Tier 3: Cross-Encoder Ensemble
        cross_encoder_path = get_latest_version_path("models", "combined-reranker-adapt")
        if cross_encoder_path:
            reranker_paths["cross_encoder"] = cross_encoder_path
        
        logger.info(f"Pipeline initialized with:")
        logger.info(f"  - Bi-Encoder: {bi_encoder_path}")
        logger.info(f"  - Light Reranker: {reranker_paths.get('light_ranking')}")
        logger.info(f"  - Cross-Encoder: {reranker_paths.get('cross_encoder')}")
        logger.info(f"  - FAISS Index: {faiss_index_path}")

# Model loading process:
# 1. Tìm latest version của mỗi model type
# 2. Load model weights và configuration
# 3. Move models to appropriate device (CPU/GPU)
# 4. Initialize FAISS index cho vector search
```

##### **Bước 2: Query Processing & Embedding - "Xử Lý Input"**

```python
# core/retrieval.py - Query processing
class RetrievalEngine:
    def __init__(self, bi_encoder_path, faiss_index_path, content_map_path, index_to_aid_path):
        """Initialize retrieval engine."""
        
        # Load Bi-Encoder model
        self.model = SentenceTransformer(bi_encoder_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load content mappings
        self.content_map = load_json(content_map_path)
        self.index_to_aid = load_json(index_to_aid_path)
        
        logger.info(f"Retrieval engine initialized:")
        logger.info(f"  - Model: {bi_encoder_path}")
        logger.info(f"  - FAISS index: {self.faiss_index.ntotal} documents")
        logger.info(f"  - Content map: {len(self.content_map)} articles")
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve top-k documents cho query."""
        
        # 1. Encode query thành embedding
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        
        # 2. Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, top_k)
        
        # 3. Map indices back to documents
        results = []
        for sim_score, idx in zip(similarities[0], indices[0]):
            if idx != -1:  # Valid index
                aid = self.index_to_aid[str(idx)]
                content = self.content_map.get(aid, "")
                
                results.append({
                    "aid": aid,
                    "content": content,
                    "similarity_score": float(sim_score),
                    "index": int(idx)
                })
        
        # 4. Sort by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results

# Query processing flow:
# Input: "Luật về đất đai quy định gì?"
# 1. Tokenize: [CLS] Luật về đất đai quy định gì? [SEP]
# 2. Encode: [0.123, -0.456, 0.789, ...] (768 dimensions)
# 3. FAISS search: Find top 100 similar vectors
# 4. Map back: Convert indices to document content
```

##### **Bước 3: Light Reranking - "Lọc Sơ Bộ"**

```python
# core/reranking.py - Light reranking implementation
class RerankingEngine:
    def rank_light(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Light reranking với PhoBERT model."""
        
        # Load light ranking model
        model_name = "light_ranking"
        if model_name not in self.models:
            logger.warning(f"Light ranking model {model_name} not available")
            return documents
        
        model, tokenizer = self.models[model_name]
        
        # Prepare input pairs
        sentence_pairs = [(query, doc["content"]) for doc in documents]
        
        # Get predictions
        scores = self._predict_batch(model_name, sentence_pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["light_ranking_score"] = score
        
        # Sort by light ranking score
        documents.sort(key=lambda x: x["light_ranking_score"], reverse=True)
        
        return documents

# Light ranking process:
# Input: Top 100 documents từ Tier 1
# 1. Tạo query-document pairs
# 2. Encode pairs với PhoBERT model
# 3. Get classification scores (0-1)
# 4. Sort documents by scores
# Output: Top 80 documents với light ranking scores
```

##### **Bước 4: Cross-Encoder Ensemble - "Ranking Cuối Cùng"**

```python
# core/reranking.py - Cross-encoder ensemble
class EnsembleCrossEncoder:
    def __init__(self, adapt_model, base_model, adapt_weight=0.7, base_weight=0.3):
        """Ensemble của ADAPT-enhanced và base model."""
        
        self.adapt_model = adapt_model
        self.base_model = base_model
        self.adapt_weight = adapt_weight
        self.base_weight = base_weight
        
        # Normalize weights
        total_weight = adapt_weight + base_weight
        self.adapt_weight /= total_weight
        self.base_weight /= total_weight
    
    def __call__(self, **inputs):
        """Get ensemble predictions."""
        
        # Get predictions from both models
        adapt_output = self.adapt_model(**inputs)
        base_output = self.base_model(**inputs)
        
        # Weighted ensemble
        ensemble_output = (
            self.adapt_weight * adapt_output + 
            self.base_weight * base_output
        )
        
        return ensemble_output

def rank_cross(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-encoder ranking với ensemble model."""
    
    # Load cross-encoder model
    model_name = "cross_encoder"
    if model_name not in self.models:
        logger.warning(f"Cross-encoder model {model_name} not available")
        return documents
    
    model, tokenizer = self.models[model_name]
    
    # Prepare input pairs
    sentence_pairs = [(query, doc["content"]) for doc in documents]
    
    # Get ensemble predictions
    scores = self._predict_batch(model_name, sentence_pairs)
    
    # Add cross-encoder scores
    for doc, score in zip(documents, scores):
        doc["cross_encoder_score"] = score
    
    # Sort by cross-encoder score
    documents.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    
    return documents

# Cross-encoder process:
# Input: Top 80 documents từ Tier 2
# 1. Tạo query-document pairs
# 2. Encode với ensemble model (70% ADAPT + 30% Base)
# 3. Get regression scores
# 4. Sort documents by final scores
# Output: Top 10 documents với cross-encoder scores
```

##### **Bước 5: Score Aggregation & Final Ranking - "Tổng Hợp Kết Quả"**

```python
# core/pipeline.py - Score aggregation
def _combine_scores(self, retrieval_results, light_results, cross_results):
    """Kết hợp scores từ tất cả tiers."""
    
    # Create final results với combined scores
    final_results = []
    
    for doc in cross_results:
        # Get scores from all tiers
        retrieval_score = doc.get("similarity_score", 0.0)
        light_score = doc.get("light_ranking_score", 0.0)
        cross_score = doc.get("cross_encoder_score", 0.0)
        
        # Calculate weighted final score
        final_score = (
            0.2 * retrieval_score +      # Tier 1: 20% weight
            0.3 * light_score +          # Tier 2: 30% weight
            0.5 * cross_score            # Tier 3: 50% weight
        )
        
        # Create final result
        final_result = {
            "aid": doc["aid"],
            "content": doc["content"],
            "scores": {
                "retrieval": retrieval_score,
                "light_ranking": light_score,
                "cross_encoder": cross_score,
                "final": final_score
            },
            "rank": len(final_results) + 1
        }
        
        final_results.append(final_result)
    
    # Sort by final score
    final_results.sort(key=lambda x: x["scores"]["final"], reverse=True)
    
    return final_results

# Score aggregation process:
# 1. Lấy scores từ tất cả 3 tiers
# 2. Tính weighted average (20% + 30% + 50%)
# 3. Sort by final score
# 4. Add ranking information
# Output: Final ranked results với comprehensive scores
```

##### **Bước 6: Pipeline Execution Flow - "Toàn Bộ Quá Trình"**

```python
# core/pipeline.py - Main pipeline execution
def predict(self, query: str, top_k_retrieval=100, top_k_light=80, top_k_final=10):
    """Execute complete 3-tier pipeline."""
    
    start_time = time.time()
    
    # Tier 1: Bi-Encoder Retrieval
    logger.info("🔄 Tier 1: Bi-Encoder Retrieval")
    retrieval_results = self.retrieval_engine.retrieve(query, top_k_retrieval)
    logger.info(f"✅ Retrieved {len(retrieval_results)} documents")
    
    # Tier 2: Light Reranking
    logger.info("🔄 Tier 2: Light Reranking")
    light_results = self.reranking_engine.rank_light(query, retrieval_results[:top_k_light])
    logger.info(f"✅ Light ranked {len(light_results)} documents")
    
    # Tier 3: Cross-Encoder Ensemble
    logger.info("🔄 Tier 3: Cross-Encoder Ensemble")
    cross_results = self.reranking_engine.rank_cross(query, light_results[:top_k_final])
    logger.info(f"✅ Cross-encoded {len(cross_results)} documents")
    
    # Final aggregation
    logger.info("🔄 Final Score Aggregation")
    final_results = self._combine_scores(retrieval_results, light_results, cross_results)
    
    # Add metadata
    execution_time = time.time() - start_time
    pipeline_metadata = {
        "query": query,
        "execution_time_seconds": execution_time,
        "tier_results": {
            "retrieval_count": len(retrieval_results),
            "light_ranking_count": len(light_results),
            "cross_encoder_count": len(cross_results),
            "final_count": len(final_results)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Pipeline completed in {execution_time:.2f}s")
    logger.info(f"📊 Final results: {len(final_results)} documents")
    
    return final_results

# Complete pipeline flow:
# Input: "Luật về đất đai quy định gì?"
# 1. Tier 1: Bi-Encoder + FAISS → Top 100 documents
# 2. Tier 2: Light Reranker → Top 80 documents  
# 3. Tier 3: Cross-Encoder Ensemble → Top 10 documents
# 4. Score Aggregation → Final ranked results
# Output: Ranked legal documents với comprehensive scores
```

#### 9.3.4 Model Directory Structure

```
models/
├── bi-encoder_20250127_143022/       # Timestamped model version
│   ├── config.json                   # Model configuration
│   ├── pytorch_model.bin            # Model weights
│   ├── training_metadata.json       # Training information
│   ├── hpo_results.json             # Hyperparameter optimization results
│   └── performance_metrics.json     # Model performance data
├── light-ranking_20250127_143045/    # Light ranking model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── training_metadata.json
│   └── hpo_results.json
├── combined-reranker-adapt_20250127_143108/  # Cross-encoder ensemble
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── training_metadata.json
│   ├── ensemble_config.json         # Ensemble configuration
│   └── hpo_results.json
└── faiss_index/                      # FAISS index storage
    ├── faiss_index.bin
    ├── aid_map.json
    └── index_to_aid.json
```

#### 9.3.2 Centralized Model Configuration (`config/models.py`)

```python
MODEL_TYPES = {
    "bi_encoder": {
        "display_name": "Bi-Encoder",
        "directory_prefix": "bi-encoder",
        "status_key": "bi_encoder",
        "training_stage": "bi_encoder",
        "data_type": "triplets",
        "score_field": "similarity_score",
        "loading_type": "sentence_transformers"
    },
    "light_reranker": {
        "display_name": "Light Reranker",
        "directory_prefix": "light-ranking",
        "status_key": "light_reranker",
        "training_stage": "light_ranking",
        "data_type": "classification",
        "score_field": "classification_score",
        "loading_type": "transformers"
    },
    "cross_encoder": {
        "display_name": "Cross-Encoder Ensemble",
        "directory_prefix": "combined-reranker-adapt",
        "status_key": "cross_encoder",
        "training_stage": "cross_encoder",
        "data_type": "pairs",
        "score_field": "ensemble_score",
        "loading_type": "ensemble"
    }
}

MODEL_DIRECTORY_MAPPING = {
    "bi_encoder": "bi-encoder",
    "light_reranker": "light-ranking",
    "cross_encoder": "combined-reranker-adapt",
}
```

#### 9.3.3 Model Status Tracking

```python
def get_model_status() -> Dict[str, Dict[str, Any]]:
    """Checks for the existence, path, and size of all model types."""
    models_dir = Path(config.paths.model_dir)
    status = {}
    
    for key, dir_prefix in MODEL_DIRECTORY_MAPPING.items():
        # Find latest model version
        latest_path = get_latest_version_path("models", dir_prefix)
        
        if latest_path and latest_path.exists():
            status[key] = {
                "status": "ready",
                "exists": True,
                "path": str(latest_path),
                "version": latest_path.name,
                "size_mb": _get_directory_size_mb(latest_path),
                "last_modified": latest_path.stat().st_mtime,
                "metadata": _load_model_metadata(latest_path)
            }
        else:
            status[key] = {
                "status": "not_ready",
                "exists": False,
                "path": None,
                "version": None,
                "size_mb": 0,
                "last_modified": None,
                "metadata": None
            }
    
    return status
```

### 9.4 Workflow Checkpoint Management

#### 9.4.1 Checkpoint System Architecture

```python
class WorkflowCheckpoint:
    """Manages workflow checkpoints for resuming interrupted runs."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(self, stage: str, status: str, metadata: Dict[str, Any]):
        """Save a checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"
        checkpoint_data = {
            "stage": stage,
            "status": status,
            "timestamp": get_timestamp(),
            "metadata": metadata
        }
        
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a stage."""
        checkpoint_file = self.checkpoint_dir / f"{stage}_checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint for {stage}: {e}")
        return None
```

#### 9.4.2 Checkpoint Data Structure

```python
# Example checkpoint data
checkpoint_data = {
    "stage": "bi_encoder",
    "status": "completed",
    "timestamp": "20250127_143022",
    "metadata": {
        "model_path": "models/bi-encoder_20250127_143022",
        "training_duration": 120.5,
        "final_loss": 0.0234,
        "training_samples": 1500,
        "validation_samples": 300,
        "hpo_results": {
            "best_score": 0.87,
            "best_params": {"learning_rate": 2e-5, "batch_size": 16}
        },
        "device_used": "cuda",
        "gpu_memory_used_gb": 8.5,
        "checkpoint_file": "checkpoints/bi_encoder_training.json"
    }
}
```

### 9.5 Advanced Techniques & Optimization: Từ Phân Tử Đến Hợp Chất Siêu Phức Tạp

#### 9.5.1 Hard Negative Mining: Tìm Kiếm Negative Examples Khó Nhất

**Hard Negative Mining là kỹ thuật quan trọng để cải thiện chất lượng training data:**

##### **Nguyên Lý Cơ Bản**

```python
# training/hard_negative_mining.py - Hard Negative Mining implementation
class HardNegativeMiner:
    def __init__(self, model, device: str = "cpu"):
        """Initialize hard negative miner với pre-trained model."""
        
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Mining parameters
        self.similarity_threshold = 0.7  # Threshold để xác định hard negatives
        self.top_k = 5                  # Số lượng hard negatives cần tìm
        self.mining_ratio = 0.3         # Tỷ lệ hard negatives trong training data
    
    def mine_hard_negatives(
        self, 
        queries: List[str], 
        positive_docs: List[str], 
        negative_candidates: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """Tìm hard negative examples từ negative candidates."""
        
        hard_negatives = []
        
        for query in queries:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            query_embedding = query_embedding.to(self.device)
            
            # Encode negative candidates
            negative_embeddings = self.model.encode(negative_candidates, convert_to_tensor=True)
            negative_embeddings = negative_embeddings.to(self.device)
            
            # Calculate similarities
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(1), 
                negative_embeddings.unsqueeze(0), 
                dim=2
            ).squeeze(0)
            
            # Find hardest negatives (highest similarity với query)
            # Hard negatives là những negative examples có similarity cao với query
            # Điều này làm cho training khó hơn và cải thiện model performance
            hard_negative_indices = torch.topk(similarities, k=top_k, largest=True).indices
            
            for idx in hard_negative_indices:
                if similarities[idx] >= similarity_threshold:
                    hard_negatives.append(negative_candidates[idx])
        
        # Remove duplicates và return
        return list(set(hard_negatives))

# Ví dụ sử dụng:
# Input:
# - queries: ["Luật về đất đai quy định gì?"]
# - positive_docs: ["Điều 15. Quyền sử dụng đất..."]
# - negative_candidates: ["Điều 20. Thuế đất đai...", "Điều 25. Xử phạt...", ...]
#
# Process:
# 1. Encode query: [0.123, -0.456, 0.789, ...]
# 2. Encode negatives: [[0.111, -0.444, 0.777, ...], [0.222, -0.333, 0.666, ...], ...]
# 3. Calculate similarities: [0.85, 0.72, 0.45, ...]
# 4. Find hardest: indices [0, 1] (highest similarities)
# 5. Return: ["Điều 20. Thuế đất đai...", "Điều 25. Xử phạt..."]
```

##### **Advanced Hard Negative Mining với Dynamic Threshold**

```python
# training/run_light_ranking.py - Advanced HNM implementation
def _mine_hard_negatives_with_new_model(
    self, 
    queries: List[str], 
    positive_docs: List[str], 
    negative_candidates: List[str],
    hpo_params: Dict[str, Any]
) -> List[str]:
    """Advanced hard negative mining với dynamic threshold và adaptive mining."""
    
    # Get HPO-optimized parameters
    similarity_threshold = hpo_params.get("similarity_threshold", 0.7)
    top_k = hpo_params.get("top_k", 5)
    mining_ratio = hpo_params.get("mining_ratio", 0.3)
    
    # Adaptive threshold based on data distribution
    if len(queries) > 100:  # Large dataset
        # Use percentile-based threshold
        all_similarities = []
        for query in queries[:100]:  # Sample first 100 queries
            query_emb = self.model.encode([query], convert_to_tensor=True)
            neg_embs = self.model.encode(negative_candidates[:50], convert_to_tensor=True)
            
            sims = F.cosine_similarity(query_emb.unsqueeze(1), neg_embs.unsqueeze(0), dim=2)
            all_similarities.extend(sims.squeeze(0).tolist())
        
        # Use 80th percentile as threshold
        similarity_threshold = np.percentile(all_similarities, 80)
        logger.info(f"Adaptive threshold: {similarity_threshold:.3f}")
    
    # Mine hard negatives
    hard_negatives = self.miner.mine_hard_negatives(
        queries, positive_docs, negative_candidates,
        top_k=top_k, similarity_threshold=similarity_threshold
    )
    
    # Quality control: ensure diversity
    diverse_hard_negatives = self._ensure_diversity(hard_negatives, queries)
    
    # Log mining statistics
    mining_stats = {
        "total_candidates": len(negative_candidates),
        "hard_negatives_found": len(diverse_hard_negatives),
        "mining_ratio": len(diverse_hard_negatives) / len(negative_candidates),
        "similarity_threshold": similarity_threshold,
        "top_k": top_k
    }
    
    logger.info(f"Hard Negative Mining completed: {mining_stats}")
    
    return diverse_hard_negatives

def _ensure_diversity(self, hard_negatives: List[str], queries: List[str]) -> List[str]:
    """Đảm bảo tính đa dạng của hard negatives."""
    
    diverse_negatives = []
    used_content_patterns = set()
    
    for negative in hard_negatives:
        # Extract content pattern (first few words)
        content_pattern = " ".join(negative.split()[:5])
        
        if content_pattern not in used_content_patterns:
            diverse_negatives.append(negative)
            used_content_patterns.add(content_pattern)
        
        if len(diverse_negatives) >= len(hard_negatives) * 0.8:  # Keep 80% diversity
            break
    
    return diverse_negatives
```

#### 9.5.2 ADAPT Domain Adaptation: Thích Ứng Với Pháp Luật Việt Nam

**ADAPT (Adaptive Domain-Adversarial Training) giúp model thích ứng với domain pháp luật:**

##### **Nguyên Lý ADAPT**

```python
# training/run_bi_encoder.py - ADAPT implementation
def apply_adapt_technique(self, model: Any, domain_data: list) -> Any:
    """Áp dụng ADAPT technique cho domain adaptation."""
    
    # Domain data là legal text từ Việt Nam
    # Mục tiêu: làm cho model hiểu tốt hơn về pháp luật Việt Nam
    
    class ADAPTModel(nn.Module):
        def __init__(self, base_model, domain_classifier_dim=768):
            super().__init__()
            
            # Base model (Vietnamese Bi-Encoder)
            self.base_model = base_model
            
            # Domain classifier để phân biệt source domain vs target domain
            self.domain_classifier = nn.Sequential(
                nn.Linear(768, 384),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(384, 192),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(192, 2)  # 2 classes: source vs target domain
            )
            
            # Task-specific head cho contrastive learning
            self.task_head = nn.Linear(768, 768)
            
            # Gradient reversal layer cho adversarial training
            self.gradient_reversal = GradientReversalLayer(alpha=0.1)
        
        def forward(self, input_ids, attention_mask, domain_labels=None):
            # Get base model output
            base_output = self.base_model(input_ids, attention_mask)
            shared_features = base_output.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Task-specific output (contrastive learning)
            task_output = self.task_head(shared_features)
            
            # Domain classification (adversarial)
            if domain_labels is not None:
                # Apply gradient reversal để fool domain classifier
                reversed_features = self.gradient_reversal(shared_features)
                domain_logits = self.domain_classifier(reversed_features)
                return task_output, domain_logits
            
            return task_output
    
    # Gradient Reversal Layer implementation
    class GradientReversalLayer(nn.Module):
        def __init__(self, alpha=0.1):
            super().__init__()
            self.alpha = alpha
        
        def forward(self, x):
            return x
        
        def backward(self, grad_output):
            # Reverse gradient direction
            return -self.alpha * grad_output
    
    # Create ADAPT model
    adapt_model = ADAPTModel(model)
    adapt_model.to(self.device)
    
    # Prepare domain data
    domain_dataset = self._create_domain_dataset(domain_data)
    
    # Train ADAPT model
    self._train_adapt_model(adapt_model, domain_dataset)
    
    return adapt_model

def _create_domain_dataset(self, domain_data: list) -> List[Dict]:
    """Tạo dataset cho domain adaptation."""
    
    domain_examples = []
    
    for item in domain_data:
        # Source domain: general Vietnamese text
        source_text = item.get("source_text", "")
        if source_text:
            domain_examples.append({
                "text": source_text,
                "domain_label": 0  # Source domain
            })
        
        # Target domain: legal text
        target_text = item.get("target_text", "")
        if target_text:
            domain_examples.append({
                "text": target_text,
                "domain_label": 1  # Target domain (legal)
            })
    
    return domain_examples

def _train_adapt_model(self, adapt_model: nn.Module, domain_dataset: List[Dict]):
    """Train ADAPT model với domain adaptation."""
    
    # Setup training
    optimizer = AdamW(adapt_model.parameters(), lr=1e-5)
    domain_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    adapt_model.train()
    for epoch in range(3):  # Few epochs for adaptation
        total_loss = 0.0
        
        for batch in domain_dataset:
            # Prepare input
            text = batch["text"]
            domain_label = batch["domain_label"]
            
            # Tokenize
            inputs = self.tokenizer(
                text, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)
            
            domain_labels = torch.tensor([domain_label]).to(self.device)
            
            # Forward pass
            task_output, domain_logits = adapt_model(
                inputs["input_ids"], 
                inputs["attention_mask"], 
                domain_labels
            )
            
            # Calculate domain classification loss
            domain_loss = domain_criterion(domain_logits, domain_labels)
            
            # Backward pass
            domain_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += domain_loss.item()
        
        avg_loss = total_loss / len(domain_dataset)
        logger.info(f"ADAPT Epoch {epoch+1}, Domain Loss: {avg_loss:.4f}")
```

##### **ADAPT Training Strategy**

```python
# training/run_reranker.py - ADAPT training strategy
def create_ensemble_strategy(self, adapt_model: Any, base_model: Any) -> Dict[str, Any]:
    """Tạo ensemble strategy với ADAPT và base model."""
    
    # Ensemble configuration
    ensemble_config = {
        "adapt_weight": 0.7,      # ADAPT model weight
        "base_weight": 0.3,       # Base model weight
        "ensemble_method": "weighted_average",
        "adaptation_steps": 1000,
        "domain_data_ratio": 0.3
    }
    
    # Create ensemble model
    ensemble_model = EnsembleReranker(
        adapt_model=adapt_model,
        base_model=base_model,
        adapt_weight=ensemble_config["adapt_weight"],
        base_weight=ensemble_config["base_weight"]
    )
    
    # Training strategy
    training_strategy = {
        "phase_1": {
            "description": "Base model training",
            "epochs": 3,
            "learning_rate": 2e-5,
            "focus": "General task learning"
        },
        "phase_2": {
            "description": "ADAPT adaptation",
            "epochs": 2,
            "learning_rate": 1e-5,
            "focus": "Domain adaptation"
        },
        "phase_3": {
            "description": "Ensemble fine-tuning",
            "epochs": 1,
            "learning_rate": 5e-6,
            "focus": "Ensemble optimization"
        }
    }
    
    return {
        "ensemble_model": ensemble_model,
        "ensemble_config": ensemble_config,
        "training_strategy": training_strategy
    }

# Ensemble model implementation
class EnsembleReranker(torch.nn.Module):
    def __init__(self, adapt_model, base_model, adapt_weight=0.7, base_weight=0.3):
        super().__init__()
        
        self.adapt_model = adapt_model
        self.base_model = base_model
        
        # Normalize weights
        total_weight = adapt_weight + base_weight
        self.adapt_weight = adapt_weight / total_weight
        self.base_weight = base_weight / total_weight
        
        logger.info(f"Ensemble weights: ADAPT={self.adapt_weight:.2f}, Base={self.base_weight:.2f}")
    
    def forward(self, input_ids, attention_mask):
        # Get predictions from both models
        adapt_output = self.adapt_model(input_ids, attention_mask)
        base_output = self.base_model(input_ids, attention_mask)
        
        # Weighted ensemble
        ensemble_output = (
            self.adapt_weight * adapt_output + 
            self.base_weight * base_output
        )
        
        return ensemble_output
    
    def save_pretrained(self, save_directory):
        """Save ensemble model."""
        # Save individual models
        adapt_dir = Path(save_directory) / "adapt_model"
        base_dir = Path(save_directory) / "base_model"
        
        self.adapt_model.save_pretrained(adapt_dir)
        self.base_model.save_pretrained(base_dir)
        
        # Save ensemble configuration
        ensemble_config = {
            "adapt_weight": self.adapt_weight,
            "base_weight": self.base_weight,
            "ensemble_method": "weighted_average"
        }
        
        config_path = Path(save_directory) / "ensemble_config.json"
        with open(config_path, "w") as f:
            json.dump(ensemble_config, f, indent=2)
```

#### 9.5.3 HPO Implementation: Hyperparameter Optimization với Optuna

**Hyperparameter Optimization giúp tìm ra best parameters cho mỗi model:**

##### **Optuna Integration**

```python
# training/hpo.py - Optuna-based HPO implementation
class HyperparameterOptimizer:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize HPO với Optuna."""
        
        self.config = self._get_default_config()
        if config_path:
            self.config.update(self._load_config(config_path))
        
        # Optuna study configuration
        self.study_config = {
            "sampler": "tpe",           # Tree-structured Parzen Estimator
            "pruner": "median",         # Median pruner for early stopping
            "direction": "maximize",    # Maximize objective function
            "n_trials": 50,             # Number of trials
            "timeout": 3600             # Timeout in seconds
        }
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function cho Optuna optimization."""
        
        # Suggest hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            "epochs": trial.suggest_int("epochs", 2, 8),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
            "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 8)
        }
        
        # Advanced parameters cho specific techniques
        if self.config.get("use_hard_negative_mining", False):
            params.update({
                "similarity_threshold": trial.suggest_float("similarity_threshold", 0.5, 0.9),
                "mining_ratio": trial.suggest_float("mining_ratio", 0.1, 0.5),
                "top_k": trial.suggest_int("top_k", 3, 10)
            })
        
        if self.config.get("use_adapt", False):
            params.update({
                "adapt_weight": trial.suggest_float("adapt_weight", 0.5, 0.9),
                "domain_data_ratio": trial.suggest_float("domain_data_ratio", 0.2, 0.5),
                "adaptation_steps": trial.suggest_int("adaptation_steps", 500, 2000)
            })
        
        # Train model với suggested parameters
        try:
            model = self._train_with_params(params)
            score = self._evaluate_model(model)
            
            # Log trial results
            trial.set_user_attr("params", params)
            trial.set_user_attr("model_path", str(model))
            
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float("-inf")  # Return worst possible score
    
    def _train_with_params(self, params: Dict[str, Any]) -> Any:
        """Train model với given parameters."""
        
        # Setup training environment
        training_config = {
            "model_name": self.config["model_name"],
            "data_path": self.config["data_path"],
            "output_dir": f"models/hpo_trial_{uuid.uuid4().hex[:8]}",
            **params
        }
        
        # Initialize trainer
        trainer = self._get_trainer_class()(training_config)
        
        # Train model
        model_path = trainer.train()
        
        return model_path
    
    def _evaluate_model(self, model_path: str) -> float:
        """Evaluate model và return score."""
        
        try:
            # Load trained model
            model = self._load_model(model_path)
            
            # Run evaluation
            eval_results = self._run_evaluation(model)
            
            # Calculate composite score
            score = self._calculate_composite_score(eval_results)
            
            return score
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0
    
    def _calculate_composite_score(self, eval_results: Dict[str, Any]) -> float:
        """Tính composite score từ evaluation results."""
        
        # Weighted combination của multiple metrics
        weights = {
            "precision": 0.3,
            "recall": 0.2,
            "f1": 0.3,
            "ndcg": 0.2
        }
        
        composite_score = 0.0
        
        for metric, weight in weights.items():
            if metric in eval_results:
                metric_value = eval_results[metric]
                if isinstance(metric_value, (list, tuple)):
                    # Use average nếu metric là list
                    metric_value = sum(metric_value) / len(metric_value)
                composite_score += weight * metric_value
        
        return composite_score
    
    def optimize(self, n_trials: int = None) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        
        if n_trials:
            self.study_config["n_trials"] = n_trials
        
        # Create Optuna study
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            direction="maximize"
        )
        
        # Run optimization
        logger.info(f"Starting HPO with {self.study_config['n_trials']} trials...")
        study.optimize(self.objective, n_trials=self.study_config["n_trials"])
        
        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        best_trial = study.best_trial
        
        # Create results summary
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "best_trial": {
                "number": best_trial.number,
                "user_attrs": best_trial.user_attrs
            },
            "study_summary": {
                "n_trials": len(study.trials),
                "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            },
            "optimization_history": [
                {
                    "trial": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name
                }
                for trial in study.trials
            ]
        }
        
        logger.info(f"HPO completed!")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return results
```

##### **HPO Results Analysis**

```python
# training/run_bi_encoder.py - HPO results analysis
def analyze_hpo_results(self, hpo_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze HPO results và tạo insights."""
    
    analysis = {
        "best_parameters": hpo_results["best_params"],
        "performance_improvement": {},
        "parameter_importance": {},
        "recommendations": []
    }
    
    # Analyze performance improvement
    if "baseline_score" in self.config:
        baseline = self.config["baseline_score"]
        best_score = hpo_results["best_score"]
        improvement = ((best_score - baseline) / baseline) * 100
        
        analysis["performance_improvement"] = {
            "baseline_score": baseline,
            "best_score": best_score,
            "absolute_improvement": best_score - baseline,
            "relative_improvement_percent": improvement
        }
    
    # Analyze parameter importance
    if "optimization_history" in hpo_results:
        trials = hpo_results["optimization_history"]
        completed_trials = [t for t in trials if t["state"] == "COMPLETE"]
        
        if len(completed_trials) > 1:
            # Calculate parameter importance using correlation
            param_importance = self._calculate_parameter_importance(completed_trials)
            analysis["parameter_importance"] = param_importance
    
    # Generate recommendations
    analysis["recommendations"] = self._generate_hpo_recommendations(hpo_results)
    
    return analysis

def _calculate_parameter_importance(self, trials: List[Dict]) -> Dict[str, float]:
    """Tính parameter importance dựa trên correlation với performance."""
    
    param_importance = {}
    
    # Get all parameter names
    all_params = set()
    for trial in trials:
        all_params.update(trial["params"].keys())
    
    # Calculate correlation cho mỗi parameter
    for param in all_params:
        param_values = []
        scores = []
        
        for trial in trials:
            if param in trial["params"]:
                param_values.append(trial["params"][param])
                scores.append(trial["value"])
        
        if len(param_values) > 1:
            # Calculate correlation coefficient
            correlation = np.corrcoef(param_values, scores)[0, 1]
            if not np.isnan(correlation):
                param_importance[param] = abs(correlation)
    
    # Sort by importance
    param_importance = dict(sorted(
        param_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return param_importance

def _generate_hpo_recommendations(self, hpo_results: Dict[str, Any]) -> List[str]:
    """Tạo recommendations dựa trên HPO results."""
    
    recommendations = []
    
    # Performance-based recommendations
    best_score = hpo_results["best_score"]
    if best_score > 0.9:
        recommendations.append("Excellent performance achieved! Consider production deployment.")
    elif best_score > 0.8:
        recommendations.append("Good performance. Consider fine-tuning specific parameters.")
    else:
        recommendations.append("Performance below target. Review data quality and model architecture.")
    
    # Parameter-based recommendations
    best_params = hpo_results["best_params"]
    
    if best_params.get("learning_rate", 0) < 1e-5:
        recommendations.append("Low learning rate detected. Consider increasing for faster convergence.")
    
    if best_params.get("batch_size", 0) < 16:
        recommendations.append("Small batch size. Consider increasing if memory allows.")
    
    if best_params.get("epochs", 0) > 5:
        recommendations.append("High epoch count. Consider early stopping to prevent overfitting.")
    
    # Study-based recommendations
    study_summary = hpo_results["study_summary"]
    completion_rate = study_summary["n_completed"] / study_summary["n_trials"]
    
    if completion_rate < 0.8:
        recommendations.append("Low completion rate. Review trial timeout and error handling.")
    
    if study_summary["n_pruned"] > study_summary["n_completed"] * 0.5:
        recommendations.append("High pruning rate. Consider adjusting pruner parameters.")
    
    return recommendations
```

### 9.6 System Integration & Monitoring: Từ Hợp Chất Đến Hệ Thống Hoàn Chỉnh

#### 9.6.1 Workflow Orchestration: Quản Lý Toàn Bộ Pipeline

**Workflow orchestration trong LawBot quản lý toàn bộ quá trình từ data preparation đến model deployment:**

##### **Workflow Architecture**

```python
# run_workflow.py - Workflow orchestration
class WorkflowRunner:
    def __init__(self, stages: List[str], force_restart: bool = False):
        """Initialize workflow runner với stages và restart policy."""
        
        self.stages = stages
        self.force_restart = force_restart
        
        # Workflow components
        self.checkpoint_manager = WorkflowCheckpoint()
        self.progress_tracker = ProgressTracker(len(stages))
        self.device_info = self._detect_device()
        
        # Stage definitions
        self.stage_definitions = {
            "data_preparation": {
                "script": "data_processing/run_preparation.py",
                "dependencies": [],
                "timeout": 300,  # 5 minutes
                "retry_count": 2
            },
            "bi_encoder": {
                "script": "training/run_bi_encoder.py",
                "dependencies": ["data_preparation"],
                "timeout": 1800,  # 30 minutes
                "retry_count": 1
            },
            "light_ranking": {
                "script": "training/run_light_ranking.py",
                "dependencies": ["data_preparation"],
                "timeout": 1800,  # 30 minutes
                "retry_count": 1
            },
            "cross_encoder": {
                "script": "training/run_reranker.py",
                "dependencies": ["data_preparation"],
                "timeout": 3600,  # 1 hour
                "retry_count": 1
            },
            "faiss_index": {
                "script": "training/run_create_faiss_index.py",
                "dependencies": ["bi_encoder"],
                "timeout": 1800,  # 30 minutes
                "retry_count": 1
            },
            "evaluation": {
                "script": "evaluation/run_evaluation.py",
                "dependencies": ["bi_encoder", "light_ranking", "cross_encoder", "faiss_index"],
                "timeout": 600,   # 10 minutes
                "retry_count": 2
            }
        }
    
    def run_workflow(self) -> bool:
        """Execute complete workflow với stage management."""
        
        start_time = time.time()
        logger.info(f"🚀 Starting LawBot workflow with {len(self.stages)} stages")
        
        # Validate stages
        if not self.validate_stages():
            logger.error("❌ Stage validation failed")
            return False
        
        # Setup environment
        self._setup_environment()
        
        # Execute stages
        success_count = 0
        failed_stages = []
        
        for i, stage in enumerate(self.stages):
            stage_start_time = time.time()
            
            # Check dependencies
            if not self._check_dependencies(stage):
                logger.error(f"❌ Dependencies not met for stage: {stage}")
                failed_stages.append(stage)
                continue
            
            # Check checkpoint (skip if already completed)
            if not self.force_restart and self.checkpoint_manager.load_checkpoint(stage):
                logger.info(f"⏭️  Skipping {stage} (checkpoint exists)")
                success_count += 1
                continue
            
            # Start stage
            logger.info(f"🔄 Starting stage {i+1}/{len(self.stages)}: {stage}")
            self.progress_tracker.start_step(stage)
            
            # Execute stage
            stage_success = self.run_stage(stage)
            
            if stage_success:
                # Save checkpoint
                checkpoint_data = {
                    "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                    "duration_seconds": time.time() - stage_start_time,
                    "device_used": str(self.device_info.get("device", "unknown")),
                    "gpu_memory_used": self.device_info.get("gpu_memory_used_gb", 0)
                }
                
                self.checkpoint_manager.save_checkpoint(stage, "completed", checkpoint_data)
                self.progress_tracker.complete_step(stage, True)
                
                logger.info(f"✅ Stage {stage} completed successfully")
                success_count += 1
                
                # Generate stage report
                self._generate_stage_report(stage, checkpoint_data)
                
            else:
                # Stage failed
                self.progress_tracker.complete_step(stage, False)
                failed_stages.append(stage)
                
                logger.error(f"❌ Stage {stage} failed")
                
                # Generate failure report
                self._generate_failure_report(stage, time.time() - stage_start_time)
                
                # Check if we should continue
                if not self._should_continue_after_failure(stage):
                    break
        
        # Generate workflow summary
        total_time = time.time() - start_time
        self._generate_workflow_summary(success_count, failed_stages, total_time)
        
        # Return success status
        workflow_success = len(failed_stages) == 0
        logger.info(f"🎉 Workflow completed: {'SUCCESS' if workflow_success else 'FAILED'}")
        logger.info(f"📊 Results: {success_count}/{len(self.stages)} stages successful")
        
        return workflow_success
```

##### **Checkpoint Management System**

```python
# run_workflow.py - Checkpoint management
class WorkflowCheckpoint:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager."""
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint file patterns
        self.checkpoint_patterns = {
            "data_preparation": "data_preparation_checkpoint.json",
            "bi_encoder": "bi_encoder_checkpoint.json",
            "light_ranking": "light_ranking_checkpoint.json",
            "cross_encoder": "cross_encoder_checkpoint.json",
            "faiss_index": "faiss_index_checkpoint.json",
            "evaluation": "evaluation_checkpoint.json"
        }
    
    def save_checkpoint(self, stage: str, status: str, metadata: Dict[str, Any]):
        """Save checkpoint cho stage."""
        
        checkpoint_file = self.checkpoint_dir / self.checkpoint_patterns[stage]
        
        checkpoint_data = {
            "stage": stage,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "checkpoint_version": "1.0"
        }
        
        # Save checkpoint
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Checkpoint saved for {stage}: {status}")
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint cho stage."""
        
        checkpoint_file = self.checkpoint_dir / self.checkpoint_patterns[stage]
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                
                # Validate checkpoint
                if self._validate_checkpoint(checkpoint_data):
                    logger.info(f"📂 Checkpoint loaded for {stage}")
                    return checkpoint_data
                else:
                    logger.warning(f"⚠️  Invalid checkpoint for {stage}, clearing...")
                    self.clear_checkpoint(stage)
                    
            except Exception as e:
                logger.error(f"❌ Failed to load checkpoint for {stage}: {e}")
                self.clear_checkpoint(stage)
        
        return None
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Validate checkpoint data."""
        
        required_fields = ["stage", "status", "timestamp", "metadata"]
        
        # Check required fields
        for field in required_fields:
            if field not in checkpoint_data:
                return False
        
        # Check timestamp validity
        try:
            timestamp = datetime.fromisoformat(checkpoint_data["timestamp"])
            if timestamp < datetime.now() - timedelta(hours=24):  # 24 hour expiry
                return False
        except:
            return False
        
        # Check status validity
        valid_statuses = ["completed", "in_progress", "failed"]
        if checkpoint_data["status"] not in valid_statuses:
            return False
        
        return True
    
    def clear_checkpoint(self, stage: str):
        """Clear checkpoint cho stage."""
        
        checkpoint_file = self.checkpoint_dir / self.checkpoint_patterns[stage]
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"🗑️  Checkpoint cleared for {stage}")
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get status của tất cả checkpoints."""
        
        status = {}
        
        for stage, pattern in self.checkpoint_patterns.items():
            checkpoint_file = self.checkpoint_dir / pattern
            
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, "r") as f:
                        checkpoint_data = json.load(f)
                    
                    status[stage] = {
                        "exists": True,
                        "status": checkpoint_data.get("status", "unknown"),
                        "timestamp": checkpoint_data.get("timestamp", ""),
                        "metadata": checkpoint_data.get("metadata", {})
                    }
                except:
                    status[stage] = {"exists": False, "status": "corrupted"}
            else:
                status[stage] = {"exists": False, "status": "not_found"}
        
        return status
```

#### 9.6.2 Performance Monitoring: Real-time System Health

**Performance monitoring system theo dõi real-time health và performance của LawBot:**

##### **System Health Dashboard**

```python
# core/utils/system_check.py - System health monitoring
def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_info": get_device_info(),
        "model_status": get_model_status(),
        "faiss_status": get_faiss_index_status(),
        "data_status": _get_data_status(),
        "workflow_status": _get_workflow_status(),
        "performance_metrics": _get_performance_metrics(),
        "health_score": _calculate_health_score()
    }

def _get_data_status() -> Dict[str, Any]:
    """Get data status và freshness."""
    
    try:
        # Validate training data paths
        from config.paths import validate_training_data_paths
        validation_results = validate_training_data_paths()
        
        # Check data freshness
        data_freshness = _check_data_freshness()
        
        return {
            "training_data_validation": validation_results,
            "data_freshness": data_freshness,
            "data_quality_scores": _calculate_data_quality_scores(validation_results)
        }
    except Exception as e:
        return {"error": str(e)}

def _check_data_freshness() -> Dict[str, Any]:
    """Kiểm tra tính mới của data."""
    
    freshness_info = {}
    
    # Check processed data directories
    features_dir = Path("features")
    processed_dirs = [
        d for d in features_dir.iterdir()
        if d.is_dir() and d.name.startswith("processed_data_")
    ]
    
    if processed_dirs:
        # Get latest directory
        latest_dir = max(processed_dirs, key=lambda d: d.stat().st_mtime)
        latest_timestamp = datetime.fromtimestamp(latest_dir.stat().st_mtime)
        
        # Calculate age
        age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        freshness_info = {
            "latest_data_dir": latest_dir.name,
            "latest_timestamp": latest_timestamp.isoformat(),
            "age_hours": round(age_hours, 2),
            "freshness_status": "fresh" if age_hours < 24 else "stale" if age_hours < 72 else "outdated"
        }
    else:
        freshness_info = {
            "status": "no_processed_data",
            "message": "No processed data directories found"
        }
    
    return freshness_info

def _calculate_data_quality_scores(validation_results: Dict[str, Any]) -> Dict[str, float]:
    """Tính quality scores cho data."""
    
    quality_scores = {}
    
    for tier in ["tier_1", "tier_2", "tier_3"]:
        tier_data = validation_results.get(tier, {})
        
        if tier_data.get("status") == "ready":
            # Calculate quality score based on multiple factors
            file_count_score = min(tier_data.get("file_count", 0) / 10, 1.0)
            content_length_score = min(tier_data.get("avg_content_length", 0) / 100, 1.0)
            diversity_score = min(tier_data.get("unique_queries", 0) / 20, 1.0)
            
            # Weighted average
            quality_scores[tier] = (
                0.4 * file_count_score + 
                0.3 * content_length_score + 
                0.3 * diversity_score
            )
        else:
            quality_scores[tier] = 0.0
    
    return quality_scores

def _get_workflow_status() -> Dict[str, Any]:
    """Get workflow status và checkpoint information."""
    
    try:
        from run_workflow import WorkflowCheckpoint
        checkpoint_manager = WorkflowCheckpoint()
        
        return {
            "checkpoints": checkpoint_manager.get_checkpoint_status(),
            "workflow_health": _calculate_workflow_health()
        }
    except Exception as e:
        return {"error": str(e)}

def _calculate_workflow_health() -> Dict[str, Any]:
    """Tính workflow health score."""
    
    try:
        from run_workflow import WorkflowCheckpoint
        checkpoint_manager = WorkflowCheckpoint()
        
        checkpoint_status = checkpoint_manager.get_checkpoint_status()
        
        # Calculate completion rate
        total_stages = len(checkpoint_status)
        completed_stages = sum(
            1 for status in checkpoint_status.values() 
            if status.get("status") == "completed"
        )
        
        completion_rate = completed_stages / total_stages if total_stages > 0 else 0
        
        # Calculate health score
        health_score = completion_rate * 100
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "completion_rate": round(completion_rate, 3),
            "health_score": round(health_score, 1),
            "health_status": health_status,
            "total_stages": total_stages,
            "completed_stages": completed_stages
        }
    except Exception as e:
        return {"error": str(e)}

def _get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics và benchmarks."""
    
    metrics = {}
    
    try:
        # Model performance metrics
        model_status = get_model_status()
        
        for model_name, status in model_status.items():
            if status.get("exists", False):
                metrics[model_name] = {
                    "size_mb": status.get("size_mb", 0),
                    "last_modified": status.get("last_modified", 0),
                    "status": status.get("status", "unknown")
                }
        
        # FAISS index metrics
        faiss_status = get_faiss_index_status()
        if faiss_status.get("exists", False):
            metrics["faiss_index"] = {
                "document_count": faiss_status.get("document_count", 0),
                "dimensions": faiss_status.get("dimensions", 0),
                "index_size_mb": faiss_status.get("index_size_mb", 0)
            }
        
        # System performance metrics
        system_info = get_device_info()
        metrics["system"] = {
            "cpu_count": system_info.get("hardware", {}).get("cpu_count", 0),
            "memory_total_gb": system_info.get("hardware", {}).get("memory_total_gb", 0),
            "memory_available_gb": system_info.get("hardware", {}).get("memory_available_gb", 0),
            "gpu_count": system_info.get("gpu", {}).get("gpu_count", 0),
            "cuda_available": system_info.get("gpu", {}).get("cuda_available", False)
        }
        
    except Exception as e:
        metrics["error"] = str(e)
    
    return metrics

def _calculate_health_score() -> float:
    """Tính overall health score của system."""
    
    try:
        # Get component scores
        data_status = _get_data_status()
        workflow_status = _get_workflow_status()
        model_status = get_model_status()
        faiss_status = get_faiss_index_status()
        
        # Calculate component scores
        data_score = _calculate_data_health_score(data_status)
        workflow_score = _calculate_workflow_health_score(workflow_status)
        model_score = _calculate_model_health_score(model_status)
        faiss_score = _calculate_faiss_health_score(faiss_status)
        
        # Weighted average
        health_score = (
            0.3 * data_score +      # Data quality: 30%
            0.3 * workflow_score +  # Workflow health: 30%
            0.25 * model_score +    # Model status: 25%
            0.15 * faiss_score      # FAISS index: 15%
        )
        
        return round(health_score, 2)
        
    except Exception as e:
        logger.error(f"Failed to calculate health score: {e}")
        return 0.0

def _calculate_data_health_score(data_status: Dict[str, Any]) -> float:
    """Tính data health score."""
    
    try:
        validation_results = data_status.get("training_data_validation", {})
        
        if "overall" in validation_results:
            overall_status = validation_results["overall"]
            
            if overall_status.get("status") == "ready":
                return 100.0
            elif overall_status.get("status") == "missing_data":
                return 50.0
            else:
                return 25.0
        
        return 0.0
        
    except:
        return 0.0

def _calculate_workflow_health_score(workflow_status: Dict[str, Any]) -> float:
    """Tính workflow health score."""
    
    try:
        workflow_health = workflow_status.get("workflow_health", {})
        
        if "health_score" in workflow_health:
            return workflow_health["health_score"]
        
        return 0.0
        
    except:
        return 0.0

def _calculate_model_health_score(model_status: Dict[str, Any]) -> float:
    """Tính model health score."""
    
    try:
        total_models = len(model_status)
        ready_models = sum(
            1 for status in model_status.values() 
            if status.get("status") == "ready"
        )
        
        if total_models > 0:
            return (ready_models / total_models) * 100
        
        return 0.0
        
    except:
        return 0.0

def _calculate_faiss_health_score(faiss_status: Dict[str, Any]) -> float:
    """Tính FAISS index health score."""
    
    try:
        if faiss_status.get("exists", False):
            document_count = faiss_status.get("document_count", 0)
            
            if document_count > 10000:
                return 100.0
            elif document_count > 5000:
                return 75.0
            elif document_count > 1000:
                return 50.0
            else:
                return 25.0
        
        return 0.0
        
    except:
        return 0.0
```

---

### 9.7 User Experience & Interface: Từ Hệ Thống Đến Người Dùng Cuối

#### 9.7.1 Streamlit Application Architecture

**Streamlit app trong LawBot cung cấp giao diện intuitive cho end users:**

##### **App Structure**

```python
# app/app.py - Main Streamlit application
import streamlit as st
from app.pages import search, analysis, system

def main():
    """Main Streamlit application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="LawBot v8.3 - Legal AI Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("⚖️ LawBot v8.3")
    st.sidebar.markdown("**Legal AI Assistant**")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Search", "📊 Analysis", "⚙️ System", "📚 About"]
    )
    
    # Page routing
    if page == "🔍 Search":
        search.show()
    elif page == "📊 Analysis":
        analysis.show()
    elif page == "⚙️ System":
        system.show()
    elif page == "📚 About":
        show_about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version:** v8.3")
    st.sidebar.markdown("**Last Updated:** 2025-01-27")
```

##### **Search Page Implementation**

```python
# app/pages/search.py - Search interface
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_random_questions() -> List[str]:
    """Load random legal questions cho demo."""
    
    try:
        # Load từ training data
        from core.utils.io import load_jsonl
        from pathlib import Path
        
        training_data_path = Path("features/processed_data/training_data.jsonl")
        if training_data_path.exists():
            data = load_jsonl(str(training_data_path))
            questions = [item.get("query", "") for item in data if item.get("query")]
            return questions[:10]  # Return first 10 questions
        else:
            # Fallback questions
            return [
                "Luật về đất đai quy định gì?",
                "Quyền sử dụng đất được xác định như thế nào?",
                "Thuế đất đai được tính toán ra sao?",
                "Xử phạt vi phạm đất đai như thế nào?",
                "Thủ tục chuyển đổi mục đích sử dụng đất?"
            ]
    except Exception as e:
        st.error(f"Failed to load questions: {e}")
        return []

def show():
    """Show search page."""
    
    st.title("🔍 Legal Question Search")
    st.markdown("Hỏi đáp pháp lý với AI assistant")
    
    # Initialize pipeline
    pipeline = get_pipeline_lazy()
    
    if not pipeline or not pipeline.is_ready:
        st.error("❌ Pipeline not ready. Please check system status.")
        return
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_input(
            "Nhập câu hỏi pháp lý:",
            placeholder="Ví dụ: Luật về đất đai quy định gì?",
            help="Nhập câu hỏi pháp lý bằng tiếng Việt"
        )
        
        # Search parameters
        col_params1, col_params2, col_params3 = st.columns(3)
        
        with col_params1:
            top_k_final = st.slider("Số kết quả:", 1, 10, 5)
        
        with col_params2:
            similarity_threshold = st.slider("Ngưỡng tương đồng:", 0.0, 1.0, 0.7, 0.1)
        
        with col_params3:
            use_ensemble = st.checkbox("Sử dụng ensemble", value=True)
    
    with col2:
        # Random questions
        st.markdown("**💡 Câu hỏi mẫu:**")
        random_questions = load_random_questions()
        
        for i, question in enumerate(random_questions[:5]):
            if st.button(f"Q{i+1}", key=f"q{i}"):
                st.session_state.query = question
                st.rerun()
    
    # Search execution
    if query:
        if st.button("🔍 Tìm kiếm", type="primary"):
            with st.spinner("Đang tìm kiếm..."):
                try:
                    # Execute search
                    results = pipeline.predict(
                        query=query,
                        top_k_final=top_k_final,
                        similarity_threshold=similarity_threshold,
                        use_ensemble=use_ensemble
                    )
                    
                    # Display results
                    display_search_results(query, results, pipeline)
                    
                except Exception as e:
                    st.error(f"❌ Search failed: {e}")
                    st.exception(e)
```

#### 9.7.2 Analysis Page: Comprehensive Evaluation

**Analysis page cung cấp detailed insights về model performance:**

##### **Auto-evaluation System**

```python
# app/pages/analysis.py - Analysis interface
def show_auto_evaluation():
    """Show auto-evaluation interface."""
    
    st.markdown("### 🔄 Auto-evaluation System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Auto-evaluation** tự động chạy comprehensive evaluation cho toàn bộ pipeline:
        
        - **Tier 1**: Bi-Encoder retrieval performance
        - **Tier 2**: Light reranker classification accuracy
        - **Tier 3**: Cross-encoder ensemble ranking
        - **Overall**: Pipeline end-to-end performance
        """)
    
    with col2:
        if st.button("🚀 Run Auto-evaluation", type="primary"):
            run_auto_evaluation()
    
    # Latest evaluation results
    st.markdown("#### 📊 Latest Evaluation Results:")
    
    try:
        latest_results = load_latest_comprehensive_evaluation()
        if latest_results:
            display_evaluation_results(latest_results)
        else:
            st.info("ℹ️ No evaluation results found. Run auto-evaluation first.")
    except Exception as e:
        st.error(f"❌ Failed to load results: {e}")

def run_auto_evaluation():
    """Run comprehensive auto-evaluation."""
    
    with st.spinner("🔄 Running comprehensive evaluation..."):
        try:
            # Import evaluation function
            from evaluation.run_evaluation import run_comprehensive_evaluation
            
            # Run evaluation
            results = run_comprehensive_evaluation(None)
            
            if results:
                st.success("✅ Auto-evaluation completed successfully!")
                st.balloons()
                
                # Display results
                display_evaluation_results(results)
                
                # Save results
                save_evaluation_results(results)
                
            else:
                st.error("❌ Auto-evaluation failed")
                
        except Exception as e:
            st.error(f"❌ Auto-evaluation error: {e}")
            st.exception(e)
```

---

### 9.8 Best Practices & Recommendations

#### 9.8.1 Data Management
- **Always use centralized paths**: Sử dụng `config/paths.py` thay vì hardcode paths
- **Validate data before training**: Kiểm tra data quality trước mỗi training stage
- **Maintain data lineage**: Lưu trữ metadata về nguồn gốc và xử lý data
- **Regular data freshness checks**: Tự động kiểm tra tính mới của data

#### 9.8.2 Feature Management
- **Use timestamped directories**: Tạo versioned directories cho mỗi data processing run
- **Implement feature validation**: Kiểm tra tính toàn vẹn của features trước khi sử dụng
- **Maintain feature registry**: Theo dõi tất cả features và versions
- **Automated feature discovery**: Tự động tìm features mới nhất

#### 9.8.3 Model Versioning
- **Consistent naming convention**: Sử dụng format `{model_name}_{timestamp}`
- **Comprehensive metadata**: Lưu trữ đầy đủ thông tin training và performance
- **Model lineage tracking**: Theo dõi dependencies giữa models và data
- **Performance benchmarking**: So sánh performance giữa các versions

#### 9.8.4 Checkpoint Management
- **Regular checkpointing**: Lưu checkpoint sau mỗi stage completion
- **Metadata preservation**: Lưu trữ đầy đủ metadata trong checkpoints
- **Recovery procedures**: Implement procedures để resume từ checkpoints
- **Checkpoint cleanup**: Tự động cleanup old checkpoints

#### 9.8.5 Advanced Techniques
- **Hard Negative Mining**: Sử dụng adaptive threshold và diversity control
- **ADAPT Implementation**: Proper gradient reversal và domain adaptation
- **HPO Strategy**: Optuna integration với early stopping và pruning
- **Ensemble Methods**: Weighted combination với performance monitoring

#### 9.8.6 System Integration
- **Workflow Orchestration**: Stage dependency management và checkpointing
- **Performance Monitoring**: Real-time health dashboard và metrics
- **Error Handling**: Comprehensive error handling và recovery procedures
- **Logging Strategy**: Structured logging với different levels

---

### 8.4 Kết luận (Updated: 2025-01-27)

LawBot v8.3 đã đạt được những thành tựu đáng kể trong việc xây dựng hệ thống AI hỏi đáp pháp luật Việt Nam với kiến trúc 3 tầng hiện đại. Hệ thống đã tích hợp thành công các kỹ thuật machine learning tiên tiến như contrastive learning, hard negative mining, ADAPT domain adaptation, ensemble learning, và advanced hyperparameter optimization.

**Những điểm nổi bật:**
- **Kiến trúc robust**: 3-tier architecture với mỗi tầng được tối ưu hóa
- **Kỹ thuật ML hiện đại**: Sử dụng các techniques mới nhất trong NLP
- **Advanced HPO**: Hyperparameter optimization với Optuna, Bayesian search, và early stopping
- **Comprehensive Evaluation**: Multi-tier evaluation với precision, recall, F1, NDCG, MRR, quality metrics
- **Centralized Configuration**: Quản lý tập trung paths, models, và validation
- **Performance Monitoring**: Real-time performance tracking và automated optimization
- **Unified Reports Storage**: Consolidated evaluation reports trong single directory
- **MLOps practices**: Comprehensive logging, monitoring, và error handling
- **User experience**: Giao diện Streamlit intuitive và responsive
- **Performance**: Tối ưu hóa cho speed và accuracy
- **Workflow automation**: Hệ thống workflow tự động với checkpoint management

**Latest Achievements (2025-01-27):**
- **Centralized Path Management**: Tự động tìm thư mục processed data mới nhất
- **Automated Data Validation**: Tự động kiểm tra tính mới của dữ liệu training
- **Smart Workflow**: Tự động skip data_preparation nếu dữ liệu đã sẵn sàng
- **Performance Optimization**: Cải thiện hiệu suất training và inference
- **Comprehensive Evaluation**: Multi-tier evaluation với metrics đa dạng
- **Advanced HPO**: Hyperparameter optimization với Optuna và early stopping

**Hướng phát triển:**
- Tiếp tục cải thiện performance và scalability
- Thêm advanced ML techniques và architectures
- Mở rộng sang microservices architecture
- Tăng cường monitoring và observability
- Phát triển enterprise features
- Advanced HPO với multi-objective optimization
- Real-time performance monitoring và automated tuning

LawBot đã tạo nền tảng vững chắc cho việc phát triển hệ thống AI pháp luật trong tương lai, với khả năng mở rộng và thích ứng với các yêu cầu mới. Hệ thống workflow automation giờ đây hoạt động ổn định và đáng tin cậy, với comprehensive evaluation và advanced optimization capabilities.

**Source Code Integration:**
- **Core Pipeline**: `core/pipeline.py` - LegalQAPipeline với 3-tier architecture
- **Retrieval Engine**: `core/retrieval.py` - RetrievalEngine với FAISS integration
- **Reranking Engine**: `core/reranking.py` - RerankingEngine với ensemble models
- **Configuration**: `config/paths.py` - Centralized path management
- **Models**: `config/models.py` - Centralized model configuration
- **UI Components**: `app/pages/` - Streamlit pages với caching và optimization
- **Training**: `training/` - Training scripts với HPO và validation
- **Evaluation**: `evaluation/` - Comprehensive evaluation system

**Data Management & Versioning:**
- **Versioning Utilities**: `core/utils/versioning.py` - Timestamp-based versioning
- **System Monitoring**: `core/utils/system_check.py` - Model status và health checks
- **Workflow Management**: `run_workflow.py` - Checkpoint và stage management
- **Path Management**: `config/paths.py` - Centralized data paths và validation
- **Model Configuration**: `config/models.py` - Centralized model management

---

## PHỤ LỤC

### A. Cấu hình chi tiết

#### A.1 Environment Configuration
```bash
# requirements.txt - Core dependencies
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
streamlit>=1.25.0
optuna>=3.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
pydantic>=2.0.0
pyyaml>=6.0
```

#### A.2 Configuration Files
```yaml
# config/default.yml - Default configuration
training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 5
  warmup_steps: 100
  weight_decay: 0.01
  
evaluation:
  metrics: ["precision", "recall", "f1", "ndcg", "mrr"]
  top_k_values: [1, 3, 5, 10, 20, 50, 100]
  
system:
  cache_ttl: 3600
  max_workers: 4
  log_level: "INFO"
```

### B. API Reference

#### B.1 Core Pipeline API
```python
# LegalQAPipeline class
class LegalQAPipeline:
    def __init__(self, bi_encoder_path: Optional[str] = None,
                 reranker_paths: Optional[Union[str, List[str]]] = None)
    
    def predict(self, query: str, top_k_final: int = 5,
                similarity_threshold: float = 0.7,
                use_ensemble: bool = True) -> List[Dict[str, Any]]
    
    def is_ready(self) -> bool
    def get_loaded_model_versions(self) -> Dict[str, str]
```

#### B.2 Retrieval Engine API
```python
# RetrievalEngine class
class RetrievalEngine:
    def __init__(self, bi_encoder_path: str, faiss_index_path: str,
                 content_map_path: str, index_to_aid_path: str)
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]
    def get_index_info(self) -> Dict[str, Any]
```

#### B.3 Reranking Engine API
```python
# RerankingEngine class
class RerankingEngine:
    def __init__(self, model_paths: Dict[str, str])
    
    def rank_light(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]
    def rank_cross(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]
```

### C. Troubleshooting Guide

#### C.1 Common Issues & Solutions

**Issue 1: CUDA Out of Memory**
```bash
# Solution: Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python training/run_bi_encoder.py

# Or reduce batch size
python training/run_bi_encoder.py --batch-size 8
```

**Issue 2: Import Errors**
```bash
# Solution: Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python training/run_bi_encoder.py
```

**Issue 3: FAISS Index Corrupted**
```bash
# Solution: Rebuild index
rm -f features/faiss_index.bin
python training/run_create_faiss_index.py
```

#### C.2 Recovery Procedures

**Reset Failed Training:**
```bash
# Clear checkpoints
rm -f checkpoints/*_training.json

# Restart training
python training/run_light_ranking.py
```

**Restore from Backup:**
```bash
# Restore models
cp -r models/backup/* models/

# Restore features
cp -r features/backup/* features/
```

### D. Performance Benchmarks

#### D.1 Training Performance
```
Bi-Encoder Training:
- HPO optimization với Optuna
- Training time: ~20 seconds
- Enhancement: Contrastive Learning + ADAPT

Light Reranker Training:
- HPO optimization với Optuna
- Training time: ~19 seconds
- Enhancement: Independent ADAPT training

Cross-Encoder Training:
- HPO optimization với Optuna
- Training time: ~29 seconds
- Enhancement: Ensemble + ADAPT

Overall System:
- Training Speed: HPO optimization với Optuna
- Memory Management: Basic memory handling
- Config Consistency: 100% centralized configuration
- Quality Score Logic: Tier-specific thresholds với adjusted scoring
```

#### D.2 Inference Performance
```
Query Processing:
- Single query: <100ms
- Batch processing (100 queries): <5 seconds
- Memory usage: <2GB

FAISS Search:
- Index size: 17,989 documents
- Search time: <50ms per query
- Memory footprint: 53MB
```

#### D.3 Model Performance Metrics (Actual Results từ Evaluation)
```
Tier 1 (Bi-Encoder):
- Precision: 1.0
- Recall: 0.867
- F1 Score: 0.81
- NDCG: 0.992
- MRR: 1.0
- Quality Score: 0.633

Tier 2 (Light Reranker):
- Precision: 1.0
- Recall: 0.867
- F1 Score: 0.81
- NDCG: 0.990
- MRR: 1.0
- Quality Score: 1.0

Tier 3 (Cross-Encoder):
- Precision: 1.0
- Recall: 0.867
- F1 Score: 0.81
- NDCG: 0.987
- MRR: 1.0
- Quality Score: 0.6

Combined System:
- Precision: 1.0
- Recall: 0.867
- F1 Score: 0.81
- NDCG: 1.0
- MRR: 1.0
- Quality Score: 0.8

System-wide Improvements:
- Config Optimization: top_k_final=5 cho 3-5 kết quả
- Metrics Accuracy: effective_k logic đã được sửa chính xác
- Quality Score Logic: Tier-specific thresholds với adjusted scoring
```

### 8.2 Performance & Quality Improvements

#### 8.2.1 Training Performance Optimization

**Training Speed Improvements:**
- **Bi-Encoder**: HPO optimization với Optuna
- **Light Reranker**: HPO optimization với Optuna
- **Cross-Encoder**: HPO optimization với Optuna
- **Overall System**: HPO optimization với Optuna

**Memory Management:**
- **Memory Usage**: Basic memory handling với cleanup
- **Batch Processing**: Standard batch processing
- **Model Loading**: Standard model loading
- **Cache Management**: Basic caching strategy

#### 8.2.2 Quality Score & Config Optimization

**Config Optimization:**
- **top_k_final**: `5` phù hợp với yêu cầu 3-5 kết quả cuối cùng
- **K-values**: `[3, 5, 10]` phù hợp với nhu cầu thực tế
- **Config Consistency**: 100% sử dụng centralized configuration
- **Validation**: Automated config validation với schema checking

**Quality Score Logic:**
- **Tier-specific Thresholds**: Adjusted scoring cho từng tier
- **Tier 1 (Retrieval)**: Tier-specific thresholds với adjusted scoring
- **Tier 2 (Light Reranker)**: Tier-specific thresholds với adjusted scoring
- **Tier 3 (Cross-Encoder)**: Tier-specific thresholds với adjusted scoring

**Metrics Calculation:**
- **effective_k Logic**: Sửa logic để tính chính xác recall và quality scores
- **Precision Optimization**: Tier-specific precision thresholds
- **F1 Score Enhancement**: Balanced scoring với weighted metrics

### 8.3 Technical Enhancements

#### 8.3.1 Advanced HPO với Optuna Integration

**Hyperparameter Optimization:**
- **Optuna Integration**: Advanced HPO với early stopping
- **Search Space**: Optimized search space cho từng tier
- **Early Stopping**: Intelligent early stopping để tránh overfitting
- **Multi-objective**: Balance giữa accuracy và training speed

**Performance Monitoring:**
- **Basic Tracking**: Performance metrics logging
- **HPO Optimization**: Hyperparameter optimization với Optuna
- **Resource Management**: Basic resource allocation
- **Progress Logging**: Progress tracking với logging

#### 8.3.2 ADAPT Domain Adaptation Enhancement
**Domain Adaptation:**
- **Vietnamese Legal Domain**: Specialized adaptation cho pháp luật Việt Nam
- **Enhanced Training**: Improved training với domain-specific data
- **Cross-lingual Support**: Better handling của Vietnamese text
- **Legal Terminology**: Specialized vocabulary cho legal terms

**Training Improvements:**
- **Adaptive Thresholds**: Dynamic threshold adjustment
- **Domain Expertise**: Legal domain knowledge integration
- **Performance Boost**: HPO optimization với Optuna
- **Generalization**: Basic domain adaptation

#### 8.3.3 Hard Negative Mining với Adaptive Threshold

**Intelligent Mining:**
- **Adaptive Threshold**: Dynamic threshold adjustment
- **Quality Control**: Automated quality assessment
- **Negative Selection**: Intelligent negative example selection
- **Training Stability**: Improved training stability

**Implementation Details:**
```python
class AdaptiveHardNegativeMining:
    def __init__(self, initial_threshold=0.5):
        self.threshold = initial_threshold
        self.adaptation_rate = 0.1
        
    def update_threshold(self, current_performance):
        """Adaptive threshold update based on performance"""
        if current_performance > 0.8:
            self.threshold += self.adaptation_rate
        elif current_performance < 0.6:
            self.threshold -= self.adaptation_rate
        
        self.threshold = max(0.1, min(0.9, self.threshold))
        return self.threshold
    
    def mine_hard_negatives(self, candidates, positive_score):
        """Mine hard negatives with adaptive threshold"""
        hard_negatives = []
        for candidate in candidates:
            if candidate['score'] > positive_score - self.threshold:
                hard_negatives.append(candidate)
        return hard_negatives
```

#### 8.3.4 Ensemble Learning với Weighted Combination

**Ensemble Strategy:**
- **Model Combination**: Weighted combination của multiple models
- **Performance Optimization**: 70% ADAPT-enhanced + 30% Base model
- **Score Aggregation**: Intelligent score combination
- **Confidence Scoring**: Confidence-based weighting

**Implementation Benefits:**
- **Accuracy Improvement**: HPO optimization với Optuna
- **Robustness**: Basic ensemble approach
- **Performance**: Balanced performance với weighted combination
- **Scalability**: Basic ensemble architecture

### F. Data Management & Versioning Reference
- **Section 9**: Data, Feature & Model Versioning Management
- **Core Files**: `config/paths.py`, `config/models.py`, `core/utils/versioning.py`
- **Workflow**: `run_workflow.py` với checkpoint management
- **Monitoring**: `core/utils/system_check.py` với system health dashboard

---

## 9. DOCUMENTATION ARCHITECTURE & TECHNICAL GUIDES

### 9.1 Comprehensive Documentation Structure

**LawBot** đã được xây dựng với hệ thống tài liệu kỹ thuật toàn diện, bao gồm các guide chuyên biệt cho từng thành phần:

#### **📚 Core Technical Documentation:**

1. **`LAWBOT_MLOPS_TECHNICAL_GUIDE.md`** - Hướng dẫn kỹ thuật MLOPs
   - **FAISS Index Management**: Vector search optimization với mathematical foundations
   - **HPO (Hyperparameter Optimization)**: Optuna-based optimization với statistical analysis
   - **HNM (Hard Negative Mining)**: Training data quality improvement với adaptive thresholds
   - **Training Pipeline**: Unified training engine với error handling và recovery
   - **Software Architecture Patterns**: Template Method, Strategy, Factory, Observer, Singleton, Command
   - **Advanced Code Techniques**: Context Manager, Decorator, Chain of Responsibility, Builder

2. **`LAWBOT_EVALUATION_GUIDE.md`** - Hướng dẫn đánh giá toàn diện
   - **3-Tier Evaluation Architecture**: Bi-Encoder, Light Reranker, Cross-Encoder
   - **Mathematical Metrics**: Precision@K, Recall@K, F1@K, MRR@K, NDCG@K với công thức chi tiết
   - **Quality Score Logic**: Tier-specific thresholds với adjusted scoring
   - **Performance Analysis**: Real-time monitoring và automated optimization
   - **Comprehensive Evaluation**: Multi-tier evaluation với unified reporting

3. **`LAWBOT_UI_TECHNICAL_GUIDE.md`** - Hướng dẫn kiến trúc UI
   - **Streamlit Architecture**: Modular page system với custom navigation
   - **UI Components**: Custom components và responsive design
   - **State Management**: Session state với caching strategies
   - **Performance Optimization**: Lazy loading, memory management, caching
   - **User Experience**: Interactive elements và visual feedback

4. **`LAWBOT_MODELS_TRAINING_GUIDE.md`** - Hướng dẫn training models
   - **Training Pipeline**: Automated workflow với centralized configuration
   - **Model Architecture**: PhoBERT, Vietnamese Bi-Encoder, Cross-Encoder
   - **Training Strategies**: Domain adaptation, ensemble learning, checkpoint management
   - **Performance Monitoring**: Real-time metrics và automated optimization

5. **`DATASET_MANAGEMENT_GUIDE.md`** - Hướng dẫn quản lý dữ liệu
   - **Data Processing**: Automated pipeline với validation và quality control
   - **Data Versioning**: Version control cho datasets và processed data
   - **Data Validation**: Quality checks và automated cleaning
   - **Storage Optimization**: Efficient storage và retrieval strategies

6. **`USER_REQUEST_PROCESSING_FLOW.md`** - Luồng xử lý request
   - **Request Flow**: End-to-end processing từ user input đến response
   - **Pipeline Integration**: 3-tier architecture với caching và optimization
   - **Error Handling**: Robust error handling và recovery mechanisms
   - **Performance Monitoring**: Real-time performance tracking

7. **`QUICK_START.md`** - Hướng dẫn khởi động nhanh
   - **Setup Instructions**: Step-by-step setup và configuration
   - **Basic Usage**: Quick examples và common use cases
   - **Troubleshooting**: Common issues và solutions
   - **Best Practices**: Recommended workflows và optimization tips

### 9.2 Technical Implementation Details

#### **🏗️ Architecture Patterns Implemented:**

```python
# Design Patterns trong LawBot
design_patterns = {
    "Template Method": "BaseTrainingScript.run() - Training workflow skeleton",
    "Strategy": "Model loading strategies (auto-discovery vs manual)",
    "Factory": "RerankingEngine.load_model() - Model creation",
    "Observer": "Logging system với multiple handlers",
    "Singleton": "ConfigLoader - Single configuration instance",
    "Command": "TrainingEngine - Command-based training operations",
    "Context Manager": "GPUMemoryManager - Resource management",
    "Decorator": "Performance monitoring decorators",
    "Chain of Responsibility": "Error handling chain",
    "Builder": "ModelConfigBuilder - Configuration building"
}
```

#### **🔧 Advanced Code Techniques:**

```python
# Advanced Techniques Implementation
advanced_techniques = {
    "Lazy Loading": "Pipeline loading only when needed",
    "Memory Management": "GPU memory optimization và cleanup",
    "Caching Strategies": "Multi-level caching với TTL optimization",
    "Error Recovery": "Automatic recovery mechanisms",
    "Performance Monitoring": "Real-time metrics và optimization",
    "Batch Processing": "Efficient batch operations cho large datasets",
    "Async Processing": "Non-blocking operations cho better UX"
}
```

#### **📊 Mathematical Foundations:**

```python
# Mathematical Formulas Implemented
mathematical_formulas = {
    "Cosine Similarity": "cos(θ) = (A·B) / (||A|| × ||B||)",
    "Precision@K": "P@K = |Relevant ∩ Retrieved[:K]| / |Retrieved[:K]|",
    "Recall@K": "R@K = |Relevant ∩ Retrieved[:K]| / |Relevant|",
    "F1@K": "F1@K = 2 × (P@K × R@K) / (P@K + R@K)",
    "MRR@K": "MRR@K = (1/|Relevant|) × Σ(1/rank_i)",
    "NDCG@K": "NDCG@K = DCG@K / IDCG@K",
    "Quality Score": "Q = α × S + β × C + γ × R"
}
```

### 9.3 Mathematical Foundations & Algorithms

#### **🧮 Core Mathematical Concepts:**

1. **Vector Similarity & Search:**
   - **Cosine Similarity**: `cos(θ) = (A·B) / (||A|| × ||B||)`
   - **L2 Normalization**: `||v||₂ = √(v₁² + v₂² + ... + vₙ²)`
   - **Dot Product**: `A·B = Σ(aᵢ × bᵢ)`

2. **Information Retrieval Metrics:**
   - **Precision@K**: `P@K = |Relevant ∩ Retrieved[:K]| / |Retrieved[:K]|`
   - **Recall@K**: `R@K = |Relevant ∩ Retrieved[:K]| / |Relevant|`
   - **F1@K**: `F1@K = 2 × (P@K × R@K) / (P@K + R@K)`
   - **MRR@K**: `MRR@K = (1/|Relevant|) × Σ(1/rank_i)`
   - **NDCG@K**: `NDCG@K = DCG@K / IDCG@K` với `DCG@K = Σ(relevance_i / log₂(i + 1))`

3. **Quality Scoring Algorithms:**
   - **Tier-specific Thresholds**: Adjusted scoring cho từng tier
   - **Relevance Formula**: `R = α × S + β × Q` với α, β là weights
   - **Performance Indicators**: Statistical significance testing với t-test

4. **Optimization Algorithms:**
   - **HPO with Optuna**: Bayesian optimization với TPE sampler
   - **Early Stopping**: Convergence criteria với patience và min_delta
   - **Parameter Validation**: Mathematical constraints cho hyperparameters

#### **📈 Performance Analysis Formulas:**

```python
# Performance Metrics Calculation
performance_formulas = {
    "Cache Hit Rate": "HR = Hits / Total_Requests",
    "Cache Efficiency": "CE = HR × (1 - Cache_Size/Max_Cache_Size)",
    "Memory Efficiency": "ME = (Cache_Size / Max_Cache_Size) × HR",
    "Optimization AUC": "AUC = Σ(yi × Δxi) where Δxi = xi+1 - xi",
    "Statistical Significance": "t = (x̄ - μ₀) / (s/√n) với confidence level 95%"
}
```

#### **🎯 Algorithm Complexity Analysis:**

```python
# Time & Space Complexity
complexity_analysis = {
    "FAISS Search": "O(log n) với approximate search, O(n) với exact search",
    "Embedding Generation": "O(n × d) với n documents, d dimensions",
    "HPO Optimization": "O(t × e × b) với t trials, e epochs, b batch_size",
    "Batch Processing": "O(n/b) với n total items, b batch_size",
    "Caching Operations": "O(1) average case, O(n) worst case"
}
```

---

## 10. PHÂN TÍCH ƯU NHƯỢC ĐIỂM & HẠN CHẾ HỆ THỐNG

### **✅ Ưu điểm chính:**

1. **Kiến trúc 3-Tầng hiện đại với Comprehensive Enhancement:**

   **🚀 Tier 1 (Bi-Encoder) - Retrieval Engine:**
   - **ADAPT Enhancement**: Domain adaptation cho legal expertise
   - **HNM Enhancement**: Hard negative mining cho improved training data quality
   - **HPO Enhancement**: Hyperparameter optimization với Optuna
   - **FAISS Optimization**: Vector search optimization cho high-speed retrieval

   **⚡ Tier 2 (Light Reranker) - Filtering Engine:**
   - **ADAPT Enhancement**: Independent domain adaptation training
   - **HNM Enhancement**: Intelligent negative selection và enrichment
   - **HPO Enhancement**: Automated hyperparameter tuning
   - **Performance Optimization**: Fast filtering với enhanced accuracy

   **🎯 Tier 3 (Cross-Encoder Ensemble) - Final Ranking:**
   - **ADAPT Enhancement**: Dual model domain adaptation (PhoBERT-base-v2 + PhoBERT-large)
   - **HNM Enhancement**: Hard negative mining cho improved training data quality
   - **HPO Enhancement**: Hyperparameter optimization cho optimal ensemble performance
   - **Ensemble Strategy**: Weighted combination (70% base + 30% large) với confidence calibration

2. **MLOPs techniques tiên tiến:**
   - **FAISS Vector Search**: Approximate nearest neighbor với optimization
   - **HPO với Optuna**: Bayesian optimization cho hyperparameters
   - **HNM (Hard Negative Mining)**: Adaptive threshold với intelligent mining
   - **ADAPT Enhancement**: Domain adaptation cho pháp luật Việt Nam

3. **Performance optimization:**
   - **Caching Strategy**: Multi-level caching (Streamlit + file-based)
   - **Batch Processing**: Memory-efficient processing cho large datasets
   - **GPU Acceleration**: CUDA support với memory management

4. **Comprehensive Evaluation:**
   - **Multi-tier Metrics**: Precision@K, Recall@K, F1@K, MRR@K, NDCG@K
   - **Quality Scoring**: Tier-specific thresholds với adjusted scoring
   - **Statistical Analysis**: T-test significance testing

### **⚠️ Nhược điểm và hạn chế:**

1. **Model Limitations:**
   - **PhoBERT Architecture**: Chỉ hỗ trợ tiếng Việt, không đa ngôn ngữ
   - **Context Length**: Giới hạn 512 tokens có thể ảnh hưởng đến long documents
   - **Training Data Dependency**: Phụ thuộc vào chất lượng legal corpus

2. **Performance Constraints:**
   - **Memory Usage**: PhoBERT-large (30%) có thể gây memory pressure
   - **Inference Latency**: Ensemble processing tăng thời gian response
   - **Scalability**: Single-node architecture, không có distributed processing

3. **Data Quality Issues:**
   - **Legal Corpus Coverage**: Có thể thiếu một số văn bản pháp luật mới
   - **Annotation Quality**: Training data quality phụ thuộc vào manual annotation
   - **Domain Specificity**: Chỉ specialized cho pháp luật Việt Nam

4. **Technical Debt:**
   - **Code Complexity**: Ensemble logic có thể khó maintain
   - **Configuration Management**: Multiple config files có thể gây confusion
   - **Error Handling**: Limited error recovery mechanisms

### **🛠️ Giải pháp cụ thể:**

1. **Model Optimization:**
   ```python
   # Giải pháp cho memory usage
   def optimize_memory_usage():
       # Gradient checkpointing cho PhoBERT-large
       model.gradient_checkpointing_enable()
       
       # Mixed precision training
       scaler = GradScaler()
       
       # Dynamic batch sizing
       batch_size = calculate_optimal_batch_size(available_memory)
   ```

2. **Performance Improvement:**
   ```python
   # Giải pháp cho inference latency
   def optimize_inference():
       # Model quantization
       quantized_model = torch.quantization.quantize_dynamic(model)
       
       # Batch inference với optimal size
       optimal_batch_size = find_optimal_batch_size()
       
       # Async processing
       results = await process_batch_async(queries, optimal_batch_size)
   ```

3. **Data Quality Enhancement:**
   ```python
   # Giải pháp cho data quality
   def enhance_data_quality():
       # Automated data validation
       validation_rules = create_legal_data_validation_rules()
       
       # Data augmentation techniques
       augmented_data = apply_legal_specific_augmentation(original_data)
       
       # Continuous data monitoring
       data_quality_metrics = monitor_data_quality_continuously()
   ```

4. **Architecture Improvement:**
   ```python
   # Giải pháp cho scalability
   def improve_scalability():
       # Microservices architecture
       services = split_into_microservices(pipeline)
       
       # Load balancing
       load_balancer = implement_round_robin_balancing()
       
       # Distributed processing
       distributed_pipeline = implement_distributed_processing()
   ```

### **📊 Đánh giá tổng thể:**

- **Strengths Score**: 9.0/10 (Kiến trúc hiện đại, MLOPs tiên tiến, UI optimization)
- **Weaknesses Score**: 4.5/10 (Đã giải quyết major performance issues)
- **Improvement Potential**: 7.5/10 (Còn room cho advanced features)
- **Production Readiness**: 8.5/10 (Đã sẵn sàng cho production với current optimizations)

---

## 🎯 **TỔNG KẾT DOCUMENTATION ARCHITECTURE**

### **✅ Đã Hoàn Thiện:**

1. **Comprehensive Coverage**: Tất cả thành phần chính đều có documentation chi tiết
2. **Mathematical Foundation**: Công thức toán học và thuật toán được giải thích đầy đủ
3. **Code Examples**: Ví dụ thực tế từ source code với implementation details
4. **Architecture Patterns**: Design patterns và best practices được document
5. **Performance Analysis**: Metrics và optimization strategies được phân tích
6. **User Experience**: UI/UX guidelines và responsive design principles
7. **Operational Guidelines**: Setup, deployment, và troubleshooting guides

### **🚀 Benefits của Documentation System:**

- **Developer Onboarding**: New developers có thể hiểu system nhanh chóng
- **Knowledge Transfer**: Technical knowledge được preserve và share
- **Maintenance & Updates**: Dễ dàng maintain và update system
- **Quality Assurance**: Consistent implementation và best practices
- **Performance Optimization**: Mathematical insights cho optimization
- **Troubleshooting**: Comprehensive guides cho problem solving
- **Scalability**: Architecture patterns cho future expansion

### **📋 Documentation Standards:**

- **Consistency**: Tất cả docs follow cùng format và structure
- **Accuracy**: Bám sát source code và actual implementation
- **Completeness**: Cover tất cả aspects từ high-level đến low-level
- **Maintainability**: Dễ dàng update khi system thay đổi
- **Accessibility**: Clear language và visual aids cho better understanding

---

**Tài liệu này được tạo bởi LawBot Development Team**
**Phiên bản: v8.3 | Ngày cập nhật: 2025-01-21 (UI Optimization & Safe Device Handling)**
**Liên hệ: dev-team@lawbot.com**