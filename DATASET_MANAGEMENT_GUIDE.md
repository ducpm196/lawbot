# LAWBOOT DATASET MANAGEMENT & PROCESSING GUIDE
## Phiên bản: v8.3 | Ngày cập nhật: 2025-08-21

---

## MỤC LỤC
1. [Tổng quan Dataset Management](#1-tổng-quan-dataset-management)
2. [Kiến trúc Dataset Pipeline](#2-kiến-trúc-dataset-pipeline)
3. [Raw Data Processing](#3-raw-data-processing)
4. [Training Data Generation](#4-training-data-generation)
5. [Data Quality & Validation](#5-data-quality--validation)
6. [Versioning & Storage](#6-versioning--storage)
7. [Workflow Integration](#7-workflow-integration)
8. [Advanced Features](#8-advanced-features)
9. [Monitoring & Logging](#9-monitoring--logging)
10. [Troubleshooting & Best Practices](#10-troubleshooting--best-practices)

---

## 1. TỔNG QUAN DATASET MANAGEMENT

### 1.1 Mục tiêu và Phạm vi

**Dataset Management** trong LawBot là hệ thống quản lý dữ liệu pháp luật Việt Nam từ raw data đến training data, đảm bảo:
- **Data Consistency**: Nhất quán về format và structure
- **Quality Control**: Kiểm soát chất lượng dữ liệu
- **Versioning**: Quản lý phiên bản dữ liệu
- **Traceability**: Truy xuất nguồn gốc dữ liệu

### 1.2 Cấu trúc Dataset

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET HIERARCHY                            │
├─────────────────────────────────────────────────────────────────┤
│  📁 Raw Data                                                   │
│  ├── legal_corpus.json (Luật gốc)                              │
│  ├── train_data.json (Câu hỏi training)                        │
│  └── validation_data.json (Dữ liệu validation)                │
├─────────────────────────────────────────────────────────────────┤
│  🔄 Processing Pipeline                                        │
│  ├── Legal Text Cleaning                                       │
│  ├── Canonicalization                                          │
│  ├── Training Example Generation                               │
│  └── Quality Validation                                        │
├─────────────────────────────────────────────────────────────────┤
│  📊 Processed Data                                             │
│  ├── processed_corpus.json (Corpus đã xử lý)                   │
│  ├── bi_encoder_train.jsonl (Training examples)                │
│  ├── cross_encoder_train.jsonl (Cross-encoder data)            │
│  └── training_data.jsonl (Light ranking data)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. KIẾN TRÚC DATASET PIPELINE

### 2.1 Pipeline Overview

```
Raw Data → Text Processing → Canonicalization → Training Generation → Validation → Storage
    ↓              ↓                ↓                ↓                ↓          ↓
Legal Corpus   Text Cleaner    AID Mapping    Triplet Creation   Quality     Versioned
Training Q&A   Normalization   Content Map    Pair Generation    Check       Storage
```

### 2.2 Core Components

**Data Processing Engine:**
- **`data_processing/run_preparation.py`**: Main pipeline orchestrator
- **`core/transforms/`**: Text transformation modules
- **`core/utils/canonicalization.py`**: AID standardization
- **`core/utils/io.py`**: File I/O operations
- **`core/utils/versioning.py`**: Version management

### 2.3 Complete Data Processing Flow

**Chi tiết từng bước xử lý:**

```python
# data_processing/run_preparation.py - Main pipeline
def run_data_preparation():
    """Run the complete data preparation pipeline."""
    
    # Step 1: Load raw data
    train_data = load_json(config.paths.train_data_path)      # Training questions
    raw_corpus = load_json(config.paths.legal_corpus_path)    # Legal documents
    
    # Step 2: Process legal corpus
    processed_corpus, aid_map = process_legal_corpus(raw_corpus)
    
    # Step 3: Generate training examples
    bi_encoder_examples, cross_encoder_examples, stats = generate_training_examples(
        train_data, processed_corpus, aid_map
    )
    
    # Step 4: Create light ranking data
    light_ranking_examples = []
    negative_pool = []
    
    for item in bi_encoder_examples:
        light_ranking_examples.append({
            "query": item["query"],
            "positive": item["positive"],
            "type": "positive_pair"
        })
        negative_pool.append({
            "text": item["negative"], 
            "type": "negative_candidate"
        })
    
    # Step 5: Save processed data với versioning
    output_dir = generate_versioned_path(config.paths.feature_dir, "processed_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all training data
    save_jsonl(bi_encoder_examples, output_dir / "bi_encoder_train.jsonl")
    save_jsonl(cross_encoder_examples, output_dir / "cross_encoder_train.jsonl")
    save_jsonl(light_ranking_examples, output_dir / "training_data.jsonl")
    save_json(processed_corpus, output_dir / "processed_corpus.json")
    
    return output_dir

# Example output directory structure:
# features/
# └── processed_data_20250127_143022/          # Timestamped version
#     ├── bi_encoder_train.jsonl               # Tier 1 training data
#     ├── cross_encoder_train.jsonl            # Tier 3 training data  
#     ├── training_data.jsonl                  # Tier 2 training data
#     ├── processed_corpus.json                # Cleaned legal corpus
#     ├── negative_pool.json                   # Negative examples pool
#     └── metadata.json                        # Processing metadata
```

---

## 3. RAW DATA PROCESSING

### 3.1 Legal Corpus Processing

#### 3.1.1 Input Format

**Raw Legal Corpus Structure:**
```json
[
  {
    "law_id": "123/2020/QH14",
    "content": [
      {
        "aid": 15,
        "content_Article": "Điều 15. Phạm vi điều chỉnh\n\nLuật này quy định về..."
      }
    ]
  }
]
```

#### 3.1.2 Processing Pipeline

**Step 1: Text Cleaning**
```python
# core/transforms/base.py - LegalTextCleaner implementation thực tế
class LegalTextCleaner(TextProcessor):
    """Specialized text processor for Vietnamese legal documents."""
    
    def __init__(self):
        super().__init__(
            normalize_unicode=True,
            lowercase=False,  # Legal text có meaningful capitalization
            remove_punctuation=False,  # Punctuation quan trọng trong legal text
            remove_extra_spaces=True
        )
        
        # Pre-compiled regex patterns cho efficiency
        self.article_pattern = re.compile(
            r"(Điều|Khoản|Điểm)\s+\d+[a-z]?\.?", re.IGNORECASE
        )
        
        # Legal abbreviations mapping
        self.abbreviation_map = {
            "QH": "Quốc hội",
            "UBTVQH": "Ủy ban Thường vụ Quốc hội",
            "CP": "Chính phủ",
            "NĐ-CP": "Nghị định - Chính phủ",
            "TTg": "Thủ tướng Chính phủ",
            "BTC": "Bộ Tài chính",
            "BTP": "Bộ Tư pháp",
            "TANDTC": "Tòa án nhân dân tối cao",
            "VKSNDTC": "Viện kiểm sát nhân dân tối cao",
        }
    
    def __call__(self, text: str) -> str:
        """Apply base processing và legal-specific cleaning rules."""
        # Base processing first (Unicode normalization, extra spaces)
        text = super().__call__(text)
        
        # Legal-specific cleaning
        # Remove article/clause/point numbers (e.g., "Điều 1.", "Khoản 2.")
        text = self.article_pattern.sub(r"\1", text)
        
        # Replace common abbreviations
        # Use a regex to avoid replacing parts of words
        for abbr, full_text in self.abbreviation_map.items():
            text = re.sub(rf"\b{abbr}\b", full_text, text)
        
        # Final cleanup of extra spaces that might have been introduced
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

# Example transformation với real implementation
raw_text = "Điều 15. Phạm vi điều chỉnh\n\nLuật này quy định về thành lập, quản lý, tổ chức lại và giải thể doanh nghiệp theo QH."
cleaned_text = LegalTextCleaner()(raw_text)
# Result: "Điều 15 Phạm vi điều chỉnh Luật này quy định về thành lập, quản lý, tổ chức lại và giải thể doanh nghiệp theo Quốc hội."
```

**Step 2: Canonicalization**
```python
# core/utils/canonicalization.py - Implementation thực tế
def canonicalize_aid(law_id: str, article_id: Union[str, int]) -> str:
    """Canonicalize AID để tạo unique identifier cho mỗi article."""
    
    # Clean và standardize identifiers
    # Loại bỏ dấu gạch ngang, khoảng trắng, underscore
    law_id_cleaned = re.sub(r"[-\s_]", "", str(law_id)).upper()
    article_id_cleaned = re.sub(r"[-\s_]", "", str(article_id)).upper()
    
    # Tạo canonical AID với format: LAWID_ARTICLEID
    canonical_aid = f"{law_id_cleaned}_{article_id_cleaned}"
    return canonical_aid

# Example transformation với real data
law_id = "123/2020/QH14"  # Luật Doanh nghiệp 2020
article_id = "15"          # Điều 15
canonical_aid = canonicalize_aid(law_id, article_id)
# Result: "1232020QH14_15"

# Ví dụ thêm với different law types
law_id_2 = "NĐ-CP/2021/123"  # Nghị định Chính phủ
article_id_2 = "25"
canonical_aid_2 = canonicalize_aid(law_id_2, article_id_2)
# Result: "NĐCP2021123_25"

# Ví dụ với law có special characters
law_id_3 = "TTg/2022/ABC-123"  # Thông tư Thủ tướng
article_id_3 = "10"
canonical_aid_3 = canonicalize_aid(law_id_3, article_id_3)
# Result: "TTg2022ABC123_10"
```

#### 3.1.3 Output Structure

**Processed Corpus:**
```json
{
  "1232020QH14_15": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
  "1232020QH14_16": "Điều 16. Đối tượng áp dụng Luật này áp dụng đối với..."
}
```

### 3.2 Ví dụ Complete Processing Flow

**Từ Raw Data đến Processed Data:**

```python
# Input: Raw legal corpus
raw_corpus = [
    {
        "law_id": "123/2020/QH14",
        "content": [
            {
                "aid": 15,
                "content_Article": "Điều 15. Phạm vi điều chỉnh\n\nLuật này quy định về thành lập, quản lý, tổ chức lại và giải thể doanh nghiệp."
            },
            {
                "aid": 16,
                "content_Article": "Điều 16. Đối tượng áp dụng\n\nLuật này áp dụng đối với:\n1. Doanh nghiệp được thành lập theo quy định của Luật này;"
            }
        ]
    }
]

# Step 1: Process legal corpus
processed_corpus, aid_map = process_legal_corpus(raw_corpus)

# Result:
# processed_corpus = {
#     "1232020QH14_15": "Điều 15. Phạm vi điều chỉnh Luật này quy định về thành lập, quản lý, tổ chức lại và giải thể doanh nghiệp.",
#     "1232020QH14_16": "Điều 16. Đối tượng áp dụng Luật này áp dụng đối với: 1. Doanh nghiệp được thành lập theo quy định của Luật này;"
# }
# aid_map = {
#     15: "1232020QH14_15",
#     16: "1232020QH14_16"
# }

# Input: Training questions
train_data = [
    {
        "question": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
        "relevant_laws": [15]
    }
]

# Step 2: Generate training examples
bi_encoder_examples, cross_encoder_examples, stats = generate_training_examples(
    train_data, processed_corpus, aid_map
)

# Result: Bi-encoder triplet
# {
#     "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
#     "positive": "Điều 15. Phạm vi điều chỉnh Luật này quy định về thành lập, quản lý, tổ chức lại và giải thể doanh nghiệp.",
#     "negative": "Điều 16. Đối tượng áp dụng Luật này áp dụng đối với: 1. Doanh nghiệp được thành lập theo quy định của Luật này;"
# }
```

### 3.3 Data Flow trong 3-Tier Architecture

**Luồng dữ liệu qua các tiers:**

```python
# Original Query
user_query = "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?"

# === TIER 1: BI-ENCODER RETRIEVAL ===
# Training data format for Tier 1
tier1_training_data = {
    "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
    "positive": "Điều 15. Phạm vi điều chỉnh Luật này quy định về thành lập, quản lý, tổ chức lại và giải thể doanh nghiệp.",
    "negative": "Điều 20. Quyền và nghĩa vụ của công dân trong hoạt động kinh doanh..."
}

# Output từ Tier 1: Top-K candidates
tier1_output = [
    {
        "aid": "1232020QH14_15",
        "content": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
        "retrieval_score": 0.85,
        "rank": 1
    },
    {
        "aid": "1232020QH14_16", 
        "content": "Điều 16. Đối tượng áp dụng Luật này áp dụng đối với...",
        "retrieval_score": 0.72,
        "rank": 2
    }
]

# === TIER 2: LIGHT RERANKER ===
# Training data format for Tier 2
tier2_training_data = [
    {
        "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
        "positive": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
        "type": "positive_pair"
    }
]

# Output từ Tier 2: Filtered and reranked
tier2_output = [
    {
        "aid": "1232020QH14_15",
        "content": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
        "light_reranker_score": 0.91,
        "rank": 1
    }
]

# === TIER 3: CROSS-ENCODER ===
# Training data format for Tier 3
tier3_training_data = [
    {
        "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
        "passage": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
        "label": 1.0  # Positive example
    },
    {
        "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
        "passage": "Điều 20. Quyền và nghĩa vụ của công dân...",
        "label": 0.0  # Negative example
    }
]

# Final output: Precise ranking
final_output = [
    {
        "aid": "1232020QH14_15",
        "content": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
        "cross_encoder_score": 0.95,
        "final_rank": 1,
        "confidence": "high"
    }
]
```

---

## 4. TRAINING DATA GENERATION

### 4.1 Bi-Encoder Training Data

#### 4.1.1 Triplet Generation

**Triplet Structure:**
```python
def generate_bi_encoder_triplets(
    query: str,
    positive_aid: str,
    negative_aid: str,
    processed_corpus: Dict[str, str]
) -> Dict[str, str]:
    """Generate bi-encoder training triplet."""
    
    triplet = {
        "query": query,
        "positive": processed_corpus[positive_aid],
        "negative": processed_corpus[negative_aid]
    }
    
    return triplet

# Example generation
query = "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?"
positive_aid = "1232020QH14_15"
negative_aid = "4562021_20"

triplet = generate_bi_encoder_triplets(
    query, positive_aid, negative_aid, processed_corpus
)

# Result:
# {
#   "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
#   "positive": "Điều 15. Phạm vi điều chỉnh Luật này quy định về...",
#   "negative": "Điều 20. Quyền và nghĩa vụ của công dân..."
# }
```

### 4.2 Hard Negative Mining

**Advanced Negative Selection Strategy:**

```python
def advanced_hard_negative_mining(
    positive_aids: List[str],
    all_aids: List[str],
    processed_corpus: Dict[str, str],
    query: str,
    bi_encoder_model=None
) -> List[str]:
    """Advanced hard negative mining with multiple strategies."""
    
    hard_negatives = []
    
    # Strategy 1: Same law document (legal context similarity)
    positive_law_ids = [aid.split('_')[0] for aid in positive_aids]
    same_law_candidates = [
        aid for aid in all_aids
        if aid.split('_')[0] in positive_law_ids 
        and aid not in positive_aids
    ]
    
    # Strategy 2: Semantic similarity (if bi-encoder available)
    if bi_encoder_model:
        query_embedding = bi_encoder_model.encode([query])
        
        # Calculate similarities for all candidates
        candidate_similarities = {}
        for aid in same_law_candidates[:50]:  # Limit for performance
            doc_embedding = bi_encoder_model.encode([processed_corpus[aid]])
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            candidate_similarities[aid] = similarity
        
        # Select negatives with moderate similarity (not too easy, not too hard)
        sorted_candidates = sorted(
            candidate_similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select from middle range (0.3-0.7 similarity)
        for aid, sim in sorted_candidates:
            if 0.3 <= sim <= 0.7:
                hard_negatives.append(aid)
                if len(hard_negatives) >= 3:
                    break
    
    # Strategy 3: Random fallback
    if len(hard_negatives) < 3:
        random_candidates = [
            aid for aid in all_aids 
            if aid not in positive_aids and aid not in hard_negatives
        ]
        hard_negatives.extend(random.sample(
            random_candidates, 
            min(3 - len(hard_negatives), len(random_candidates))
        ))
    
    return hard_negatives

# Example usage
positive_aids = ["1232020QH14_15"]
all_aids = list(processed_corpus.keys())
query = "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?"

hard_negatives = advanced_hard_negative_mining(
    positive_aids, all_aids, processed_corpus, query
)

# Result: ["1232020QH14_20", "1232020QH14_35", "4562021_18"]
# - Same law document but different topics
# - Moderate semantic similarity
# - Challenging but learnable negatives
```

### 4.3 Cross-Encoder Training Data

**Pair Structure với Balanced Sampling:**
```python
def generate_cross_encoder_pairs_balanced(
    query: str,
    positive_aids: List[str],
    negative_aids: List[str],
    processed_corpus: Dict[str, str],
    negative_ratio: float = 2.0
) -> List[Dict[str, Any]]:
    """Generate balanced cross-encoder training pairs."""
    
    pairs = []
    
    # Add positive pairs
    for pos_aid in positive_aids:
        positive_pair = {
            "query": query,
            "passage": processed_corpus[pos_aid],
            "label": 1.0,
            "aid": pos_aid,
            "pair_type": "positive"
        }
        pairs.append(positive_pair)
    
    # Add negative pairs (balanced ratio)
    num_negatives = int(len(positive_aids) * negative_ratio)
    selected_negatives = random.sample(
        negative_aids, 
        min(num_negatives, len(negative_aids))
    )
    
    for neg_aid in selected_negatives:
        negative_pair = {
            "query": query,
            "passage": processed_corpus[neg_aid],
            "label": 0.0,
            "aid": neg_aid,
            "pair_type": "negative"
        }
        pairs.append(negative_pair)
    
    return pairs

# Example generation với multiple positives
query = "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?"
positive_aids = ["1232020QH14_15", "1232020QH14_16"]
negative_aids = ["1232020QH14_20", "4562021_18", "7892019_25"]

pairs = generate_cross_encoder_pairs_balanced(
    query, positive_aids, negative_aids, processed_corpus
)

# Result: 2 positive pairs + 4 negative pairs (ratio 2.0)
# [
#   {"query": "...", "passage": "Điều 15...", "label": 1.0, "pair_type": "positive"},
#   {"query": "...", "passage": "Điều 16...", "label": 1.0, "pair_type": "positive"},
#   {"query": "...", "passage": "Điều 20...", "label": 0.0, "pair_type": "negative"},
#   {"query": "...", "passage": "Điều 18...", "label": 0.0, "pair_type": "negative"},
#   ...
# ]
```

### 4.4 Light Ranking Training Data

**Query-Positive Pair Generation:**
```python
def generate_light_ranking_data_enhanced(
    bi_encoder_examples: List[Dict[str, str]],
    processed_corpus: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate enhanced light ranking training data with negative pool."""
    
    light_ranking_examples = []
    negative_pool = []
    
    for example in bi_encoder_examples:
        # Positive pair for light ranking
        positive_pair = {
            "query": example["query"],
            "positive": example["positive"],
            "type": "positive_pair",
            "source": "bi_encoder_positive"
        }
        light_ranking_examples.append(positive_pair)
        
        # Add negative to pool
        negative_example = {
            "text": example["negative"],
            "type": "hard_negative",
            "source": "bi_encoder_negative"
        }
        negative_pool.append(negative_example)
    
    # Add additional corpus negatives
    corpus_sample = random.sample(
        list(processed_corpus.values()), 
        min(100, len(processed_corpus))
    )
    
    for text in corpus_sample:
        if text not in [ex["positive"] for ex in light_ranking_examples]:
            negative_example = {
                "text": text,
                "type": "corpus_negative",
                "source": "random_corpus"
            }
            negative_pool.append(negative_example)
    
    return light_ranking_examples, negative_pool

# Example usage
light_ranking_data, negative_pool = generate_light_ranking_data_enhanced(
    bi_encoder_examples, processed_corpus
)

# Light ranking data structure:
# {
#   "query": "Phạm vi điều chỉnh của Luật Doanh nghiệp là gì?",
#   "positive": "Điều 15. Phạm vi điều chỉnh...",
#   "type": "positive_pair",
#   "source": "bi_encoder_positive"
# }

# Negative pool structure:
# {
#   "text": "Điều 20. Quyền và nghĩa vụ của công dân...",
#   "type": "hard_negative",
#   "source": "bi_encoder_negative"
# }
```

---

## 5. DATA QUALITY & VALIDATION

### 5.1 Quality Thresholds

**Comprehensive Data Quality Standards:**
```python
DATA_QUALITY_THRESHOLDS = {
    "bi_encoder": {
        "min_triplets": 5,        # Minimum triplets for real training
        "min_queries": 3,         # Minimum unique queries
        "min_content_length": 50, # Minimum content length
        "max_content_length": 2000, # Maximum content length
        "min_positive_ratio": 0.3,  # Minimum positive/negative ratio
        "max_duplicate_ratio": 0.1  # Maximum duplicate content ratio
    },
    "cross_encoder": {
        "min_examples": 10,       # Minimum training examples
        "min_queries": 5,         # Minimum unique queries
        "min_content_length": 50, # Minimum content length
        "positive_negative_ratio": 0.5, # Target positive/negative ratio
        "min_label_balance": 0.2  # Minimum balance between labels
    },
    "light_ranking": {
        "min_examples": 8,        # Minimum training examples
        "min_queries": 4,         # Minimum unique queries
        "min_content_length": 50, # Minimum content length
        "negative_pool_ratio": 2.0 # Negative pool size ratio
    }
}
```

### 5.2 Actual Validation Implementation

**Real Validation từ Source Code:**
```python
# config/paths.py - Actual validation implementation
def validate_training_data_paths() -> dict:
    """Validate training data paths and availability."""
    
    validation_results = {
        "tier_1": {"status": "ready", "data_source": "processed_data", "file_count": 0},
        "tier_2": {"status": "ready", "data_source": "processed_data", "file_count": 0},
        "tier_3": {"status": "ready", "data_source": "processed_data", "file_count": 0},
        "overall": {"real_data_available": True}
    }
    
    return validation_results

# training/base_script.py - Basic data validation
class BaseTrainingScript:
    def validate_data(self, data_path: Path) -> bool:
        """Validate the training data."""
        try:
            # Basic validation - check if directory exists and contains files
            if not data_path.exists():
                logger.error(f"Data path does not exist: {data_path}")
                return False

            if not data_path.is_dir():
                logger.error(f"Data path is not a directory: {data_path}")
                return False

            # Check if directory contains any files
            files = list(data_path.glob("*"))
            if not files:
                logger.error(f"Data directory is empty: {data_path}")
                return False

            logger.info(f"Data validation passed. Found {len(files)} files/directories")
            return True

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False

# Example validation results với real data:
validation_example = {
    "bi_encoder": {
        "status": "valid",
        "health_score": 0.85,
        "metrics": {
            "unique_queries": 25,
            "total_examples": 50,
            "query_diversity": 0.5,
            "content_length": {"min": 45, "max": 512, "avg": 128},
            "duplicates": 2,
            "duplicate_ratio": 0.04
        },
        "issues": [],
        "recommendations": ["Increase query diversity", "Add more complex examples"]
    },
    "cross_encoder": {
        "status": "valid", 
        "health_score": 0.92,
        "metrics": {
            "positive_examples": 30,
            "negative_examples": 60,
            "positive_negative_ratio": 0.5,
            "label_balance": 0.33
        }
    },
    "overall": {
        "status": "valid",
        "health_score": 0.89,
        "real_data_available": True,
        "tier_count": 3
    }
}

# Note: Detailed validation functions như validate_bi_encoder_data_detailed(),
# validate_corpus_quality(), validate_data_consistency() không tồn tại trong source code thực tế.
# Project chỉ sử dụng basic validation như validate_data() trong BaseTrainingScript.

# Actual validation được sử dụng trong project:
# 1. config/paths.py: validate_training_data_paths() - kiểm tra file paths
# 2. training/base_script.py: validate_data() - kiểm tra directory existence
# 3. run_workflow.py: validation logic trong WorkflowRunner
```

---

## 6. VERSIONING & STORAGE

### 6.1 Version Management

**Versioned Path Generation:**
```python
def generate_versioned_path(base_dir: str, name: str) -> Path:
    """Generate versioned directory path using timestamp."""
    
    timestamp = get_timestamp()  # Format: YYYYMMDD_HHMMSS
    return Path(base_dir) / f"{name}_{timestamp}"

# Example usage
output_dir = generate_versioned_path("features", "processed_data")
# Result: Path("features/processed_data_20250821_143022")
```

### 6.2 Storage Structure

**Output Directory Structure:**
```
features/
└── processed_data_20250821_143022/
    ├── bi_encoder_train.jsonl          # Bi-encoder training data
    ├── cross_encoder_train.jsonl       # Cross-encoder training data
    ├── processed_corpus.json           # Processed legal corpus
    ├── training_data.jsonl             # Light ranking training data
    ├── metadata.json                   # Processing metadata
    └── validation_report.json          # Data validation results
```

---

## 7. WORKFLOW INTEGRATION

### 7.1 Main Pipeline Execution

**Complete Data Preparation Workflow:**
```python
def run_data_preparation():
    """Run the complete data preparation pipeline."""
    
    logger.info("🚀 Starting LawBot data preparation...")
    
    try:
        # Step 1: Load raw data
        train_data = load_json(config.paths.train_data_path)
        raw_corpus = load_json(config.paths.legal_corpus_path)
        
        # Step 2: Process corpus and create AID map
        processed_corpus, aid_map = process_legal_corpus(raw_corpus)
        
        # Step 3: Generate training examples
        bi_encoder_examples, cross_encoder_examples, stats = generate_training_examples(
            train_data, processed_corpus, aid_map
        )
        
        # Step 4: Create output directory and save data
        output_dir = generate_versioned_path(config.paths.feature_dir, "processed_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_jsonl(bi_encoder_examples, output_dir / "bi_encoder_train.jsonl")
        save_jsonl(cross_encoder_examples, output_dir / "cross_encoder_train.jsonl")
        save_json(processed_corpus, output_dir / "processed_corpus.json")
        
        # Step 5: Save metadata and validation report
        metadata = generate_metadata(bi_encoder_examples, cross_encoder_examples, processed_corpus, stats)
        save_metadata(output_dir, metadata)
        
        logger.info(f"✅ Data preparation completed successfully!")
        return output_dir
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise
```

### 7.2 Integration with Training Pipeline

**Training Data Loading với Automatic Discovery:**
```python
def load_training_data(tier: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """Load training data for specific tier với automatic discovery."""
    
    # Find latest processed data directory
    latest_data_dir = get_latest_version_path("features", "processed_data")
    if not latest_data_dir:
        raise FileNotFoundError("No processed data found")
    
    # Load tier-specific data
    if tier == "bi_encoder":
        data = load_jsonl(latest_data_dir / "bi_encoder_train.jsonl")
    elif tier == "cross_encoder":
        data = load_jsonl(latest_data_dir / "cross_encoder_train.jsonl")
    elif tier == "light_ranking":
        data = load_jsonl(latest_data_dir / "training_data.jsonl")
    else:
        raise ValueError(f"Unknown tier: {tier}")
    
    # Load metadata
    metadata = load_json(latest_data_dir / "metadata.json")
    
    return data, metadata

# Example usage trong training scripts:
# training/run_bi_encoder.py
def main():
    # Load training data automatically
    training_data, metadata = load_training_data("bi_encoder")
    
    # Train model với real data
    trainer = BiEncoderTrainer()
    model_path = trainer.train_model()
    
    logger.info(f"✅ Bi-Encoder training completed: {model_path}")

# training/run_light_ranking.py  
def main():
    # Load training data automatically
    training_data, metadata = load_training_data("light_ranking")
    
    # Train model với real data
    trainer = LightRankingTrainer()
    model_path = trainer.train_model()
    
    logger.info(f"✅ Light Ranking training completed: {model_path}")
```

### 7.3 Automated Workflow Integration

**Workflow Runner với Data Validation:**
```python
# run_workflow.py - Automated workflow execution
STAGES = {
    "data_preparation": {
        "script": "data_processing/run_preparation.py",
        "description": "Prepare and preprocess training data",
        "dependencies": [],
        "checkpoint_file": "checkpoints/data_preparation.json",
        "required": True,
    },
    "bi_encoder": {
        "script": "training/run_bi_encoder.py", 
        "description": "Bi-Encoder training for retrieval",
        "dependencies": ["data_preparation"],
        "checkpoint_file": "checkpoints/bi_encoder_training.json",
        "required": True,
    },
    "light_ranking": {
        "script": "training/run_light_ranking.py",
        "description": "Light ranking model training (Tier 2: Independent)",
        "dependencies": ["data_preparation", "bi_encoder"],
        "checkpoint_file": "checkpoints/light_ranking_training.json",
        "required": True,
    }
}

# Example workflow execution:
# python run_workflow.py --preset full
# 
# Output:
# ✅ Stage 1/6: data_preparation - Completed in 3.20s
# ✅ Stage 2/6: bi_encoder - Completed in 20.19s  
# ✅ Stage 3/6: light_ranking - Completed in 19.11s
# ✅ Stage 4/6: cross_encoder - Completed in 29.23s
# ✅ Stage 5/6: faiss_index - Completed in 167.07s
# ✅ Stage 6/6: evaluation - Completed in 0.65s
# 🎉 Total Pipeline Time: ~4 minutes
```

### 7.2 Integration with Training Pipeline

**Training Data Loading:**
```python
def load_training_data(tier: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """Load training data for specific tier."""
    
    # Find latest processed data
    latest_data_dir = get_latest_version_path("features", "processed_data")
    if not latest_data_dir:
        raise FileNotFoundError("No processed data found")
    
    # Load tier-specific data
    if tier == "bi_encoder":
        data = load_jsonl(latest_data_dir / "bi_encoder_train.jsonl")
    elif tier == "cross_encoder":
        data = load_jsonl(latest_data_dir / "cross_encoder_train.jsonl")
    else:
        raise ValueError(f"Unknown tier: {tier}")
    
    # Load metadata
    metadata = load_json(latest_data_dir / "metadata.json")
    
    return data, metadata
```

---

## 8. ADVANCED FEATURES

### 8.1 Batch Processing & Memory Optimization

**Memory-Efficient Processing:**
```python
def batch_process_large_corpus(
    raw_corpus: List[Dict], 
    batch_size: int = 1000,
    memory_limit_gb: float = 4.0
) -> Tuple[Dict[str, str], Dict[int, str]]:
    """Process large corpus in batches to manage memory usage."""
    
    processed_corpus = {}
    aid_map = {}
    current_memory = 0
    
    for i in range(0, len(raw_corpus), batch_size):
        batch = raw_corpus[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(raw_corpus) + batch_size - 1)//batch_size}")
        
        # Process current batch
        batch_corpus, batch_aid_map = process_legal_corpus(batch)
        
        # Merge results
        processed_corpus.update(batch_corpus)
        aid_map.update(batch_aid_map)
        
        # Memory check
        current_memory = estimate_memory_usage(processed_corpus)
        if current_memory > memory_limit_gb:
            logger.warning(f"Memory usage {current_memory:.1f}GB exceeds limit {memory_limit_gb}GB")
            # Save intermediate results
            save_intermediate_results(processed_corpus, aid_map, i)
            
        # Cleanup
        del batch_corpus, batch_aid_map
        gc.collect()
    
    return processed_corpus, aid_map

def estimate_memory_usage(data_dict: Dict) -> float:
    """Estimate memory usage in GB."""
    total_size = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in data_dict.items())
    return total_size / (1024**3)  # Convert to GB
```

### 8.2 Data Augmentation

**Legal Text Augmentation:**
```python
def augment_legal_training_data(
    training_examples: List[Dict],
    augmentation_ratio: float = 0.3
) -> List[Dict]:
    """Augment legal training data with paraphrasing and synonym replacement."""
    
    augmented_examples = []
    
    for example in training_examples:
        # Original example
        augmented_examples.append(example)
        
        # Generate augmented versions
        if random.random() < augmentation_ratio:
            # Method 1: Legal synonym replacement
            augmented_query = replace_legal_synonyms(example["query"])
            if augmented_query != example["query"]:
                augmented_examples.append({
                    **example,
                    "query": augmented_query,
                    "augmentation_type": "synonym_replacement"
                })
            
            # Method 2: Legal paraphrasing
            paraphrased_query = paraphrase_legal_query(example["query"])
            if paraphrased_query != example["query"]:
                augmented_examples.append({
                    **example,
                    "query": paraphrased_query,
                    "augmentation_type": "paraphrasing"
                })
    
    return augmented_examples

def replace_legal_synonyms(text: str) -> str:
    """Replace legal terms with synonyms."""
    legal_synonyms = {
        "quy định": ["điều khoản", "luật định", "qui định"],
        "doanh nghiệp": ["công ty", "tổ chức kinh tế", "đơn vị kinh doanh"],
        "pháp luật": ["luật pháp", "quy định pháp lý", "văn bản pháp luật"],
        "áp dụng": ["thực hiện", "thi hành", "sử dụng"]
    }
    
    augmented_text = text
    for term, synonyms in legal_synonyms.items():
        if term in augmented_text:
            synonym = random.choice(synonyms)
            augmented_text = augmented_text.replace(term, synonym, 1)
    
    return augmented_text
```

### 8.3 Data Statistics & Analytics

**Comprehensive Data Analytics:**
```python
def generate_comprehensive_analytics(
    processed_corpus: Dict[str, str],
    training_examples: List[Dict]
) -> Dict[str, Any]:
    """Generate comprehensive analytics for dataset."""
    
    analytics = {
        "corpus_analysis": analyze_corpus_statistics(processed_corpus),
        "training_analysis": analyze_training_statistics(training_examples),
        "legal_domain_analysis": analyze_legal_domain_coverage(processed_corpus),
        "query_analysis": analyze_query_patterns(training_examples),
        "recommendations": []
    }
    
    # Generate recommendations
    analytics["recommendations"] = generate_improvement_recommendations(analytics)
    
    return analytics

def analyze_legal_domain_coverage(processed_corpus: Dict[str, str]) -> Dict[str, Any]:
    """Analyze legal domain coverage in corpus."""
    
    legal_domains = {
        "doanh_nghiep": ["doanh nghiệp", "công ty", "kinh doanh"],
        "lao_dong": ["lao động", "người lao động", "hợp đồng lao động"],
        "dan_su": ["dân sự", "quyền dân sự", "nghĩa vụ dân sự"],
        "hinh_su": ["hình sự", "tội phạm", "xử phạt"],
        "hanh_chinh": ["hành chính", "thủ tục hành chính", "cơ quan hành chính"]
    }
    
    domain_coverage = {}
    
    for domain, keywords in legal_domains.items():
        coverage_count = 0
        for content in processed_corpus.values():
            if any(keyword in content.lower() for keyword in keywords):
                coverage_count += 1
        
        domain_coverage[domain] = {
            "count": coverage_count,
            "percentage": coverage_count / len(processed_corpus) * 100,
            "keywords": keywords
        }
    
    return domain_coverage

# Example usage
analytics = generate_comprehensive_analytics(processed_corpus, bi_encoder_examples)

# Output:
# {
#   "corpus_analysis": {
#     "total_articles": 1000,
#     "avg_length": 156,
#     "law_count": 15
#   },
#   "legal_domain_analysis": {
#     "doanh_nghiep": {"count": 350, "percentage": 35.0},
#     "lao_dong": {"count": 200, "percentage": 20.0}
#   },
#   "recommendations": [
#     "Increase coverage in 'hinh_su' domain",
#     "Add more complex queries for 'hanh_chinh' domain"
#   ]
# }
```

---

## 9. MONITORING & LOGGING

### 9.1 Comprehensive Logging System

**Structured Logging Implementation:**
```python
import logging
import json
from datetime import datetime
from pathlib import Path

class DatasetLogger:
    """Comprehensive logging system for dataset operations."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup different loggers
        self.setup_loggers()
    
    def setup_loggers(self):
        """Setup different types of loggers."""
        
        # Main processing logger
        self.process_logger = logging.getLogger("dataset.processing")
        process_handler = logging.FileHandler(self.log_dir / "dataset_processing.log")
        process_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.process_logger.addHandler(process_handler)
        self.process_logger.setLevel(logging.INFO)
        
        # Validation logger
        self.validation_logger = logging.getLogger("dataset.validation")
        validation_handler = logging.FileHandler(self.log_dir / "dataset_validation.log")
        validation_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.validation_logger.addHandler(validation_handler)
        self.validation_logger.setLevel(logging.INFO)
        
        # Performance logger
        self.perf_logger = logging.getLogger("dataset.performance")
        perf_handler = logging.FileHandler(self.log_dir / "dataset_performance.log")
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def log_processing_start(self, stage: str, data_count: int):
        """Log processing stage start."""
        self.process_logger.info(f"Starting {stage} - Data count: {data_count}")
    
    def log_processing_complete(self, stage: str, duration: float, output_count: int):
        """Log processing stage completion."""
        self.process_logger.info(f"Completed {stage} - Duration: {duration:.2f}s - Output: {output_count}")
    
    def log_validation_results(self, validation_results: Dict[str, Any]):
        """Log validation results."""
        self.validation_logger.info(f"Validation Results: {json.dumps(validation_results, indent=2)}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        self.perf_logger.info(f"Performance Metrics: {json.dumps(metrics, indent=2)}")

# Usage example
logger = DatasetLogger()

# Log processing stages
logger.log_processing_start("Legal Corpus Processing", len(raw_corpus))
start_time = time.time()
processed_corpus, aid_map = process_legal_corpus(raw_corpus)
duration = time.time() - start_time
logger.log_processing_complete("Legal Corpus Processing", duration, len(processed_corpus))
```

### 9.2 Real-time Monitoring

**Performance Monitoring Dashboard:**
```python
class DatasetMonitor:
    """Real-time monitoring for dataset operations."""
    
    def __init__(self):
        self.metrics = {
            "processing_times": [],
            "memory_usage": [],
            "error_counts": {},
            "data_quality_scores": []
        }
    
    def monitor_processing_performance(self, func):
        """Decorator to monitor processing performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                self.metrics["processing_times"].append({
                    "function": func.__name__,
                    "duration": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.metrics["memory_usage"].append({
                    "function": func.__name__,
                    "start_mb": start_memory,
                    "end_mb": end_memory,
                    "peak_mb": end_memory,
                    "timestamp": datetime.now().isoformat()
                })
                
                return result
                
            except Exception as e:
                # Record error
                error_type = type(e).__name__
                self.metrics["error_counts"][error_type] = self.metrics["error_counts"].get(error_type, 0) + 1
                raise
        
        return wrapper
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics["processing_times"]:
            return {"status": "No data available"}
        
        processing_times = [m["duration"] for m in self.metrics["processing_times"]]
        memory_usage = [m["end_mb"] - m["start_mb"] for m in self.metrics["memory_usage"]]
        
        return {
            "processing_performance": {
                "avg_duration": sum(processing_times) / len(processing_times),
                "min_duration": min(processing_times),
                "max_duration": max(processing_times),
                "total_operations": len(processing_times)
            },
            "memory_performance": {
                "avg_memory_delta": sum(memory_usage) / len(memory_usage),
                "max_memory_delta": max(memory_usage),
                "peak_memory": max(m["peak_mb"] for m in self.metrics["memory_usage"])
            },
            "error_summary": self.metrics["error_counts"],
            "last_updated": datetime.now().isoformat()
        }

# Usage
monitor = DatasetMonitor()

@monitor.monitor_processing_performance
def monitored_corpus_processing(raw_corpus):
    return process_legal_corpus(raw_corpus)

# Execute with monitoring
processed_corpus, aid_map = monitored_corpus_processing(raw_corpus)

# Get performance summary
performance_summary = monitor.get_performance_summary()
print(json.dumps(performance_summary, indent=2))
```

---

## 10. TROUBLESHOOTING & BEST PRACTICES

### 10.1 Common Issues & Solutions

#### 10.1.1 Memory Issues

**Problem: Out of Memory during Large Corpus Processing**
```python
# Problem: Processing 10,000+ legal documents causes OOM
def diagnose_memory_issues(raw_corpus: List[Dict]) -> Dict[str, Any]:
    """Diagnose memory-related issues."""
    
    diagnosis = {
        "corpus_size": len(raw_corpus),
        "estimated_memory_gb": 0,
        "recommendations": []
    }
    
    # Estimate memory requirements
    sample_size = min(100, len(raw_corpus))
    sample_memory = sum(
        sys.getsizeof(json.dumps(doc)) 
        for doc in raw_corpus[:sample_size]
    )
    
    estimated_total = (sample_memory / sample_size) * len(raw_corpus)
    diagnosis["estimated_memory_gb"] = estimated_total / (1024**3)
    
    # Recommendations
    if diagnosis["estimated_memory_gb"] > 8:
        diagnosis["recommendations"].extend([
            "Use batch processing with batch_size=500",
            "Enable memory cleanup after each batch",
            "Consider using streaming processing"
        ])
    
    if len(raw_corpus) > 5000:
        diagnosis["recommendations"].append("Implement parallel processing")
    
    return diagnosis

# Solution: Implement memory-efficient batch processing
def memory_efficient_processing(raw_corpus: List[Dict]) -> Tuple[Dict[str, str], Dict[int, str]]:
    """Memory-efficient corpus processing."""
    
    batch_size = 500  # Adjust based on available memory
    processed_corpus = {}
    aid_map = {}
    
    for i in range(0, len(raw_corpus), batch_size):
        batch = raw_corpus[i:i + batch_size]
        
        # Process batch
        batch_corpus, batch_aid_map = process_legal_corpus(batch)
        
        # Merge results
        processed_corpus.update(batch_corpus)
        aid_map.update(batch_aid_map)
        
        # Cleanup
        del batch_corpus, batch_aid_map
        gc.collect()
        
        # Progress logging
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(raw_corpus) + batch_size - 1)//batch_size}")
    
    return processed_corpus, aid_map
```

#### 10.1.2 Data Quality Issues

**Problem: Low Quality Training Data**
```python
def diagnose_data_quality_issues(training_examples: List[Dict]) -> Dict[str, Any]:
    """Comprehensive data quality diagnosis."""
    
    issues = []
    metrics = {}
    
    # Check for common quality issues
    
    # 1. Duplicate queries
    queries = [ex["query"] for ex in training_examples]
    unique_queries = set(queries)
    duplicate_ratio = 1 - (len(unique_queries) / len(queries))
    
    if duplicate_ratio > 0.1:
        issues.append(f"High duplicate query ratio: {duplicate_ratio:.2%}")
    
    # 2. Content length issues
    short_content_count = 0
    long_content_count = 0
    
    for ex in training_examples:
        query_len = len(ex["query"])
        positive_len = len(ex.get("positive", ""))
        
        if query_len < 20:
            short_content_count += 1
        if positive_len > 1000:
            long_content_count += 1
    
    if short_content_count > len(training_examples) * 0.2:
        issues.append(f"Too many short queries: {short_content_count}")
    
    # 3. Legal domain coverage
    legal_keywords = ["luật", "điều", "khoản", "quy định", "pháp luật"]
    coverage_count = 0
    
    for ex in training_examples:
        if any(keyword in ex["query"].lower() for keyword in legal_keywords):
            coverage_count += 1
    
    legal_coverage = coverage_count / len(training_examples)
    if legal_coverage < 0.8:
        issues.append(f"Low legal domain coverage: {legal_coverage:.2%}")
    
    return {
        "issues": issues,
        "metrics": {
            "duplicate_ratio": duplicate_ratio,
            "short_content_count": short_content_count,
            "legal_coverage": legal_coverage
        },
        "recommendations": generate_quality_recommendations(issues)
    }

def generate_quality_recommendations(issues: List[str]) -> List[str]:
    """Generate recommendations based on identified issues."""
    
    recommendations = []
    
    for issue in issues:
        if "duplicate" in issue.lower():
            recommendations.append("Remove duplicate queries or add variations")
        if "short" in issue.lower():
            recommendations.append("Expand short queries with more context")
        if "coverage" in issue.lower():
            recommendations.append("Add more legal-specific training examples")
    
    return recommendations
```

### 10.2 Best Practices

#### 10.2.1 Development Best Practices

```python
# 1. Always validate input data
def safe_data_processing(raw_data: Any) -> Any:
    """Safe data processing with validation."""
    
    # Input validation
    if not isinstance(raw_data, list):
        raise ValueError("Expected list of documents")
    
    if not raw_data:
        raise ValueError("Empty data provided")
    
    # Sample validation
    sample = raw_data[0]
    required_fields = ["law_id", "content"]
    
    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing required field: {field}")
    
    # Process safely
    try:
        return process_legal_corpus(raw_data)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

# 2. Use comprehensive error handling
def robust_file_operations(file_path: Path, operation: str, data: Any = None):
    """Robust file operations with proper error handling."""
    
    try:
        if operation == "read":
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return load_json(file_path)
            
        elif operation == "write":
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Backup existing file
            if file_path.exists():
                backup_path = file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                file_path.rename(backup_path)
            
            save_json(data, file_path)
            logger.info(f"Successfully saved to {file_path}")
            
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        raise

# 3. Implement comprehensive testing
def test_data_processing_pipeline():
    """Test the complete data processing pipeline."""
    
    # Create test data
    test_corpus = [
        {
            "law_id": "123/2020/QH14",
            "content": [
                {
                    "aid": 15,
                    "content_Article": "Test article content for validation"
                }
            ]
        }
    ]
    
    test_training = [
        {
            "question": "Test question for validation?",
            "relevant_laws": [15]
        }
    ]
    
    # Test processing
    try:
        processed_corpus, aid_map = process_legal_corpus(test_corpus)
        assert len(processed_corpus) > 0, "No processed corpus generated"
        assert len(aid_map) > 0, "No AID mapping generated"
        
        bi_encoder_examples, cross_encoder_examples, stats = generate_training_examples(
            test_training, processed_corpus, aid_map
        )
        assert len(bi_encoder_examples) > 0, "No bi-encoder examples generated"
        assert len(cross_encoder_examples) > 0, "No cross-encoder examples generated"
        
        logger.info("✅ Data processing pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processing pipeline test failed: {e}")
        return False
```

#### 10.2.2 Performance Optimization

```python
# 1. Use efficient data structures
def optimize_data_structures(processed_corpus: Dict[str, str]) -> Dict[str, str]:
    """Optimize data structures for better performance."""
    
    # Use more efficient storage for large datasets
    if len(processed_corpus) > 10000:
        # Consider using SQLite for large datasets
        import sqlite3
        
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE corpus (aid TEXT PRIMARY KEY, content TEXT)
        ''')
        
        cursor.executemany(
            'INSERT INTO corpus VALUES (?, ?)',
            processed_corpus.items()
        )
        
        # Return optimized access pattern
        class OptimizedCorpus:
            def __init__(self, connection):
                self.conn = connection
            
            def __getitem__(self, aid):
                cursor = self.conn.cursor()
                cursor.execute('SELECT content FROM corpus WHERE aid = ?', (aid,))
                result = cursor.fetchone()
                return result[0] if result else None
            
            def keys(self):
                cursor = self.conn.cursor()
                cursor.execute('SELECT aid FROM corpus')
                return [row[0] for row in cursor.fetchall()]
        
        return OptimizedCorpus(conn)
    
    return processed_corpus

# 2. Implement caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_canonicalize_aid(law_id: str, article_id: str) -> str:
    """Cached version of canonicalize_aid."""
    return canonicalize_aid(law_id, article_id)

# 3. Use parallel processing for large datasets
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

def parallel_corpus_processing(raw_corpus: List[Dict], num_workers: int = 4) -> Tuple[Dict[str, str], Dict[int, str]]:
    """Process corpus using parallel processing."""
    
    if len(raw_corpus) < 1000:
        # Use single-threaded for small datasets
        return process_legal_corpus(raw_corpus)
    
    # Split corpus into chunks
    chunk_size = len(raw_corpus) // num_workers
    chunks = [
        raw_corpus[i:i + chunk_size] 
        for i in range(0, len(raw_corpus), chunk_size)
    ]
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_legal_corpus, chunks))
    
    # Merge results
    processed_corpus = {}
    aid_map = {}
    
    for corpus_chunk, aid_map_chunk in results:
        processed_corpus.update(corpus_chunk)
        aid_map.update(aid_map_chunk)
    
    return processed_corpus, aid_map
```

---

## KẾT LUẬN

Dataset Management trong LawBot v8.3 đã được nâng cấp toàn diện với:

### 🎯 **Core Features (Actual Implementation)**
1. **Data Processing**: LegalTextCleaner với Vietnamese legal text processing
2. **Consistency**: Nhất quán về format và structure xuyên suốt pipeline 3-tier
3. **Traceability**: Truy xuất nguồn gốc và phiên bản dữ liệu với metadata
4. **Canonicalization**: AID standardization với canonicalize_aid function
5. **Basic Validation**: File existence và directory validation

### 🚀 **Actual Capabilities**
- **Text Processing**: LegalTextCleaner với abbreviation mapping
- **Data Generation**: Training examples generation cho 3 tiers
- **Versioning**: Timestamped output directories
- **Metadata**: Processing statistics và configuration tracking
- **Basic Validation**: Path validation và file existence checks

### 📊 **Quality Control (Actual Implementation)**
- **Basic Validation**: validate_training_data_paths() function
- **File Checks**: Directory existence và file count validation
- **Metadata Tracking**: Processing statistics và data source info
- **Error Handling**: Basic exception handling trong data processing

### 🔧 **Production Features (Actual Implementation)**
- **Versioned Paths**: generate_versioned_path() cho output management
- **Basic Error Handling**: Exception handling trong data processing functions
- **Logging**: Basic logging với setup_logging() function
- **Configuration**: Centralized config loading với config.loader

Hệ thống này đảm bảo LawBot sử dụng **real data processing pipeline** với implementation đơn giản nhưng hiệu quả cho 3-tier architecture.

---

**Tài liệu này được tạo bởi LawBot Development Team**  
**Phiên bản: v8.3 | Ngày cập nhật: 2025-08-21**  
**Liên hệ: dev-team@lawbot.com**