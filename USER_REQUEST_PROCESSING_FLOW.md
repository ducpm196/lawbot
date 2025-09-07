# Luồng Xử Lý Yêu Cầu User - LawBot

## Mục Lục
1. [Tổng Quan Hệ Thống](#tổng-quan-hệ-thống)
2. [Kiến Trúc Tổng Thể](#kiến-trúc-tổng-thể)
3. [Luồng Xử Lý Chính](#luồng-xử-lý-chính)
4. [Chi Tiết Từng Bước](#chi-tiết-từng-bước)
5. [Xử Lý Lỗi và Fallback](#xử-lý-lỗi-và-fallback)
6. [Tối Ưu Hóa Hiệu Suất](#tối-ưu-hóa-hiệu-suất)
7. [Ví Dụ Thực Tế](#ví-dụ-thực-tế)

---

## 1. Tổng Quan Hệ Thống

LawBot là một hệ thống hỏi-đáp pháp luật thông minh sử dụng kiến trúc 3-tier để xử lý các yêu cầu của người dùng một cách hiệu quả và chính xác.

### 1.1 Mục Tiêu Chính
- **Xử lý câu hỏi pháp luật** một cách tự động và chính xác
- **Tìm kiếm tài liệu pháp luật** liên quan từ cơ sở dữ liệu lớn
- **Đưa ra câu trả lời** dựa trên thông tin pháp luật có sẵn
- **Hỗ trợ đa dạng loại câu hỏi** về các lĩnh vực pháp luật khác nhau

### 1.2 Đặc Điểm Nổi Bật
- **Kiến trúc modular**: Dễ dàng mở rộng và bảo trì
- **Xử lý batch**: Hỗ trợ xử lý nhiều câu hỏi cùng lúc
- **Cache thông minh**: Tối ưu hóa hiệu suất và thời gian phản hồi
- **Fallback mechanism**: Đảm bảo hệ thống hoạt động ngay cả khi có lỗi

---

## 2. Kiến Trúc Tổng Thể

### 2.1 Sơ Đồ Kiến Trúc

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Streamlit UI   │───▶│  Main Pipeline  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Search Page   │    │ Retrieval Engine│
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Analysis Page   │    │Reranking Engine │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  System Page    │    │  Content Maps   │
                       └─────────────────┘    └─────────────────┘
```

### 2.2 Các Thành Phần Chính

#### 2.2.1 Frontend Layer (Streamlit)
- **`app.py`**: Điểm vào chính của ứng dụng
- **`pages/search.py`**: Trang tìm kiếm và xử lý câu hỏi
- **`pages/analysis.py`**: Trang phân tích kết quả
- **`pages/system.py`**: Trang quản lý hệ thống

#### 2.2.2 Core Processing Layer
- **`LegalQAPipeline`**: Điều phối toàn bộ quá trình xử lý
- **`RetrievalEngine`**: Tìm kiếm tài liệu liên quan
- **`RerankingEngine`**: Sắp xếp lại kết quả theo độ liên quan

#### 2.2.3 Data Layer
- **FAISS Index**: Chỉ mục vector để tìm kiếm nhanh
- **Content Maps**: Bản đồ nội dung tài liệu
- **Parent Law Mapping**: Liên kết giữa các văn bản pháp luật

### 2.3 Luồng Dữ Liệu

```
User Query → Text Processing → Vector Encoding → FAISS Search → 
Content Retrieval → Reranking → Result Formatting → User Response
```

---

## 3. Luồng Xử Lý Chính

### 3.1 Khởi Tạo Hệ Thống

Khi ứng dụng khởi động, hệ thống sẽ:

1. **Load Configuration**: Đọc cấu hình từ `config/default.yml`
2. **Initialize Pipeline**: Khởi tạo `LegalQAPipeline`
3. **Load Models**: Tải các model bi-encoder và reranker
4. **Setup Indexes**: Khởi tạo FAISS index và content maps
5. **Validate Dependencies**: Kiểm tra tính khả dụng của các thành phần

### 3.2 Xử Lý Yêu Cầu User

#### 3.2.1 Nhận Input
- User nhập câu hỏi vào giao diện Streamlit
- Hệ thống validate và preprocess câu hỏi
- Chuẩn bị context cho việc xử lý

#### 3.2.2 Tìm Kiếm (Retrieval)
- **Vector Encoding**: Chuyển đổi câu hỏi thành vector embedding
- **FAISS Search**: Tìm kiếm trong index để lấy top-k kết quả
- **Content Mapping**: Lấy nội dung chi tiết của các kết quả

#### 3.2.3 Sắp Xếp Lại (Reranking)
- **Score Calculation**: Tính toán điểm số liên quan
- **Context Enhancement**: Bổ sung thông tin bối cảnh
- **Final Ranking**: Sắp xếp kết quả theo độ chính xác

#### 3.2.4 Trả Về Kết Quả
- **Format Results**: Định dạng kết quả cho người dùng
- **Display Information**: Hiển thị thông tin chi tiết
- **Provide Context**: Cung cấp bối cảnh pháp luật liên quan

---

## 4. Chi Tiết Từng Bước

### 4.1 Bước 1: Khởi Tạo và Validation

```python
# Trong LegalQAPipeline.__init__() - Source code thực tế
def __init__(
    self,
    bi_encoder_path: Optional[Path] = None,
    reranker_paths: Optional[Dict[str, Path]] = None,
    faiss_index_path: Optional[Path] = None,
    content_map_path: Optional[Path] = None,
    index_to_aid_path: Optional[Path] = None,
):
    """Initialize the LegalQA Pipeline."""
    self.is_ready = False
    self.loaded_model_paths = {}

    try:
        # Ensure parent law mapping is available before proceeding
        logger.info("Validating parent law mapping availability...")
    if not ensure_parent_law_mapping():
            logger.warning(
                "Parent law mapping validation failed, but continuing with pipeline initialization"
            )
        else:
            logger.info("✅ Parent law mapping validation successful")

        # Load Retriever với path resolution
        paths = config.paths
        bi_encoder_to_load = bi_encoder_path or get_latest_version_path(
            paths.model_dir, "bi-encoder"
        )
        faiss_to_load = faiss_index_path or paths.faiss_index_path
        content_map_to_load = content_map_path or paths.content_map_path
        index_to_aid_to_load = index_to_aid_path or paths.index_to_aid_path

    self.retriever = RetrievalEngine(
            bi_encoder_path=bi_encoder_to_load,
            faiss_index_path=faiss_to_load,
            content_map_path=content_map_to_load,
            index_to_aid_path=index_to_aid_to_load
        )

        # Load Reranker với config
    self.reranker = RerankingEngine(reranker_configs=reranker_configs)
        
        # Set pipeline ready status
        self.is_ready = (
            self.retriever.is_ready and 
            self.reranker.is_ready
        )
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        self.is_ready = False
```

**Ví dụ thực tế:**
- Hệ thống kiểm tra xem có đủ file mapping không
- Nếu thiếu, sẽ log warning và tiếp tục với chức năng hạn chế
- Đảm bảo pipeline vẫn hoạt động được

### 4.2 Bước 2: Xử Lý Câu Hỏi

```python
# Trong search.py - main() function thực tế
def main():
    """Main search page function."""
    # Load pipeline với caching
    pipeline = load_pipeline()
    
    if pipeline and pipeline.is_ready:
        # Get search results với parameters thực tế
        results = pipeline.predict(
            query,
            top_k_retrieval=params["top_k_retrieval"],
            top_k_light=params["top_k_light_reranking"],
            top_k_final=params["top_k_final"],
    )
    
    return results

@st.cache_resource(ttl=config.app.cache_pipeline_ttl_seconds)
def load_pipeline(force_cpu=False):
    """Load and cache pipeline to avoid reloading on each interaction."""
    try:
        pipeline = LegalQAPipeline()
        if not pipeline.is_ready:
            return None
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None
```

**Ví dụ thực tế:**
- User nhập: "Quy định về thời gian làm việc của nhân viên"
- Hệ thống xử lý: Loại bỏ từ thừa, chuẩn hóa format
- Kết quả: "thời gian làm việc nhân viên"

### 4.3 Bước 3: Tìm Kiếm Vector

```python
# Trong LegalQAPipeline.predict() - 3-Tier Architecture thực tế
def predict(
    self,
    query: str,
    top_k_retrieval: Optional[int] = None,
    top_k_light: Optional[int] = None,
    top_k_final: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Process query through 3-tier architecture."""
    
    try:
        # === TIER 1: BI-ENCODER RETRIEVAL ===
        logger.info(f"🎯 Tier 1 - Bi-Encoder retrieval (top_k={top_k_retrieval})")
        retrieval_results = self.retriever.retrieve(query, top_k_retrieval or 100)
        
        if not retrieval_results:
            logger.warning("No retrieval results found")
            return []
        
        # === TIER 2: LIGHT RERANKING ===
        logger.info(f"⚡ Tier 2 - Light reranking (top_k={top_k_light})")
        light_results = self.reranker.rank_light(
            query, retrieval_results[:top_k_light or 80]
        )
        
        # === TIER 3: CROSS-ENCODER ENSEMBLE ===
        logger.info(f"🎯 Tier 3 - Cross-encoder ensemble (top_k={top_k_final})")
        final_results = self.reranker.rank_cross(
            query, light_results[:20]  # Top 20 for final ranking
        )
        
        # Sort by final score và return top_k_final
        final_results = sorted(
            final_results, key=lambda x: x["final_score"], reverse=True
        )[:top_k_final]
        
        logger.info(f"✅ Final ranking completed. Returning {len(final_results)} results")
        return final_results
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return []
```

**Ví dụ thực tế:**
- Input query: "Quy định về thời gian làm việc"
- Tier 1: Bi-Encoder retrieval → 100 candidates
- Tier 2: Light reranking → 80 candidates  
- Tier 3: Cross-encoder ensemble → 5 final results

### 4.4 Bước 4: Sắp Xếp Lại Kết Quả

```python
# Trong RerankingEngine - 2-tier reranking thực tế
class RerankingEngine:
    """Unified reranking component với light và cross-encoder."""
    
    def rank_light(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Light reranking với PhoBERT model"""
        if not self.is_ready:
            return documents
        
        # Sử dụng light reranker model
        light_model = self.models.get("light_reranker")
        if light_model:
            return self._rank_with_model("light_reranker", query, documents)
        
        return documents
    
    def rank_cross(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-encoder reranking với ensemble model"""
        if not self.is_ready:
            return documents
        
        # Sử dụng cross-encoder model
        cross_model = self.models.get("cross_encoder")
        if cross_model:
            return self._rank_with_model("cross_encoder", query, documents)
        
        return documents
    
    def _rank_with_model(self, model_name: str, query: str, documents: List[Dict]) -> List[Dict]:
        """Rank documents với specific model."""
        model_info = self.models[model_name]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Calculate scores for each document
        for doc in documents:
            inputs = tokenizer(
                query, doc["content"],
                truncation=True, padding=True,
                max_length=256, return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.softmax(outputs.logits, dim=1)[0, 1].item()
                doc[f"{model_name}_score"] = score
        
        # Sort by model score
        return sorted(documents, key=lambda x: x[f"{model_name}_score"], reverse=True)
```

**Ví dụ thực tế:**
- Input: 80 candidates từ Tier 2 light reranking
- Cross-encoder: Ensemble model với ADAPT enhancement
- Output: Top 5 kết quả cuối cùng được sắp xếp theo final_score

---

## 5. Xử Lý Lỗi và Fallback

### 5.1 Các Loại Lỗi Thường Gặp

#### 5.1.1 Lỗi Model Loading
```python
try:
    self.bi_encoder = SentenceTransformer(bi_encoder_path)
except Exception as e:
    logger.error(f"Failed to load bi-encoder: {e}")
    # Fallback: Sử dụng model mặc định
    self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
```

#### 5.1.2 Lỗi FAISS Index
```python
try:
    self.faiss_index = faiss.read_index(faiss_index_path)
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    # Fallback: Sử dụng search đơn giản
    self.use_simple_search = True
```

#### 5.1.3 Lỗi Content Mapping
```python
if not self.corpus_content:
    logger.warning("Content map not available, using fallback")
    # Fallback: Trả về thông tin cơ bản
    return self._fallback_retrieval(query, top_k)
```

### 5.2 Chiến Lược Fallback

1. **Graceful Degradation**: Hạ cấp chức năng thay vì dừng hoàn toàn
2. **Multiple Fallback Levels**: Nhiều tầng fallback khác nhau
3. **User Notification**: Thông báo rõ ràng về tình trạng hệ thống
4. **Automatic Recovery**: Tự động khôi phục khi có thể

---

## 6. Tối Ưu Hóa Hiệu Suất

### 6.1 Caching Strategy

#### 6.1.1 Pipeline Caching (Actual Implementation)
```python
@st.cache_resource(ttl=config.app.cache_pipeline_ttl_seconds)
def load_pipeline(force_cpu=False):
    """Load and cache pipeline to avoid reloading on each interaction."""
    logger.info("Loading pipeline (force_cpu=%s)", force_cpu)
    try:
        # Set CUDA environment variables to avoid issues
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Force CPU if requested
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("Using CPU mode")
            st.info("Đang sử dụng CPU mode")

        # Memory cleanup before loading
        gc.collect()

        start_time = time.time()
        pipeline = LegalQAPipeline()
        load_time = time.time() - start_time

        if not pipeline.is_ready:
            logger.error("Pipeline failed to initialize")
            return None

        logger.info(f"Pipeline loaded successfully in {load_time:.2f}s")
        return pipeline

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None
```

#### 6.1.2 Question Caching (Actual Implementation)
```python
@st.cache_data(ttl=config.app.cache_questions_ttl_seconds, show_spinner=False)
def load_random_questions() -> Tuple[List[str], List[str]]:
    """Load random questions from datasets for UI display."""
    try:
        # Load training questions
        training_questions = []
        test_questions = []
        
        # Load từ validation sets
        for tier in ["tier1", "tier2", "tier3"]:
            try:
                file_path = Path(f"features/validation_sets/{tier}_validation.jsonl")
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            if "question" in data:
                                training_questions.append(data["question"])
            except Exception as e:
                logger.warning(f"Failed to load {tier} questions: {e}")

        # Shuffle và return
        if training_questions:
            random.shuffle(training_questions)
            training_questions = training_questions[:20]  # Limit to 20 questions
        
        return training_questions, test_questions

    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return [], []
```

### 6.2 Memory Management (Actual Implementation)

#### 6.2.1 Garbage Collection
```python
# Memory cleanup before loading - từ search.py
gc.collect()

# Pipeline cleanup method
def cleanup(self):
    """Clean up resources used by the pipeline."""
    try:
        if hasattr(self, "retriever"):
            self.retriever.cleanup()
        if hasattr(self, "reranker"):
            self.reranker.cleanup()

    except Exception as e:
        logger.warning(f"Error during pipeline cleanup: {e}")
```

#### 6.2.2 Parameter Optimization
```python
def calculate_optimal_parameters(
    final_results_count: int, search_aggressiveness: str
) -> Dict[str, int]:
    """Calculate optimal search parameters based on requirements."""
    
    # Base parameters từ config
    base_retrieval = 100
    base_light = 80
    
    # Adjust based on aggressiveness
    if search_aggressiveness == "Cao":
        multiplier = 1.5
    elif search_aggressiveness == "Trung bình":
        multiplier = 1.0
    else:  # "Thấp"
        multiplier = 0.7
    
    # Calculate optimal values
    optimal_retrieval = int(base_retrieval * multiplier)
    optimal_light = int(base_light * multiplier)
    
    return {
        "top_k_retrieval": max(final_results_count * 10, optimal_retrieval),
        "top_k_light_reranking": max(final_results_count * 5, optimal_light),
        "top_k_final": final_results_count,
    }
```

---

## 7. Ví Dụ Thực Tế (Complete Flow)

### 7.1 Luồng Xử Lý Hoàn Chỉnh

**Input:** User nhập câu hỏi "Quy định về thời gian làm việc của nhân viên"

#### Bước 1: Khởi tạo và Validation
```python
# Streamlit app khởi động
pipeline = load_pipeline()  # Cached với TTL=7200s
if pipeline.is_ready:
    logger.info("✅ Pipeline ready for queries")
```

#### Bước 2: Parameter Calculation
```python
# User chọn: final_results_count=5, search_aggressiveness="Trung bình"
params = calculate_optimal_parameters(5, "Trung bình")
# Result: {
#   "top_k_retrieval": 100,
#   "top_k_light_reranking": 80, 
#   "top_k_final": 5
# }
```

#### Bước 3: 3-Tier Processing
```python
# Pipeline.predict() được gọi với parameters
results = pipeline.predict(
    query="Quy định về thời gian làm việc của nhân viên",
    top_k_retrieval=100,    # Tier 1: Bi-Encoder
    top_k_light=80,         # Tier 2: Light Reranker  
    top_k_final=5           # Tier 3: Cross-Encoder Ensemble
)

# Tier 1: Bi-Encoder Retrieval
# - Input: "Quy định về thời gian làm việc của nhân viên"
# - Process: Vector encoding + FAISS search
# - Output: 100 candidates với retrieval_score

# Tier 2: Light Reranking
# - Input: Top 80 từ Tier 1
# - Process: PhoBERT-base-v2 reranking
# - Output: 80 candidates với light_reranker_score

# Tier 3: Cross-Encoder Ensemble
# - Input: Top 20 từ Tier 2
# - Process: Ensemble (70% ADAPT + 30% Base)
# - Output: 5 final results với final_score
```

#### Bước 4: Result Display
```python
# Display results trong Streamlit UI
for i, result in enumerate(results):
    st.markdown(f"### 📄 Kết quả {i+1}")
    st.markdown(f"**AID:** {result['aid']}")
    st.markdown(f"**Nội dung:** {result['content'][:200]}...")
    
    # Score display với actual field names
    st.markdown(f"**Retrieval Score:** {result.get('retrieval_score', 0):.3f}")
    st.markdown(f"**Light Reranker Score:** {result.get('light_reranker_score', 0):.3f}")
    st.markdown(f"**Cross Encoder Score:** {result.get('cross_encoder_score', 0):.3f}")
    st.markdown(f"**Final Score:** {result.get('final_score', 0):.3f}")
```

### 7.2 Performance Monitoring

```python
# Performance tracking trong search.py
start_time = time.time()
with st.spinner("🔎 Đang tìm kiếm... vui lòng đợi"):
    results = pipeline.predict(query, **params)
end_time = time.time()

processing_time = end_time - start_time
logger.info(f"Query processed in {processing_time:.2f}s, found {len(results)} results")

# Display performance metrics
st.success(f"✅ Tìm thấy {len(results)} kết quả trong {processing_time:.2f}s")
```

### 7.3 Error Handling Example

```python
# Error handling trong pipeline.predict()
try:
    # === TIER 1: BI-ENCODER RETRIEVAL ===
    retrieval_results = self.retriever.retrieve(query, top_k_retrieval or 100)
    
    if not retrieval_results:
        logger.warning("No retrieval results found")
        return []
    
    # === TIER 2 & 3: Continue processing ===
    # ...
    
except Exception as e:
    logger.error(f"Prediction failed: {e}", exc_info=True)
    return []  # Return empty results instead of crashing
```

---

**Tài liệu này được cập nhật với source code thực tế từ LawBot v8.3**  
**Ngày cập nhật: 2025-08-21**  
**Phiên bản: Complete Flow Documentation**