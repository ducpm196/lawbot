# LawBot v8.3 - Quick Start Guide (Centralized Management & Optimized Workflow)

## 🚀 **Cài đặt nhanh (Updated: 2025-09-08 - Centralized Management & Optimized Workflow)**

### **1. Clone & Setup**
```bash
# Clone repository
git clone https://github.com/your-org/lawbot.git
cd lawbot

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### **2. Kiểm tra environment**
```bash
# Kiểm tra Python version
python --version  # Python 3.8+

# Kiểm tra CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Kiểm tra dependencies
python -c "import sentence_transformers, faiss, optuna; print('All packages installed')"
```

## 🔄 **Workflow Commands**

### **1. Automated Pipeline (Recommended)**
```bash
# Chạy toàn bộ pipeline với centralized path management
python run_workflow.py --preset full

# Chạy training stages only (Bi-Encoder + Light Ranking + Cross-Encoder)
python run_workflow.py --preset training

# Chạy retrieval stages only (Bi-Encoder + FAISS)
python run_workflow.py --preset retrieval

# Chạy evaluation only
python run_workflow.py --preset evaluation_only

# Force restart (ignore checkpoints)
python run_workflow.py --preset full --force-restart
```

### Lưu ý quan trọng về dữ liệu (v8.3 - Updated: 2025-08-21)
- **Real-data only**: Hệ thống chỉ sử dụng dữ liệu thật được tạo bởi `data_processing/run_preparation.py`
- **Centralized Path Management**: Tất cả đường dẫn được quản lý tập trung tại `config/paths.py`
- **Centralized Model Configuration**: Tất cả cấu hình model được quản lý tập trung tại `config/models.py`
- **Automated Data Discovery**: Hệ thống tự động tìm thư mục processed data mới nhất với timestamp qua `find_latest_processed_data_dir()`
- **Data Freshness Validation**: Workflow tự động kiểm tra tính mới của dữ liệu và re-run `data_preparation` khi cần thiết
- **Dependency Validation**: Workflow tự động validate dependencies giữa các tiers
- **Model Inheritance**: Tier 3 Cross-Encoder inherits ADAPT-enhanced model từ Tier 2 Light Reranker
- **Single ADAPT**: Chỉ PhoBERT-base-v2 có ADAPT (từ Tier 2), PhoBERT-large là base model
- **Advanced HPO**: Hyperparameter optimization với Optuna và early stopping
- **Comprehensive Evaluation**: Multi-tier evaluation với precision, recall, F1, NDCG, MRR, quality metrics
- **Unified Reports Storage**: Consolidated evaluation reports trong single `reports/` directory
- **Config Optimization**: `top_k_retrieval: 20`, `top_k_light: 10`, `top_k_final: 5` theo source code thực tế
- **Quality Score Logic**: Tier-specific thresholds cho từng tier với adjusted scoring
- **Device Management**: Safe device handling với meta tensor và offloaded model support
- **Error Recovery**: Multiple fallback strategies cho model loading và graceful degradation

**Các file dữ liệu cần thiết:**
- `bi_encoder_train.jsonl` (Tier 1)
- `cross_encoder_train.jsonl` (Tier 3)
- `processed_corpus.json` (Tier 1 & 3)
- `training_data.jsonl` (Tier 2)
- `negative_pool.jsonl` (Tier 2)

Hệ thống sẽ tự động kiểm tra bằng `config/paths.validate_training_data_paths()` và log chi tiết trong `run_workflow.py`.

### **2. Manual Step-by-Step**
```bash
# Stage 1: Data Preparation
python data_processing/run_preparation.py

# Stage 2: Bi-Encoder Training
python training/run_bi_encoder.py

# Stage 3: Light Ranking Training
python training/run_light_ranking.py

# Stage 4: Cross-Encoder Training
python training/run_reranker.py

# Stage 5: FAISS Index Creation
python training/run_create_faiss_index.py

# Stage 6: Evaluation
python evaluation/run_evaluation.py
```

### **3. Custom Stage Selection**
```bash
# Chạy specific stages
python run_workflow.py --stages data_preparation bi_encoder

# Chạy với dependencies
python run_workflow.py --stages bi_encoder light_ranking cross_encoder

# Chạy evaluation với trained models
python run_workflow.py --stages evaluation
```

## 🧹 **Environment Management**

### **1. Cleanup & Reset**
```bash
# Xóa checkpoints
rm -rf checkpoints/*

# Xóa models cũ
rm -rf models/*

# Xóa features cũ
rm -rf features/*

# Xóa reports cũ
rm -rf reports/*

# Reset hoàn toàn
python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ['checkpoints', 'models', 'features', 'reports']]"
```

### **2. Selective Cleanup**
```bash
# Xóa specific model
rm -rf models/light-ranking_*

# Xóa specific checkpoint
rm -f checkpoints/light_ranking_training.json

# Xóa specific features
rm -f features/faiss_index.bin
```

### **3. Backup & Restore**
```bash
# Backup models
cp -r models models_backup_$(date +%Y%m%d_%H%M%S)

# Backup features
cp -r features features_backup_$(date +%Y%m%d_%H%M%S)

# Restore from backup
cp -r models_backup_20250816_221146/* models/
```

## 📊 **Monitoring & Debugging**

### **1. Real-time Monitoring**
```bash
# Monitor training progress
tail -f logs/training/*.log

# Monitor specific stage
tail -f logs/training/light_ranking_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### **2. Log Analysis**
```bash
# Tìm errors
grep -r "ERROR\|Exception\|Failed" logs/

# Tìm training progress
grep -r "Epoch.*Loss" logs/training/

# Tìm evaluation results
grep -r "Accuracy\|F1\|NDCG" logs/evaluation/

# Tìm performance metrics
grep -r "Score\|Similarity\|Recall" logs/
```

### **3. Performance Testing**
```bash
# Test retrieval engine
python -c "from core.retrieval import RetrievalEngine; r = RetrievalEngine('models/bi-encoder_20250816_221146', 'features/faiss_index.bin', 'features/aid_map.json', 'features/index_to_aid.json'); print('Index info:', r.get_index_info())"

# Test pipeline
python -c "from core.pipeline import LegalQAPipeline; p = LegalQAPipeline(); results = p.predict('Luật về đất đai?', top_k_final=3); print(f'Results: {len(results)}')"

# Test model loading
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('models/bi-encoder_20250816_221146'); print('Model loaded:', m is not None)"

# Test comprehensive evaluation
python -c "from app.pages.analysis import run_comprehensive_evaluation; results = run_comprehensive_evaluation(None); print('Evaluation completed:', results is not None)"

# Test reports loading
python -c "from app.pages.analysis import load_latest_comprehensive_evaluation; results = load_latest_comprehensive_evaluation(); print('Reports loaded:', results is not None)"
```

## 🏷️ **Model Management**

### **1. Model Status Check**
```bash
# Kiểm tra model status
python -c "from core.utils.system_check import get_model_status; print('Models:', get_model_status())"

# Kiểm tra FAISS index
python -c "from core.utils.system_check import get_faiss_index_status; print('FAISS:', get_faiss_index_status())"

# Kiểm tra device info
python -c "from core.utils.system_check import get_device_info; print('Device:', get_device_info())"

# Kiểm tra centralized paths
python -c "from config.paths import validate_training_data_paths; print('Path validation:', validate_training_data_paths())"

# Kiểm tra centralized model configuration
python -c "from config.models import MODEL_TYPES, MODEL_STATUS_KEYS; print('Model Types:', list(MODEL_TYPES.keys())); print('Status Keys:', MODEL_STATUS_KEYS)"
```

### **2. Model Versioning**
```bash
# List model versions
ls -la models/

# Check model metadata
cat models/bi-encoder_20250816_221146/training_metadata.json

# Check HPO results
cat models/light-ranking_20250816_221255/hpo_results.json

# Check model size
du -sh models/*/
```

### **3. Model Comparison**
```bash
# Compare model performance
python evaluation/run_evaluation.py --compare-models

# Compare specific models
python evaluation/run_evaluation.py --model1 bi-encoder_20250816_221146 --model2 bi-encoder_20250816_203414
```

## 🔧 **Configuration & Tuning (Updated: 2025-08-21 - Quality Score & Config Optimization)**

### **1. Centralized Configuration Management**

#### **1.1 Model Configuration (`config/models.py`)**
```bash
# Kiểm tra model types
python -c "from config.models import MODEL_TYPES; print('Available models:', [m['display_name'] for m in MODEL_TYPES.values()])"

# Kiểm tra model directory mapping
python -c "from config.models import MODEL_DIRECTORY_MAPPING; print('Model directories:', MODEL_DIRECTORY_MAPPING)"

# Lấy display name cho model
python -c "from config.models import get_display_name; print('Bi-Encoder display:', get_display_name('bi_encoder'))"

# Lấy model config
python -c "from config.models import get_model_config; print('Bi-Encoder config:', get_model_config('bi_encoder'))"
```

#### **1.2 Path Configuration (`config/paths.py`)**
```bash
# Validate training data paths
python -c "from config.paths import validate_training_data_paths; result = validate_training_data_paths(); print('Validation result:', result['overall']['status'])"

# Get training data path cho specific tier
python -c "from config.paths import get_training_data_path; print('Bi-Encoder path:', get_training_data_path('bi_encoder'))"

# Get latest processed data directory
python -c "from config.paths import get_latest_processed_data_dir; print('Latest data dir:', get_latest_processed_data_dir())"
```

### **2. HPO (Hyperparameter Optimization)**
```bash
# Run HPO cho Light Ranking
python training/run_light_ranking.py --hpo --trials 100

# Run HPO cho Cross-Encoder
python training/run_reranker.py --hpo --trials 150

# Run HPO với custom study
python training/run_light_ranking.py --hpo --study-name custom_study --trials 200

# Advanced HPO với Optuna
python training/run_bi_encoder.py --hpo --trials 50

# HPO với early stopping
python training/run_light_ranking.py --hpo --trials 100 --early-stopping --patience 10

# Check HPO results
python -c "import json; results = json.load(open('models/light_ranking_latest/hpo_results.json')); print('Best params:', results['best_params']); print('Best score:', results['best_score'])"
```

### **2. Configuration Updates**
```bash
# Update config
python -c "from config.loader import config; print('Current config:', config.model_dump())"

# Validate config
python -c "from config.schemas import validate_config; validate_config()"

# Update specific settings
python -c "import yaml; config = yaml.safe_load(open('config/default.yml')); config['training']['batch_size'] = 32; yaml.dump(config, open('config/default.yml', 'w'))"

# Kiểm tra centralized paths
python -c "from config.paths import TRAINING_DATA_PATHS; print('Training paths:', TRAINING_DATA_PATHS)"

# Kiểm tra centralized model configuration
python -c "from config.models import MODEL_TYPES, MODEL_STATUS_KEYS; print('Model types:', list(MODEL_TYPES.keys())); print('Status keys:', MODEL_STATUS_KEYS)"

# Validate training data paths
python -c "from config.paths import validate_training_data_paths; result = validate_training_data_paths(); print('Path validation:', result['overall']['status'])"

# Get latest processed data directory
python -c "from config.paths import get_latest_processed_data_dir; print('Latest data dir:', get_latest_processed_data_dir())"
```

### **3. Environment Variables**
```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set log level
export LAWBOT_LOG_LEVEL=DEBUG

# Set model cache
export TRANSFORMERS_CACHE=/path/to/cache

# Set FAISS thread count
export OMP_NUM_THREADS=8
```

### **4. Recent Config & Quality Score Updates (2025-08-21)**
```bash
# Kiểm tra config mới theo source code thực tế
python -c "from config.loader import config; print('top_k_retrieval:', config.app.top_k_retrieval); print('top_k_light:', config.app.top_k_light); print('top_k_final:', config.app.top_k_final)"

# Test quality score logic mới
python -c "
from app.pages.analysis import calculate_quality_score
# Test Tier 1 (Retrieval) scores
tier1_scores = [0.52, 0.53, 0.54]
quality = calculate_quality_score(tier1_scores, 3)
print(f'Tier 1 Quality: {quality:.3f} (Expected: 1.0)')
"

# Kiểm tra k_values mới và model inheritance
python -c "
from app.pages.analysis import run_comprehensive_evaluation
print('K-values updated to [3, 5, 10] for realistic evaluation')
print('Model inheritance: Tier 3 inherits ADAPT-enhanced model từ Tier 2')
"
```

## 🚀 **Application Usage**

### **1. Start Streamlit App**
```bash
# Start app
python run_app.py

# Start với custom port
python run_app.py --port 8502

# Start với custom config
python run_app.py --config custom_config.yml
```

### **2. API Usage**
```bash
# Health check
curl http://localhost:8501/health

# Query processing
curl -X POST http://localhost:8501/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Luật về đất đai quy định gì?", "top_k": 5}'

# Model status
curl http://localhost:8501/api/status
```

### **3. Batch Processing**
```bash
# Process multiple queries
python -c "
from core.pipeline import LegalQAPipeline
p = LegalQAPipeline()
queries = ['Luật về đất đai?', 'Quyền sử dụng đất?', 'Thuế đất đai?']
for q in queries:
    results = p.predict(q, top_k_final=3)
    print(f'Query: {q}, Results: {len(results)}')
"
```

## 🐛 **Troubleshooting (Updated: 2025-08-21 - Quality Score & Config Issues)**

### **1. Common Issues**
```bash
# CUDA out of memory (handled)
export CUDA_VISIBLE_DEVICES=""
python training/run_light_ranking.py

# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python training/run_bi_encoder.py

# Port conflicts
lsof -ti:8501 | xargs kill -9
python run_app.py
```

### **2. Debug Commands**
```bash
# Check Python path
python -c "import sys; print('Python path:', sys.path)"

# Check CUDA installation
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Check file permissions
ls -la models/ features/ logs/

# Check disk space
df -h
```

### **3. Recovery Procedures**
```bash
# Reset failed training
rm -f checkpoints/*_training.json
python training/run_light_ranking.py

# Rebuild FAISS index
rm -f features/faiss_index.bin
python training/run_create_faiss_index.py

# Restore from backup
cp -r models/backup/* models/
```

### **4. Recent Quality Score & Config Issues (2025-08-21)**
```bash
# Kiểm tra quality score logic
python -c "
from app.pages.analysis import calculate_quality_score
# Test với scores thấp (Tier 1 & 3)
low_scores = [0.52, 0.53, 0.54]
quality = calculate_quality_score(low_scores, 3)
print(f'Low scores quality: {quality:.3f} (Should be 1.0 for retrieval/ensemble)')
"

# Kiểm tra config consistency
python -c "
from config.loader import config
print(f'Config check: top_k_final={config.app.top_k_final}, k_values=[3,5,10]')
print('Expected: top_k_final=5 for 3-5 results, k_values realistic for evaluation')
"

# Reset quality score cache nếu cần
python -c "
from app.pages.analysis import clear_comprehensive_evaluation_cache
clear_comprehensive_evaluation_cache()
print('Quality score cache cleared')
"
```

## 📈 **Performance Optimization**

### **1. GPU Optimization**
```bash
# Set optimal CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Monitor GPU memory
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

### **2. Memory Optimization**
```bash
# Set optimal batch sizes
export LAWBOT_BATCH_SIZE=16
export LAWBOT_GRADIENT_ACCUMULATION_STEPS=4

# Monitor memory usage
watch -n 1 'free -h && echo "---" && df -h'
```

### **3. Parallel Processing**
```bash
# Run multiple stages in parallel
python training/run_bi_encoder.py &
python training/run_light_ranking.py &
wait

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
python training/run_reranker.py --multi-gpu
```

## 🔍 **Data Analysis**

### **1. Corpus Analysis**
```bash
# Analyze legal corpus
python -c "
import json
from pathlib import Path
corpus_path = Path('features/processed_data/processed_corpus.json')
if corpus_path.exists():
    data = json.load(open(corpus_path))
    print('Total articles:', len(data))
    print('Sample content:', list(data.values())[0][:100] if data else 'N/A')
else:
    print('Corpus not found')
"

# Check data distribution
python -c "
import json
from pathlib import Path
from collections import Counter
corpus_path = Path('features/processed_data/processed_corpus.json')
if corpus_path.exists():
    data = json.load(open(corpus_path))
    print('Total articles:', len(data))
    print('Sample AIDs:', list(data.keys())[:5])
else:
    print('Corpus not found')
"
```

### **2. Feature Analysis**
```bash
# Analyze FAISS index
python -c "
import faiss
from pathlib import Path
index_path = Path('models/faiss_index/faiss_index.bin')
if index_path.exists():
    index = faiss.read_index(str(index_path))
    print('Index size:', index.ntotal)
    print('Dimensions:', index.d)
    print('Is trained:', index.is_trained)
else:
    print('FAISS index not found')
"

# Check training data
python -c "
from pathlib import Path
from core.utils.io import load_jsonl
data_path = Path('features/processed_data/training_data.jsonl')
if data_path.exists():
    data = load_jsonl(str(data_path))
    print('Training samples:', len(data))
    print('Sample structure:', list(data[0].keys()) if data else 'N/A')
else:
    print('Training data not found')
"
```

## 📚 **Documentation & Help**

### **1. Help Commands**
```bash
# Show workflow help
python run_workflow.py --help

# Show training help
python training/run_light_ranking.py --help

# Show evaluation help
python evaluation/run_evaluation.py --help
```

### **2. List Available Options**
```bash
# List workflow stages
python run_workflow.py --list-stages

# List presets
python run_workflow.py --list-presets

# List model types
python -c "from core.utils.system_check import get_model_status; print('Available models:', list(get_model_status().keys()))"
```

### **3. Version Information**
```bash
# Check LawBot version
python -c "import lawbot; print('LawBot version:', lawbot.__version__)"

# Check all package versions
pip list | grep -E "(torch|transformers|faiss|optuna|streamlit)"

# Check git status
git status
git log --oneline -5
```

---

**🚀 Ready to use LawBot v8.3!** 

For detailed information, see [README.md](README.md).
For advanced usage, see the training and evaluation documentation.

## 🔧 **Key Features v8.3 (Updated: 2025-08-21)**

- **3-Tier Architecture**: Bi-Encoder Retrieval, Light Reranker, Cross-Encoder Ensemble
- **Centralized Configuration**: Unified path and model management
- **Automated Workflow**: Smart data validation and stage management
- **Advanced HPO**: Hyperparameter optimization with Optuna
- **Comprehensive Evaluation**: Multi-tier performance metrics với tier-specific quality scoring
- **Real-time Monitoring**: System health and performance tracking
- **Config Optimization**: `top_k_final: 5` phù hợp với yêu cầu 3-5 kết quả cuối cùng
- **Quality Score Logic**: Tier-specific thresholds với adjusted scoring cho từng tier
- **Metrics Calculation**: Sửa logic effective_k để tính chính xác recall và quality scores
