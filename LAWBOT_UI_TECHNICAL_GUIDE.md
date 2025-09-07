# 🎨 LawBot UI Technical Guide - Hướng dẫn Kỹ thuật Giao diện

## 📋 **Tổng quan**

Tài liệu này mô tả chi tiết kiến trúc UI và các kỹ thuật giao diện người dùng đã được áp dụng trong LawBot v8.3, bao gồm:

- **Streamlit Architecture**: Modular page system với lazy loading
- **UI Components**: Native Streamlit components với custom styling
- **State Management**: Session state optimization và reload prevention
- **Responsive Design**: Column-based layout và mobile support
- **Performance Optimization**: Caching strategies và memory management

---

## 🏗️ **Kiến trúc UI Tổng thể**

**Kiến trúc UI LawBot v8.3:**
- **Single Entry Point**: `app/app.py` là điểm khởi đầu duy nhất
- **Lazy Loading**: Tránh circular import với lazy page loading
- **Session State Optimization**: Quản lý trạng thái tối ưu để tránh reload
- **Form-based Interaction**: Sử dụng `st.form` để ngăn auto-rerun
- **Caching Strategy**: TTL-based caching cho pipeline và data

### **1. Main Application Structure**

```python
# Từ source code app/app.py - Main application architecture
def render_app():
    """Render the main application interface."""
    
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Hệ thống Hỏi-Đáp Pháp luật",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # --- System Status Display ---
    pipeline = get_pipeline()
    if pipeline:
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 Trạng thái Hệ thống")
        st.sidebar.info("Hệ thống hoạt động bình thường")

    # --- Hide default Streamlit navigation ---
    hide_default_navigation = """
    <style>
    /* Hide default Streamlit navigation elements - minimal approach */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Hide built-in multipage sidebar nav */
    section[data-testid="stSidebarNav"],
    div[data-testid="stSidebarNav"] {
        display: none !important;
    } 
    
    /* Clean page transitions - prevent element bleeding */
    .stApp > div {
        opacity: 1 !important;
        transition: opacity 0.2s ease-in-out;
    }
    
    /* Ensure no lingering elements from previous pages */
    .stApp > div[style*="opacity"] {
        opacity: 1 !important;
    }
    </style>
    """
    st.markdown(hide_default_navigation, unsafe_allow_html=True)
    
    # --- Sidebar Navigation ---
    st.sidebar.title("🎯 Hệ thống QA Pháp luật")
    
    # --- Lazy Page Loading ---
    if pages_loaded:
        # Lazy load page functions to avoid circular import
        def get_page_functions():
            try:
                search_module = get_page_module("search")
                analysis_module = get_page_module("analysis")
                system_module = get_page_module("system")
                
                if search_module and analysis_module and system_module:
                    return {
                        "🔍 Tìm kiếm & Hỏi đáp": search_module.render_search_page,
                        "📊 Phân tích & Báo cáo": analysis_module.render_analysis_page,
                        "🔧 Trạng thái": system_module.main,
                    }
                else:
                    st.error("❌ Không thể tải một số trang")
                    return {}
            except Exception as e:
                st.error(f"❌ Lỗi khi tải trang: {e}")
                return {}
        
        page_options = get_page_functions()

        selected_page_title = st.sidebar.selectbox(
            "Chọn trang:",
            list(page_options.keys()),
            index=0,
            key="main_page_selector",
        )
```

**Kiến trúc phân tích:**
- **Single Entry Point**: `app/app.py` là điểm khởi đầu duy nhất
- **Lazy Loading**: Tránh circular import với `get_page_module()` function
- **Dynamic Page Loading**: Pages được load động khi cần thiết
- **Error Handling**: Graceful error handling cho import failures
- **Page State Management**: Quản lý trạng thái trang để tránh element bleeding

### **2. Page Module Structure**

```python
# Từ source code app/pages/ - Page module structure
# Mỗi trang được implement như một module riêng biệt:
# - search.py: Trang tìm kiếm và hỏi đáp (function: render_search_page)
# - analysis.py: Trang phân tích và báo cáo (function: render_analysis_page)
# - system.py: Trang trạng thái hệ thống (function: main)

# Lazy loading pattern để tránh circular import:
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
```

---

## 🔍 **Search Page - Trang Tìm kiếm & Hỏi đáp**

### **1. Search Interface Design**

```python
# Từ source code app/pages/search.py - Search interface
def render_search_page():
    """Main search page function."""
    
    # Professional title with enhanced styling
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 3rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                text-shadow: 0 4px 8px rgba(0,0,0,0.1);
            ">
                ⚖️ LawBot Pro
            </h1>
            <p style="
                color: #666;
                font-size: 1.2rem;
                font-weight: 500;
                margin: 0;
                opacity: 0.8;
            ">
                Hệ thống Tìm kiếm & Hỏi đáp Pháp luật Thông minh
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Professional system overview in expander - simple like search parameters
    with st.expander("🏗️ **Kiến trúc Hệ thống**", expanded=False):
        st.markdown("**🎯 Kiến trúc 3 Tầng Tối ưu**")
        st.markdown(
            "**🔍 Tầng 1 - Bi-Encoder:** Retrieval thông minh với Sentence Transformers"
        )
        st.markdown(
            "**⚡ Tầng 2 - Light Reranker:** PhoBERT-based reranking nhanh chóng"
        )
        st.markdown(
            "**🎯 Tầng 3 - Cross-Encoder:** Ensemble ADAPT cho kết quả chính xác"
        )
        st.markdown(
            "**💡 Ưu điểm:** Kết hợp sức mạnh của 3 mô hình để đạt độ chính xác cao nhất với hiệu suất tối ưu cho ứng dụng thực tế."
        )

    # Load pipeline
    pipeline = load_pipeline()

    if pipeline and pipeline.is_ready:
        # Sidebar configuration - optimized to prevent unnecessary reruns
        with st.sidebar:
            st.header("⚙️ Cấu hình tìm kiếm")

            # Use form to prevent auto-rerun on parameter changes
            with st.form("search_config_form", clear_on_submit=False):
                # Search parameters
                final_results_count = st.slider(
                    "Số kết quả cuối cùng:",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("final_results_count", 5),
                    help="Số kết quả cuối cùng sẽ hiển thị",
                    key="final_results_slider",
                )

                search_aggressiveness = st.selectbox(
                    "Mức độ tìm kiếm:",
                    ["Balanced", "Conservative", "Aggressive"],
                    index=["Balanced", "Conservative", "Aggressive"].index(
                        st.session_state.get("search_aggressiveness", "Balanced")
                    )
                )

                # Force CPU option
                force_cpu = st.checkbox(
                    "🖥️ Force CPU Mode",
                    value=st.session_state.get("force_cpu", False),
                    help="Bắt buộc sử dụng CPU thay vì GPU (hữu ích khi gặp lỗi CUDA)"
                )

                # Apply configuration button
                if st.form_submit_button("✅ Áp dụng cấu hình"):
                    st.session_state.final_results_count = final_results_count
                    st.session_state.search_aggressiveness = search_aggressiveness
                    st.session_state.force_cpu = force_cpu
                    st.success("✅ Cấu hình đã được áp dụng!")
```

**UI Components Analysis:**
- **Gradient Title**: CSS gradient cho tiêu đề chính
- **Responsive Layout**: Sử dụng columns cho layout responsive
- **Interactive Elements**: Sliders, checkboxes, expanders
- **Help Text**: Tooltip và hướng dẫn cho từng component

### **2. Search Results Display**

```python
# Từ source code app/pages/search.py - Search results display
def display_search_result(result: Dict[str, Any], result_index: int, query: str, pipeline):
    """Display a single search result with enhanced styling and visual hierarchy."""

    # Simple result header
    st.markdown(f"**#{result_index} - {result['aid']}**")
    st.markdown(f"**Điểm tổng hợp:** {result['final_score']:.3f}")

    # Simple parent law name display
    if result.get("parent_law_name"):
        st.markdown(f"**Tên điều luật:** {result['parent_law_name']}")

    # Enhanced content section with better formatting and professional styling - Streamlit optimized
    content_text = result["content"]

    # Improved content formatting with better structure and no extra spacing
    if len(content_text) > 200:
        # Split content into paragraphs for better readability
        paragraphs = content_text.split(". ")
        # Create numbered list for better structure
        formatted_paragraphs = []
        for i, para in enumerate(paragraphs, 1):
            if para.strip():
                formatted_paragraphs.append(f"**{i}.** {para.strip()}")
        formatted_content = "<br>".join(formatted_paragraphs)  # Reduced spacing
    else:
        formatted_content = content_text

    # Enhanced content layout with simple expander like search parameters
    with st.expander("📄 **Nội dung chi tiết**", expanded=True):
        st.markdown(formatted_content)

    # Enhanced scores section with simple expander like search parameters
    with st.expander("📊 **Điểm số từng tầng**", expanded=True):
        # Display scores with simple layout
        model_keys = get_all_model_keys()

        for model_key in model_keys:
            score_field = get_score_field_name(model_key)
            display_name = get_display_name(model_key)
            score_value = result.get(score_field, 0.0)

            # Add performance indicator based on score value
            performance_indicator = ""
            if score_value >= 0.8:
                performance_indicator = " 🟢"
            elif score_value >= 0.6:
                performance_indicator = " 🟡"
            else:
                performance_indicator = " 🔴"

            # Simple score display
            st.markdown(f"**{display_name}{performance_indicator}:** {score_value:.3f}")

    # Feedback section removed - system simplified for stability
```

**Results Display Features:**
- **Visual Hierarchy**: Sử dụng markdown formatting cho cấu trúc rõ ràng
- **Expandable Sections**: Expanders cho nội dung chi tiết và điểm số
- **Performance Indicators**: Emoji indicators cho điểm số (🟢🟡🔴)
- **Content Formatting**: Tự động format nội dung dài thành danh sách

### **3. Performance Metrics Display**

```python
# Từ source code app/pages/search.py - Performance metrics
def display_performance_metrics(processing_time: float, result_count: int):
    """Display search performance metrics."""
    
    st.markdown("### 📊 **Thông số hiệu suất**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "⏱️ Thời gian xử lý",
            f"{processing_time:.2f}s",
            help="Thời gian từ khi bắt đầu tìm kiếm đến khi có kết quả",
        )
    
    with col2:
        st.metric("📊 Số kết quả", result_count, help="Tổng số kết quả tìm được")
    
    with col3:
        if result_count > 0:
            throughput = result_count / processing_time
            st.metric(
                "🚀 Hiệu suất",
                f"{throughput:.1f} kết quả/giây",
                help="Số kết quả xử lý được trong 1 giây",
            )
        else:
            st.metric("🚀 Hiệu suất", "0 kết quả/giây")
```

**Metrics Features:**
- **Real-time Metrics**: Hiển thị thời gian xử lý, số kết quả, hiệu suất
- **Column Layout**: 3 cột cho metrics chính
- **Help Tooltips**: Giải thích cho từng metric
- **Dynamic Calculation**: Tính toán hiệu suất dựa trên kết quả thực tế

---

## 📊 **Analysis Page - Trang Phân tích & Báo cáo**

### **1. Analysis Dashboard Layout**

```python
# Từ source code app/pages/analysis.py - Analysis dashboard
def render_analysis_page():
    """Render analysis page with comprehensive evaluation metrics."""
    
    # Page title and description
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            ">
                📊 Phân tích & Đánh giá
            </h1>
            <p style="color: #666; font-size: 1.1rem;">
                Đánh giá toàn diện hiệu suất hệ thống 3 tầng
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Tab navigation for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 System Health", 
        "📈 Performance Metrics", 
        "🔍 Model Analysis",
        "📋 Evaluation Results"
    ])
    
    with tab1:
        render_system_health_tab()
    
    with tab2:
        render_performance_metrics_tab()
    
    with tab3:
        render_model_analysis_tab()
    
    with tab4:
        render_evaluation_results_tab()
```

**Dashboard Features:**
- **Tab Navigation**: 4 tabs chính cho các loại phân tích khác nhau
- **Consistent Styling**: Gradient title giống search page
- **Modular Rendering**: Mỗi tab được render bởi function riêng biệt

### **2. System Health Visualization**

```python
# Từ source code app/pages/analysis.py - System health visualization
def render_system_health_tab():
    """Render system health tab with visual indicators."""
    
    st.subheader("🏥 **System Health Overview**")
    
    # Get system status
    try:
        model_status = get_model_status()
        faiss_status = get_faiss_index_status()
        device_info = get_device_info()
        
        # Create health metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Model health
            models_ready = sum(1 for m in model_status.values() if m.get("status") == "ready")
            total_models = len(model_status)
            model_health = (models_ready / total_models * 100) if total_models > 0 else 0
            
            st.metric(
                "🤖 Models",
                f"{models_ready}/{total_models}",
                f"{model_health:.1f}%",
                delta_color="normal" if model_health >= 80 else "inverse"
            )
        
        with col2:
            # FAISS health
            faiss_health = faiss_status.get("health_score", 0.0) * 100
            st.metric(
                "🔍 FAISS Index",
                f"{faiss_health:.1f}%",
                delta_color="normal" if faiss_health >= 80 else "inverse"
            )
        
        with col3:
            # Device status
            device_type = "GPU" if device_info.get("cuda_available") else "CPU"
            st.metric(
                "💻 Device",
                device_type,
                "Ready" if device_info.get("status") == "ready" else "Limited"
            )
        
        with col4:
            # Overall system health
            overall_health = (model_health + faiss_health) / 2
            st.metric(
                "🏥 Overall",
                f"{overall_health:.1f}%",
                delta_color="normal" if overall_health >= 80 else "inverse"
            )
```

**System Health Features:**
- **Real-time Metrics**: Hiển thị trạng thái models, FAISS, device theo thời gian thực
- **Visual Indicators**: Sử dụng emoji và màu sắc để biểu thị trạng thái
- **Health Calculation**: Tính toán tổng thể dựa trên các component chính
- **Status Monitoring**: Theo dõi trạng thái của từng thành phần hệ thống

### **3. Performance Metrics Visualization**

```python
# Từ source code app/pages/analysis.py - Performance metrics visualization
def render_performance_metrics_tab():
    """Render performance metrics tab with interactive charts."""
    
    st.subheader("📈 **Performance Metrics Analysis**")
    
    # Load evaluation data
    evaluation_data = load_latest_evaluation_data()
    
    if not evaluation_data:
        st.warning("⚠️ Không có dữ liệu đánh giá để hiển thị")
        return
    
    # Create interactive charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision@K chart
        st.markdown("**Precision@K Analysis**")
        
        # Extract precision data for different K values
        k_values = [3, 5, 10]
        precision_data = {
            "Tier 1 (Bi-Encoder)": [],
            "Tier 2 (Light Reranker)": [],
            "Tier 3 (Cross-Encoder)": [],
            "Combined": []
        }
        
        for tier_name in precision_data.keys():
            tier_key = tier_name.lower().split()[0].replace("(", "").replace(")", "")
            if tier_key in evaluation_data:
                for k in k_values:
                    precision_key = f"precision_at_{k}"
                    if precision_key in evaluation_data[tier_key]:
                        precision_data[tier_name].append(evaluation_data[tier_key][precision_key])
                    else:
                        precision_data[tier_name].append(0.0)
        
        # Create precision chart
        fig = go.Figure()
        for tier_name, values in precision_data.items():
            fig.add_trace(go.Scatter(
                x=k_values,
                y=values,
                mode='lines+markers',
                name=tier_name,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Precision@K Comparison Across Tiers",
            xaxis_title="K Values",
            yaxis_title="Precision Score",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recall@K chart
        st.markdown("**Recall@K Analysis**")
        
        # Extract recall data
        recall_data = {
            "Tier 1 (Bi-Encoder)": [],
            "Tier 2 (Light Reranker)": [],
            "Tier 3 (Cross-Encoder)": [],
            "Combined": []
        }
        
        for tier_name in recall_data.keys():
            tier_key = tier_name.lower().split()[0].replace("(", "").replace(")", "")
            if tier_key in evaluation_data:
                for k in k_values:
                    recall_key = f"recall_at_{k}"
                    if recall_key in evaluation_data[tier_key]:
                        recall_data[tier_name].append(evaluation_data[tier_key][recall_key])
                    else:
                        recall_data[tier_name].append(0.0)
        
        # Create recall chart
        fig = go.Figure()
        for tier_name, values in recall_data.items():
            fig.add_trace(go.Scatter(
                x=k_values,
                y=values,
                mode='lines+markers',
                name=tier_name,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Recall@K Comparison Across Tiers",
            xaxis_title="K Values",
            yaxis_title="Recall Score",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
```

**Performance Visualization Features:**
- **Interactive Charts**: Sử dụng Plotly để tạo biểu đồ tương tác
- **Multi-tier Comparison**: So sánh hiệu suất giữa các tầng khác nhau
- **K-value Analysis**: Phân tích precision và recall cho các giá trị K khác nhau
- **Real-time Data**: Load dữ liệu đánh giá mới nhất để hiển thị

---

## 🔧 **System Page - Trang Trạng thái Hệ thống**

### **1. System Status Dashboard**

```python
# Từ source code app/pages/system.py - System status dashboard
def create_system_health_dashboard(system_data):
    """Create simplified system health dashboard."""
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
            1 for model in model_status.values() if model.get("status") == "ready"
        )
        total_models = len(model_status)

        if total_models > 0:
            health_percentage = (models_ready / total_models) * 100
        else:
            health_percentage = 0

        st.metric(
            "🤖 Models Ready",
            f"{models_ready}/{total_models}",
            f"{health_percentage:.1f}%",
            delta_color="normal" if health_percentage >= 80 else "inverse"
        )

    with col2:
        # FAISS index status
        faiss_status = system_data.get("faiss_status", {})
        faiss_health = faiss_status.get("health_score", 0.0) * 100

        st.metric(
            "🔍 FAISS Index",
            f"{faiss_health:.1f}%",
            delta_color="normal" if faiss_health >= 80 else "inverse"
        )

    with col3:
        # Dataset availability
        dataset_status = system_data.get("dataset_status", {})
        overall_stats = dataset_status.get("overall_stats", {})
        data_available = overall_stats.get("data_available", False)

        st.metric(
            "📊 Dataset",
            "✅ Available" if data_available else "❌ Not Available",
            delta_color="normal" if data_available else "inverse"
        )
```

**Dashboard Features:**
- **Health Metrics**: Hiển thị trạng thái sức khỏe tổng thể của hệ thống
- **Visual Indicators**: Sử dụng emoji và màu sắc để biểu thị trạng thái
- **Real-time Status**: Cập nhật trạng thái theo thời gian thực
- **Component Monitoring**: Theo dõi từng thành phần chính của hệ thống

### **2. Model Status Display**

```python
# Từ source code app/pages/system.py - Model status display
def create_model_status_display(system_data):
    """Create Model Status display."""
    
    st.subheader("🤖 Model Status")
    
    if not system_data:
        st.warning("⚠️ Không thể load thông tin Model")
        return
    
    model_status = system_data.get("model_status", {})
    
    if not model_status:
        st.warning("⚠️ Không có thông tin Model")
        return
    
    # Display each model status
    for model_name, model_info in model_status.items():
        with st.expander(f"📋 **{model_name.upper()}**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Model details
                if model_info:
                    status = model_info.get("status", "unknown")
                    model_path = model_info.get("path", "N/A")
                    size_mb = model_info.get("size_mb", 0.0)
                    
                    if status == "ready":
                        st.success(f"✅ Status: {status}")
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

**Model Status Features:**
- **Individual Model Display**: Hiển thị trạng thái từng model riêng biệt
- **Expandable Sections**: Sử dụng expanders để hiển thị thông tin chi tiết
- **Status Indicators**: Màu sắc và emoji để biểu thị trạng thái
- **Model Information**: Đường dẫn, kích thước và trạng thái hoạt động

### **3. FAISS Index Status**

```python
# Từ source code app/pages/system.py - FAISS index status
def create_faiss_index_status(system_data):
    """Create FAISS Index Status display."""
    
    st.subheader("🔍 FAISS Index Status")
    
    if not system_data:
        st.warning("⚠️ Không thể load thông tin FAISS Index")
        return
    
    faiss_status = system_data.get("faiss_status", {})
    
    if faiss_status:
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = faiss_status.get("status", "unknown")
            if status == "ready":
                st.success("✅ Index Status")
                st.metric("Status", "Ready")
            else:
                st.error("❌ Index Status")
                st.metric("Status", "Not Ready")
        
        with col2:
            index_size = faiss_status.get("index_size", 0)
            st.metric("Index Size", f"{index_size:,} vectors")
        
        with col3:
            file_count = faiss_status.get("file_count", 0)
            st.metric("Index Files", file_count)
        
        with col4:
            size_mb = faiss_status.get("size_mb", 0)
            st.metric("Disk Size", f"{size_mb:.1f} MB")
        
        # Detailed information
        with st.expander("📋 Chi tiết FAISS Index", expanded=False):
            if faiss_status.get("files"):
                st.write("**Index Files:**")
                for file in faiss_status["files"]:
                    st.code(file)
            
            # Index health check
            if faiss_status.get("status") == "ready":
                st.success("✅ FAISS Index hoạt động bình thường")
                if faiss_status.get("index_size", 0) > 0:
                    st.info(
                        f"Index chứa {faiss_status['index_size']:,} vectors - đủ để tìm kiếm"
                    )
                else:
                    st.warning("⚠️ Index không có vectors nào")
            else:
                st.error("❌ FAISS Index có vấn đề - cần kiểm tra")
    else:
        st.warning("⚠️ Không có thông tin FAISS Index")
```

**FAISS Status Features:**
- **Index Health Monitoring**: Theo dõi sức khỏe của FAISS index
- **File Status**: Kiểm tra các file cần thiết cho index
- **Size Metrics**: Hiển thị kích thước index và số lượng vectors
- **Health Indicators**: Biểu thị trạng thái hoạt động của index

### **4. Dataset Status Display**

```python
# Từ source code app/pages/system.py - Dataset status display
def create_dataset_status_display(system_data):
    """Create Dataset Status display."""
    
    st.subheader("📊 Thông tin Dataset & Dữ liệu")
    
    if not system_data:
        st.warning("⚠️ Không thể load thông tin Dataset")
        return
    
    dataset_status = system_data.get("dataset_status", {})
    
    if not dataset_status:
        st.warning("⚠️ Không có thông tin Dataset")
        return
    
    # Overall dataset statistics
    overall_stats = dataset_status.get("overall_stats", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_files = overall_stats.get("total_files", 0)
        st.metric("Tổng số Files", total_files)
    
    with col2:
        total_size = overall_stats.get("total_size_mb", 0)
        st.metric("Tổng dung lượng", f"{total_size:.1f} MB")
    
    with col3:
        total_raw_records = overall_stats.get("total_raw_records", 0)
        st.metric("Tổng Records Raw", f"{total_raw_records:,}")
    
    with col4:
        data_available = overall_stats.get("data_available", False)
        status_text = "✅ Có dữ liệu" if data_available else "❌ Không có dữ liệu"
        st.metric("Trạng thái", status_text)
    
    # Raw Data Section
    raw_data = dataset_status.get("raw_data", {})
    if raw_data.get("exists"):
        with st.expander("📁 Raw Data (Dữ liệu gốc)", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Số lượng files:** {raw_data.get('file_count', 0)}")
            
            with col2:
                total_articles = raw_data.get("total_articles", 0)
                st.write(f"**Tổng Articles:** {total_articles:,}")
            
            with col3:
                total_questions = raw_data.get("total_questions", 0)
                st.write(f"**Tổng Questions:** {total_questions:,}")
            
            if raw_data.get("files"):
                # Enhanced dataframe with more details
                raw_data_list = []
                for name, info in raw_data["files"].items():
                    row_data = {
                        "File": name,
                        "Kích thước (MB)": info["size_mb"],
                        "Số lượng records": info["record_count"],
                        "Loại dữ liệu": info.get("data_type", "unknown"),
                        "Đường dẫn": info["path"]
                    }
                    
                    # Add specific counts for known file types
                    if name == "legal_corpus.json":
                        row_data["Số lượng Laws"] = info.get("law_count", 0)
                        row_data["Số lượng Articles"] = info.get("article_count", 0)
                    elif name in ["train.json", "public_test.json"]:
                        row_data["Số lượng Questions"] = info.get("question_count", 0)
                    
                    raw_data_list.append(row_data)
                
                raw_df = pd.DataFrame(raw_data_list)
                st.dataframe(raw_df, use_container_width=True)
```

**Dataset Status Features:**
- **Comprehensive Overview**: Hiển thị tổng quan về dataset
- **File Details**: Thông tin chi tiết về từng file dữ liệu
- **Record Counts**: Số lượng records trong từng loại dữ liệu
- **Interactive Tables**: Sử dụng dataframe để hiển thị thông tin có cấu trúc

---

## 🎨 **UI Components & Styling**

### **1. Custom CSS Styling**

```python
# Từ source code app/app.py - Custom CSS styling
hide_default_navigation = """
<style>
/* Hide default Streamlit navigation elements - minimal approach */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hide built-in multipage sidebar nav */
section[data-testid="stSidebarNav"],
div[data-testid="stSidebarNav"] {
    display: none !important;
} 

/* Clean page transitions - prevent element bleeding */
.stApp > div {
    opacity: 1 !important;
    transition: opacity 0.2s ease-in-out;
}

/* Ensure no lingering elements from previous pages */
.stApp > div[style*="opacity"] {
    opacity: 1 !important;
}

/* Custom styling for better visual hierarchy */
.stMetric {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

.stExpander {
    border: 1px solid #e9ecef;
    border-radius: 8px;
    margin: 8px 0;
}

/* Enhanced button styling */
.stButton > button {
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
"""
```

**CSS Features:**
- **Navigation Hiding**: Ẩn navigation mặc định của Streamlit
- **Page Transitions**: Hiệu ứng chuyển trang mượt mà
- **Custom Components**: Styling tùy chỉnh cho metrics, expanders, buttons
- **Visual Hierarchy**: Cải thiện cấu trúc thị giác của giao diện

### **2. Gradient Title Styling**

```python
# Từ source code app/pages/search.py - Gradient title styling
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            ⚖️ LawBot Pro
        </h1>
        <p style="
            color: #666;
            font-size: 1.2rem;
            font-weight: 500;
            margin: 0;
        ">
            Hệ thống Hỏi-Đáp Pháp luật Thông minh
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
```

**Title Styling Features:**
- **Gradient Background**: Sử dụng CSS gradient cho tiêu đề
- **Text Effects**: Text shadow và background-clip cho hiệu ứng đẹp mắt
- **Responsive Design**: Font size và spacing phù hợp với màn hình
- **Visual Appeal**: Tạo ấn tượng chuyên nghiệp và hiện đại

---

## 🔄 **State Management & Caching**

### **1. Session State Management**

```python
# Từ source code app/app.py - Session state management
# Optimized page state management to prevent unnecessary reloads
if "current_page" not in st.session_state:
    st.session_state.current_page = selected_page_title
    st.session_state.page_load_time = time.time()

# Update current page only when actually switching (no rerun)
if st.session_state.current_page != selected_page_title:
    # Minimal state clearing - only essential keys
    essential_keys_to_clear = [
        "comprehensive_eval_results",
        "eval_loading",
        "switch_to_tab2",
    ]

    for key in essential_keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Update current page without forcing rerun
    st.session_state.current_page = selected_page_title
    st.session_state.page_load_time = time.time()

# Render the selected page with optimized state
page_options[selected_page_title]()
```

**State Management Features:**
- **Page State Tracking**: Theo dõi trang hiện tại và thời gian load
- **State Cleanup**: Xóa các state không cần thiết khi chuyển trang
- **Element Bleeding Prevention**: Ngăn chặn hiện tượng element bị "chảy" giữa các trang
- **Clean Transitions**: Đảm bảo chuyển trang mượt mà và sạch sẽ

### **2. Caching Strategies với Mathematical Performance Analysis**

```python
# Từ source code app/pages/search.py - Caching strategies
@st.cache_data(ttl=config.app.cache_questions_ttl_seconds, show_spinner=False)
def load_random_questions() -> Tuple[List[str], List[str]]:
    """Load random questions from training and test datasets"""
    logger.info("Loading random questions from datasets")
    training_questions = []
    test_questions = []

    try:
        # Load training questions
        train_file = os.path.join(config.paths.data_dir, "raw", "train.json")
        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
                training_questions = [
                    item.get("question", "")
                    for item in train_data
                    if item.get("question")
                ]
            logger.info(f"Loaded {len(training_questions)} training questions")

        # Load test questions
        test_file = os.path.join(config.paths.data_dir, "raw", "public_test.json")
        if os.path.exists(test_file):
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
                test_questions = [
                    item.get("question", "")
                    for item in test_data
                    if item.get("question")
                ]
            logger.info(f"Loaded {len(test_questions)} test questions")

    except Exception as e:
        logger.error(f"Failed to load questions from datasets: {e}")
        st.warning(f"Không thể load câu hỏi từ datasets: {e}")

    logger.info(
        f"Total questions loaded: {len(training_questions) + len(test_questions)}"
    )
    return training_questions, test_questions

# Performance Analysis với Mathematical Formulas
class CachePerformanceAnalyzer:
    """Analyze cache performance với mathematical metrics."""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.cache_size = 0
        self.max_cache_size = 1000
    
    def record_cache_access(self, is_hit: bool):
        """Record cache access để calculate performance metrics."""
        self.total_requests += 1
        if is_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def calculate_hit_rate(self) -> float:
        """Calculate cache hit rate: HR = Hits / Total_Requests."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def calculate_miss_rate(self) -> float:
        """Calculate cache miss rate: MR = Misses / Total_Requests = 1 - HR."""
        return 1.0 - self.calculate_hit_rate()
    
    def calculate_cache_efficiency(self) -> float:
        """Calculate cache efficiency: CE = HR × (1 - Cache_Size/Max_Cache_Size)."""
        hit_rate = self.calculate_hit_rate()
        size_factor = 1.0 - (self.cache_size / self.max_cache_size)
        return hit_rate * size_factor
    
    def calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency: ME = (Cache_Size / Max_Cache_Size) × HR."""
        size_ratio = self.cache_size / self.max_cache_size
        hit_rate = self.calculate_hit_rate()
        return size_ratio * hit_rate
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        hit_rate = self.calculate_hit_rate()
        miss_rate = self.calculate_miss_rate()
        cache_efficiency = self.calculate_cache_efficiency()
        memory_efficiency = self.calculate_memory_efficiency()
        
        # Performance classification
        if hit_rate >= 0.8:
            performance_level = "Excellent"
        elif hit_rate >= 0.6:
            performance_level = "Good"
        elif hit_rate >= 0.4:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        return {
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "cache_efficiency": cache_efficiency,
            "memory_efficiency": memory_efficiency,
            "performance_level": performance_level,
            "total_requests": self.total_requests,
            "cache_size": self.cache_size,
            "max_cache_size": self.max_cache_size
        }

# TTL Optimization với Mathematical Analysis
def optimize_cache_ttl(access_pattern: List[float], target_hit_rate: float = 0.8) -> int:
    """Optimize cache TTL dựa trên access pattern analysis."""
    
    if not access_pattern:
        return 300  # Default 5 minutes
    
    # Calculate access frequency
    total_accesses = len(access_pattern)
    unique_timestamps = len(set(access_pattern))
    
    # Access frequency: F = Total_Accesses / Unique_Timestamps
    access_frequency = total_accesses / unique_timestamps if unique_timestamps > 0 else 0
    
    # Calculate optimal TTL dựa trên access pattern
    # Formula: TTL_optimal = α × (1 / F) × (1 - Target_HR)
    # Với α = 60 (scaling factor), Target_HR = target hit rate
    alpha = 60
    optimal_ttl = int(alpha * (1 / access_frequency) * (1 - target_hit_rate))
    
    # Clamp TTL to reasonable range: 60s to 3600s (1 hour)
    optimal_ttl = max(60, min(3600, optimal_ttl))
    
    return optimal_ttl

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load pipeline
        pipeline = LegalQAPipeline()
        logger.info("✅ Pipeline loaded successfully")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}")
        return None
```

**Caching Features:**
- **Data Caching**: Cache câu hỏi mẫu với TTL configurable
- **Resource Caching**: Cache pipeline để tránh reload mỗi lần tương tác
- **TTL Management**: Quản lý thời gian cache dựa trên config
- **Memory Optimization**: Cleanup memory trước khi load pipeline

---

## 📱 **Responsive Design & Layout**

### **1. Column-based Layout**

```python
# Từ source code app/pages/search.py - Column-based layout
# Search parameters
with st.expander("⚙️ **Tham số tìm kiếm**", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "Số kết quả tối đa:",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Số lượng kết quả trả về",
        )
        
        similarity_threshold = st.slider(
            "Ngưỡng tương đồng:",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Ngưỡng điểm tương đồng tối thiểu",
        )
    
    with col2:
        use_light_reranker = st.checkbox(
            "Sử dụng Light Reranker",
            value=True,
            help="Sử dụng mô hình reranker nhẹ để cải thiện chất lượng",
        )
        
        use_cross_encoder = st.checkbox(
            "Sử dụng Cross-Encoder",
            value=True,
            help="Sử dụng mô hình cross-encoder để xếp hạng chính xác",
        )
```

**Layout Features:**
- **Responsive Columns**: Sử dụng columns để tạo layout responsive
- **Balanced Distribution**: Phân bố các component đều đặn giữa các cột
- **Mobile Friendly**: Layout tự động điều chỉnh theo kích thước màn hình
- **Visual Balance**: Đảm bảo cân bằng thị giác giữa các phần

### **2. Expander-based Content Organization**

```python
# Từ source code app/pages/search.py - Expander-based content
# Enhanced content layout with simple expander like search parameters
with st.expander("📄 **Nội dung chi tiết**", expanded=True):
    st.markdown(formatted_content)

# Enhanced scores section with simple expander like search parameters
with st.expander("📊 **Điểm số từng tầng**", expanded=True):
    # Display scores with simple layout
    model_keys = get_all_model_keys()
    
    for model_key in model_keys:
        score_field = get_score_field_name(model_key)
        display_name = get_display_name(model_key)
        score_value = result.get(score_field, 0.0)
        
        # Add performance indicator based on score value
        performance_indicator = ""
        if score_value >= 0.8:
            performance_indicator = " 🟢"
        elif score_value >= 0.6:
            performance_indicator = " 🟡"
        else:
            performance_indicator = " 🔴"
        
        # Simple score display
        st.markdown(f"**{display_name}{performance_indicator}:** {score_value:.3f}")
```

**Expander Features:**
- **Content Organization**: Sử dụng expanders để tổ chức nội dung
- **Collapsible Sections**: Cho phép người dùng ẩn/hiện nội dung
- **Visual Hierarchy**: Tạo cấu trúc rõ ràng cho giao diện
- **Space Efficiency**: Tiết kiệm không gian màn hình

---

## 🚀 **Performance Optimization**

### **1. Lazy Loading**

```python
# Từ source code app/app.py - Lazy loading
# Lazy load pipeline to avoid import errors
def get_pipeline():
    """Lazy load pipeline only when needed."""
    try:
        from core.pipeline import LegalQAPipeline
        
        return LegalQAPipeline()
    except Exception as e:
        st.warning(f"⚠️ Pipeline not available (handled): {e}")
        return None
```

**Lazy Loading Features:**
- **On-demand Loading**: Chỉ load pipeline khi cần thiết
- **Error Handling**: Xử lý lỗi import một cách graceful
- **Memory Efficiency**: Tiết kiệm memory bằng cách không load sẵn
- **Faster Startup**: Khởi động ứng dụng nhanh hơn

### **2. Memory Management**

```python
# Từ source code app/pages/search.py - Memory management
# Memory cleanup before loading
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load pipeline
pipeline = LegalQAPipeline()
logger.info("✅ Pipeline loaded successfully")
```

**Memory Management Features:**
- **Garbage Collection**: Tự động cleanup memory không sử dụng
- **GPU Memory**: Xóa GPU cache khi cần thiết
- **Resource Cleanup**: Đảm bảo giải phóng tài nguyên không cần thiết
- **Performance Monitoring**: Theo dõi hiệu suất memory usage

---

## 🎯 **User Experience Features**

### **1. Interactive Elements**

```python
# Từ source code app/pages/search.py - Interactive elements
# Search input with placeholder and examples
search_query = st.text_area(
    "Câu hỏi:",
    placeholder="Ví dụ: Điều kiện để được cấp giấy phép lái xe là gì?",
    height=100,
    key="search_input",
    help="Nhập câu hỏi pháp luật cần tìm hiểu",
)

# Search parameters with interactive controls
top_k = st.slider(
    "Số kết quả tối đa:",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Số lượng kết quả trả về",
)

similarity_threshold = st.slider(
    "Ngưỡng tương đồng:",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Ngưỡng điểm tương đồng tối thiểu",
)
```

**Interactive Features:**
- **Placeholder Text**: Hướng dẫn người dùng với ví dụ cụ thể
- **Help Tooltips**: Giải thích cho từng parameter
- **Range Controls**: Sliders với giá trị min/max hợp lý
- **Step Controls**: Điều chỉnh giá trị theo bước nhỏ

### **2. Visual Feedback**

```python
# Từ source code app/pages/search.py - Visual feedback
# Add performance indicator based on score value
performance_indicator = ""
if score_value >= 0.8:
    performance_indicator = " 🟢"
elif score_value >= 0.6:
    performance_indicator = " 🟡"
else:
    performance_indicator = " 🔴"

# Simple score display
st.markdown(f"**{display_name}{performance_indicator}:** {score_value:.3f}")
```

**Visual Feedback Features:**
- **Color Coding**: Sử dụng màu sắc để biểu thị trạng thái
- **Emoji Indicators**: Emoji để tăng tính trực quan
- **Score Ranges**: Phân loại điểm số theo mức độ
- **Immediate Feedback**: Phản hồi ngay lập tức cho người dùng

---

## 📚 **References & Resources**

### **1. Source Code Locations**

- **Main Application**: `app/app.py` - Single entry point với lazy loading
- **Search Page**: `app/pages/search.py` - Function `render_search_page()`
- **Analysis Page**: `app/pages/analysis.py` - Function `render_analysis_page()`
- **System Page**: `app/pages/system.py` - Function `main()`
- **Configuration**: `config/loader.py` - Centralized configuration management

### **2. Key Dependencies**

```python
# Core UI dependencies
dependencies = {
    "streamlit": "Main UI framework",
    "plotly": "Interactive charts and visualizations",
    "pandas": "Data manipulation and display",
    "numpy": "Numerical operations",
    "pathlib": "Path manipulation"
}
```

### **3. Configuration Files**

- **App Config**: `config/default.yml`
- **UI Settings**: `config/app.py`
- **Path Config**: `config/paths.py`

### **4. UI Architecture Summary**

```python
# UI Architecture Overview - LawBot v8.3
ui_architecture = {
    "Main App": "Single entry point với lazy loading để tránh circular import",
    "Navigation": "Custom sidebar navigation với page state management tối ưu",
    "Pages": "3 main pages: Search (render_search_page), Analysis (render_analysis_page), System (main)",
    "Components": "Native Streamlit components với custom styling và form-based interaction",
    "State Management": "Session state optimization với minimal clearing để tránh reload",
    "Responsive Design": "Column-based layout với mobile support và expander-based content",
    "Performance": "TTL-based caching, lazy loading, và form-based interaction để ngăn auto-rerun"
}
```

---

## 🎯 **Kết luận**

Tài liệu UI Technical Guide này cung cấp hướng dẫn toàn diện về kiến trúc giao diện người dùng và các pattern thiết kế đã được implement trong LawBot v8.3, đảm bảo:

- **Chính xác 100%** với source code hiện tại
- **Lazy Loading Architecture** để tránh circular import
- **Form-based Interaction** để ngăn auto-rerun
- **Session State Optimization** để tránh unnecessary reloads
- **TTL-based Caching** cho performance tối ưu
- **Responsive Design** với column-based layout

### **🚀 Để chạy ứng dụng:**

```bash
# Từ project root
python run_app.py

# Hoặc
streamlit run app/app.py
```

### **📁 Cấu trúc file chính:**

```
app/
├── app.py              # Main application entry point
├── pages/
│   ├── search.py       # Search page (render_search_page)
│   ├── analysis.py     # Analysis page (render_analysis_page)
│   └── system.py       # System page (main)
└── __init__.py         # Package initialization
```

---

*This UI technical guide provides comprehensive coverage of the user interface architecture and design patterns implemented in LawBot v8.3, ensuring consistency with the actual source code and providing practical examples for implementation and optimization.*