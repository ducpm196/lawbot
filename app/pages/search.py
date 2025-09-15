#!/usr/bin/env python3
"""
Search Page - LawBot Application
===============================

Optimized search functionality with 3-tier architecture support.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import streamlit as st
import random
import gc
import pandas as pd

try:
    from core.pipeline import LegalQAPipeline
    from core.utils.logging_manager import get_logger
    from core.utils.parent_law_manager import ensure_parent_law_mapping
    from config.loader import (
        config,
        get_display_name,
        get_score_field_name,
        get_all_model_keys,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info(
        "Please ensure the project structure is correct and dependencies are installed. "
        "Try running the app from the project root directory."
    )
    st.stop()

# Setup app logger
logger = get_logger("search_page")

# No custom CSS needed - using Streamlit native components


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


def calculate_optimal_parameters(
    final_results_count: int, search_aggressiveness: str
) -> Dict[str, int]:
    """Calculate optimal search parameters based on user preferences and system performance."""
    # Base parameters - Tối ưu hóa Recall: Candidate pool cân bằng
    base_retrieval = 120  # Tăng từ 100 → 150 (50% increase) - cân bằng hơn
    base_light_reranking = 100  # Tăng từ 80 → 100 (25% increase) - cân bằng hơn

    # Performance-based multipliers (fine-tuned based on empirical data)
    performance_multipliers = {
        "Conservative": {"retrieval": 0.85, "light": 0.85},  # Balanced precision
        "Balanced": {"retrieval": 1.0, "light": 1.0},  # Default performance
        "Aggressive": {"retrieval": 1.25, "light": 1.15},  # Higher recall
    }

    # Scale based on final results count with diminishing returns
    if final_results_count <= 5:
        retrieval_multiplier = 1.0
        light_multiplier = 1.0
    elif final_results_count <= 10:
        retrieval_multiplier = 1.15  # Reduced from 1.2
        light_multiplier = 1.08  # Reduced from 1.1
    elif final_results_count <= 15:
        retrieval_multiplier = 1.25  # Reduced from 1.5
        light_multiplier = 1.15  # Reduced from 1.3
    else:
        retrieval_multiplier = 1.35  # Cap at reasonable level
        light_multiplier = 1.20

    # Calculate base values
    top_k_retrieval = int(base_retrieval * retrieval_multiplier)
    top_k_light_reranking = int(base_light_reranking * light_multiplier)

    # Apply search aggressiveness
    aggressiveness = performance_multipliers.get(
        search_aggressiveness, performance_multipliers["Balanced"]
    )
    top_k_retrieval = int(top_k_retrieval * aggressiveness["retrieval"])
    top_k_light_reranking = int(top_k_light_reranking * aggressiveness["light"])

    # Ensure minimum values for quality
    top_k_retrieval = max(top_k_retrieval, 50)
    top_k_light_reranking = max(top_k_light_reranking, 40)

    # Apply upper bounds for performance - Cân bằng Recall + Performance
    top_k_retrieval = min(top_k_retrieval, 150)  # Cap retrieval at 50 (cân bằng hơn)
    top_k_light_reranking = min(
        top_k_light_reranking, 120
    )  # Cap light reranking at 120 (cân bằng hơn)

    # Get model-specific limits from config
    try:
        light_reranker_top_k = config.reranker_pipeline.light_reranker["top_k"]
        top_k_light_reranking = min(top_k_light_reranking, light_reranker_top_k)
    except (KeyError, AttributeError):
        logger.warning(
            "Could not get light_reranker top_k from config, using calculated value"
        )

    # Validate final parameters
    if top_k_retrieval < top_k_light_reranking:
        logger.warning("Retrieval count < Light reranking count, adjusting...")
        top_k_retrieval = max(top_k_light_reranking + 20, top_k_retrieval)

    logger.info(
        f"🔧 Calculated parameters: retrieval={top_k_retrieval}, light={top_k_light_reranking}, final={final_results_count}"
    )

    return {
        "top_k_retrieval": top_k_retrieval,
        "top_k_light_reranking": top_k_light_reranking,
        "top_k_final": final_results_count,
    }


def display_performance_metrics(start_time: float, end_time: float, result_count: int):
    """Display performance metrics for the search operation."""
    processing_time = end_time - start_time

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


# Function removed - no longer needed with simple expander approach


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


def main():
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
                    value=st.session_state.get("final_results_count", 3),
                    help="Số kết quả cuối cùng sẽ hiển thị",
                    key="final_results_slider",
                )

                search_aggressiveness = st.selectbox(
                    "Mức độ tìm kiếm:",
                    ["Balanced", "Conservative", "Aggressive"],
                    index=["Balanced", "Conservative", "Aggressive"].index(
                        st.session_state.get("search_aggressiveness", "Balanced")
                    ),
                    help="Conservative: Ít kết quả, độ chính xác cao. Aggressive: Nhiều kết quả, độ bao phủ cao",
                    key="aggressiveness_select",
                )

                # Force CPU option
                force_cpu = st.checkbox(
                    "🖥️ Force CPU Mode",
                    value=st.session_state.get("force_cpu", False),
                    help="Bắt buộc sử dụng CPU thay vì GPU (hữu ích khi gặp lỗi CUDA)",
                    key="force_cpu_checkbox",
                )

                # Apply configuration button
                if st.form_submit_button("✅ Áp dụng cấu hình"):
                    # Save form values to session state
                    st.session_state.final_results_count = final_results_count
                    st.session_state.search_aggressiveness = search_aggressiveness
                    st.session_state.force_cpu = force_cpu
                    st.success("✅ Cấu hình đã được áp dụng!")
                    st.rerun()  # Force rerun to update UI

            # Display current configuration
            if force_cpu:
                st.warning("⚠️ CPU mode được kích hoạt. Hiệu suất có thể chậm hơn.")

            # Reload pipeline button
            if st.button("🔄 Reload Pipeline"):
                st.cache_resource.clear()
                st.success("✅ Cache đã được xóa!")

            # Model versions display
            with st.sidebar.expander("📦 Phiên bản Models đang chạy", expanded=False):
                model_versions = pipeline.get_loaded_model_versions()
                if model_versions:
                    for name, version in model_versions.items():
                        st.text(f"{name}:")
                        st.code(version, language=None)
                else:
                    st.text("Không thể xác định phiên bản.")

        # Initialize session state efficiently (only once)
        if "search_page_initialized" not in st.session_state:
            st.session_state.search_page_initialized = True
            st.session_state.current_question = (
                "Người lao động được nghỉ phép bao nhiêu ngày?"
            )
            st.session_state.searching = False
            st.session_state.last_query = ""
            st.session_state.results_cache = {}
            st.session_state.search_params_cache = {}

        # Main search interface with professional styling
        with st.container():
            st.markdown("### 🔍 Nhập câu hỏi")

            # Random question button in sidebar - optimized to prevent rerun
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 🎲 Câu hỏi mẫu")
                
                # Use form to prevent auto-rerun
                with st.form("random_question_form", clear_on_submit=False):
                    if st.form_submit_button(
                        "🎲 Lấy câu hỏi ngẫu nhiên",
                        help="Lấy câu hỏi ngẫu nhiên từ dataset",
                        use_container_width=True,
                    ):
                        try:
                            training_questions, test_questions = load_random_questions()
                            all_questions = training_questions + test_questions
                            if all_questions:
                                random_question = random.choice(all_questions)
                                st.session_state.current_question = random_question
                                st.session_state.last_query = random_question
                                st.success(f"✅ Đã chọn câu hỏi: {random_question[:50]}...")
                            else:
                                st.warning("Không có câu hỏi mẫu")
                        except Exception as e:
                            st.error(f"Lỗi khi load câu hỏi ngẫu nhiên: {e}")

            # Search input - optimized to prevent unnecessary reruns
            query = st.text_input(
                "Nhập câu hỏi của bạn:",
                value=st.session_state.current_question,
                help="Nhập câu hỏi về pháp luật hoặc sử dụng tính năng câu hỏi ngẫu nhiên",
                key="query_input",
                placeholder="Ví dụ: Người lao động được nghỉ phép bao nhiêu ngày?",
                on_change=lambda: None,  # Prevent auto-rerun
            )

        # Update current question only when form is submitted
        if query and query != st.session_state.last_query:
            st.session_state.current_question = query

        # Searching state is already initialized above

        # Search button - optimized to prevent unnecessary reruns
        with st.form("search_form", clear_on_submit=False):
            search_clicked = st.form_submit_button(
                "🔍 Tìm kiếm", type="primary", disabled=st.session_state.searching
            )

            if search_clicked and not st.session_state.searching:
                if query:
                    st.session_state.searching = True
                    st.session_state.last_query = query
                    logger.info(f"Processing query: {query[:100]}...")
                    try:
                        # Get parameters from form values (not session state)
                        final_results_count = st.session_state.final_results_slider
                        search_aggressiveness = st.session_state.aggressiveness_select
                        
                        # Debug logging
                        logger.info(f"🔍 Search parameters: final_results_count={final_results_count}, search_aggressiveness={search_aggressiveness}")
                        
                        # Calculate optimal parameters
                        params = calculate_optimal_parameters(
                            final_results_count, search_aggressiveness
                        )
                        logger.info(f"🔧 Calculated search parameters: {params}")
                        logger.info(f"🔧 Pipeline will use top_k={params['top_k_final']} for final results")

                        # Display calculated parameters
                        with st.expander("📊 Thông số tìm kiếm được tính toán tự động"):
                            st.markdown(
                                f"**🎯 Tầng 1 - Retrieval:** {params['top_k_retrieval']} ứng viên"
                            )
                            st.markdown(
                                f"**⚡ Tầng 2 - Light Reranking:** {params['top_k_light_reranking']} ứng viên"
                            )
                            st.markdown(
                                f"**🎯 Tầng 3 - Final Reranking:** {params['top_k_final']} kết quả cuối cùng"
                            )
                            st.markdown(f"**⚙️ Mức độ tìm kiếm:** {search_aggressiveness}")

                        # Performance monitoring
                        start_time = time.time()
                        with st.spinner("🔎 Đang tìm kiếm... vui lòng đợi"):
                            # Get results from pipeline with calculated parameters
                            results = pipeline.predict(
                                query,
                                top_k=params["top_k_final"],  # ✅ Use top_k parameter
                            )
                        end_time = time.time()

                        processing_time = end_time - start_time
                        logger.info(
                            f"Query processed in {processing_time:.2f}s, found {len(results) if results else 0} results"
                        )

                        if results:
                            # Display performance metrics - simple styling
                            display_performance_metrics(start_time, end_time, len(results))
                            
                            # Debug: Show actual vs expected results
                            expected_results = params["top_k_final"]
                            actual_results = len(results)
                            if actual_results != expected_results:
                                st.warning(f"⚠️ Debug: Expected {expected_results} results, got {actual_results} results")
                            else:
                                st.success(f"✅ Debug: Got exactly {actual_results} results as expected")

                            # Results display section
                            st.markdown("---")
                            st.markdown("### 📊 Kết quả tìm kiếm")
                            st.info("Hiển thị kết quả tìm kiếm với thông tin chi tiết")

                            # Display results in tabs
                            tab1, tab2 = st.tabs(["📄 Xem chi tiết", "📊 So sánh điểm số"])

                            with tab1:
                                st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")

                                # Results summary - simple styling
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "🎯 Kết quả cao nhất",
                                        f"{results[0]['final_score']:.3f}",
                                    )
                                with col2:
                                    st.metric(
                                        "📊 Điểm trung bình",
                                        f"{sum(r['final_score'] for r in results)/len(results):.3f}",
                                    )
                                with col3:
                                    st.metric("🔍 Tổng kết quả", len(results))

                                # Display each result with styled scores
                                for i, result in enumerate(results, 1):
                                    with st.expander(
                                        f"#{i} - {result['aid']} (Điểm: {result['final_score']:.3f})",
                                        expanded=(i <= 3),
                                    ):
                                        display_search_result(result, i, query, pipeline)

                            with tab2:
                                # Score comparison chart using centralized configuration
                                scores_data = {
                                    "Kết quả": [f"#{i+1}" for i in range(len(results))],
                                }

                                # Add scores for each model type
                                for model_key in get_all_model_keys():
                                    score_field = get_score_field_name(model_key)
                                    display_name = get_display_name(model_key)
                                    scores_data[display_name] = [
                                        r.get(score_field, 0.0) for r in results
                                    ]

                                df_scores = pd.DataFrame(scores_data)

                                # Display metrics for the primary score (light reranker)
                                light_rerank_scores = scores_data.get(
                                    "Điểm light rerank", []
                                )
                                if light_rerank_scores:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Điểm cao nhất",
                                            f"{max(light_rerank_scores):.3f}",
                                        )
                                    with col2:
                                        st.metric(
                                            "Điểm thấp nhất",
                                            f"{min(light_rerank_scores):.3f}",
                                        )
                                    with col3:
                                        st.metric(
                                            "Điểm trung bình",
                                            f"{sum(light_rerank_scores)/len(light_rerank_scores):.3f}",
                                        )

                                # Chart
                                st.bar_chart(df_scores.set_index("Kết quả"))
                        else:
                            st.warning(
                                "❌ Không tìm thấy kết quả phù hợp. Hãy thử câu hỏi khác."
                            )
                    except Exception as e:
                        st.error(f"❌ Lỗi khi tìm kiếm: {str(e)}")
                        st.info("💡 Gợi ý: Thử giảm số lượng kết quả hoặc sử dụng CPU mode")
                    finally:
                        st.session_state.searching = False
    else:
        st.error("❌ Không thể tải pipeline. Vui lòng kiểm tra lại hệ thống.")


def render_search_page():
    """Wrapper for app router to render the search page."""
    main()


if __name__ == "__main__":
    main()
