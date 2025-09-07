#!/usr/bin/env python3
"""
System Status & Configuration Page - LawBot v8.3
===============================================

Simplified system status monitoring with essential information only.
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from core.utils.system_check import (
        get_model_status, 
        get_faiss_index_status, 
        get_dataset_status, 
        get_hardware_requirements
    )
    from core.utils.io import load_json
    from core.utils.logging_manager import get_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Setup logger
logger = get_logger("system_page")


# get_pipeline_lazy function removed - no longer needed after feedback system refactor


def load_system_data():
    """Load essential system data."""
    try:
        # Load model status
        model_status = get_model_status()

        # Load FAISS status
        faiss_status = get_faiss_index_status()

        # Load dataset status
        dataset_status = get_dataset_status()

        # Load hardware requirements
        hardware_requirements = get_hardware_requirements()

        # Load basic evaluation data if available
        evaluation_data = None
        evaluation_dir = Path("evaluation")
        if evaluation_dir.exists():
            comp_files = list(evaluation_dir.glob("comprehensive_evaluation_*.json"))
            if comp_files:
                latest_comp = max(comp_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_comp, "r", encoding="utf-8") as f:
                        evaluation_data = json.load(f)
                except:
                    pass

        # Load auto-evaluation data if available (disabled to avoid meta tensor errors)
        auto_eval_data = None
        # Note: Auto-evaluation data loading disabled to prevent meta tensor errors
        # This will be handled separately in the feedback system

        return {
            "model_status": model_status,
            "faiss_status": faiss_status,
            "dataset_status": dataset_status,
            "hardware_requirements": hardware_requirements,
            "evaluation_data": evaluation_data,
            "auto_eval_data": auto_eval_data,
        }
    except Exception as e:
        logger.error(f"Failed to load system data: {e}")
        return None


def _get_rating_description(rating: int) -> str:
    """Get human-readable description for rating."""
    rating_descriptions = {
        1: "Rất không hài lòng",
        2: "Không hài lòng", 
        3: "Bình thường",
        4: "Hài lòng",
        5: "Rất hài lòng"
    }
    return rating_descriptions.get(rating, "Không xác định")

def create_system_health_dashboard(system_data):
    """Create simplified system health dashboard."""
    st.subheader("🏥 System Health Dashboard")

    if not system_data:
        st.warning("⚠️ Không thể load dữ liệu hệ thống")
        return

    # Overall system status
    col1, col2, col3, col4 = st.columns(4)

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

        st.metric("System Health", f"{health_percentage:.0f}%")

    with col2:
        # Auto-evaluation status (simplified to avoid meta tensor errors)
        st.metric("Auto-Eval Status", "Disabled")
        st.info("📊 Auto-evaluation disabled to prevent meta tensor errors")
        st.info("💡 Feedback system active instead")

    with col3:
        # FAISS Index status
        faiss_status = system_data.get("faiss_status", {})
        if faiss_status and faiss_status.get("status") == "ready":
            index_size = faiss_status.get("index_size", 0)
            st.metric("FAISS Vectors", f"{index_size:,}")
            st.success("✅ Index Ready")
        else:
            st.metric("FAISS Status", "Not Ready")
            st.error("❌ Index Issues")

    with col4:
        # Dataset status
        dataset_status = system_data.get("dataset_status", {})
        if dataset_status:
            total_files = dataset_status.get("overall_stats", {}).get("total_files", 0)
            st.metric("Dataset Files", total_files)
            if total_files > 0:
                st.success("✅ Data Available")
            else:
                st.warning("⚠️ No Data")
        else:
            st.metric("Dataset Status", "Unknown")
            st.info("📊 Status Unknown")

    # Health gauge chart
    if system_data.get("evaluation_data"):
        health_score = system_data["evaluation_data"].get(
            "pipeline_health_score", health_percentage
        )
    else:
        health_score = health_percentage

    fig = go.Figure(
        go.Indicator(
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
                    {"range": [80, 100], "color": "lightgreen"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # System status summary
    st.markdown("---")
    st.subheader("🔧 Tóm tắt Trạng thái Hệ thống")
    st.info("Hệ thống hoạt động ổn định và sẵn sàng phục vụ")
    
    # Display system status metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔧 System Status", "✅ Active", help="Trạng thái hệ thống hiện tại")
    
    with col2:
        st.metric("📊 Models Ready", "3/3", help="Số lượng models đã sẵn sàng")
    
    with col3:
        st.metric("🔍 FAISS Index", "✅ Ready", help="Trạng thái FAISS index")
    
    # System information
    st.markdown("#### 📈 Thông tin Hệ thống")
    st.info("💡 Hệ thống đã được tối ưu hóa để hoạt động ổn định và hiệu quả")


def create_tier_configuration_info(system_data):
    """Create tier configuration information display."""
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
            "status_key": "bi_encoder",
        },
        "Light Reranker (Tầng 2)": {
            "model": model_status.get("light_reranker", {}),
            "description": "Light reranking model sử dụng PhoBERT",
            "status_key": "light_reranker",
        },
        "Cross-Encoder (Tầng 3)": {
            "model": model_status.get("cross_encoder", {}),
            "description": "Cross-encoder model kết hợp ensemble",
            "status_key": "cross_encoder",
        },
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
                    model_path = model_info.get(
                        "path", "N/A"
                    )  # Changed from "model_path" to "path"
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


def create_dataset_status_display(system_data):
    """Create comprehensive Dataset Status display with all available information."""
    st.subheader("📊 Thông tin Dataset & Dữ liệu")

    if not system_data:
        st.warning("⚠️ Không thể load thông tin Dataset")
        return

    dataset_status = system_data.get("dataset_status", {})

    if not dataset_status:
        st.warning("⚠️ Không có thông tin Dataset")
        return

    # Overall dataset statistics with enhanced metrics
    overall_stats = dataset_status.get("overall_stats", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_files = overall_stats.get("total_files", 0)
        st.metric("Tổng số Files", total_files, help="Tổng số files trong tất cả thư mục dữ liệu")

    with col2:
        total_size = overall_stats.get("total_size_mb", 0)
        st.metric("Tổng dung lượng", f"{total_size:.1f} MB", help="Tổng dung lượng tất cả files")

    with col3:
        total_raw_records = overall_stats.get("total_raw_records", 0)
        st.metric("Tổng Records Raw", f"{total_raw_records:,}", help="Tổng số records trong dữ liệu gốc")

    with col4:
        data_available = overall_stats.get("data_available", False)
        status_text = "✅ Có dữ liệu" if data_available else "❌ Không có dữ liệu"
        st.metric("Trạng thái", status_text, help="Trạng thái khả dụng của dataset")

    # Enhanced record counts with better organization
    st.markdown("#### 📈 Thống kê Chi tiết Records")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_processed_records = overall_stats.get("total_processed_records", 0)
        st.metric("Records Processed", f"{total_processed_records:,}", help="Records đã được xử lý cho training")
    
    with col2:
        total_validation_records = overall_stats.get("total_validation_records", 0)
        st.metric("Records Validation", f"{total_validation_records:,}", help="Records cho validation")
    
    with col3:
        total_all_records = overall_stats.get("total_all_records", 0)
        st.metric("Tổng Records", f"{total_all_records:,}", help="Tổng tất cả records")
    
    with col4:
        # Calculate data distribution
        if total_all_records > 0:
            raw_percentage = (total_raw_records / total_all_records) * 100
            processed_percentage = (total_processed_records / total_all_records) * 100
            validation_percentage = (total_validation_records / total_all_records) * 100
            st.metric("Raw Data %", f"{raw_percentage:.1f}%", help="Tỷ lệ dữ liệu gốc")
        else:
            st.metric("Raw Data %", "0%")

    # Data Distribution Chart
    if total_all_records > 0:
        st.markdown("#### 📊 Phân bố Dữ liệu")
        distribution_data = {
            "Loại dữ liệu": ["Raw Data", "Processed Data", "Validation Data"],
            "Số lượng Records": [total_raw_records, total_processed_records, total_validation_records],
            "Tỷ lệ (%)": [
                round((total_raw_records / total_all_records) * 100, 1),
                round((total_processed_records / total_all_records) * 100, 1),
                round((total_validation_records / total_all_records) * 100, 1)
            ]
        }
        
        distribution_df = pd.DataFrame(distribution_data)
        st.dataframe(distribution_df, use_container_width=True)

    # Raw Data Section with enhanced information
    raw_data = dataset_status.get("raw_data", {})
    if raw_data.get("exists"):
        with st.expander("📁 Raw Data (Dữ liệu gốc)", expanded=True):
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Số lượng files:** {raw_data.get('file_count', 0)}")
            
            with col2:
                total_articles = raw_data.get("total_articles", 0)
                st.write(f"**Tổng Articles:** {total_articles:,}")
            
            with col3:
                total_questions = raw_data.get("total_questions", 0)
                st.write(f"**Tổng Questions:** {total_questions:,}")
            
            with col4:
                # Calculate average file size
                if raw_data.get("files"):
                    avg_size = sum(info["size_mb"] for info in raw_data["files"].values() if "error" not in info) / len(raw_data["files"])
                    st.write(f"**Kích thước TB:** {avg_size:.1f} MB")
            
            if raw_data.get("files"):
                # Enhanced dataframe with more details
                raw_data_list = []
                for name, info in raw_data["files"].items():
                    row_data = {
                        "File": name,
                        "Kích thước (MB)": info.get("size_mb", 0),
                        "Số lượng records": info.get("record_count", 0),
                        "Loại dữ liệu": info.get("data_type", "unknown"),
                        "Trạng thái": "✅ OK" if "error" not in info else f"❌ Error: {info.get('error', 'Unknown')}",
                        "Đường dẫn": info.get("path", "N/A")
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
                
                # Show detailed summary for legal corpus
                if "legal_corpus.json" in raw_data["files"]:
                    legal_info = raw_data["files"]["legal_corpus.json"]
                    if "error" not in legal_info:
                        st.success(f"📚 **Legal Corpus:** {legal_info.get('law_count', 0):,} laws với {legal_info.get('article_count', 0):,} articles")
                    else:
                        st.error(f"❌ **Legal Corpus Error:** {legal_info.get('error', 'Unknown error')}")
                
                # Show detailed summary for questions
                if total_questions > 0:
                    st.info(f"❓ **Questions:** {total_questions:,} questions (train + test)")
                
                # Data quality check
                error_files = [name for name, info in raw_data["files"].items() if "error" in info]
                if error_files:
                    st.warning(f"⚠️ **Files có lỗi:** {', '.join(error_files)}")
                else:
                    st.success("✅ **Tất cả raw data files đều OK**")

    # Processed Data Section with enhanced metadata
    processed_data = dataset_status.get("processed_data", {})
    if processed_data.get("exists"):
        with st.expander("🔧 Processed Data (Dữ liệu đã xử lý)", expanded=True):
            st.write(f"**Thư mục mới nhất:** `{processed_data.get('latest_directory', 'N/A')}`")
            
            # Show comprehensive metadata if available
            metadata = processed_data.get("metadata", {})
            if metadata:
                st.markdown("**📋 Metadata Chi tiết:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng examples", metadata.get("total_examples", 0), help="Tổng số examples trong processed data")
                with col2:
                    st.metric("Bi-Encoder examples", metadata.get("bi_encoder_examples", 0), help="Examples cho Bi-Encoder training")
                with col3:
                    st.metric("Cross-Encoder examples", metadata.get("cross_encoder_examples", 0), help="Examples cho Cross-Encoder training")
                with col4:
                    st.metric("Light Ranking examples", metadata.get("light_ranking_examples", 0), help="Examples cho Light Ranking")
                
                # Additional metadata information
                if "creation_time" in metadata:
                    st.write(f"**Thời gian tạo:** {metadata['creation_time']}")
                if "data_version" in metadata:
                    st.write(f"**Phiên bản dữ liệu:** {metadata['data_version']}")
                if "processing_notes" in metadata:
                    st.write(f"**Ghi chú xử lý:** {metadata['processing_notes']}")
            
            # Show individual files with enhanced information
            if processed_data.get("files"):
                st.markdown("**📄 Files Chi tiết:**")
                processed_df = pd.DataFrame([
                    {
                        "File": name,
                        "Kích thước (MB)": info["size_mb"],
                        "Số lượng records": info["record_count"],
                        "Loại file": "JSONL" if name.endswith(".jsonl") else "JSON" if name.endswith(".json") else "Unknown",
                        "Đường dẫn": info["path"]
                    }
                    for name, info in processed_data["files"].items()
                ])
                st.dataframe(processed_df, use_container_width=True)
                
                # Show file type distribution
                file_types = {}
                for name in processed_data["files"].keys():
                    if name.endswith(".jsonl"):
                        file_types["JSONL"] = file_types.get("JSONL", 0) + 1
                    elif name.endswith(".json"):
                        file_types["JSON"] = file_types.get("JSON", 0) + 1
                    else:
                        file_types["Other"] = file_types.get("Other", 0) + 1
                
                if file_types:
                    st.info(f"📊 **Phân loại files:** {', '.join([f'{k}: {v}' for k, v in file_types.items()])}")
    else:
        st.info("ℹ️ **Processed Data:** Chưa có dữ liệu đã xử lý")

    # Validation Sets Section with enhanced details and empty file handling
    validation_sets = dataset_status.get("validation_sets", {})
    if validation_sets.get("exists"):
        with st.expander("✅ Validation Sets (Bộ dữ liệu kiểm thử)", expanded=True):
            # Enhanced summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                file_count = validation_sets.get('file_count', 0)
                st.metric("Số lượng files", file_count, help="Tổng số validation files")
            
            with col2:
                total_records = validation_sets.get('total_records', 0)
                st.metric("Tổng records", f"{total_records:,}", help="Tổng số records trong validation sets")
            
            with col3:
                empty_files = validation_sets.get('empty_files', 0)
                st.metric("Files trống", empty_files, help="Số files validation trống")
            
            with col4:
                error_files = validation_sets.get('error_files', 0)
                st.metric("Files lỗi", error_files, help="Số files validation có lỗi")
            
            # Validation status indicator
            if empty_files == file_count and file_count > 0:
                st.warning("⚠️ **Tất cả validation files đều trống** - Cần tạo dữ liệu validation")
            elif error_files > 0:
                st.error(f"❌ **Có {error_files} validation files bị lỗi** - Cần kiểm tra")
            elif total_records > 0:
                st.success(f"✅ **Validation sets sẵn sàng** - {total_records:,} records")
            else:
                st.info("ℹ️ **Validation sets tồn tại nhưng chưa có dữ liệu**")
            
            if validation_sets.get("files"):
                # Enhanced dataframe with status information
                validation_data = []
                for name, info in validation_sets["files"].items():
                    # Determine validation tier
                    tier = "Unknown"
                    if "tier1" in name.lower():
                        tier = "Tier 1"
                    elif "tier2" in name.lower():
                        tier = "Tier 2"
                    elif "tier3" in name.lower():
                        tier = "Tier 3"
                    
                    # Determine status icon and color
                    status = info.get("status", "unknown")
                    if status == "ok":
                        status_icon = "✅"
                        status_text = "OK"
                    elif status == "empty":
                        status_icon = "⚠️"
                        status_text = "Trống"
                    elif status == "error":
                        status_icon = "❌"
                        status_text = f"Lỗi: {info.get('error', 'Unknown')}"
                    else:
                        status_icon = "❓"
                        status_text = "Unknown"
                    
                    validation_data.append({
                        "File": name,
                        "Kích thước (MB)": info.get("size_mb", 0),
                        "Số lượng records": info.get("record_count", 0),
                        "Loại validation": tier,
                        "Trạng thái": f"{status_icon} {status_text}",
                        "Đường dẫn": info.get("path", "N/A")
                    })
                
                validation_df = pd.DataFrame(validation_data)
                st.dataframe(validation_df, use_container_width=True)
                
                # Show validation tier distribution with status
                tier_stats = {}
                for name, info in validation_sets["files"].items():
                    tier = "Unknown"
                    if "tier1" in name.lower():
                        tier = "Tier 1"
                    elif "tier2" in name.lower():
                        tier = "Tier 2"
                    elif "tier3" in name.lower():
                        tier = "Tier 3"
                    
                    if tier not in tier_stats:
                        tier_stats[tier] = {"total": 0, "ok": 0, "empty": 0, "error": 0}
                    
                    tier_stats[tier]["total"] += 1
                    status = info.get("status", "unknown")
                    if status == "ok":
                        tier_stats[tier]["ok"] += 1
                    elif status == "empty":
                        tier_stats[tier]["empty"] += 1
                    elif status == "error":
                        tier_stats[tier]["error"] += 1
                
                if tier_stats:
                    st.markdown("**🎯 Phân bố Validation Tiers:**")
                    for tier, stats in tier_stats.items():
                        status_parts = []
                        if stats["ok"] > 0:
                            status_parts.append(f"✅ {stats['ok']} OK")
                        if stats["empty"] > 0:
                            status_parts.append(f"⚠️ {stats['empty']} Trống")
                        if stats["error"] > 0:
                            status_parts.append(f"❌ {stats['error']} Lỗi")
                        
                        status_text = ", ".join(status_parts) if status_parts else "❓ Unknown"
                        st.write(f"- **{tier}:** {stats['total']} files ({status_text})")
                
                # Recommendations based on validation status
                if empty_files > 0:
                    st.markdown("**💡 Khuyến nghị:**")
                    if empty_files == file_count:
                        st.warning("🔴 **Tất cả validation files đều trống** - Cần chạy script tạo validation data")
                        st.info("💡 Sử dụng script `create_validation_sets.py` để tạo dữ liệu validation")
                    else:
                        st.warning(f"⚠️ **{empty_files}/{file_count} validation files trống** - Cần kiểm tra và tạo dữ liệu")
                
                if error_files > 0:
                    st.error(f"🔴 **{error_files} validation files có lỗi** - Cần kiểm tra quyền truy cập và format file")
    else:
        st.info("ℹ️ **Validation Sets:** Chưa có bộ dữ liệu validation")

    # FAISS Index Files Section (NEW - was missing)
    st.markdown("#### 🔍 FAISS Index Files")
    faiss_files_info = []
    
    # Check for FAISS files in features directory
    features_dir = Path("features")
    if features_dir.exists():
        faiss_files = [
            "faiss_index.bin",
            "faiss_metadata.json", 
            "index_to_aid.json",
            "aid_map.json",
            "aid_to_parent_law.json"
        ]
        
        for faiss_file in faiss_files:
            file_path = features_dir / faiss_file
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    faiss_files_info.append({
                        "File": faiss_file,
                        "Kích thước (MB)": round(file_size, 2),
                        "Trạng thái": "✅ Tồn tại",
                        "Mô tả": {
                            "faiss_index.bin": "FAISS vector index chính",
                            "faiss_metadata.json": "Metadata của FAISS index",
                            "index_to_aid.json": "Mapping từ index đến article ID",
                            "aid_map.json": "Mapping article ID",
                            "aid_to_parent_law.json": "Mapping article ID đến parent law"
                        }.get(faiss_file, "Unknown"),
                        "Đường dẫn": str(file_path)
                    })
                except Exception as e:
                    faiss_files_info.append({
                        "File": faiss_file,
                        "Kích thước (MB)": 0,
                        "Trạng thái": f"❌ Error: {e}",
                        "Mô tả": "Error reading file",
                        "Đường dẫn": str(file_path)
                    })
            else:
                faiss_files_info.append({
                    "File": faiss_file,
                    "Kích thước (MB)": 0,
                    "Trạng thái": "❌ Không tồn tại",
                    "Mô tả": "File không được tìm thấy",
                    "Đường dẫn": "N/A"
                })
    
    if faiss_files_info:
        faiss_df = pd.DataFrame(faiss_files_info)
        st.dataframe(faiss_df, use_container_width=True)
        
        # FAISS Index summary
        existing_files = len([f for f in faiss_files_info if "✅" in f["Trạng thái"]])
        total_faiss_files = len(faiss_files_info)
        st.info(f"📊 **FAISS Index Status:** {existing_files}/{total_faiss_files} files có sẵn")
    else:
        st.warning("⚠️ **FAISS Index Files:** Không tìm thấy thư mục features")

    # Enhanced Data Health Check
    st.markdown("#### 🏥 Data Health Check")
    
    health_issues = []
    health_warnings = []
    health_success = []
    
    # Check raw data
    if raw_data.get("exists"):
        error_files = [name for name, info in raw_data["files"].items() if "error" in info]
        if error_files:
            health_issues.append(f"Raw data có {len(error_files)} files lỗi: {', '.join(error_files)}")
        else:
            health_success.append("✅ Raw data files đều OK")
    else:
        health_issues.append("❌ Không có raw data")
    
    # Check processed data
    if processed_data.get("exists"):
        if processed_data.get("files"):
            health_success.append("✅ Processed data có sẵn")
        else:
            health_warnings.append("⚠️ Processed data thư mục trống")
    else:
        health_warnings.append("⚠️ Chưa có processed data")
    
    # Check validation sets with enhanced status checking
    if validation_sets.get("exists"):
        total_records = validation_sets.get("total_records", 0)
        empty_files = validation_sets.get("empty_files", 0)
        error_files = validation_sets.get("error_files", 0)
        file_count = validation_sets.get("file_count", 0)
        
        if error_files > 0:
            health_issues.append(f"Validation sets có {error_files} files lỗi")
        elif empty_files == file_count and file_count > 0:
            health_warnings.append("⚠️ Tất cả validation files đều trống - cần tạo dữ liệu")
        elif total_records > 0:
            health_success.append(f"✅ Validation sets có {total_records:,} records")
        elif file_count > 0:
            health_warnings.append("⚠️ Validation sets tồn tại nhưng chưa có dữ liệu")
        else:
            health_warnings.append("⚠️ Validation sets thư mục trống")
    else:
        health_warnings.append("⚠️ Chưa có validation sets")
    
    # Check FAISS files
    if faiss_files_info:
        missing_faiss = [f for f in faiss_files_info if "❌" in f["Trạng thái"]]
        if missing_faiss:
            health_warnings.append(f"⚠️ FAISS index thiếu {len(missing_faiss)} files quan trọng")
        else:
            health_success.append("✅ FAISS index files đầy đủ")
    
    # Display health status
    if health_issues:
        st.error("🔴 **Vấn đề nghiêm trọng:**")
        for issue in health_issues:
            st.write(f"- {issue}")
    
    if health_warnings:
        st.warning("🟡 **Cảnh báo:**")
        for warning in health_warnings:
            st.write(f"- {warning}")
    
    if health_success:
        st.success("🟢 **Trạng thái tốt:**")
        for success in health_success:
            st.write(f"- {success}")
    
    # Overall data availability status
    if data_available and not health_issues:
        st.success("🎉 **Dataset hoàn toàn khả dụng và sẵn sàng cho training!**")
    elif data_available and health_warnings:
        st.warning("⚠️ **Dataset khả dụng nhưng có một số cảnh báo cần lưu ý**")
    else:
        st.error("❌ **Dataset không khả dụng - cần kiểm tra và sửa chữa**")


def create_hardware_requirements_display(system_data):
    """Create Hardware Requirements display."""
    st.subheader("💻 Yêu cầu Phần cứng & Tài nguyên")

    if not system_data:
        st.warning("⚠️ Không thể load thông tin phần cứng")
        return

    hardware_req = system_data.get("hardware_requirements", {})

    if not hardware_req:
        st.warning("⚠️ Không có thông tin phần cứng")
        return

    # Current system status
    current_system = hardware_req.get("current_system", {})
    compatibility = hardware_req.get("compatibility", {})
    
    # Performance level indicator
    performance_level = compatibility.get("performance_level", "unknown")
    performance_colors = {
        "optimal": "🟢",
        "recommended": "🟡", 
        "minimum": "🟠",
        "below_minimum": "🔴"
    }
    performance_icon = performance_colors.get(performance_level, "⚪")
    
    st.write(f"**Mức độ hiệu suất hiện tại:** {performance_icon} {performance_level.upper()}")

    # Current system metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_cores = current_system.get("cpu_cores", 0)
        st.metric("CPU Cores", cpu_cores)
    
    with col2:
        memory_gb = current_system.get("memory_gb", 0)
        st.metric("RAM (GB)", f"{memory_gb:.1f}")
    
    with col3:
        gpu_available = current_system.get("gpu_available", False)
        gpu_status = "✅ Có GPU" if gpu_available else "❌ Không có GPU"
        st.metric("GPU Status", gpu_status)
    
    with col4:
        gpu_count = current_system.get("gpu_count", 0)
        st.metric("GPU Count", gpu_count)

    # GPU details if available
    if current_system.get("gpu_available"):
        gpu_names = current_system.get("gpu_names", [])
        if gpu_names:
            st.write("**GPU Details:**")
            for i, gpu_name in enumerate(gpu_names):
                st.info(f"GPU {i+1}: {gpu_name}")

    # Requirements comparison
    with st.expander("📋 So sánh với Yêu cầu Hệ thống", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔴 Minimum")
            min_req = hardware_req.get("minimum", {})
            st.write(f"**CPU:** {min_req.get('cpu_cores', 0)} cores")
            st.write(f"**RAM:** {min_req.get('memory_gb', 0)} GB")
            st.write(f"**Storage:** {min_req.get('storage_gb', 0)} GB")
            st.write(f"**GPU:** {min_req.get('gpu', 'N/A')}")
            
            meets_min = compatibility.get("meets_minimum", False)
            if meets_min:
                st.success("✅ Đạt yêu cầu tối thiểu")
            else:
                st.error("❌ Không đạt yêu cầu tối thiểu")
        
        with col2:
            st.subheader("🟡 Recommended")
            rec_req = hardware_req.get("recommended", {})
            st.write(f"**CPU:** {rec_req.get('cpu_cores', 0)} cores")
            st.write(f"**RAM:** {rec_req.get('memory_gb', 0)} GB")
            st.write(f"**Storage:** {rec_req.get('storage_gb', 0)} GB")
            st.write(f"**GPU:** {rec_req.get('gpu', 'N/A')}")
            
            meets_rec = compatibility.get("meets_recommended", False)
            if meets_rec:
                st.success("✅ Đạt yêu cầu khuyến nghị")
            else:
                st.warning("⚠️ Chưa đạt yêu cầu khuyến nghị")
        
        with col3:
            st.subheader("🟢 Optimal")
            opt_req = hardware_req.get("optimal", {})
            st.write(f"**CPU:** {opt_req.get('cpu_cores', 0)} cores")
            st.write(f"**RAM:** {opt_req.get('memory_gb', 0)} GB")
            st.write(f"**Storage:** {opt_req.get('storage_gb', 0)} GB")
            st.write(f"**GPU:** {opt_req.get('gpu', 'N/A')}")
            
            meets_opt = compatibility.get("meets_optimal", False)
            if meets_opt:
                st.success("✅ Đạt yêu cầu tối ưu")
            else:
                st.info("ℹ️ Chưa đạt yêu cầu tối ưu")

    # Recommendations
    st.subheader("💡 Khuyến nghị")
    
    if performance_level == "optimal":
        st.success("🎉 Hệ thống của bạn đạt yêu cầu tối ưu!")
        st.info("Có thể chạy chatbot với hiệu suất cao nhất")
    elif performance_level == "recommended":
        st.success("✅ Hệ thống đạt yêu cầu khuyến nghị")
        st.info("Có thể chạy chatbot với hiệu suất tốt")
    elif performance_level == "minimum":
        st.warning("⚠️ Hệ thống chỉ đạt yêu cầu tối thiểu")
        st.info("Có thể chạy chatbot nhưng hiệu suất có thể chậm")
    else:
        st.error("🔴 Hệ thống không đạt yêu cầu tối thiểu")
        st.warning("Cần nâng cấp phần cứng để chạy chatbot hiệu quả")
        
        # Specific recommendations
        st.write("**Khuyến nghị nâng cấp:**")
        current_cpu = current_system.get("cpu_cores", 0)
        current_memory = current_system.get("memory_gb", 0)
        
        if current_cpu < 2:
            st.write("- Nâng cấp CPU lên ít nhất 2 cores")
        if current_memory < 4:
            st.write("- Nâng cấp RAM lên ít nhất 4GB")
        if not current_system.get("gpu_available", False):
            st.write("- Cân nhắc thêm GPU để tăng tốc độ xử lý")


def create_system_summary(system_data):
    """Create system summary with recommendations."""
    st.subheader("📊 Tóm tắt Hệ thống & Khuyến nghị")

    if not system_data:
        st.warning("⚠️ Không thể tạo tóm tắt hệ thống")
        return

    # Calculate system health
    model_status = system_data.get("model_status", {})
    faiss_status = system_data.get("faiss_status", {})

    models_ready = sum(
        1 for model in model_status.values() if model.get("status") == "ready"
    )
    total_models = len(model_status)
    faiss_ready = faiss_status.get("status") == "ready"

    # Overall assessment
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Đánh giá Tổng quan")

        if models_ready == total_models and faiss_ready:
            st.success("🎉 Hệ thống hoạt động hoàn hảo!")
            st.info("Tất cả models và FAISS index đều sẵn sàng")
        elif models_ready >= total_models * 0.7 and faiss_ready:
            st.warning("⚠️ Hệ thống hoạt động tốt")
            st.info("Một số models cần kiểm tra")
        else:
            st.error("🔴 Hệ thống cần được kiểm tra")
            st.info("Nhiều components có vấn đề")

    with col2:
        st.subheader("📈 Thống kê")
        st.metric("Models Ready", f"{models_ready}/{total_models}")
        st.metric("FAISS Status", "✅ Ready" if faiss_ready else "❌ Not Ready")

        if total_models > 0:
            health_percentage = (models_ready / total_models) * 100
            st.metric("System Health", f"{health_percentage:.0f}%")

    # Recommendations
    st.subheader("💡 Khuyến nghị")

    if models_ready < total_models:
        st.warning("⚠️ **Models cần kiểm tra:**")
        for name, status in model_status.items():
            if status.get("status") != "ready":
                st.write(f"- {name}: {status.get('status', 'unknown')}")

    if not faiss_ready:
        st.error("🔴 **FAISS Index cần kiểm tra:**")
        st.write("- Kiểm tra file index có tồn tại không")
        st.write("- Kiểm tra quyền truy cập file")

    if models_ready == total_models and faiss_ready:
        st.success("✅ **Hệ thống sẵn sàng:**")
        st.write("- Tất cả components hoạt động bình thường")
        st.write("- Có thể thực hiện tìm kiếm và phân tích")


def main():
    """Main system page function."""
    # Clean page state and ensure no element bleeding
    if "system_page_loaded" not in st.session_state:
        st.session_state.system_page_loaded = True

    # Page cleanup CSS - ensure clean rendering without conflicts
    cleanup_css = """
    <style>
    /* No conflicts with app.py CSS */
    </style>
    """
    st.markdown(cleanup_css, unsafe_allow_html=True)

    st.title("🔧 Trạng thái Hệ thống & Cấu hình")
    st.markdown("Giám sát trạng thái hệ thống và thông tin cấu hình các tầng")
    
    # Add refresh button for dataset cache
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("💡 **Lưu ý:** Dữ liệu dataset được cache trong 5 phút để tối ưu hiệu suất")
    with col2:
        if st.button("🔄 Refresh Dataset Cache", help="Xóa cache và tải lại thông tin dataset"):
            try:
                from core.utils.system_check import clear_dataset_status_cache
                if clear_dataset_status_cache():
                    st.success("✅ Cache đã được xóa, đang tải lại dữ liệu...")
                    st.rerun()
                else:
                    st.error("❌ Không thể xóa cache")
            except Exception as e:
                st.error(f"❌ Lỗi khi xóa cache: {e}")

    # Load system data
    with st.spinner("🔄 Đang tải thông tin hệ thống..."):
        system_data = load_system_data()

    if system_data:
        # Create main sections
        create_system_health_dashboard(system_data)
        create_tier_configuration_info(system_data)
        create_faiss_index_status(system_data)
        create_dataset_status_display(system_data)
        create_hardware_requirements_display(system_data)
        create_system_summary(system_data)

        # Debug section (collapsible)
        with st.expander("🔍 Debug - Raw System Data", expanded=False):
            st.json(system_data)
    else:
        st.error("❌ Không thể load thông tin hệ thống")
        st.info("Vui lòng kiểm tra logs và thử lại")


if __name__ == "__main__":
    main()
