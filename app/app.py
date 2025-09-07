#!/usr/bin/env python3
"""
LawBot - Main Application
=========================

Main Streamlit application with a modular page architecture.
This is the single entry point for the user interface.
"""

import streamlit as st
import sys
from pathlib import Path
import os
import time
from datetime import datetime

# Lazy import pages to avoid circular import
pages_loaded = True

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


# Lazy load pipeline to avoid import errors
def get_pipeline():
    """Lazy load pipeline only when needed."""
    try:
        from core.pipeline import LegalQAPipeline

        return LegalQAPipeline()
    except Exception as e:
        st.warning(f"⚠️ Pipeline not available: {e}")
        return None


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

        # --- Main Content ---
        if selected_page_title in page_options:
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

    # --- Footer ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**LawBot v8.3** - Hệ thống QA Pháp luật thông minh")
    st.sidebar.markdown("Powered by AI & Legal Knowledge Base")


def main():
    """Main application entry point."""
    render_app()


if __name__ == "__main__":
    main()
