#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Streamlit app to verify everything works
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

st.title("TimeLense Test")
st.write("Testing if the app runs properly...")

try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    st.success("✅ Core libraries imported successfully")
    
    # Test intelligence modules
    from intelligence.llm_insight_generator import LLMInsightGenerator
    st.success("✅ LLM Insight Generator imported")
    
    from intelligence.column_detector import SmartColumnDetector
    st.success("✅ Smart Column Detector imported")
    
    from intelligence.smart_configurator import SmartDatasetConfigurator
    st.success("✅ Smart Dataset Configurator imported")
    
    from visualization.chronosphere_viz import ChronosphereVisualizer
    st.success("✅ Chronosphere Visualizer imported")
    
    from utils.sample_generator import SampleDatasetGenerator
    st.success("✅ Sample Dataset Generator imported")
    
    st.success("🎉 All modules imported successfully!")
    
    # Test data loading
    try:
        sample_path = 'data/processed/events_sample.csv'
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path, nrows=100)
            st.success(f"✅ Sample data loaded: {len(df)} rows")
        else:
            st.warning("Sample data file not found")
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
    
except ImportError as e:
    st.error(f"Import error: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")

st.write("Test completed!")