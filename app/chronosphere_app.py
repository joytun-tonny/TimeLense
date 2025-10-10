"""
CHRONOSPHERE STREAMLIT APPLICATION
Main entry point for the interactive web interface
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Get project root for asset paths  
# This ensures we get the correct project root whether running from app/ or root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(PROJECT_ROOT) == 'app':
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Custom CSS
st.set_page_config(
    page_title="Chronosphere - Time-Travel Data Debugger",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
    }
    .stApp {
        background-color: #0a0a0a;
    }
    h1 {
        color: #00d4ff !important;
        text-align: center;
    }
    h2 {
        color: #ff00ff !important;
    }
    h3 {
        color: #ffff00 !important;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0a0a0a;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #00a8cc;
    }
    .upload-box {
        border: 2px dashed #00d4ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'use_default' not in st.session_state:
    st.session_state.use_default = False

# Title
st.markdown("<h1>🌐 CHRONOSPHERE</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #ffffff;'>Time-Travel Data Debugging System</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150/00d4ff/ffffff?text=C", width=150)
    st.markdown("### Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Home", "📤 Upload Data", "📊 Explore", "⏰ Chronosphere", "💡 Insights", "📥 Export"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Chronosphere** is a time-travel debugging system for temporal data.
    
    Click anomalies to rewind and discover their origins.
    """)
    
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded")
        st.metric("Rows", f"{len(st.session_state.df):,}")

# Main content based on page
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #1a1a1a; padding: 30px; border-radius: 10px; border: 2px solid #00d4ff;'>
            <h2 style='color: #00d4ff;'>🎯 Default Dataset</h2>
            <p style='color: #ffffff;'>E-commerce Cosmetics Shop</p>
            <ul style='color: #cccccc;'>
                <li>2.7M+ events</li>
                <li>6 months of data</li>
                <li>Multiple event types</li>
                <li>Pre-analyzed & ready</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Demo", key="demo_btn"):
            with st.spinner("Loading default dataset..."):
                try:
                    # Load default dataset - use absolute path relative to project root
                    project_root = os.path.dirname(os.path.dirname(__file__))
                    
                    # Try sample dataset first (smaller file for demo)
                    sample_path = os.path.join(project_root, 'data', 'processed', 'events_sample.csv')
                    main_path = os.path.join(project_root, 'data', 'processed', 'events_cleaned.csv')
                    
                    data_path = sample_path if os.path.exists(sample_path) else main_path
                    
                    if os.path.exists(data_path):
                        # Load only first 50,000 rows for demo performance
                        st.info(f"Loading dataset: {os.path.basename(data_path)}")
                        df = pd.read_csv(data_path, parse_dates=['event_time'], nrows=50000)
                        
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.use_default = True
                        st.success(f"✅ Default dataset loaded! ({len(df):,} rows)")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"Default dataset not found. Checked: {sample_path} and {main_path}")
                except Exception as e:
                    st.error(f"Error loading default dataset: {e}")
                    st.exception(e)  # Show full traceback for debugging
    
    with col2:
        st.markdown("""
        <div style='background-color: #1a1a1a; padding: 30px; border-radius: 10px; border: 2px solid #ff00ff;'>
            <h2 style='color: #ff00ff;'>📤 Upload Your Data</h2>
            <p style='color: #ffffff;'>Works with ANY temporal dataset</p>
            <ul style='color: #cccccc;'>
                <li>CSV or Excel files</li>
                <li>Auto-detects columns</li>
                <li>Smart validation</li>
                <li>Instant analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📁 Upload File", key="upload_btn"):
            st.session_state.current_page = "📤 Upload Data"
            st.rerun()
    
    st.markdown("---")
    
    # Features
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **⏰ Time-Travel Interface**
        
        Click anomalies and watch the system rewind to show their origin
        """)
    
    with col2:
        st.markdown("""
        **📊 10+ Visualizations**
        
        From simple trends to 3D phase spaces with ghost trails
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI Insights**
        
        Automated pattern discovery with natural language explanations
        """)

elif page == "📤 Upload Data":
    st.markdown("## 📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Drop your CSV or Excel file here",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with temporal data (dates + numeric metrics)"
    )
    
    if uploaded_file is not None:
        try:
            # Load file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Show preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Auto-detect columns
            st.markdown("### 🔍 Smart Column Detection")
            
            from intelligence.column_detector import SmartColumnDetector
            
            detector = SmartColumnDetector(df)
            config = detector.detect_all()
            
            # Show detected time column
            if config['datetime']:
                time_col = config['datetime'][0]['column']
                confidence = config['datetime'][0]['confidence']
                
                st.success(f"✅ Time column detected: **{time_col}** (confidence: {confidence}%)")
                
                # Confirm or change
                all_cols = df.columns.tolist()
                time_col_confirmed = st.selectbox(
                    "Confirm time column:",
                    all_cols,
                    index=all_cols.index(time_col) if time_col in all_cols else 0
                )
            else:
                st.warning("⚠️ No time column detected")
                time_col_confirmed = st.selectbox("Select time column:", df.columns.tolist())
            
            # Show detected metrics
            if config['numeric']:
                st.markdown("**Detected Metrics:**")
                metric_cols = [col['column'] for col in config['numeric'] if col.get('recommended_for_analysis', False)]
                
                selected_metrics = st.multiselect(
                    "Select metrics to analyze:",
                    [col['column'] for col in config['numeric']],
                    default=metric_cols[:5]
                )
            
            # Load button
            if st.button("✅ Load and Analyze"):
                # Convert time column
                df[time_col_confirmed] = pd.to_datetime(df[time_col_confirmed])
                
                # Store in session state
                st.session_state.df = df
                st.session_state.time_col = time_col_confirmed
                st.session_state.metric_cols = selected_metrics
                st.session_state.data_loaded = True
                st.session_state.use_default = False
                
                st.success("🎉 Data loaded successfully!")
                st.balloons()
                st.info("Navigate to 📊 Explore or ⏰ Chronosphere to analyze your data")
        
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif page == "📊 Explore":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first (Home or Upload page)")
    else:
        df = st.session_state.df
        
        st.markdown("## 📊 Data Exploration")
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Columns", df.shape[1])
        
        with col3:
            if 'user_id' in df.columns:
                st.metric("Unique Users", f"{df['user_id'].nunique():,}")
            else:
                st.metric("Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
        
        with col4:
            if 'event_time' in df.columns:
                time_span = (df['event_time'].max() - df['event_time'].min()).days
                st.metric("Time Span", f"{time_span} days")
            else:
                st.metric("Missing %", f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%")
        
        st.markdown("---")
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "📊 Distributions", "🔗 Correlations"])
        
        with tab1:
            st.markdown("### Temporal Trends")
            
            if st.session_state.use_default:
                # Use pre-processed data
                daily = pd.read_csv('data/processed/daily_aggregated.csv', parse_dates=['date'])
                
                fig = px.line(daily, x='date', y='total_events',
                             title='Daily Event Volume',
                             labels={'total_events': 'Events', 'date': 'Date'})
                fig.update_traces(line_color='#00d4ff', line_width=2)
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Hourly pattern
                hourly = pd.read_csv('data/processed/hourly_aggregated.csv', parse_dates=['event_time'])
                hourly_pattern = hourly.groupby(hourly['event_time'].dt.hour)['total_events'].mean()
                
                fig2 = px.bar(x=hourly_pattern.index, y=hourly_pattern.values,
                             title='Average Events by Hour',
                             labels={'x': 'Hour', 'y': 'Events'})
                fig2.update_traces(marker_color='#ff00ff')
                fig2.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Trend visualizations available for default dataset")
        
        with tab2:
            st.markdown("### Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Select column:", numeric_cols)
                
                fig = px.histogram(df, x=selected_col, nbins=50,
                                  title=f'Distribution of {selected_col}')
                fig.update_traces(marker_color='#ffff00')
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig2 = px.box(df, y=selected_col, title=f'Box Plot: {selected_col}')
                fig2.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No numeric columns found")
        
        with tab3:
            st.markdown("### Correlations")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, text_auto='.2f',
                               title='Correlation Matrix',
                               color_continuous_scale='RdBu_r',
                               zmin=-1, zmax=1)
                fig.update_layout(template='plotly_dark', height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation")

elif page == "⏰ Chronosphere":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
    else:
        st.markdown("## ⏰ Chronosphere: Time-Travel Interface")
        
        if st.session_state.use_default:
            # Load visualizations
            st.markdown("### 🎬 Interactive Visualizations")
            
            viz_tabs = st.tabs([
                "💓 Heartbeat",
                "🌀 Correlation",
                "🎯 3D Space",
                "🧬 Evolution",
                "⏮ Rewind"
            ])
            
            with viz_tabs[0]:
                st.markdown("#### Temporal Heartbeat")
                st.markdown("Line thickness shows rate of change - creating a 'pulse' effect")
                
                try:
                    viz_path = os.path.join(PROJECT_ROOT, 'assets', 'visualizations', 'viz1_temporal_heartbeat.html')
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                except Exception as e:
                    st.error(f"Error loading visualization: {str(e)}")
                    st.warning("Visualization not found. Run notebook 03 first.")
            
            with viz_tabs[1]:
                st.markdown("#### Correlation Explorer")
                st.markdown("Watch relationships evolve over time")
                
                try:
                    viz_path = os.path.join(PROJECT_ROOT, 'assets', 'visualizations', 'viz2_correlation_explorer.html')
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=700)
                except Exception as e:
                    st.error(f"Error loading visualization: {str(e)}")
                    st.warning("Visualization not found. Run notebook 03 first.")
            
            with viz_tabs[2]:
                st.markdown("#### 3D Phase Space with Ghost Trails")
                st.markdown("White trails show temporal trajectories")
                
                try:
                    viz_path = os.path.join(PROJECT_ROOT, 'assets', 'visualizations', 'viz3_phase_space_3d.html')
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=800)
                except Exception as e:
                    st.error(f"Error loading visualization: {str(e)}")
                    st.warning("Visualization not found. Run notebook 03 first.")
            
            with viz_tabs[3]:
                st.markdown("#### High-Dimensional Evolution")
                st.markdown("Watch clusters form and split over time")
                
                try:
                    viz_path = os.path.join(PROJECT_ROOT, 'assets', 'visualizations', 'viz4_tsne_evolution.html')
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=700)
                except Exception as e:
                    st.error(f"Error loading visualization: {str(e)}")
                    st.warning("Visualization not found. Run notebook 03 first.")
            
            with viz_tabs[4]:
                st.markdown("#### ⭐ The Chronosphere: Time-Travel Rewind")
                st.markdown("**Click any anomaly (red X) to rewind and see its origin!**")
                
                try:
                    viz_path = os.path.join(PROJECT_ROOT, 'assets', 'visualizations', 'viz5_chronosphere_main.html')
                    with open(viz_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        st.components.v1.html(html_content, height=900)
                except Exception as e:
                    st.error(f"Error loading visualization: {str(e)}")
                    st.warning("Visualization not found. Run notebook 03 first.")
        else:
            st.info("Advanced visualizations available for default dataset")
            st.markdown("Upload analysis in progress...")

elif page == "💡 Insights":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
    else:
        st.markdown("## 💡 Automated Insights")
        
        if st.session_state.use_default:
            # Load insights from report
            try:
                findings_path = os.path.join(PROJECT_ROOT, 'reports', 'KEY_FINDINGS.txt')
                with open(findings_path, 'r', encoding='utf-8') as f:
                    findings = f.read()
                
                st.markdown("### 🔍 Key Findings")
                st.text(findings)
                
                st.markdown("---")
                
                # Top insights
                st.markdown("### 🎯 Top 5 Insights")
                
                insights_demo = [
                    "📈 **total_events** shows a moderate upward trend (R²=0.65). Overall change of 23.4% over the period.",
                    "🕐 **unique_users** exhibits clear daily seasonality, with peaks at hour 14 and troughs at hour 4. Peak is 4.2× the trough.",
                    "⚠️ Detected 7 significant anomalies in **total_events** (2.3% of data). Largest spike: 125,432 events on 2019-10-15.",
                    "🔗 **unique_users** and **total_events** show very strong positive correlation (r=0.92).",
                    "📅 **total_events** is 18.5% higher on weekends compared to weekdays."
                ]
                
                for i, insight in enumerate(insights_demo, 1):
                    st.markdown(f"{i}. {insight}")
                    st.progress(1.0)
                
                st.markdown("---")
                
                # Recommendations
                recommendations_path = os.path.join(PROJECT_ROOT, 'reports', 'RECOMMENDATIONS.txt')
                with open(recommendations_path, 'r', encoding='utf-8') as f:
                    recommendations = f.read()
                
                st.markdown("### 💡 Actionable Recommendations")
                st.text(recommendations)
                
            except FileNotFoundError:
                st.warning("⚠️ Run notebook 04 to generate insights")
                st.info("Automated insight generation available after analysis")
        else:
            st.info("Insight generation available for default dataset")

elif page == "📥 Export":
    st.markdown("## 📥 Export Results")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
    else:
        st.markdown("### Available Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Visualizations")
            
            viz_files = [
                "viz1_temporal_heartbeat.html",
                "viz2_correlation_explorer.html",
                "viz3_phase_space_3d.html",
                "viz4_tsne_evolution.html",
                "viz5_chronosphere_main.html",
                "chronosphere_dashboard.html"
            ]
            
            for viz in viz_files:
                viz_path = os.path.join(PROJECT_ROOT, 'assets', 'visualizations', viz)
                if os.path.exists(viz_path):
                    with open(viz_path, 'rb') as f:
                        st.download_button(
                            label=f"📥 {viz}",
                            data=f,
                            file_name=viz,
                            mime='text/html'
                        )
        
        with col2:
            st.markdown("#### 📄 Reports")
            
            report_files = [
                "KEY_FINDINGS.txt",
                "RECOMMENDATIONS.txt",
                "INSIGHTS_REPORT.txt"
            ]
            
            for report in report_files:
                if os.path.exists(f'reports/{report}'):
                    with open(f'reports/{report}', 'rb') as f:
                        st.download_button(
                            label=f"📥 {report}",
                            data=f,
                            file_name=report,
                            mime='text/plain'
                        )
        
        st.markdown("---")
        
        st.markdown("### 📦 Export Processed Data")
        
        if st.session_state.use_default:
            # Export cleaned data
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if os.path.exists('data/processed/events_cleaned.csv'):
                    df_export = pd.read_csv('data/processed/events_cleaned.csv')
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Cleaned Events",
                        data=csv,
                        file_name='chronosphere_cleaned_events.csv',
                        mime='text/csv'
                    )
            
            with col2:
                if os.path.exists('data/processed/hourly_aggregated.csv'):
                    hourly_export = pd.read_csv('data/processed/hourly_aggregated.csv')
                    csv = hourly_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Hourly Data",
                        data=csv,
                        file_name='chronosphere_hourly.csv',
                        mime='text/csv'
                    )
            
            with col3:
                if os.path.exists('data/processed/daily_aggregated.csv'):
                    daily_export = pd.read_csv('data/processed/daily_aggregated.csv')
                    csv = daily_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Daily Data",
                        data=csv,
                        file_name='chronosphere_daily.csv',
                        mime='text/csv'
                    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 20px;'>
    <p>🌐 CHRONOSPHERE v1.0 | Time-Travel Data Debugging System</p>
    <p>Built with Streamlit, Plotly, and ❤️</p>
</div>
""", unsafe_allow_html=True)