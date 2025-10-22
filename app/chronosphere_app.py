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
    page_title="TimeLense - Time-Travel Data Debugger",
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
st.markdown("<h1>🌐 TimeLense</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #ffffff;'>Time-Travel Data Debugging System</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Home", "📤 Upload Data", "📊 Explore", "⏰ Chronosphere", "💡 Insights", "📥 Export"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **TimeLense** is a time-travel debugging system for temporal data.
    
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
    
    # Add tabs for upload vs demo
    upload_tabs = st.tabs(["📁 Upload File", "🎮 Try Demo Data"])
    
    with upload_tabs[0]:
        uploaded_file = st.file_uploader(
            "Drop your CSV or Excel file here",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file with temporal data (dates + numeric metrics)"
        )
    
    with upload_tabs[1]:
        st.markdown("### 🎮 Try Sample Datasets")
        st.markdown("Experience advanced visualizations with realistic demo data")
        
        try:
            from utils.sample_generator import SampleDatasetGenerator
            
            sample_options = {
                'E-commerce Metrics': 'Online store performance data (90 days, hourly)',
                'IoT Sensor Data': 'Environmental sensor readings (30 days, 15-minute intervals)', 
                'Financial Data': 'Stock market time series (1 year, daily)',
                'Website Analytics': 'Web traffic and conversion metrics (60 days, hourly)'
            }
            
            selected_sample = st.selectbox(
                "Choose a sample dataset:",
                list(sample_options.keys()),
                format_func=lambda x: f"{x}: {sample_options[x]}"
            )
            
            if st.button("🚀 Load Sample Dataset", key="load_sample"):
                with st.spinner(f"Generating {selected_sample}..."):
                    try:
                        # Generate the selected dataset
                        if selected_sample == 'E-commerce Metrics':
                            df = SampleDatasetGenerator.generate_ecommerce_data()
                            time_col = 'timestamp'
                            metric_cols = ['visitors', 'page_views', 'orders', 'revenue', 'conversion_rate', 'avg_order_value', 'bounce_rate']
                        elif selected_sample == 'IoT Sensor Data':
                            df = SampleDatasetGenerator.generate_iot_sensor_data()
                            time_col = 'datetime'
                            metric_cols = ['temperature_c', 'humidity_pct', 'pressure_hpa', 'light_intensity', 'air_quality_index']
                        elif selected_sample == 'Financial Data':
                            df = SampleDatasetGenerator.generate_financial_data()
                            time_col = 'date'
                            metric_cols = ['stock_price', 'volume', 'market_cap', 'rsi', 'ma_20', 'ma_50', 'beta', 'daily_return']
                        elif selected_sample == 'Website Analytics':
                            df = SampleDatasetGenerator.generate_website_analytics()
                            time_col = 'timestamp'
                            metric_cols = ['sessions', 'page_views', 'avg_session_duration', 'bounce_rate', 'conversion_rate', 'conversions', 'pages_per_session']
                        
                        # Show preview
                        st.success(f"✅ Sample dataset generated: {selected_sample}")
                        st.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
                        
                        st.markdown("### 📋 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Smart configuration
                        from intelligence.smart_configurator import SmartDatasetConfigurator
                        
                        configurator = SmartDatasetConfigurator(df, time_col, metric_cols)
                        optimized_df = configurator.get_optimized_dataset()
                        
                        # Store in session state
                        st.session_state.df = optimized_df
                        st.session_state.time_col = time_col
                        st.session_state.metric_cols = metric_cols
                        st.session_state.configurator = configurator
                        st.session_state.data_loaded = True
                        st.session_state.use_default = False
                        
                        # Show insights
                        st.markdown("### 🧠 Smart Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 💡 Dataset Insights")
                            insights = configurator.get_insights()
                            for insight in insights[:3]:  # Show top 3
                                st.info(insight)
                        
                        with col2:
                            st.markdown("#### 🎯 Available Visualizations")
                            recommended_viz = configurator.config.get('recommended_visualizations', [])
                            st.success(f"✅ **{len(recommended_viz)} advanced visualizations** ready!")
                            st.markdown("Ready to explore in **⏰ Chronosphere**")
                        
                        st.balloons()
                        st.success("🎉 Sample dataset loaded and optimized! Navigate to ⏰ Chronosphere to explore.")
                        
                    except Exception as e:
                        st.error(f"Error generating sample data: {str(e)}")
                        st.exception(e)
        
        except ImportError:
            st.warning("Sample data generator not available")
        except Exception as e:
            st.error(f"Error loading sample data options: {str(e)}")
    
    # Continue with existing upload logic
    uploaded_file = uploaded_file if 'uploaded_file' in locals() else None
    
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
                
                # Smart configuration
                with st.spinner("🧠 Analyzing dataset characteristics..."):
                    try:
                        from intelligence.smart_configurator import SmartDatasetConfigurator
                        
                        configurator = SmartDatasetConfigurator(df, time_col_confirmed, selected_metrics)
                        optimized_df = configurator.get_optimized_dataset()
                        
                        # Store in session state
                        st.session_state.df = optimized_df
                        st.session_state.time_col = time_col_confirmed
                        st.session_state.metric_cols = selected_metrics
                        st.session_state.configurator = configurator
                        st.session_state.data_loaded = True
                        st.session_state.use_default = False
                        
                        st.success("🎉 Data loaded and optimized successfully!")
                        
                        # Show smart insights
                        st.markdown("### 🧠 Smart Dataset Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 💡 Dataset Insights")
                            insights = configurator.get_insights()
                            for insight in insights:
                                st.info(insight)
                        
                        with col2:
                            st.markdown("#### 🚀 Performance Notes")
                            perf_recommendations = configurator.get_performance_recommendations()
                            for rec in perf_recommendations:
                                st.warning(rec)
                        
                        # Show recommended visualizations
                        st.markdown("#### 🎯 Recommended Visualizations")
                        recommended_viz = configurator.config.get('recommended_visualizations', [])
                        
                        viz_names = {
                            'dashboard': '📊 Multi-Metric Dashboard',
                            'heartbeat': '💓 Temporal Heartbeat', 
                            'correlation_explorer': '🌀 Correlation Explorer',
                            'correlation_evolution': '🌀 Correlation Evolution',
                            '3d_phase': '🎯 3D Phase Space',
                            'anomaly_landscape': '🏔️ Anomaly Landscape',
                            'pattern_radar': '🎯 Pattern Radar',
                            'clustering_evolution': '🧬 Clustering Evolution',
                            'flow_field': '🌊 Flow Field',
                            'heatmap': '🔥 Temporal Heatmap',
                            'rewind': '⏮ Time-Travel Rewind',
                            'evolution': '🧬 High-Dimensional Evolution',
                            'parallel_coords': '📊 Parallel Coordinates'
                        }
                        
                        recommended_names = [viz_names.get(viz, viz) for viz in recommended_viz[:8]]
                        st.success(f"✅ **{len(recommended_names)} visualizations** optimized for your data: {', '.join(recommended_names)}")
                        
                        st.balloons()
                        st.info("🚀 Navigate to **⏰ Chronosphere** to explore your optimized visualizations!")
                        
                    except ImportError:
                        # Fallback without smart configuration
                        st.session_state.df = df
                        st.session_state.time_col = time_col_confirmed
                        st.session_state.metric_cols = selected_metrics
                        st.session_state.data_loaded = True
                        st.session_state.use_default = False
                        
                        st.success("🎉 Data loaded successfully!")
                        st.info("Navigate to 📊 Explore or ⏰ Chronosphere to analyze your data")
                    except Exception as e:
                        st.error(f"Error during smart configuration: {str(e)}")
                        # Fallback
                        st.session_state.df = df
                        st.session_state.time_col = time_col_confirmed
                        st.session_state.metric_cols = selected_metrics
                        st.session_state.data_loaded = True
                        st.session_state.use_default = False
                        st.info("Data loaded with basic configuration")
        
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
                # Generate trend visualizations for uploaded dataset
                if hasattr(st.session_state, 'time_col') and st.session_state.time_col in df.columns:
                    time_col = st.session_state.time_col
                    
                    # Ensure time column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        try:
                            df[time_col] = pd.to_datetime(df[time_col])
                        except:
                            st.error(f"Could not convert {time_col} to datetime format")
                            st.stop()
                    
                    # Get numeric columns for trend analysis
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if numeric_cols:
                        # Let user select which metric to visualize
                        selected_metric = st.selectbox(
                            "Select metric for trend analysis:",
                            numeric_cols,
                            key="trend_metric_selector"
                        )
                        
                        # Daily trend
                        df_sorted = df.sort_values(time_col)
                        
                        # Create daily aggregation
                        daily_data = df_sorted.groupby(df_sorted[time_col].dt.date)[selected_metric].agg(['mean', 'sum', 'count']).reset_index()
                        daily_data[time_col] = pd.to_datetime(daily_data[time_col])
                        
                        # Choose aggregation method
                        agg_method = st.radio(
                            "Aggregation method:",
                            ["mean", "sum", "count"],
                            horizontal=True,
                            key="trend_agg_method"
                        )
                        
                        # Show summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Days", len(daily_data))
                        with col2:
                            st.metric("Min Value", f"{daily_data[agg_method].min():.2f}")
                        with col3:
                            st.metric("Max Value", f"{daily_data[agg_method].max():.2f}")
                        with col4:
                            trend_slope = np.polyfit(range(len(daily_data)), daily_data[agg_method], 1)[0]
                            trend_direction = "📈" if trend_slope > 0 else "📉" if trend_slope < 0 else "➡️"
                            st.metric("Trend", trend_direction)
                        
                        fig = px.line(daily_data, x=time_col, y=agg_method,
                                     title=f'Daily {selected_metric} - {agg_method.title()}',
                                     labels={agg_method: f'{selected_metric} ({agg_method})', time_col: 'Date'})
                        fig.update_traces(line_color='#00d4ff', line_width=2)
                        fig.update_layout(template='plotly_dark', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Hourly pattern (only if we have intraday data)
                        if len(daily_data) > 1:
                            hourly_pattern = df.groupby(df[time_col].dt.hour)[selected_metric].mean()
                            
                            # Only show hourly pattern if we have data across multiple hours
                            if len(hourly_pattern) > 1:
                                fig2 = px.bar(x=hourly_pattern.index, y=hourly_pattern.values,
                                             title=f'Average {selected_metric} by Hour',
                                             labels={'x': 'Hour', 'y': f'Average {selected_metric}'})
                                fig2.update_traces(marker_color='#ff00ff')
                                fig2.update_layout(template='plotly_dark', height=400)
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("💡 Hourly patterns not available - data appears to be daily aggregated")
                            
                            # Day of week pattern
                            dow_pattern = df.groupby(df[time_col].dt.day_name())[selected_metric].mean()
                            # Reorder days
                            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            dow_pattern = dow_pattern.reindex([day for day in day_order if day in dow_pattern.index])
                            
                            fig3 = px.bar(x=dow_pattern.index, y=dow_pattern.values,
                                         title=f'Average {selected_metric} by Day of Week',
                                         labels={'x': 'Day of Week', 'y': f'Average {selected_metric}'})
                            fig3.update_traces(marker_color='#00ff00')
                            fig3.update_layout(template='plotly_dark', height=400)
                            st.plotly_chart(fig3, use_container_width=True)
                        
                        # Monthly pattern (if data spans multiple months)
                        if (df[time_col].max() - df[time_col].min()).days > 60:
                            monthly_data = df.groupby(df[time_col].dt.to_period('M'))[selected_metric].agg(['mean', 'sum', 'count'])
                            monthly_data.index = monthly_data.index.to_timestamp()
                            
                            fig4 = px.line(monthly_data, x=monthly_data.index, y=agg_method,
                                          title=f'Monthly {selected_metric} - {agg_method.title()}',
                                          labels={agg_method: f'{selected_metric} ({agg_method})', 'index': 'Month'})
                            fig4.update_traces(line_color='#ffaa00', line_width=2)
                            fig4.update_layout(template='plotly_dark', height=400)
                            st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.warning("No numeric columns found for trend analysis")
                else:
                    st.warning("⚠️ No time column detected. Please configure your dataset first.")
        
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
            # Load pre-built visualizations for default dataset
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
            # Dynamic visualizations for uploaded datasets
            st.markdown("### 🚀 Dynamic Visualizations for Your Data")
            
            df = st.session_state.df
            time_col = getattr(st.session_state, 'time_col', None)
            metric_cols = getattr(st.session_state, 'metric_cols', [])
            
            if not time_col or not metric_cols:
                st.warning("⚠️ Please configure time and metric columns in the Upload page first")
            else:
                # Generate visualizations dynamically
                from visualization.chronosphere_viz import ChronosphereVisualizer
            
            try:
                # Initialize visualizer
                visualizer = ChronosphereVisualizer(df, time_col)
                
                viz_tabs = st.tabs([
                    "💓 Heartbeat",
                    "🌀 Correlation",
                    "📊 Parallel Coords", 
                    "🔥 Heatmap",
                    "🎯 3D Phase",
                    "🧬 Evolution",
                    "⏮ Rewind"
                ])
                
                with viz_tabs[0]:
                    st.markdown("#### Temporal Heartbeat")
                    st.markdown("Line thickness shows rate of change - creating a 'pulse' effect")
                    
                    if metric_cols:
                        selected_metric = st.selectbox("Select metric for heartbeat:", metric_cols, key="heartbeat_metric")
                        
                        with st.spinner("Generating temporal heartbeat..."):
                            fig = visualizer.temporal_heartbeat(selected_metric)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No metrics available for heartbeat visualization")
                
                with viz_tabs[1]:
                    st.markdown("#### Correlation Explorer")
                    st.markdown("Watch relationships evolve over time")
                    
                    if len(metric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_metric = st.selectbox("X-axis metric:", metric_cols, key="corr_x")
                        with col2:
                            y_metric = st.selectbox("Y-axis metric:", [m for m in metric_cols if m != x_metric], key="corr_y")
                        
                        # Check for entity column for color coding
                        entity_col = getattr(st.session_state, 'entity_column', None)
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        color_col = None
                        if categorical_cols:
                            color_col = st.selectbox("Color by (optional):", ['None'] + categorical_cols, key="corr_color")
                            if color_col == 'None':
                                color_col = None
                        
                        with st.spinner("Generating correlation explorer..."):
                            fig = visualizer.temporal_correlation_explorer(x_metric, y_metric, color_col)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 2 metrics for correlation visualization")
                
                with viz_tabs[2]:
                    st.markdown("#### Parallel Coordinates")
                    st.markdown("Visualize multidimensional patterns across all metrics")
                    
                    if len(metric_cols) >= 3:
                        selected_cols = st.multiselect("Select columns for parallel coordinates:", 
                                                     metric_cols, 
                                                     default=metric_cols[:5], 
                                                     key="parallel_cols")
                        
                        if selected_cols:
                            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            color_col = None
                            if categorical_cols:
                                color_col = st.selectbox("Color by (optional):", ['None'] + categorical_cols, key="parallel_color")
                                if color_col == 'None':
                                    color_col = None
                            
                            with st.spinner("Generating parallel coordinates..."):
                                fig = visualizer.parallel_coordinates(selected_cols, color_col)
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 3 metrics for parallel coordinates")
                
                with viz_tabs[3]:
                    st.markdown("#### Temporal Heatmap")
                    st.markdown("Discover daily and hourly patterns in your data")
                    
                    if metric_cols:
                        selected_metric = st.selectbox("Select metric for heatmap:", metric_cols, key="heatmap_metric")
                        
                        # Create time-based grouping columns
                        df_viz = df.copy()
                        df_viz['hour'] = pd.to_datetime(df_viz[time_col]).dt.hour
                        df_viz['day_of_week'] = pd.to_datetime(df_viz[time_col]).dt.dayofweek
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            x_group = st.selectbox("X-axis grouping:", ['hour', 'day_of_week'], key="heatmap_x")
                        with col2:
                            y_group = st.selectbox("Y-axis grouping:", ['day_of_week', 'hour'], key="heatmap_y")
                        
                        with st.spinner("Generating temporal heatmap..."):
                            # Update visualizer data with time columns
                            visualizer.df = df_viz
                            fig = visualizer.temporal_heatmap(selected_metric, x_group, y_group)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No metrics available for heatmap")
                
                with viz_tabs[4]:
                    st.markdown("#### 3D Phase Space")
                    st.markdown("Explore your data in 3D space with temporal trails")
                    
                    if len(metric_cols) >= 3:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_metric = st.selectbox("X-axis:", metric_cols, key="3d_x")
                        with col2:
                            y_metric = st.selectbox("Y-axis:", [m for m in metric_cols if m != x_metric], key="3d_y")
                        with col3:
                            z_metric = st.selectbox("Z-axis:", [m for m in metric_cols if m not in [x_metric, y_metric]], key="3d_z")
                        
                        # Options
                        col1, col2 = st.columns(2)
                        with col1:
                            show_trails = st.checkbox("Show temporal trails", value=True, key="3d_trails")
                        with col2:
                            entity_col = None
                            entity_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                            if entity_cols:
                                entity_col = st.selectbox("Group trails by:", ['None'] + entity_cols, key="3d_entity")
                                if entity_col == 'None':
                                    entity_col = None
                        
                        with st.spinner("Generating 3D phase space..."):
                            fig = visualizer.phase_space_3d(x_metric, y_metric, z_metric, 
                                                          entity_col=entity_col, show_trails=show_trails)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 3 metrics for 3D visualization")
                
                with viz_tabs[5]:
                    st.markdown("#### High-Dimensional Evolution")
                    st.markdown("Watch your data evolve using dimensionality reduction")
                    
                    if len(metric_cols) >= 4:
                        selected_features = st.multiselect("Select features for evolution:", 
                                                         metric_cols, 
                                                         default=metric_cols[:6], 
                                                         key="evolution_features")
                        
                        if len(selected_features) >= 4:
                            n_frames = st.slider("Animation frames:", 5, 20, 10, key="evolution_frames")
                            
                            with st.spinner("Generating evolution visualization (this may take a moment)..."):
                                fig = visualizer.temporal_umap_evolution(selected_features, n_frames)
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Select at least 4 features for evolution visualization")
                    else:
                        st.warning("Need at least 4 metrics for evolution visualization")
                
                with viz_tabs[6]:
                    st.markdown("#### ⭐ Time-Travel Rewind")
                    st.markdown("Detect anomalies and rewind to see their origins")
                    
                    if metric_cols:
                        selected_metric = st.selectbox("Select metric for anomaly detection:", metric_cols, key="rewind_metric")
                        
                        # Simple anomaly detection using IQR
                        with st.spinner("Detecting anomalies..."):
                            metric_data = df[selected_metric].dropna()
                            Q1 = metric_data.quantile(0.25)
                            Q3 = metric_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            anomalies = df[(df[selected_metric] < lower_bound) | (df[selected_metric] > upper_bound)]
                            anomaly_indices = anomalies.index.tolist()
                            
                            st.info(f"Detected {len(anomaly_indices)} anomalies in {selected_metric}")
                            
                            if anomaly_indices:
                                window_hours = st.slider("Context window (hours):", 6, 168, 48, key="rewind_window")
                                
                                fig = visualizer.chronosphere_rewind_interface(selected_metric, anomaly_indices, window_hours)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show anomaly details
                                st.markdown("### 🔍 Detected Anomalies")
                                anomaly_display = anomalies[[time_col, selected_metric]].copy()
                                anomaly_display['anomaly_type'] = anomaly_display[selected_metric].apply(
                                    lambda x: 'High' if x > upper_bound else 'Low'
                                )
                                st.dataframe(anomaly_display.head(10), use_container_width=True)
                            else:
                                st.info("No anomalies detected with current thresholds")
                                
                                # Still show the main timeline
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df[time_col],
                                    y=df[selected_metric],
                                    mode='lines',
                                    name=selected_metric,
                                    line=dict(color='#00d4ff', width=2)
                                ))
                                fig.update_layout(
                                    title=f"Timeline: {selected_metric}",
                                    template='plotly_dark',
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No metrics available for rewind interface")
                
                # Advanced Visualizations Section
                st.markdown("---")
                st.markdown("### 🚀 Advanced Analytics Suite")
                st.markdown("*Next-generation visualizations for deep temporal insights*")
                
                try:
                    from visualization.upload_viz_generator import UploadVizGenerator
                    
                    # Initialize advanced generator with smart configuration
                    configurator = getattr(st.session_state, 'configurator', None)
                    
                    if configurator:
                        # Use optimized dataset and configuration
                        adv_generator = UploadVizGenerator(df, time_col, metric_cols)
                        
                        # Show smart recommendations
                        st.info(f"🧠 **Smart Configuration Active**: Optimized for {configurator.config.get('data_frequency', 'unknown')} frequency data with {len(metric_cols)} metrics")
                    else:
                        # Fallback without smart configuration
                        adv_generator = UploadVizGenerator(df, time_col, metric_cols)
                    
                    advanced_tabs = st.tabs([
                        "🏔️ Anomaly Landscape",
                        "🌊 Flow Field", 
                        "🧬 Clustering Evolution",
                        "🎯 Pattern Radar",
                        "🌀 Correlation Evolution",
                        "📊 Multi-Dashboard"
                    ])
                    
                    with advanced_tabs[0]:
                        st.markdown("#### 🏔️ 3D Anomaly Landscape")
                        st.markdown("Explore anomalies across different detection methods in 3D space")
                        
                        if metric_cols:
                            # Use smart recommendation if available
                            if configurator:
                                config = configurator.get_visualization_config('anomaly_landscape')
                                default_metric = config.get('recommended_metric', metric_cols[0])
                                st.info(f"🧠 **Smart recommendation**: {default_metric} (best metric for anomaly detection)")
                            else:
                                default_metric = metric_cols[0]
                            
                            default_idx = metric_cols.index(default_metric) if default_metric in metric_cols else 0
                            anom_metric = st.selectbox("Select metric for anomaly landscape:", 
                                                     metric_cols, 
                                                     index=default_idx,
                                                     key="anom_landscape")
                            
                            with st.spinner("Generating 3D anomaly landscape..."):
                                fig = adv_generator.generate_anomaly_landscape(anom_metric)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.info("💡 **Tip**: Rotate the 3D view to explore anomaly patterns from different angles")
                        else:
                            st.warning("No metrics available for anomaly landscape")
                    
                    with advanced_tabs[1]:
                        st.markdown("#### 🌊 Temporal Flow Field")
                        st.markdown("Visualize how two metrics influence each other over time")
                        
                        if len(metric_cols) >= 2:
                            col1, col2 = st.columns(2)
                            with col1:
                                flow_x = st.selectbox("X-axis metric:", metric_cols, key="flow_x")
                            with col2:
                                flow_y = st.selectbox("Y-axis metric:", [m for m in metric_cols if m != flow_x], key="flow_y")
                            
                            with st.spinner("Generating temporal flow field..."):
                                fig = adv_generator.generate_temporal_flow_field(flow_x, flow_y)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.info("💡 **Tip**: Arrows show the direction of change - revealing causal relationships")
                        else:
                            st.warning("Need at least 2 metrics for flow field")
                    
                    with advanced_tabs[2]:
                        st.markdown("#### 🧬 Temporal Clustering Evolution")
                        st.markdown("Watch how data clusters form and evolve over time")
                        
                        if len(metric_cols) >= 2:
                            # Use smart recommendations
                            if configurator:
                                config = configurator.get_visualization_config('clustering_evolution')
                                recommended_features = config.get('recommended_features', metric_cols[:4])
                                optimal_clusters = config.get('n_clusters', 5)
                                st.info(f"🧠 **Smart recommendations**: {len(recommended_features)} features, {optimal_clusters} clusters")
                            else:
                                recommended_features = metric_cols[:4]
                                optimal_clusters = 5
                            
                            cluster_features = st.multiselect("Select features for clustering:", 
                                                            metric_cols, 
                                                            default=recommended_features, 
                                                            key="cluster_features")
                            n_clusters = st.slider("Number of clusters:", 2, 8, optimal_clusters, key="n_clusters")
                            
                            if len(cluster_features) >= 2:
                                with st.spinner("Generating clustering evolution..."):
                                    fig = adv_generator.generate_temporal_clustering_evolution(cluster_features, n_clusters)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    st.info("💡 **Tip**: Play the animation to see how clusters migrate and split over time")
                            else:
                                st.warning("Select at least 2 features for clustering")
                        else:
                            st.warning("Need at least 2 metrics for clustering")
                    
                    with advanced_tabs[3]:
                        st.markdown("#### 🎯 Pattern Detection Radar")
                        st.markdown("Comprehensive pattern analysis across all metrics")
                        
                        with st.spinner("Analyzing patterns across all metrics..."):
                            fig = adv_generator.generate_pattern_detection_radar()
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("""
                            **Pattern Types Explained:**
                            - **Trend**: Strength of linear trend (0-100%)
                            - **Seasonality**: Periodic pattern strength
                            - **Volatility**: Measure of variability
                            - **Anomalies**: Density of outliers
                            - **Growth**: Rate of change over time
                            """)
                    
                    with advanced_tabs[4]:
                        st.markdown("#### 🌀 Correlation Evolution")
                        st.markdown("Watch how correlations between metrics change over time")
                        
                        if len(metric_cols) >= 2:
                            # Use smart recommendation for window size
                            if configurator:
                                config = configurator.get_visualization_config('correlation_evolution')
                                optimal_window = config.get('window_size', 50)
                                st.info(f"🧠 **Smart recommendation**: Window size {optimal_window} (optimized for your data frequency)")
                            else:
                                optimal_window = 50
                            
                            window_size = st.slider("Analysis window size:", 
                                                   max(10, optimal_window // 2), 
                                                   min(200, optimal_window * 3), 
                                                   optimal_window, 
                                                   key="corr_window")
                            
                            with st.spinner("Generating correlation evolution..."):
                                fig = adv_generator.generate_correlation_matrix_evolution(window_size)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.info("💡 **Tip**: Press Play to see how relationships evolve - revealing structural changes")
                        else:
                            st.warning("Need at least 2 metrics for correlation evolution")
                    
                    with advanced_tabs[5]:
                        st.markdown("#### 📊 Multi-Metric Dashboard")
                        st.markdown("Comprehensive overview with trend analysis")
                        
                        with st.spinner("Generating comprehensive dashboard..."):
                            fig = adv_generator.generate_multimetric_dashboard()
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("💡 **Tip**: Dashed white lines show trend directions for each metric")
                
                except ImportError as e:
                    st.error(f"Advanced visualizations module not found: {str(e)}")
                    st.info("Please ensure all visualization modules are properly installed")
                except Exception as e:
                    st.error(f"Error generating advanced visualizations: {str(e)}")
                    st.exception(e)
                        
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
                st.exception(e)
                st.info("Please ensure your data has been properly configured with time and metric columns.")

elif page == "💡 Insights":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
        st.markdown("### 🎯 What you'll get with LLM Insights:")
        st.markdown("""
        - 🧠 **AI-Powered Analysis**: Deep pattern discovery using advanced LLMs
        - 📊 **Business Context**: Insights explained in plain English
        - 🎯 **Actionable Recommendations**: Specific steps to improve performance  
        - 🔍 **Anomaly Explanations**: Why unusual events occurred
        - 📈 **Predictive Insights**: What to expect next
        """)
    else:
        st.markdown("## 🧠 AI-Powered Insights")
        st.markdown("*Generate natural, contextual insights using advanced LLM analysis*")
        
        df = st.session_state.df
        time_col = getattr(st.session_state, 'time_col', None)
        metric_cols = getattr(st.session_state, 'metric_cols', [])
        
        if not time_col or not metric_cols:
            st.error("⚠️ Please configure time and metric columns in the Upload page first")
            st.stop()
        
        # Insights Configuration Panel
        st.markdown("### ⚙️ Analysis Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_types = st.multiselect(
                "Analysis Types:",
                ["comprehensive", "anomaly_focused", "predictive"],
                default=["comprehensive"],
                help="Select which types of analysis to perform"
            )
        
        with col2:
            use_llm = st.checkbox(
                "🤖 Enable LLM Analysis", 
                value=True,
                help="Use AI for natural language insights (may take 30-60 seconds)"
            )
            
            include_rule_based = st.checkbox(
                "📊 Include Statistical Analysis",
                value=True,
                help="Add rule-based statistical insights"
            )
        
        with col3:
            max_insights = st.slider(
                "Max Insights to Show:",
                5, 20, 10,
                help="Limit the number of insights displayed"
            )
        
        # Display current dataset info
        st.markdown("### 📋 Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Records", f"{len(df):,}")
        with col2:
            time_span = (pd.to_datetime(df[time_col]).max() - pd.to_datetime(df[time_col]).min()).days
            st.metric("Time Span", f"{time_span} days")
        with col3:
            st.metric("Metrics", len(metric_cols))
        with col4:
            completeness = (1 - df[metric_cols].isnull().sum().sum() / (len(df) * len(metric_cols))) * 100
            st.metric("Data Quality", f"{completeness:.1f}%")
        
        st.markdown("**Metrics being analyzed:** " + ", ".join(metric_cols))
        
        # Generate Insights Button
        if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
            
            if not analysis_types:
                st.error("Please select at least one analysis type")
                st.stop()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🧠 Initializing AI Insight Engine...")
                progress_bar.progress(10)
                
                # Import and initialize
                from intelligence.llm_insight_generator import SmartInsightEngine
                
                insight_engine = SmartInsightEngine(df, time_col, metric_cols)
                progress_bar.progress(20)
                
                if use_llm:
                    status_text.text("🤖 Connecting to LLM APIs...")
                    st.info("💡 **Tip**: First run may take longer as models initialize. Subsequent runs will be faster.")
                
                progress_bar.progress(30)
                
                status_text.text("📊 Analyzing statistical patterns...")
                
                # Generate insights
                results = insight_engine.generate_comprehensive_insights(use_llm=use_llm)
                
                progress_bar.progress(90)
                status_text.text("✅ Processing results...")
                
                # Store results in session state for export
                st.session_state.insights_results = results
                
                progress_bar.progress(100)
                status_text.text("🎉 Analysis complete!")
                
                # Display Results
                st.markdown("---")
                st.markdown("## 📋 Analysis Results")
                
                # Executive Summary
                if results.get('executive_summary'):
                    st.markdown("### 🎯 Executive Summary")
                    
                    with st.expander("📄 View Full Summary", expanded=True):
                        st.markdown(results['executive_summary'])
                
                # Show insights in tabs
                insights_tabs = st.tabs(["🏆 Top Insights", "🤖 AI Analysis", "📊 Statistical Analysis", "📈 Export"])
                
                with insights_tabs[0]:
                    st.markdown("### 🏆 Highest Priority Insights")
                    
                    combined_insights = results.get('combined_insights', [])
                    
                    if combined_insights:
                        # Show top insights with enhanced UI
                        for i, insight in enumerate(combined_insights[:max_insights], 1):
                            importance = insight.get('importance', 0)
                            title = insight.get('title', f"Insight {i}")
                            text = insight.get('text', '')
                            source = insight.get('source', 'unknown')
                            category = insight.get('category', 'general')
                            
                            # Priority styling
                            if importance > 80:
                                priority_color = "🔴"
                                priority_text = "HIGH"
                                border_color = "#ff4444"
                            elif importance > 60:
                                priority_color = "🟡"
                                priority_text = "MEDIUM"
                                border_color = "#ffaa00"
                            else:
                                priority_color = "🟢"
                                priority_text = "LOW"
                                border_color = "#44ff44"
                            
                            source_emoji = "🤖" if source == 'llm' else "📊"
                            
                            # Create insight card
                            st.markdown(f"""
                            <div style='border: 2px solid {border_color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: rgba(255,255,255,0.05);'>
                                <h4 style='color: {border_color}; margin-top: 0;'>
                                    {priority_color} {title}
                                </h4>
                                <div style='display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; color: #cccccc;'>
                                    <span><strong>Priority:</strong> {priority_text} ({importance}/100)</span>
                                    <span><strong>Source:</strong> {source_emoji} {source.upper()}</span>
                                    <span><strong>Category:</strong> {category.upper()}</span>
                                </div>
                                <p style='color: #ffffff; line-height: 1.6; margin-bottom: 0;'>{text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No insights generated. Please try with different settings.")
                
                with insights_tabs[1]:
                    st.markdown("### 🤖 AI-Generated Insights")
                    
                    llm_insights = results.get('llm_insights', [])
                    
                    if llm_insights:
                        st.success(f"✅ Generated {len(llm_insights)} AI insights using advanced language models")
                        
                        for insight in llm_insights:
                            title = insight.get('title', 'AI Insight')
                            text = insight.get('text', '')
                            importance = insight.get('importance', 0)
                            category = insight.get('category', 'general')
                            
                            with st.expander(f"🤖 {title} (Priority: {importance}/100)"):
                                st.markdown(f"**Category:** {category.title()}")
                                st.markdown(text)
                    else:
                        if use_llm:
                            st.warning("⚠️ LLM insights could not be generated. This may be due to:")
                            st.markdown("""
                            - API rate limits or connectivity issues
                            - API keys not configured (optional for basic functionality)
                            - Model temporarily unavailable
                            
                            The system has fallback modes and statistical insights are still available.
                            """)
                        else:
                            st.info("💡 LLM analysis was disabled. Enable it above for AI-powered insights.")
                
                with insights_tabs[2]:
                    st.markdown("### 📊 Statistical Analysis")
                    
                    rule_insights = results.get('rule_based_insights', [])
                    
                    if rule_insights:
                        st.success(f"✅ Generated {len(rule_insights)} statistical insights")
                        
                        # Group by type
                        insight_groups = {}
                        for insight in rule_insights:
                            insight_type = insight.get('type', 'general')
                            if insight_type not in insight_groups:
                                insight_groups[insight_type] = []
                            insight_groups[insight_type].append(insight)
                        
                        for group_name, group_insights in insight_groups.items():
                            st.markdown(f"#### {group_name.replace('_', ' ').title()}")
                            
                            for insight in group_insights:
                                title = insight.get('title', 'Statistical Insight')
                                text = insight.get('text', '')
                                importance = insight.get('importance', 0)
                                
                                st.markdown(f"**{title}** (Score: {importance}/100)")
                                st.markdown(f"{text}")
                                st.markdown("---")
                    else:
                        st.info("📊 No statistical insights available. This feature may need configuration.")
                
                with insights_tabs[3]:
                    st.markdown("### 📤 Export Results")
                    
                    if results:
                        # Generate export data
                        export_data = {
                            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'dataset_info': {
                                'records': len(df),
                                'time_span_days': time_span,
                                'metrics': metric_cols
                            },
                            'analysis_config': {
                                'types': analysis_types,
                                'llm_enabled': use_llm,
                                'max_insights': max_insights
                            },
                            'results_summary': {
                                'total_insights': len(results.get('combined_insights', [])),
                                'llm_insights': len(results.get('llm_insights', [])),
                                'rule_insights': len(results.get('rule_based_insights', []))
                            }
                        }
                        
                        # Executive Summary Export
                        if results.get('executive_summary'):
                            summary_md = f"""# TimeLense AI Insights Report
Generated: {export_data['generation_time']}

## Dataset Information
- Records: {export_data['dataset_info']['records']:,}
- Time Span: {export_data['dataset_info']['time_span_days']} days
- Metrics: {', '.join(export_data['dataset_info']['metrics'])}

## Executive Summary
{results['executive_summary']}

## Detailed Insights
"""
                            
                            # Add top insights
                            for i, insight in enumerate(results.get('combined_insights', [])[:10], 1):
                                summary_md += f"""
### {i}. {insight.get('title', f'Insight {i}')}
**Priority:** {insight.get('importance', 0)}/100 | **Source:** {insight.get('source', 'unknown').upper()}

{insight.get('text', '')}

---
"""
                            
                            st.download_button(
                                "📄 Download Executive Summary (Markdown)",
                                data=summary_md,
                                file_name=f"timelense_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime='text/markdown'
                            )
                        
                        # JSON Export for API integration
                        import json
                        json_export = json.dumps({
                            'metadata': export_data,
                            'insights': results.get('combined_insights', [])
                        }, indent=2, default=str)
                        
                        st.download_button(
                            "💾 Download Full Results (JSON)",
                            data=json_export,
                            file_name=f"timelense_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime='application/json'
                        )
                        
                        # CSV Export for spreadsheet analysis
                        if results.get('combined_insights'):
                            insights_df = pd.DataFrame(results['combined_insights'])
                            csv_export = insights_df.to_csv(index=False)
                            
                            st.download_button(
                                "📊 Download Insights Table (CSV)",
                                data=csv_export,
                                file_name=f"timelense_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
                        
                        st.success("✅ Export options available above. Choose the format that works best for your workflow.")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.balloons()
                
            except ImportError as e:
                st.error(f"❌ LLM Insights module not available: {str(e)}")
                st.info("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
                
            except Exception as e:
                st.error(f"❌ Error generating insights: {str(e)}")
                st.markdown("""
                **Troubleshooting Tips:**
                1. Check your internet connection for LLM API access
                2. Verify your data has sufficient records (recommended: 100+)
                3. Ensure time column is properly formatted
                4. Try disabling LLM analysis if errors persist
                """)
                
                # Show debug info in expander
                with st.expander("🔧 Debug Information"):
                    st.exception(e)
        
        # Show usage tips
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Data Quality**
            - Use datasets with clear patterns
            - Ensure < 10% missing values
            - Include 3-5 related metrics
            - Have 30+ days of data
            """)
        
        with col2:
            st.markdown("""
            **🤖 LLM Analysis**  
            - First run may take 60+ seconds
            - Works without API keys (fallback mode)
            - Best with business context metrics
            - Handles various data frequencies
            """)
        
        with col3:
            st.markdown("""
            **📈 Analysis Types**
            - **Comprehensive**: General business insights
            - **Anomaly Focused**: Deep dive on outliers  
            - **Predictive**: Forward-looking trends
            """)
        
        # Show sample results if available
        if hasattr(st.session_state, 'insights_results') and st.session_state.insights_results:
            st.success("💾 Previous analysis results are cached. Click 'Generate AI Insights' to refresh with new settings.")

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
    <p>TimeLense | Time-Travel Data Debugging System</p>
</div>
""", unsafe_allow_html=True)