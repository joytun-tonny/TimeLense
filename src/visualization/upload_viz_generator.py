"""
Advanced Visualization Generator for Uploaded Datasets
Dynamically creates sophisticated visualizations for any temporal dataset
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class UploadVizGenerator:
    """Advanced visualization generator for uploaded datasets"""
    
    def __init__(self, df: pd.DataFrame, time_column: str, metric_columns: List[str]):
        self.df = df.copy()
        self.time_col = time_column
        self.metric_cols = metric_columns
        self.theme = self._get_theme()
        
        # Preprocess data
        self._preprocess_data()
        
    def _get_theme(self) -> Dict:
        """Chronosphere theme"""
        return {
            'primary': '#00d4ff',
            'secondary': '#ff00ff', 
            'accent': '#ffff00',
            'background': '#0a0a0a',
            'paper': '#1a1a1a',
            'text': '#ffffff',
            'grid': '#2a2a2a',
            'colors': ['#00d4ff', '#ff00ff', '#ffff00', '#00ff88', '#ff4444', '#8844ff', '#ff8800']
        }
    
    def _preprocess_data(self):
        """Preprocess data for visualizations"""
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_col]):
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        
        # Sort by time
        self.df = self.df.sort_values(self.time_col)
        
        # Add time-based features
        self.df['_hour'] = self.df[self.time_col].dt.hour
        self.df['_day_of_week'] = self.df[self.time_col].dt.dayofweek
        self.df['_month'] = self.df[self.time_col].dt.month
        self.df['_year'] = self.df[self.time_col].dt.year
        
        # Clean numeric data
        for col in self.metric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def generate_multimetric_dashboard(self) -> go.Figure:
        """Generate comprehensive multi-metric dashboard"""
        n_metrics = len(self.metric_cols)
        
        if n_metrics == 0:
            return self._empty_figure("No metrics available")
        
        # Create subplots
        rows = min(3, n_metrics)
        cols = min(2, (n_metrics + rows - 1) // rows)
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{col} Over Time" for col in self.metric_cols[:rows*cols]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        for i, metric in enumerate(self.metric_cols[:rows*cols]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Main time series
            fig.add_trace(
                go.Scatter(
                    x=self.df[self.time_col],
                    y=self.df[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=self.theme['colors'][i % len(self.theme['colors'])], width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add trend line
            if len(self.df) > 10:
                z = np.polyfit(range(len(self.df)), self.df[metric].fillna(0), 1)
                trend_line = np.poly1d(z)(range(len(self.df)))
                
                fig.add_trace(
                    go.Scatter(
                        x=self.df[self.time_col],
                        y=trend_line,
                        mode='lines',
                        line=dict(color='white', width=1, dash='dash'),
                        showlegend=False,
                        opacity=0.7
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="📊 Multi-Metric Dashboard",
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=600,
            showlegend=False
        )
        
        return fig
    
    def generate_correlation_matrix_evolution(self, window_size: int = 50) -> go.Figure:
        """Generate animated correlation matrix"""
        if len(self.metric_cols) < 2:
            return self._empty_figure("Need at least 2 metrics for correlation")
        
        # Calculate rolling correlations
        frames = []
        step_size = max(1, len(self.df) // 20)  # 20 frames max
        
        for i in range(window_size, len(self.df), step_size):
            window_data = self.df.iloc[i-window_size:i][self.metric_cols]
            corr_matrix = window_data.corr()
            
            # Create heatmap data
            frames.append(go.Frame(
                data=[go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False
                )],
                name=str(i)
            ))
        
        if not frames:
            return self._empty_figure("Not enough data for correlation evolution")
        
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        fig.update_layout(
            title="🌀 Correlation Matrix Evolution",
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=600,
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}], 
                     "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], 
                     "label": "Pause", "method": "animate"}
                ],
                "type": "buttons", "x": 0.1, "y": 1.1
            }]
        )
        
        return fig
    
    def generate_anomaly_landscape(self, metric: str) -> go.Figure:
        """Generate 3D anomaly landscape"""
        if metric not in self.df.columns:
            return self._empty_figure(f"Metric {metric} not found")
        
        # Detect anomalies using multiple methods
        data = self.df[metric].dropna()
        
        # Method 1: IQR
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        iqr_anomalies = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
        
        # Method 2: Z-score
        z_scores = np.abs(stats.zscore(data))
        zscore_anomalies = z_scores > 3
        
        # Method 3: Moving average deviation
        window = min(20, len(data) // 4)
        if window > 2:
            rolling_mean = data.rolling(window=window, center=True).mean()
            rolling_std = data.rolling(window=window, center=True).std()
            ma_anomalies = np.abs(data - rolling_mean) > 2 * rolling_std
        else:
            ma_anomalies = pd.Series([False] * len(data), index=data.index)
        
        # Create 3D surface
        fig = go.Figure()
        
        # Time grid
        time_numeric = pd.to_numeric(self.df[self.time_col])
        time_grid = np.linspace(time_numeric.min(), time_numeric.max(), 50)
        anomaly_grid = np.linspace(0, 1, 3)  # IQR, Z-score, MA
        
        # Create anomaly surface
        X, Y = np.meshgrid(time_grid, anomaly_grid)
        Z = np.zeros_like(X)
        
        # Fill Z values based on anomaly density
        for i, t in enumerate(time_grid):
            time_mask = np.abs(time_numeric - t) < (time_numeric.max() - time_numeric.min()) / 100
            if time_mask.any():
                Z[0, i] = iqr_anomalies[time_mask].sum() / time_mask.sum()
                Z[1, i] = zscore_anomalies[time_mask].sum() / time_mask.sum() 
                Z[2, i] = ma_anomalies[time_mask].sum() / time_mask.sum()
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Plasma',
            name='Anomaly Density',
            hovertemplate='Time: %{x}<br>Method: %{y}<br>Density: %{z:.2f}<extra></extra>'
        ))
        
        # Add scatter points for actual anomalies
        anomaly_mask = iqr_anomalies | zscore_anomalies | ma_anomalies
        if anomaly_mask.any():
            anomaly_data = self.df[anomaly_mask.reindex(self.df.index, fill_value=False)]
            
            fig.add_trace(go.Scatter3d(
                x=pd.to_numeric(anomaly_data[self.time_col]),
                y=[1] * len(anomaly_data),  # Middle layer
                z=anomaly_data[metric],
                mode='markers',
                marker=dict(size=8, color='red', symbol='x'),
                name='Detected Anomalies'
            ))
        
        fig.update_layout(
            title=f"🏔️ Anomaly Landscape: {metric}",
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Detection Method",
                zaxis_title="Anomaly Density / Value",
                yaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=['IQR', 'Z-Score', 'Moving Avg'])
            ),
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=700
        )
        
        return fig
    
    def generate_temporal_clustering_evolution(self, features: List[str], n_clusters: int = 5) -> go.Figure:
        """Generate temporal clustering evolution"""
        if len(features) < 2:
            return self._empty_figure("Need at least 2 features for clustering")
        
        # Prepare data
        feature_data = self.df[features].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Time windows for evolution
        window_size = max(50, len(self.df) // 10)
        step_size = max(10, len(self.df) // 15)
        
        frames = []
        
        for i in range(window_size, len(self.df), step_size):
            window_data = scaled_data[i-window_size:i]
            
            if len(window_data) < n_clusters:
                continue
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(window_data)
            
            # Dimensionality reduction for visualization
            if len(features) > 2:
                pca = PCA(n_components=2, random_state=42)
                coords_2d = pca.fit_transform(window_data)
            else:
                coords_2d = window_data[:, :2]
            
            # Create frame
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=coords_2d[:, 0],
                    y=coords_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=clusters,
                        colorscale='Set1',
                        showscale=False
                    ),
                    text=[f"Cluster {c}" for c in clusters],
                    hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                )],
                name=str(i)
            ))
        
        if not frames:
            return self._empty_figure("Not enough data for clustering evolution")
        
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        fig.update_layout(
            title=f"🧬 Temporal Clustering Evolution ({n_clusters} clusters)",
            xaxis_title="Principal Component 1" if len(features) > 2 else features[0],
            yaxis_title="Principal Component 2" if len(features) > 2 else features[1],
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=600,
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}], 
                     "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], 
                     "label": "Pause", "method": "animate"}
                ],
                "type": "buttons", "x": 0.1, "y": 1.1
            }]
        )
        
        return fig
    
    def generate_pattern_detection_radar(self) -> go.Figure:
        """Generate radar chart showing various pattern detections"""
        patterns = {}
        
        for metric in self.metric_cols:
            if metric not in self.df.columns:
                continue
                
            data = self.df[metric].dropna()
            
            if len(data) < 10:
                continue
            
            # Pattern 1: Trend strength
            x = np.arange(len(data))
            slope, _, r_value, _, _ = stats.linregress(x, data)
            trend_strength = abs(r_value) * 100
            
            # Pattern 2: Seasonality (simple autocorrelation)
            if len(data) > 24:
                autocorr = pd.Series(data).autocorr(lag=min(24, len(data)//4))
                seasonality = abs(autocorr) * 100 if not pd.isna(autocorr) else 0
            else:
                seasonality = 0
            
            # Pattern 3: Volatility
            volatility = (data.std() / data.mean() * 100) if data.mean() != 0 else 0
            volatility = min(volatility, 100)  # Cap at 100%
            
            # Pattern 4: Anomaly density
            z_scores = np.abs(stats.zscore(data))
            anomaly_density = (z_scores > 2).sum() / len(data) * 100
            
            # Pattern 5: Growth rate
            if len(data) > 1:
                growth_rate = abs((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100) if data.iloc[0] != 0 else 0
                growth_rate = min(growth_rate, 100)  # Cap at 100%
            else:
                growth_rate = 0
            
            patterns[metric] = {
                'Trend': trend_strength,
                'Seasonality': seasonality,
                'Volatility': volatility,
                'Anomalies': anomaly_density,
                'Growth': growth_rate
            }
        
        if not patterns:
            return self._empty_figure("No patterns could be detected")
        
        # Create radar chart
        fig = go.Figure()
        
        pattern_types = ['Trend', 'Seasonality', 'Volatility', 'Anomalies', 'Growth']
        
        for i, (metric, pattern_data) in enumerate(patterns.items()):
            values = [pattern_data.get(p, 0) for p in pattern_types]
            values.append(values[0])  # Close the polygon
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=pattern_types + [pattern_types[0]],
                fill='toself',
                name=metric,
                fillcolor=f"rgba{self.theme['colors'][i % len(self.theme['colors'])][4:-1]}, 0.1)",
                line=dict(color=self.theme['colors'][i % len(self.theme['colors'])], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            title="🎯 Pattern Detection Radar",
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def generate_temporal_flow_field(self, x_metric: str, y_metric: str) -> go.Figure:
        """Generate temporal flow field visualization"""
        if x_metric not in self.df.columns or y_metric not in self.df.columns:
            return self._empty_figure("Metrics not found")
        
        # Calculate gradients (flow vectors)
        x_data = self.df[x_metric].fillna(method='ffill').fillna(method='bfill')
        y_data = self.df[y_metric].fillna(method='ffill').fillna(method='bfill')
        
        if len(x_data) < 10:
            return self._empty_figure("Not enough data for flow field")
        
        # Calculate derivatives (flow directions)
        dx = np.gradient(x_data)
        dy = np.gradient(y_data)
        
        # Create grid
        x_grid = np.linspace(x_data.min(), x_data.max(), 15)
        y_grid = np.linspace(y_data.min(), y_data.max(), 15)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate flow vectors onto grid
        from scipy.interpolate import griddata
        
        points = np.column_stack((x_data.values, y_data.values))
        dx_grid = griddata(points, dx, (X, Y), method='cubic', fill_value=0)
        dy_grid = griddata(points, dy, (X, Y), method='cubic', fill_value=0)
        
        # Create flow field
        fig = go.Figure()
        
        # Background trajectory
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Trajectory',
            line=dict(color=self.theme['primary'], width=2),
            marker=dict(size=4, color=self.theme['primary'])
        ))
        
        # Flow vectors
        for i in range(0, len(X), 2):  # Skip some for clarity
            for j in range(0, len(Y[0]), 2):
                if not (np.isnan(dx_grid[i, j]) or np.isnan(dy_grid[i, j])):
                    # Normalize arrow length
                    length = np.sqrt(dx_grid[i, j]**2 + dy_grid[i, j]**2)
                    if length > 0:
                        scale = min(0.1, 1/length)  # Adaptive scaling
                        
                        fig.add_annotation(
                            x=X[i, j], y=Y[i, j],
                            ax=X[i, j] + dx_grid[i, j] * scale,
                            ay=Y[i, j] + dy_grid[i, j] * scale,
                            arrowhead=2, arrowsize=1, arrowwidth=1,
                            arrowcolor=self.theme['accent'],
                            showarrow=True
                        )
        
        fig.update_layout(
            title=f"🌊 Temporal Flow Field: {x_metric} vs {y_metric}",
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.theme['text'])
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=400
        )
        return fig
    
    def generate_all_visualizations(self) -> Dict[str, go.Figure]:
        """Generate all available visualizations"""
        viz_dict = {}
        
        try:
            viz_dict['dashboard'] = self.generate_multimetric_dashboard()
        except Exception as e:
            viz_dict['dashboard'] = self._empty_figure(f"Dashboard error: {str(e)}")
        
        try:
            viz_dict['correlation_evolution'] = self.generate_correlation_matrix_evolution()
        except Exception as e:
            viz_dict['correlation_evolution'] = self._empty_figure(f"Correlation error: {str(e)}")
        
        if self.metric_cols:
            try:
                viz_dict['anomaly_landscape'] = self.generate_anomaly_landscape(self.metric_cols[0])
            except Exception as e:
                viz_dict['anomaly_landscape'] = self._empty_figure(f"Anomaly error: {str(e)}")
        
        try:
            viz_dict['clustering_evolution'] = self.generate_temporal_clustering_evolution(self.metric_cols[:4])
        except Exception as e:
            viz_dict['clustering_evolution'] = self._empty_figure(f"Clustering error: {str(e)}")
        
        try:
            viz_dict['pattern_radar'] = self.generate_pattern_detection_radar()
        except Exception as e:
            viz_dict['pattern_radar'] = self._empty_figure(f"Pattern error: {str(e)}")
        
        if len(self.metric_cols) >= 2:
            try:
                viz_dict['flow_field'] = self.generate_temporal_flow_field(self.metric_cols[0], self.metric_cols[1])
            except Exception as e:
                viz_dict['flow_field'] = self._empty_figure(f"Flow field error: {str(e)}")
        
        return viz_dict