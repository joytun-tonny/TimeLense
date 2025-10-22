"""
Chronosphere Visualization Module
Core visualization functions for temporal data debugging
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class ChronosphereVisualizer:
    """Main visualization class for Chronosphere"""
    
    def __init__(self, df: pd.DataFrame, time_column: str):
        self.df = df.copy()
        self.time_col = time_column
        self.theme = self._get_chronosphere_theme()
        
    def _get_chronosphere_theme(self) -> Dict:
        """Get Chronosphere color theme"""
        return {
            'primary': '#00d4ff',
            'secondary': '#ff00ff',
            'accent': '#ffff00',
            'background': '#0a0a0a',
            'paper': '#1a1a1a',
            'text': '#ffffff',
            'grid': '#2a2a2a'
        }
    
    def temporal_heartbeat(self, metric_column: str, title: Optional[str] = None) -> go.Figure:
        """
        Create temporal heartbeat visualization with pulse effect
        
        Args:
            metric_column: Column to visualize
            title: Chart title
        """
        if title is None:
            title = f"Temporal Heartbeat: {metric_column}"
        
        fig = go.Figure()
        
        # Calculate rate of change
        rate_of_change = self.df[metric_column].diff().abs()
        rate_of_change = rate_of_change.fillna(0)
        
        # Normalize for marker size
        if rate_of_change.max() > 0:
            normalized_rate = (rate_of_change - rate_of_change.min()) / (rate_of_change.max() - rate_of_change.min())
            marker_sizes = 5 + normalized_rate * 15  # 5-20 pixel range
        else:
            marker_sizes = [5] * len(self.df)
        
        # Main line
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=self.df[metric_column],
            mode='lines',
            name=metric_column,
            line=dict(color=self.theme['primary'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(0, 212, 255, 0.1)",
            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Pulse markers
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=self.df[metric_column],
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=rate_of_change,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Change Rate", x=1.1)
            ),
            name='Pulse',
            showlegend=False,
            hovertemplate='<b>Change Rate:</b> %{marker.color:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=metric_column,
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def temporal_correlation_explorer(self, x_col: str, y_col: str, 
                                     color_col: Optional[str] = None,
                                     n_frames: int = 20) -> go.Figure:
        """
        Create animated correlation explorer
        
        Args:
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Optional color grouping
            n_frames: Number of animation frames
        """
        # Create time bins for animation
        df_copy = self.df.copy()
        df_copy['time_index'] = pd.factorize(pd.cut(
            pd.to_datetime(df_copy[self.time_col]).astype(np.int64),
            bins=n_frames
        ))[0]
        
        # Filter out invalid bins
        df_copy = df_copy[df_copy['time_index'] >= 0]
        
        if len(df_copy) == 0:
            # Fallback to simple scatter if animation fails
            return self._simple_scatter(x_col, y_col, color_col)
        
        fig = px.scatter(
            df_copy,
            x=x_col,
            y=y_col,
            color=color_col if color_col else None,
            animation_frame='time_index',
            hover_data=[self.time_col],
            title=f"{x_col} vs {y_col} Over Time",
            template='plotly_dark'
        )
        
        fig.update_traces(
            marker=dict(size=8, line=dict(width=1, color='white'))
        )
        
        fig.update_layout(
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=600
        )
        
        return fig
    
    def _simple_scatter(self, x_col: str, y_col: str, color_col: Optional[str] = None) -> go.Figure:
        """Simple scatter plot fallback"""
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=color_col if color_col else None,
            title=f"{x_col} vs {y_col}",
            template='plotly_dark'
        )
        return fig
    
    def phase_space_3d(self, x_col: str, y_col: str, z_col: str,
                      color_col: Optional[str] = None,
                      entity_col: Optional[str] = None,
                      show_trails: bool = True,
                      max_entities: int = 10) -> go.Figure:
        """
        Create 3D phase space with ghost trails
        
        Args:
            x_col, y_col, z_col: Columns for 3D axes
            color_col: Column for color mapping
            entity_col: Column for entity tracking
            show_trails: Whether to show trajectory trails
            max_entities: Maximum entities to show trails for
        """
        fig = go.Figure()
        
        # Main scatter
        hover_text = [
            f"Time: {t}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}"
            for t, x, y, z in zip(
                self.df[self.time_col],
                self.df[x_col],
                self.df[y_col],
                self.df[z_col]
            )
        ]
        
        color_values = self.df[color_col] if color_col and color_col in self.df.columns else self.df.index
        
        fig.add_trace(go.Scatter3d(
            x=self.df[x_col],
            y=self.df[y_col],
            z=self.df[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col if color_col else "Index")
            ),
            text=hover_text,
            hoverinfo='text',
            name='Data Points'
        ))
        
        # Add ghost trails
        if show_trails and entity_col and entity_col in self.df.columns:
            entities = self.df[entity_col].value_counts().head(max_entities).index
            
            for entity in entities:
                entity_data = self.df[self.df[entity_col] == entity].sort_values(self.time_col)
                
                if len(entity_data) > 1:
                    fig.add_trace(go.Scatter3d(
                        x=entity_data[x_col],
                        y=entity_data[y_col],
                        z=entity_data[z_col],
                        mode='lines',
                        line=dict(width=2, color='rgba(255,255,255,0.3)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        fig.update_layout(
            title="3D Temporal Phase Space",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                bgcolor=self.theme['paper'],
                xaxis=dict(backgroundcolor=self.theme['background'], gridcolor=self.theme['grid']),
                yaxis=dict(backgroundcolor=self.theme['background'], gridcolor=self.theme['grid']),
                zaxis=dict(backgroundcolor=self.theme['background'], gridcolor=self.theme['grid'])
            ),
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=700
        )
        
        return fig
    
    def temporal_umap_evolution(self, feature_cols: List[str], n_frames: int = 10) -> go.Figure:
        """
        Create t-SNE/UMAP evolution animation
        
        Args:
            feature_cols: Columns to use for dimensionality reduction
            n_frames: Number of animation frames
        """
        # Create time bins
        df_copy = self.df.copy()
        time_bins = pd.qcut(df_copy.index, q=min(n_frames, len(df_copy)), labels=False, duplicates='drop')
        df_copy['time_frame'] = time_bins
        
        frames = []
        
        for frame in sorted(df_copy['time_frame'].unique()):
            frame_data = df_copy[df_copy['time_frame'] <= frame]
            
            if len(frame_data) > 10:
                # Apply t-SNE
                features = frame_data[feature_cols].fillna(0)
                
                # Standardize
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Use PCA if too few samples for t-SNE
                if len(features_scaled) < 30:
                    reducer = PCA(n_components=2, random_state=42)
                else:
                    perplexity = min(30, len(features_scaled) - 1)
                    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                
                embedded = reducer.fit_transform(features_scaled)
                
                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=embedded[:, 0],
                        y=embedded[:, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=frame_data.index,
                            colorscale='Plasma',
                            showscale=True
                        ),
                        text=[f"Index: {i}" for i in frame_data.index],
                        hoverinfo='text'
                    )],
                    name=str(frame)
                ))
        
        if not frames:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough data for dimensionality reduction",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        fig.update_layout(
            title="High-Dimensional Evolution (t-SNE/PCA)",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ],
                x=0.1, y=1.15
            )],
            height=600
        )
        
        return fig
    
    def chronosphere_rewind_interface(self, metric_col: str, 
                                     anomaly_indices: List[int],
                                     window_hours: int = 48) -> go.Figure:
        """
        Create the main Chronosphere time-travel interface
        
        Args:
            metric_col: Metric to display
            anomaly_indices: Indices of detected anomalies
            window_hours: Context window in hours
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Full Timeline - Click Anomaly to Rewind", "Rewind Context Window"),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )
        
        # Top panel: Full timeline
        fig.add_trace(
            go.Scatter(
                x=self.df[self.time_col],
                y=self.df[metric_col],
                mode='lines',
                name='Timeline',
                line=dict(color=self.theme['primary'], width=2),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Mark anomalies
        if anomaly_indices:
            anomaly_data = self.df.iloc[anomaly_indices]
            fig.add_trace(
                go.Scatter(
                    x=anomaly_data[self.time_col],
                    y=anomaly_data[metric_col],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='white')),
                    hovertemplate='<b>ANOMALY</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Bottom panel: Show context for first anomaly
            if len(anomaly_indices) > 0:
                anomaly_idx = anomaly_indices[0]
                anomaly_time = self.df.iloc[anomaly_idx][self.time_col]
                
                # Get context window
                from datetime import timedelta
                start_time = anomaly_time - timedelta(hours=window_hours)
                end_time = anomaly_time + timedelta(hours=window_hours // 2)
                
                mask = (self.df[self.time_col] >= start_time) & (self.df[self.time_col] <= end_time)
                context_data = self.df[mask]
                
                # Context line
                fig.add_trace(
                    go.Scatter(
                        x=context_data[self.time_col],
                        y=context_data[metric_col],
                        mode='lines+markers',
                        name='Context',
                        line=dict(color=self.theme['accent'], width=2),
                        marker=dict(size=6),
                        hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Highlight the anomaly
                fig.add_trace(
                    go.Scatter(
                        x=[anomaly_time],
                        y=[self.df.iloc[anomaly_idx][metric_col]],
                        mode='markers',
                        name='Focus',
                        marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                        hovertemplate='<b>ANOMALY FOCUS</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    text=f"Rewound {window_hours}h before anomaly",
                    xref="x2", yref="y2",
                    x=start_time, y=context_data[metric_col].max(),
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=self.theme['secondary'],
                    font=dict(color=self.theme['secondary'])
                )
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            showlegend=True,
            title="⏰ Chronosphere: Time-Travel Debugging Interface",
            hovermode='closest'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text=metric_col, row=1, col=1)
        fig.update_yaxes(title_text=metric_col, row=2, col=1)
        
        return fig
    
    def parallel_coordinates(self, columns: List[str], color_col: Optional[str] = None) -> go.Figure:
        """
        Create parallel coordinates plot
        
        Args:
            columns: Columns to display as parallel axes
            color_col: Column for color coding
        """
        # Prepare dimensions
        dimensions = []
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                dimensions.append(dict(
                    range=[self.df[col].min(), self.df[col].max()],
                    label=col,
                    values=self.df[col]
                ))
        
        if not dimensions:
            fig = go.Figure()
            fig.add_annotation(text="No numeric columns found", x=0.5, y=0.5)
            return fig
        
        # Color values
        if color_col and color_col in self.df.columns:
            color_values = self.df[color_col]
        else:
            color_values = self.df.index
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=color_values,
                    colorscale='Viridis',
                    showscale=True
                ),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title='Parallel Coordinates: Multivariate Patterns',
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=600
        )
        
        return fig
    
    def temporal_heatmap(self, metric_col: str, 
                        x_groupby: str = 'hour',
                        y_groupby: str = 'day_of_week') -> go.Figure:
        """
        Create temporal heatmap
        
        Args:
            metric_col: Metric to aggregate
            x_groupby: Column for X-axis grouping
            y_groupby: Column for Y-axis grouping
        """
        # Create pivot table
        pivot_data = self.df.groupby([y_groupby, x_groupby])[metric_col].mean().reset_index()
        pivot_table = pivot_data.pivot(index=y_groupby, columns=x_groupby, values=metric_col)
        
        # Day names if day_of_week
        if y_groupby == 'day_of_week':
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            pivot_table.index = [day_names[i] if i < len(day_names) else str(i) for i in pivot_table.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Plasma',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b>, %{x}<br>Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Temporal Heatmap: {metric_col} by {x_groupby} and {y_groupby}',
            xaxis_title=x_groupby,
            yaxis_title=y_groupby,
            template='plotly_dark',
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['paper'],
            font=dict(color=self.theme['text']),
            height=500
        )
        
        return fig
    
    def trend_analysis_overview(self, metric_cols: List[str]) -> go.Figure:
        """
        Create comprehensive trend analysis overview with multiple metrics
        """
        fig = make_subplots(
            rows=len(metric_cols), 
            cols=1,
            subplot_titles=[f"{col} Trend" for col in metric_cols],
            vertical_spacing=0.05,
            shared_xaxes=True
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(metric_cols, 1):
            # Calculate moving averages
            df_sorted = self.df.sort_values(self.time_col)
            df_sorted[f'{col}_ma7'] = df_sorted[col].rolling(window=min(7, len(df_sorted)//10)).mean()
            df_sorted[f'{col}_ma30'] = df_sorted[col].rolling(window=min(30, len(df_sorted)//5)).mean()
            
            # Raw data
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[self.time_col],
                    y=df_sorted[col],
                    mode='lines',
                    name=f'{col} Raw',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.6,
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            # Moving averages
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[self.time_col],
                    y=df_sorted[f'{col}_ma7'],
                    mode='lines',
                    name=f'7-period MA',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[self.time_col],
                    y=df_sorted[f'{col}_ma30'],
                    mode='lines',
                    name=f'30-period MA',
                    line=dict(color=colors[i % len(colors)], width=3),
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )
        
        fig.update_layout(
            title="Trend Analysis Overview",
            showlegend=True,
            plot_bgcolor=self.theme['paper'],
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=200 * len(metric_cols)
        )
        
        return fig
    
    def seasonal_decomposition(self, metric_col: str) -> go.Figure:
        """
        Create seasonal decomposition visualization
        """
        df_sorted = self.df.sort_values(self.time_col).copy()
        
        # Simple seasonal decomposition using rolling statistics
        # Calculate trend (long-term moving average)
        window_size = max(min(30, len(df_sorted)//4), 3)
        df_sorted['trend'] = df_sorted[metric_col].rolling(window=window_size, center=True).mean()
        
        # Calculate seasonal component (simplified)
        df_sorted['detrended'] = df_sorted[metric_col] - df_sorted['trend']
        
        # Calculate day of week or hour seasonality based on data frequency
        if len(df_sorted) > 168:  # More than a week of hourly data
            df_sorted['time_group'] = pd.to_datetime(df_sorted[self.time_col]).dt.hour
            seasonal_period = 24
        else:
            df_sorted['time_group'] = pd.to_datetime(df_sorted[self.time_col]).dt.dayofweek
            seasonal_period = 7
        
        seasonal_pattern = df_sorted.groupby('time_group')['detrended'].mean()
        df_sorted['seasonal'] = df_sorted['time_group'].map(seasonal_pattern)
        
        # Calculate residual
        df_sorted['residual'] = df_sorted[metric_col] - df_sorted['trend'] - df_sorted['seasonal']
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Original data
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted[metric_col],
                      mode='lines', name='Original', line=dict(color=self.theme['primary'])),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['trend'],
                      mode='lines', name='Trend', line=dict(color=self.theme['secondary'])),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['seasonal'],
                      mode='lines', name='Seasonal', line=dict(color=self.theme['accent'])),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['residual'],
                      mode='lines', name='Residual', line=dict(color='#ff6b6b')),
            row=4, col=1
        )
        
        fig.update_layout(
            title=f"Seasonal Decomposition: {metric_col}",
            showlegend=False,
            plot_bgcolor=self.theme['paper'],
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=800
        )
        
        return fig
    
    def growth_rate_analysis(self, metric_col: str) -> go.Figure:
        """
        Create growth rate analysis with multiple time periods
        """
        df_sorted = self.df.sort_values(self.time_col).copy()
        
        # Calculate different period growth rates
        df_sorted['value'] = df_sorted[metric_col]
        df_sorted['pct_change_1'] = df_sorted['value'].pct_change(1) * 100  # Period-over-period
        df_sorted['pct_change_7'] = df_sorted['value'].pct_change(min(7, len(df_sorted)//10)) * 100  # 7-period
        df_sorted['pct_change_30'] = df_sorted['value'].pct_change(min(30, len(df_sorted)//5)) * 100  # 30-period
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[
                f'{metric_col} Value',
                'Period-over-Period Growth Rate (%)',
                'Multi-Period Growth Rate (%)'
            ],
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Original values
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['value'],
                      mode='lines', name=metric_col, line=dict(color=self.theme['primary'], width=2)),
            row=1, col=1
        )
        
        # Period-over-period growth
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['pct_change_1'],
                      mode='lines', name='1-Period Growth %', 
                      line=dict(color=self.theme['secondary'], width=2)),
            row=2, col=1
        )
        
        # Add zero line for growth rates
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Multi-period growth rates
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['pct_change_7'],
                      mode='lines', name='7-Period Growth %', 
                      line=dict(color=self.theme['accent'], width=2)),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_sorted[self.time_col], y=df_sorted['pct_change_30'],
                      mode='lines', name='30-Period Growth %', 
                      line=dict(color='#ff6b6b', width=2)),
            row=3, col=1
        )
        
        # Add zero line for growth rates
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(
            title=f"Growth Rate Analysis: {metric_col}",
            plot_bgcolor=self.theme['paper'],
            paper_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            height=900
        )
        
        return fig
    
    def comparative_trends(self, metric_cols: List[str], normalize: bool = True) -> go.Figure:
        """
        Create comparative trends visualization for multiple metrics
        """
        df_sorted = self.df.sort_values(self.time_col).copy()
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, col in enumerate(metric_cols):
            if normalize:
                # Normalize to 0-100 scale for comparison
                col_min = df_sorted[col].min()
                col_max = df_sorted[col].max()
                if col_max != col_min:
                    normalized_values = ((df_sorted[col] - col_min) / (col_max - col_min)) * 100
                else:
                    normalized_values = [50] * len(df_sorted)
                
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[self.time_col],
                        y=normalized_values,
                        mode='lines',
                        name=f'{col} (normalized)',
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_sorted[self.time_col],
                        y=df_sorted[col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=2),
                        yaxis=f'y{i+1}' if i > 0 else 'y'
                    )
                )
        
        title = "Comparative Trends Analysis"
        if normalize:
            title += " (Normalized 0-100)"
        
        layout_updates = {
            'title': title,
            'plot_bgcolor': self.theme['paper'],
            'paper_bgcolor': self.theme['background'],
            'font': dict(color=self.theme['text']),
            'height': 600,
            'hovermode': 'x unified'
        }
        
        if not normalize and len(metric_cols) > 1:
            # Create multiple y-axes for different scales
            layout_updates.update({
                'yaxis': dict(title=metric_cols[0], side='left'),
                'yaxis2': dict(title=metric_cols[1] if len(metric_cols) > 1 else '', side='right', overlaying='y'),
            })
        
        fig.update_layout(**layout_updates)
        
        return fig
    
    def trend_statistics_summary(self, metric_cols: List[str]) -> Dict:
        """
        Calculate comprehensive trend statistics
        """
        stats = {}
        
        for col in metric_cols:
            df_sorted = self.df.sort_values(self.time_col)
            values = df_sorted[col].dropna()
            
            if len(values) < 2:
                continue
                
            # Basic statistics
            stats[col] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'range': values.max() - values.min(),
                'trend_direction': 'increasing' if values.iloc[-1] > values.iloc[0] else 'decreasing',
                'total_change': values.iloc[-1] - values.iloc[0],
                'total_change_percent': ((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100) if values.iloc[0] != 0 else 0,
                'volatility': values.std() / values.mean() * 100 if values.mean() != 0 else 0,
            }
            
            # Growth rate statistics
            pct_changes = values.pct_change().dropna()
            if len(pct_changes) > 0:
                stats[col].update({
                    'avg_growth_rate': pct_changes.mean() * 100,
                    'growth_rate_std': pct_changes.std() * 100,
                    'max_growth_rate': pct_changes.max() * 100,
                    'min_growth_rate': pct_changes.min() * 100,
                })
        
        return stats