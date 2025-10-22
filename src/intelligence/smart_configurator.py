"""
Smart Dataset Configurator for Uploaded Data
Automatically optimizes visualization settings based on dataset characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class SmartDatasetConfigurator:
    """Automatically configures datasets for optimal visualization"""
    
    def __init__(self, df: pd.DataFrame, time_col: str, metric_cols: List[str]):
        self.df = df.copy()
        self.time_col = time_col
        self.metric_cols = metric_cols
        self.config = {}
        
        # Analyze dataset characteristics
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """Analyze dataset to determine optimal configurations"""
        
        # Time characteristics
        time_data = pd.to_datetime(self.df[self.time_col])
        time_span = (time_data.max() - time_data.min()).total_seconds()
        
        self.config.update({
            'total_rows': len(self.df),
            'time_span_seconds': time_span,
            'time_span_days': time_span / (24 * 3600),
            'time_range': (time_data.min(), time_data.max()),
            'data_frequency': self._estimate_frequency()
        })
        
        # Metric characteristics
        metric_stats = {}
        for col in self.metric_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                metric_stats[col] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'skewness': data.skew(),
                    'missing_pct': (len(self.df) - len(data)) / len(self.df) * 100,
                    'unique_values': data.nunique(),
                    'volatility': data.std() / data.mean() if data.mean() != 0 else 0
                }
        
        self.config['metric_stats'] = metric_stats
        
        # Determine optimal settings
        self._configure_visualizations()
    
    def _estimate_frequency(self) -> str:
        """Estimate the natural frequency of the data"""
        time_data = pd.to_datetime(self.df[self.time_col])
        
        if len(time_data) < 2:
            return 'unknown'
        
        # Calculate typical time differences
        time_diffs = time_data.diff().dropna()
        median_diff = time_diffs.median()
        
        # Classify frequency
        if median_diff <= timedelta(minutes=1):
            return 'sub-minute'
        elif median_diff <= timedelta(minutes=15):
            return 'minute'
        elif median_diff <= timedelta(hours=2):
            return 'hourly'
        elif median_diff <= timedelta(days=1):
            return 'daily'
        elif median_diff <= timedelta(days=7):
            return 'weekly'
        else:
            return 'monthly'
    
    def _configure_visualizations(self):
        """Configure visualization settings based on data characteristics"""
        
        # Sampling strategy for large datasets
        if self.config['total_rows'] > 10000:
            self.config['use_sampling'] = True
            self.config['sample_size'] = min(5000, self.config['total_rows'] // 2)
        else:
            self.config['use_sampling'] = False
            self.config['sample_size'] = self.config['total_rows']
        
        # Animation frame settings
        if self.config['total_rows'] > 1000:
            self.config['animation_frames'] = min(20, self.config['total_rows'] // 100)
        else:
            self.config['animation_frames'] = min(10, self.config['total_rows'] // 20)
        
        # Window sizes for rolling calculations
        if self.config['data_frequency'] in ['sub-minute', 'minute']:
            self.config['rolling_window'] = min(60, self.config['total_rows'] // 10)
        elif self.config['data_frequency'] == 'hourly':
            self.config['rolling_window'] = min(24, self.config['total_rows'] // 5)
        elif self.config['data_frequency'] == 'daily':
            self.config['rolling_window'] = min(30, self.config['total_rows'] // 3)
        else:
            self.config['rolling_window'] = min(12, self.config['total_rows'] // 2)
        
        # Clustering settings
        n_metrics = len(self.metric_cols)
        if n_metrics >= 6:
            self.config['optimal_clusters'] = min(8, max(3, n_metrics // 2))
        elif n_metrics >= 3:
            self.config['optimal_clusters'] = min(5, max(2, n_metrics - 1))
        else:
            self.config['optimal_clusters'] = 2
        
        # Anomaly detection sensitivity
        volatility_avg = np.mean([stats.get('volatility', 0) for stats in self.config['metric_stats'].values()])
        
        if volatility_avg > 1.0:  # High volatility
            self.config['anomaly_threshold'] = 3.0  # Less sensitive
        elif volatility_avg > 0.5:  # Medium volatility
            self.config['anomaly_threshold'] = 2.5
        else:  # Low volatility
            self.config['anomaly_threshold'] = 2.0  # More sensitive
        
        # Visualization recommendations
        self.config['recommended_visualizations'] = self._recommend_visualizations()
    
    def _recommend_visualizations(self) -> List[str]:
        """Recommend optimal visualizations based on data characteristics"""
        recommendations = []
        
        n_metrics = len(self.metric_cols)
        total_rows = self.config['total_rows']
        time_span_days = self.config['time_span_days']
        
        # Always recommend basic visualizations
        recommendations.extend(['dashboard', 'heartbeat'])
        
        # Correlation visualizations for multiple metrics
        if n_metrics >= 2:
            recommendations.append('correlation_explorer')
            if total_rows >= 100:
                recommendations.append('correlation_evolution')
        
        # 3D visualizations for 3+ metrics
        if n_metrics >= 3:
            recommendations.append('3d_phase')
            if total_rows >= 200:
                recommendations.append('anomaly_landscape')
        
        # Advanced analytics for substantial datasets
        if total_rows >= 100 and n_metrics >= 2:
            recommendations.append('pattern_radar')
            
        if total_rows >= 200 and n_metrics >= 2:
            recommendations.append('clustering_evolution')
            
        if total_rows >= 50 and n_metrics >= 2:
            recommendations.append('flow_field')
        
        # Time-based visualizations
        if time_span_days >= 1:
            recommendations.append('heatmap')
            
        if total_rows >= 50:
            recommendations.append('rewind')
        
        # High-dimensional analysis
        if n_metrics >= 4 and total_rows >= 100:
            recommendations.append('evolution')
            
        if n_metrics >= 3:
            recommendations.append('parallel_coords')
        
        return recommendations
    
    def get_optimized_dataset(self) -> pd.DataFrame:
        """Return optimized dataset for visualization"""
        df_optimized = self.df.copy()
        
        # Apply sampling if needed
        if self.config['use_sampling']:
            if self.config['sample_size'] < len(df_optimized):
                # Stratified sampling to preserve time distribution
                df_optimized = df_optimized.sample(
                    n=self.config['sample_size'], 
                    random_state=42
                ).sort_values(self.time_col)
        
        # Clean data
        for col in self.metric_cols:
            if col in df_optimized.columns:
                # Fill missing values with forward fill then backward fill
                df_optimized[col] = df_optimized[col].fillna(method='ffill').fillna(method='bfill')
                
                # Remove infinite values
                df_optimized[col] = df_optimized[col].replace([np.inf, -np.inf], np.nan)
                df_optimized[col] = df_optimized[col].fillna(df_optimized[col].median())
        
        return df_optimized
    
    def get_visualization_config(self, viz_type: str) -> Dict:
        """Get specific configuration for a visualization type"""
        base_config = {
            'sample_data': self.config['use_sampling'],
            'sample_size': self.config.get('sample_size', len(self.df))
        }
        
        if viz_type == 'heartbeat':
            return {**base_config, 'recommended_metric': self._get_best_metric_for_heartbeat()}
        
        elif viz_type == 'correlation_evolution':
            return {
                **base_config, 
                'window_size': self.config['rolling_window'],
                'n_frames': self.config['animation_frames']
            }
        
        elif viz_type == 'clustering_evolution':
            return {
                **base_config,
                'n_clusters': self.config['optimal_clusters'],
                'n_frames': self.config['animation_frames'],
                'recommended_features': self._get_best_features_for_clustering()
            }
        
        elif viz_type == 'anomaly_landscape':
            return {
                **base_config,
                'threshold': self.config['anomaly_threshold'],
                'recommended_metric': self._get_best_metric_for_anomalies()
            }
        
        elif viz_type == 'flow_field':
            return {
                **base_config,
                'recommended_pair': self._get_best_metric_pair()
            }
        
        elif viz_type == '3d_phase':
            return {
                **base_config,
                'recommended_triplet': self._get_best_metric_triplet()
            }
        
        else:
            return base_config
    
    def _get_best_metric_for_heartbeat(self) -> str:
        """Get the best metric for heartbeat visualization"""
        if not self.metric_cols:
            return None
        
        # Prefer metrics with high variability but not too noisy
        best_metric = None
        best_score = -1
        
        for metric in self.metric_cols:
            stats = self.config['metric_stats'].get(metric, {})
            volatility = stats.get('volatility', 0)
            missing_pct = stats.get('missing_pct', 100)
            
            # Score based on moderate volatility and low missing data
            if 0.1 <= volatility <= 2.0 and missing_pct < 20:
                score = (1 - abs(volatility - 0.5)) * (1 - missing_pct / 100)
                if score > best_score:
                    best_score = score
                    best_metric = metric
        
        return best_metric or self.metric_cols[0]
    
    def _get_best_metric_for_anomalies(self) -> str:
        """Get the best metric for anomaly detection"""
        if not self.metric_cols:
            return None
        
        # Prefer metrics with moderate volatility and good data quality
        best_metric = None
        best_score = -1
        
        for metric in self.metric_cols:
            stats = self.config['metric_stats'].get(metric, {})
            volatility = stats.get('volatility', 0)
            missing_pct = stats.get('missing_pct', 100)
            unique_ratio = stats.get('unique_values', 0) / stats.get('count', 1)
            
            # Score: moderate volatility, low missing, high uniqueness
            if volatility > 0.05 and missing_pct < 30 and unique_ratio > 0.1:
                score = volatility * (1 - missing_pct / 100) * unique_ratio
                if score > best_score:
                    best_score = score
                    best_metric = metric
        
        return best_metric or self.metric_cols[0]
    
    def _get_best_features_for_clustering(self) -> List[str]:
        """Get the best features for clustering"""
        # Select metrics with good variability and low correlation
        good_metrics = []
        
        for metric in self.metric_cols:
            stats = self.config['metric_stats'].get(metric, {})
            volatility = stats.get('volatility', 0)
            missing_pct = stats.get('missing_pct', 100)
            
            if volatility > 0.05 and missing_pct < 50:
                good_metrics.append(metric)
        
        return good_metrics[:6]  # Limit to 6 features for performance
    
    def _get_best_metric_pair(self) -> Tuple[str, str]:
        """Get the best pair of metrics for flow field"""
        if len(self.metric_cols) < 2:
            return None, None
        
        # Try to find metrics with different characteristics
        high_vol = []
        low_vol = []
        
        for metric in self.metric_cols:
            stats = self.config['metric_stats'].get(metric, {})
            volatility = stats.get('volatility', 0)
            missing_pct = stats.get('missing_pct', 100)
            
            if missing_pct < 30:
                if volatility > 0.3:
                    high_vol.append(metric)
                else:
                    low_vol.append(metric)
        
        # Prefer one high volatility and one low volatility metric
        if high_vol and low_vol:
            return high_vol[0], low_vol[0]
        elif len(self.metric_cols) >= 2:
            return self.metric_cols[0], self.metric_cols[1]
        else:
            return None, None
    
    def _get_best_metric_triplet(self) -> Tuple[str, str, str]:
        """Get the best triplet of metrics for 3D visualization"""
        if len(self.metric_cols) < 3:
            return None, None, None
        
        # Select metrics with good data quality
        good_metrics = []
        
        for metric in self.metric_cols:
            stats = self.config['metric_stats'].get(metric, {})
            missing_pct = stats.get('missing_pct', 100)
            volatility = stats.get('volatility', 0)
            
            if missing_pct < 30 and volatility > 0.01:
                good_metrics.append(metric)
        
        if len(good_metrics) >= 3:
            return good_metrics[0], good_metrics[1], good_metrics[2]
        else:
            return self.metric_cols[0], self.metric_cols[1], self.metric_cols[2]
    
    def get_insights(self) -> List[str]:
        """Generate insights about the dataset for users"""
        insights = []
        
        # Data size insights
        if self.config['total_rows'] > 10000:
            insights.append(f"📊 Large dataset ({self.config['total_rows']:,} rows) - using intelligent sampling for performance")
        elif self.config['total_rows'] < 100:
            insights.append(f"⚠️ Small dataset ({self.config['total_rows']} rows) - some advanced visualizations may be limited")
        
        # Time span insights
        days = self.config['time_span_days']
        if days > 365:
            insights.append(f"📅 Long time series ({days:.0f} days) - perfect for trend and seasonal analysis")
        elif days < 1:
            insights.append(f"⏰ Short time series ({days:.1f} days) - focus on intraday patterns")
        
        # Frequency insights
        freq = self.config['data_frequency']
        if freq in ['sub-minute', 'minute']:
            insights.append("⚡ High-frequency data detected - excellent for real-time pattern analysis")
        elif freq == 'daily':
            insights.append("📆 Daily data frequency - ideal for seasonal and weekly pattern detection")
        
        # Metric insights
        n_metrics = len(self.metric_cols)
        if n_metrics > 5:
            insights.append(f"🎯 Rich dataset with {n_metrics} metrics - all advanced visualizations available")
        elif n_metrics < 3:
            insights.append(f"📈 Limited metrics ({n_metrics}) - some visualizations may not be available")
        
        # Data quality insights
        avg_missing = np.mean([stats.get('missing_pct', 0) for stats in self.config['metric_stats'].values()])
        if avg_missing > 20:
            insights.append(f"⚠️ Significant missing data ({avg_missing:.1f}% average) - data cleaning applied")
        elif avg_missing < 5:
            insights.append("✅ High data quality - minimal missing values detected")
        
        # Volatility insights
        avg_volatility = np.mean([stats.get('volatility', 0) for stats in self.config['metric_stats'].values()])
        if avg_volatility > 1.0:
            insights.append("🌊 High volatility data - anomaly detection tuned for noisy signals")
        elif avg_volatility < 0.1:
            insights.append("📏 Stable data patterns - anomaly detection highly sensitive")
        
        return insights
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if self.config['total_rows'] > 5000:
            recommendations.append("🚀 Large dataset detected - some visualizations use sampling for better performance")
        
        if len(self.metric_cols) > 8:
            recommendations.append("📊 Many metrics detected - consider selecting key metrics for focused analysis")
        
        if self.config['animation_frames'] > 15:
            recommendations.append("🎬 High-resolution animations available - may take longer to generate")
        
        if self.config['data_frequency'] in ['sub-minute', 'minute'] and self.config['total_rows'] > 1000:
            recommendations.append("⏰ High-frequency data - aggregation may be applied for visualization clarity")
        
        return recommendations