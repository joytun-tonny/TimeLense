"""
Insight Generator for Chronosphere
Automatically extracts meaningful patterns and generates natural language insights
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class InsightGenerator:
    """Generate automated insights from temporal data"""
    
    def __init__(self, df: pd.DataFrame, time_column: str):
        self.df = df.copy()
        self.time_col = time_column
        self.insights = []
        
    def generate_all_insights(self, metric_columns: List[str]) -> List[Dict]:
        """
        Generate all insights for given metrics
        
        Args:
            metric_columns: List of numeric columns to analyze
        
        Returns:
            List of insight dictionaries
        """
        self.insights = []
        
        for col in metric_columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                # Trend insights
                self.insights.extend(self._analyze_trend(col))
                
                # Seasonality insights
                if 'hour' in self.df.columns:
                    self.insights.extend(self._analyze_seasonality(col))
                
                # Volatility insights
                self.insights.extend(self._analyze_volatility(col))
                
                # Anomaly insights
                self.insights.extend(self._analyze_anomalies(col))
        
        # Correlation insights
        if len(metric_columns) >= 2:
            self.insights.extend(self._analyze_correlations(metric_columns))
        
        # Sort by importance
        self.insights.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        return self.insights
    
    def _analyze_trend(self, column: str) -> List[Dict]:
        """Analyze trend in metric"""
        insights = []
        
        try:
            # Prepare data
            x = np.arange(len(self.df)).reshape(-1, 1)
            y = self.df[column].fillna(self.df[column].mean()).values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            
            slope = model.coef_[0]
            r_squared = model.score(x, y)
            
            # Determine trend strength
            cv = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0
            
            if abs(slope) < 0.001 * np.mean(y):
                trend_direction = "stable"
                strength = "minimal"
            elif slope > 0:
                trend_direction = "upward"
                strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
            else:
                trend_direction = "downward"
                strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.4 else "weak"
            
            # Calculate rate of change
            time_span_days = (self.df[self.time_col].max() - self.df[self.time_col].min()).days
            if time_span_days > 0:
                total_change = slope * len(self.df)
                daily_change = total_change / time_span_days
                change_pct = (total_change / np.mean(y)) * 100 if np.mean(y) != 0 else 0
            else:
                daily_change = 0
                change_pct = 0
            
            # Generate insight
            if trend_direction != "stable":
                insight_text = (
                    f"📈 {column} shows a {strength} {trend_direction} trend "
                    f"(R²={r_squared:.2f}). "
                )
                
                if abs(change_pct) > 1:
                    insight_text += f"Overall change of {change_pct:.1f}% over the period. "
                
                if abs(daily_change) > 0.01:
                    insight_text += f"Average daily change: {daily_change:+.2f}."
                
                insights.append({
                    'type': 'trend',
                    'column': column,
                    'text': insight_text,
                    'data': {
                        'direction': trend_direction,
                        'strength': strength,
                        'slope': float(slope),
                        'r_squared': float(r_squared),
                        'daily_change': float(daily_change),
                        'change_pct': float(change_pct)
                    },
                    'importance': int(r_squared * 100) if trend_direction != "stable" else 20
                })
        
        except Exception as e:
            pass
        
        return insights
    
    def _analyze_seasonality(self, column: str) -> List[Dict]:
        """Analyze seasonal patterns"""
        insights = []
        
        try:
            if 'hour' not in self.df.columns:
                return insights
            
            # Hourly analysis
            hourly_mean = self.df.groupby('hour')[column].mean()
            hourly_std = self.df.groupby('hour')[column].std()
            
            cv = hourly_std.mean() / hourly_mean.mean() if hourly_mean.mean() != 0 else 0
            
            if cv > 0.3:  # Significant variation
                peak_hour = hourly_mean.idxmax()
                trough_hour = hourly_mean.idxmin()
                peak_value = hourly_mean.max()
                trough_value = hourly_mean.min()
                ratio = peak_value / trough_value if trough_value != 0 else 0
                
                insight_text = (
                    f"🕐 {column} exhibits clear daily seasonality. "
                    f"Peak activity at hour {peak_hour} "
                    f"({peak_value:.2f}), "
                    f"lowest at hour {trough_hour} "
                    f"({trough_value:.2f}). "
                    f"Peak is {ratio:.1f}× the trough."
                )
                
                insights.append({
                    'type': 'seasonality',
                    'column': column,
                    'text': insight_text,
                    'data': {
                        'period': 'daily',
                        'peak_hour': int(peak_hour),
                        'trough_hour': int(trough_hour),
                        'peak_value': float(peak_value),
                        'trough_value': float(trough_value),
                        'ratio': float(ratio)
                    },
                    'importance': int(cv * 100)
                })
            
            # Weekend analysis
            if 'is_weekend' in self.df.columns:
                weekend_mean = self.df[self.df['is_weekend'] == 1][column].mean()
                weekday_mean = self.df[self.df['is_weekend'] == 0][column].mean()
                
                if not np.isnan(weekend_mean) and not np.isnan(weekday_mean) and weekday_mean != 0:
                    diff_pct = ((weekend_mean - weekday_mean) / weekday_mean) * 100
                    
                    if abs(diff_pct) > 10:
                        direction = "higher" if diff_pct > 0 else "lower"
                        insight_text = (
                            f"📅 {column} is {abs(diff_pct):.1f}% {direction} on weekends "
                            f"({weekend_mean:.2f}) compared to weekdays ({weekday_mean:.2f})."
                        )
                        
                        insights.append({
                            'type': 'seasonality',
                            'column': column,
                            'text': insight_text,
                            'data': {
                                'period': 'weekly',
                                'weekend_mean': float(weekend_mean),
                                'weekday_mean': float(weekday_mean),
                                'difference_pct': float(diff_pct)
                            },
                            'importance': int(abs(diff_pct))
                        })
        
        except Exception as e:
            pass
        
        return insights
    
    def _analyze_volatility(self, column: str) -> List[Dict]:
        """Analyze volatility/stability"""
        insights = []
        
        try:
            mean_val = self.df[column].mean()
            std_val = self.df[column].std()
            cv = std_val / mean_val if mean_val != 0 else 0
            
            if cv > 0.5:
                insight_text = (
                    f"⚠️ {column} shows high volatility "
                    f"(coefficient of variation: {cv:.2f}). "
                    f"Values range from {self.df[column].min():.2f} to {self.df[column].max():.2f}, "
                    f"with frequent large swings."
                )
                
                insights.append({
                    'type': 'volatility',
                    'column': column,
                    'text': insight_text,
                    'data': {
                        'cv': float(cv),
                        'std': float(std_val),
                        'mean': float(mean_val),
                        'min': float(self.df[column].min()),
                        'max': float(self.df[column].max())
                    },
                    'importance': int(cv * 50)
                })
            elif cv < 0.15:
                insight_text = (
                    f"✅ {column} is highly stable "
                    f"(coefficient of variation: {cv:.2f}). "
                    f"Consistent values around {mean_val:.2f}."
                )
                
                insights.append({
                    'type': 'volatility',
                    'column': column,
                    'text': insight_text,
                    'data': {
                        'cv': float(cv),
                        'stability': 'high'
                    },
                    'importance': 30
                })
        
        except Exception as e:
            pass
        
        return insights
    
    def _analyze_anomalies(self, column: str) -> List[Dict]:
        """Analyze anomalies"""
        insights = []
        
        try:
            # Detect anomalies using z-score
            values = self.df[column].dropna()
            if len(values) == 0:
                return insights
            
            z_scores = np.abs(stats.zscore(values))
            anomaly_threshold = 3
            anomalies = values[z_scores > anomaly_threshold]
            
            if len(anomalies) > 0:
                anomaly_pct = (len(anomalies) / len(values)) * 100
                
                # Get dates of anomalies
                anomaly_indices = values[z_scores > anomaly_threshold].index
                anomaly_dates = self.df.loc[anomaly_indices, self.time_col]
                
                # Check for clustering
                if len(anomaly_dates) > 1:
                    anomaly_dates_sorted = sorted(anomaly_dates)
                    time_diffs = [(anomaly_dates_sorted[i+1] - anomaly_dates_sorted[i]).days 
                                  for i in range(len(anomaly_dates_sorted)-1)]
                    
                    if time_diffs:
                        avg_gap = np.mean(time_diffs)
                        clustered = avg_gap < 2  # Within 2 days
                    else:
                        clustered = False
                else:
                    clustered = False
                
                insight_text = (
                    f"🎯 Detected {len(anomalies)} significant anomalies in {column} "
                    f"({anomaly_pct:.1f}% of data). "
                )
                
                if clustered:
                    insight_text += "Anomalies are clustered in time, suggesting event-driven spikes. "
                else:
                    insight_text += "Anomalies are scattered, suggesting random fluctuations. "
                
                max_anomaly = anomalies.max()
                max_anomaly_date = self.df.loc[anomalies.idxmax(), self.time_col]
                insight_text += f"Largest spike: {max_anomaly:.2f} on {max_anomaly_date.strftime('%Y-%m-%d')}."
                
                insights.append({
                    'type': 'anomaly',
                    'column': column,
                    'text': insight_text,
                    'data': {
                        'count': len(anomalies),
                        'percentage': float(anomaly_pct),
                        'clustered': clustered,
                        'max_value': float(max_anomaly),
                        'max_date': str(max_anomaly_date),
                        'anomaly_indices': anomaly_indices.tolist()[:10]  # First 10
                    },
                    'importance': int(len(anomalies) * 2)
                })
        
        except Exception as e:
            pass
        
        return insights
    
    def _analyze_correlations(self, columns: List[str]) -> List[Dict]:
        """Analyze correlations between metrics"""
        insights = []
        
        try:
            # Calculate correlation matrix
            numeric_df = self.df[columns].select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] < 2:
                return insights
            
            corr_matrix = numeric_df.corr()
            
            # Find strong correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.6:  # Strong correlation
                        direction = "positive" if corr_value > 0 else "negative"
                        strength = "very strong" if abs(corr_value) > 0.8 else "strong"
                        
                        insight_text = (
                            f"🔗 {col1} and {col2} show {strength} {direction} correlation "
                            f"(r={corr_value:.2f}). "
                        )
                        
                        if corr_value > 0:
                            insight_text += f"When {col1} increases, {col2} tends to increase as well."
                        else:
                            insight_text += f"When {col1} increases, {col2} tends to decrease."
                        
                        insights.append({
                            'type': 'correlation',
                            'columns': [col1, col2],
                            'text': insight_text,
                            'data': {
                                'correlation': float(corr_value),
                                'direction': direction,
                                'strength': strength
                            },
                            'importance': int(abs(corr_value) * 80)
                        })
        
        except Exception as e:
            pass
        
        return insights
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all insights"""
        if not self.insights:
            return "No insights generated yet. Run generate_all_insights() first."
        
        report = "=" * 70 + "\n"
        report += "CHRONOSPHERE INSIGHT REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Group by type
        insight_types = {}
        for insight in self.insights:
            insight_type = insight['type']
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        # Format by type
        type_names = {
            'trend': 'TREND ANALYSIS',
            'seasonality': 'SEASONALITY PATTERNS',
            'volatility': 'VOLATILITY ASSESSMENT',
            'anomaly': 'ANOMALY DETECTION',
            'correlation': 'CORRELATION INSIGHTS'
        }
        
        for insight_type, insights_list in insight_types.items():
            report += f"\n{type_names.get(insight_type, insight_type.upper())}\n"
            report += "-" * 70 + "\n"
            for i, insight in enumerate(insights_list, 1):
                report += f"{i}. {insight['text']}\n\n"
        
        report += "=" * 70 + "\n"
        report += f"Total Insights: {len(self.insights)}\n"
        report += "=" * 70 + "\n"
        
        return report
    
    def get_top_insights(self, n: int = 5) -> List[Dict]:
        """Get top N most important insights"""
        if not self.insights:
            return []
        
        sorted_insights = sorted(self.insights, key=lambda x: x.get('importance', 0), reverse=True)
        return sorted_insights[:n]