"""
Data Processing Module for Chronosphere
Handles cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class TemporalDataProcessor:
    """Process temporal data for Chronosphere analysis"""
    
    def __init__(self, df: pd.DataFrame, time_column: str, entity_column: Optional[str] = None):
        """
        Initialize processor
        
        Args:
            df: Input DataFrame
            time_column: Name of timestamp column
            entity_column: Optional column for entity tracking (user_id, etc.)
        """
        self.df = df.copy()
        self.time_col = time_column
        self.entity_col = entity_column
        self.original_shape = df.shape
        
        # Convert time column
        self._convert_time_column()
        
        # Sort by time
        self.df = self.df.sort_values(self.time_col).reset_index(drop=True)
        
    def _convert_time_column(self):
        """Convert time column to datetime"""
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_col]):
            try:
                # Try multiple datetime formats
                self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
            except:
                # Try unix timestamp
                try:
                    self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], unit='s')
                except:
                    try:
                        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], unit='ms')
                    except Exception as e:
                        raise ValueError(f"Could not parse time column: {e}")
    
    def clean_data(self, 
                   drop_duplicates: bool = True,
                   handle_missing: str = 'drop',
                   missing_threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean the dataset
        
        Args:
            drop_duplicates: Whether to drop duplicate rows
            handle_missing: 'drop', 'forward_fill', or 'mean'
            missing_threshold: Drop columns with more than this % missing
        """
        initial_rows = len(self.df)
        
        # Drop duplicates
        if drop_duplicates:
            self.df = self.df.drop_duplicates()
            print(f"Removed {initial_rows - len(self.df)} duplicate rows")
        
        # Handle missing values
        if handle_missing == 'drop':
            self.df = self.df.dropna(subset=[self.time_col])
        elif handle_missing == 'forward_fill':
            self.df = self.df.fillna(method='ffill')
        elif handle_missing == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        # Drop columns with too many missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            print(f"Dropped columns with >{missing_threshold*100}% missing: {cols_to_drop}")
        
        print(f"Final shape: {self.df.shape}")
        return self.df
    
    def create_temporal_features(self) -> pd.DataFrame:
        """Extract temporal features from timestamp"""
        
        # Basic temporal features
        self.df['hour'] = self.df[self.time_col].dt.hour
        self.df['day_of_week'] = self.df[self.time_col].dt.dayofweek
        self.df['day_of_month'] = self.df[self.time_col].dt.day
        self.df['month'] = self.df[self.time_col].dt.month
        self.df['year'] = self.df[self.time_col].dt.year
        self.df['quarter'] = self.df[self.time_col].dt.quarter
        
        # Derived features
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_business_hours'] = self.df['hour'].between(9, 17).astype(int)
        
        # Day names for readability
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.df['day_name'] = self.df['day_of_week'].apply(lambda x: day_names[x])
        
        print(f"Created temporal features: hour, day_of_week, month, is_weekend, etc.")
        return self.df
    
    def create_lag_features(self, columns: List[str], lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Create lag features for time series analysis
        
        Args:
            columns: List of column names to create lags for
            lags: List of lag periods
        """
        for col in columns:
            if col in self.df.columns:
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        return self.df
    
    def create_rolling_features(self, 
                               columns: List[str], 
                               windows: List[int] = [7, 30]) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            columns: Columns to compute rolling stats for
            windows: Window sizes
        """
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                for window in windows:
                    self.df[f'{col}_rolling_mean_{window}'] = self.df[col].rolling(window).mean()
                    self.df[f'{col}_rolling_std_{window}'] = self.df[col].rolling(window).std()
        
        return self.df
    
    def aggregate_temporal(self, 
                          freq: str = '1H',
                          agg_dict: Optional[Dict] = None) -> pd.DataFrame:
        """
        Aggregate data by time frequency
        
        Args:
            freq: Frequency string ('1H', '1D', '1W', '1M')
            agg_dict: Dictionary of {column: aggregation_function}
        """
        if agg_dict is None:
            # Default aggregations for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' for col in numeric_cols}
            
            # Add count
            if self.entity_col:
                agg_dict[self.entity_col] = 'nunique'
        
        aggregated = self.df.set_index(self.time_col).resample(freq).agg(agg_dict)
        aggregated = aggregated.reset_index()
        
        print(f"Aggregated to {freq}: {aggregated.shape}")
        return aggregated
    
    def get_entity_trajectories(self, entity_id) -> pd.DataFrame:
        """Get temporal trajectory for a specific entity"""
        if self.entity_col and entity_id in self.df[self.entity_col].values:
            return self.df[self.df[self.entity_col] == entity_id].sort_values(self.time_col)
        return pd.DataFrame()
    
    def get_time_range_data(self, start_time, end_time) -> pd.DataFrame:
        """Get data within a specific time range"""
        mask = (self.df[self.time_col] >= start_time) & (self.df[self.time_col] <= end_time)
        return self.df[mask]
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the dataset"""
        stats = {
            'total_rows': len(self.df),
            'time_span': (self.df[self.time_col].max() - self.df[self.time_col].min()).days,
            'start_date': self.df[self.time_col].min(),
            'end_date': self.df[self.time_col].max(),
            'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
        }
        
        if self.entity_col:
            stats['unique_entities'] = self.df[self.entity_col].nunique()
        
        return stats


class AnomalyDetector:
    """Detect anomalies in temporal data"""
    
    def __init__(self, df: pd.DataFrame, time_column: str):
        self.df = df.copy()
        self.time_col = time_column
    
    def detect_zscore(self, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using z-score method
        
        Args:
            column: Column to check for anomalies
            threshold: Number of standard deviations
        """
        from scipy import stats
        
        values = self.df[column].dropna()
        z_scores = np.abs(stats.zscore(values))
        
        anomaly_mask = pd.Series(False, index=self.df.index)
        anomaly_mask[values.index] = z_scores > threshold
        
        self.df[f'{column}_anomaly'] = anomaly_mask
        self.df[f'{column}_zscore'] = 0.0
        self.df.loc[values.index, f'{column}_zscore'] = z_scores
        
        n_anomalies = anomaly_mask.sum()
        print(f"Detected {n_anomalies} anomalies in {column} (z-score > {threshold})")
        
        return self.df
    
    def detect_iqr(self, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect anomalies using IQR method
        
        Args:
            column: Column to check
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme)
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        anomaly_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        self.df[f'{column}_anomaly_iqr'] = anomaly_mask
        
        n_anomalies = anomaly_mask.sum()
        print(f"Detected {n_anomalies} anomalies in {column} (IQR method)")
        
        return self.df
    
    def detect_isolation_forest(self, columns: List[str], contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            columns: List of columns to use for detection
            contamination: Expected proportion of anomalies
        """
        from sklearn.ensemble import IsolationForest
        
        data = self.df[columns].dropna()
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(data)
        
        anomaly_mask = pd.Series(False, index=self.df.index)
        anomaly_mask[data.index] = predictions == -1
        
        self.df['anomaly_isolation_forest'] = anomaly_mask
        
        n_anomalies = anomaly_mask.sum()
        print(f"Detected {n_anomalies} anomalies using Isolation Forest")
        
        return self.df
    
    def get_anomaly_summary(self, anomaly_column: str = 'anomaly') -> Dict:
        """Get summary of detected anomalies"""
        anomalies = self.df[self.df[anomaly_column] == True]
        
        if len(anomalies) == 0:
            return {'count': 0, 'message': 'No anomalies detected'}
        
        summary = {
            'count': len(anomalies),
            'percentage': len(anomalies) / len(self.df) * 100,
            'dates': anomalies[self.time_col].tolist(),
            'indices': anomalies.index.tolist(),
        }
        
        return summary
    
    def get_anomaly_context(self, anomaly_index: int, window_hours: int = 48) -> pd.DataFrame:
        """
        Get temporal context around an anomaly
        
        Args:
            anomaly_index: Index of the anomaly
            window_hours: Hours before and after to include
        """
        anomaly_time = self.df.loc[anomaly_index, self.time_col]
        start_time = anomaly_time - timedelta(hours=window_hours)
        end_time = anomaly_time + timedelta(hours=window_hours)
        
        mask = (self.df[self.time_col] >= start_time) & (self.df[self.time_col] <= end_time)
        context = self.df[mask].copy()
        
        # Mark the anomaly in context
        context['is_anomaly_focus'] = context.index == anomaly_index
        
        return context