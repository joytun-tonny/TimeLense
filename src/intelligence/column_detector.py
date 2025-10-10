"""
Smart Column Detector for Chronosphere
Automatically identifies column types in uploaded datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


class SmartColumnDetector:
    """Automatically detect column types in a dataset"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_types = {}
        
    def detect_all(self) -> Dict[str, Dict]:
        """Detect all column types"""
        results = {
            'datetime': self.detect_datetime_columns(),
            'numeric': self.detect_numeric_columns(),
            'categorical': self.detect_categorical_columns(),
            'entity': self.detect_entity_columns(),
            'recommendations': self.get_recommendations()
        }
        return results
    
    def detect_datetime_columns(self) -> List[Dict]:
        """
        Detect potential datetime columns
        Returns list of dicts with column name, confidence, and sample values
        """
        datetime_cols = []
        
        # Keywords that suggest datetime
        datetime_keywords = [
            'date', 'time', 'timestamp', 'datetime', 'created', 'updated',
            'dt', 'day', 'when', 'at', 'period', 'year', 'month'
        ]
        
        for col in self.df.columns:
            confidence = 0
            reason = []
            
            # Check 1: Column name contains datetime keywords
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in datetime_keywords):
                confidence += 30
                reason.append("name_suggests_datetime")
            
            # Check 2: Already datetime type
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                confidence += 70
                reason.append("already_datetime_type")
            else:
                # Check 3: Try to parse as datetime
                try:
                    sample = self.df[col].dropna().head(100)
                    parsed = pd.to_datetime(sample, errors='coerce')
                    success_rate = parsed.notna().sum() / len(sample)
                    
                    if success_rate > 0.9:
                        confidence += 50
                        reason.append(f"parseable_as_datetime_{success_rate:.0%}")
                    elif success_rate > 0.5:
                        confidence += 20
                        reason.append(f"partially_parseable_{success_rate:.0%}")
                except:
                    pass
                
                # Check 4: Check for unix timestamp
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    sample_values = self.df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        # Unix timestamp in seconds (roughly 2000-2030)
                        if sample_values.between(946684800, 1893456000).all():
                            confidence += 40
                            reason.append("unix_timestamp_seconds")
                        # Unix timestamp in milliseconds
                        elif sample_values.between(946684800000, 1893456000000).all():
                            confidence += 40
                            reason.append("unix_timestamp_milliseconds")
            
            if confidence > 20:
                datetime_cols.append({
                    'column': col,
                    'confidence': min(confidence, 100),
                    'reasons': reason,
                    'sample_values': self.df[col].dropna().head(3).tolist(),
                    'dtype': str(self.df[col].dtype)
                })
        
        # Sort by confidence
        datetime_cols.sort(key=lambda x: x['confidence'], reverse=True)
        return datetime_cols
    
    def detect_numeric_columns(self) -> List[Dict]:
        """Detect numeric columns suitable for analysis"""
        numeric_cols = []
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Get statistics
                stats = {
                    'column': col,
                    'dtype': str(self.df[col].dtype),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'missing_pct': float(self.df[col].isna().sum() / len(self.df) * 100),
                    'unique_count': int(self.df[col].nunique()),
                    'zero_count': int((self.df[col] == 0).sum())
                }
                
                # Determine if it's a good metric
                is_good_metric = (
                    stats['std'] > 0 and  # Has variation
                    stats['unique_count'] > 10 and  # Not just a few values
                    stats['missing_pct'] < 50  # Not too many missing
                )
                
                stats['recommended_for_analysis'] = is_good_metric
                numeric_cols.append(stats)
        
        return numeric_cols
    
    def detect_categorical_columns(self) -> List[Dict]:
        """Detect categorical columns"""
        categorical_cols = []
        
        for col in self.df.columns:
            # Skip numeric and datetime
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                continue
            
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            cardinality_ratio = unique_count / total_count
            
            # Good categorical: not too many unique values
            if cardinality_ratio < 0.5:
                value_counts = self.df[col].value_counts().head(10)
                
                categorical_cols.append({
                    'column': col,
                    'unique_count': unique_count,
                    'cardinality_ratio': round(cardinality_ratio, 3),
                    'top_values': value_counts.to_dict(),
                    'missing_pct': round(self.df[col].isna().sum() / len(self.df) * 100, 2),
                    'recommended_for_grouping': cardinality_ratio < 0.1
                })
        
        return categorical_cols
    
    def detect_entity_columns(self) -> List[Dict]:
        """
        Detect columns that represent entities (user_id, transaction_id, etc.)
        High cardinality, likely numeric or string IDs
        """
        entity_cols = []
        
        # Keywords suggesting entity/ID columns
        id_keywords = ['id', 'user', 'customer', 'client', 'account', 'session', 
                       'transaction', 'order', 'visitor', 'device']
        
        for col in self.df.columns:
            confidence = 0
            reasons = []
            
            # Check name
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in id_keywords):
                confidence += 40
                reasons.append("name_suggests_id")
            
            # Check cardinality
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            cardinality_ratio = unique_count / total_count
            
            # High cardinality suggests ID
            if cardinality_ratio > 0.5:
                confidence += 30
                reasons.append(f"high_cardinality_{cardinality_ratio:.1%}")
            
            # Check if values look like IDs (alphanumeric strings or large integers)
            if pd.api.types.is_string_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                sample = self.df[col].dropna().head(10).astype(str)
                # Check for UUID-like patterns or alphanumeric IDs
                if sample.str.match(r'^[A-Za-z0-9\-_]+$').all():
                    confidence += 20
                    reasons.append("alphanumeric_id_pattern")
            
            if confidence > 30:
                entity_cols.append({
                    'column': col,
                    'confidence': confidence,
                    'reasons': reasons,
                    'unique_count': unique_count,
                    'cardinality_ratio': round(cardinality_ratio, 3),
                    'sample_values': self.df[col].dropna().head(3).tolist()
                })
        
        entity_cols.sort(key=lambda x: x['confidence'], reverse=True)
        return entity_cols
    
    def get_recommendations(self) -> Dict[str, str]:
        """Get recommended column assignments"""
        datetime_cols = self.detect_datetime_columns()
        numeric_cols = self.detect_numeric_columns()
        entity_cols = self.detect_entity_columns()
        
        recommendations = {}
        
        # Recommend time column
        if datetime_cols:
            recommendations['time_column'] = datetime_cols[0]['column']
            recommendations['time_column_confidence'] = datetime_cols[0]['confidence']
        
        # Recommend metrics
        good_metrics = [col['column'] for col in numeric_cols if col.get('recommended_for_analysis', False)]
        recommendations['metric_columns'] = good_metrics[:5]  # Top 5
        
        # Recommend entity column
        if entity_cols:
            recommendations['entity_column'] = entity_cols[0]['column']
            recommendations['entity_column_confidence'] = entity_cols[0]['confidence']
        
        return recommendations
    
    def validate_for_chronosphere(self) -> Tuple[bool, List[str]]:
        """
        Validate if dataset is suitable for Chronosphere
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: Has datetime column
        datetime_cols = self.detect_datetime_columns()
        if not datetime_cols:
            issues.append("❌ No datetime column detected. Dataset must have at least one time-based column.")
        elif datetime_cols[0]['confidence'] < 50:
            issues.append(f"⚠️ Datetime column '{datetime_cols[0]['column']}' has low confidence ({datetime_cols[0]['confidence']}%). Please verify.")
        
        # Check 2: Has numeric columns
        numeric_cols = self.detect_numeric_columns()
        good_metrics = [col for col in numeric_cols if col.get('recommended_for_analysis', False)]
        if len(good_metrics) < 2:
            issues.append(f"⚠️ Only {len(good_metrics)} good numeric columns found. Recommended: at least 3.")
        
        # Check 3: Minimum rows
        if len(self.df) < 100:
            issues.append(f"❌ Dataset has only {len(self.df)} rows. Minimum required: 100.")
        
        # Check 4: Not too many missing values
        missing_pct = self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100
        if missing_pct > 50:
            issues.append(f"⚠️ Dataset has {missing_pct:.1f}% missing values. This may affect analysis quality.")
        
        # Check 5: File size check (approximate)
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 500:
            issues.append(f"⚠️ Dataset is large ({memory_mb:.0f}MB). Processing may be slow.")
        
        is_valid = not any(issue.startswith("❌") for issue in issues)
        
        return is_valid, issues


def auto_configure_dataset(df: pd.DataFrame) -> Dict:
    """
    Automatically configure dataset for Chronosphere
    Returns configuration dictionary
    """
    detector = SmartColumnDetector(df)
    detection_results = detector.detect_all()
    is_valid, issues = detector.validate_for_chronosphere()
    
    config = {
        'is_valid': is_valid,
        'issues': issues,
        'detection_results': detection_results,
        'recommended_config': {
            'time_column': detection_results['recommendations'].get('time_column'),
            'metric_columns': detection_results['recommendations'].get('metric_columns', []),
            'entity_column': detection_results['recommendations'].get('entity_column'),
            'aggregation_freq': 'auto'  # Will be determined by data density
        }
    }
    
    return config