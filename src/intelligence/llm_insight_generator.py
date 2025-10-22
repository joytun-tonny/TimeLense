"""
LLM-Powered Insight Generator for Chronosphere
Uses free APIs (Gemini, Grok) to generate very useful insights for uploaded datasets
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

# API clients for different LLM providers
import requests
import time
from datetime import datetime


class LLMInsightGenerator:
    """Generate deep insights using LLM APIs"""
    
    def __init__(self, df: pd.DataFrame, time_column: str, metric_columns: List[str]):
        self.df = df.copy()
        self.time_col = time_column
        self.metric_cols = metric_columns
        self.insights = []
        
        # API configurations
        self.api_providers = {
            'gemini': {
                'url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                'headers': {'Content-Type': 'application/json'},
                'available': True
            },
            'huggingface': {
                'url': 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1',
                'headers': {'Content-Type': 'application/json'},
                'available': True
            }
        }
        
    def generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        summary = {
            'dataset_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'time_span_days': (self.df[self.time_col].max() - self.df[self.time_col].min()).days,
                'missing_data_pct': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
            },
            'metrics_analysis': {}
        }
        
        for col in self.metric_cols:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                col_data = self.df[col].dropna()
                
                # Basic statistics
                basic_stats = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
                
                # Trend analysis
                x = np.arange(len(col_data))
                if len(col_data) > 1:
                    correlation = np.corrcoef(x, col_data)[0, 1]
                    basic_stats['trend_correlation'] = float(correlation)
                
                # Volatility
                basic_stats['coefficient_variation'] = float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                
                # Outliers using IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                basic_stats['outlier_count'] = len(outliers)
                basic_stats['outlier_percentage'] = (len(outliers) / len(col_data)) * 100
                
                summary['metrics_analysis'][col] = basic_stats
        
        # Correlation analysis
        if len(self.metric_cols) >= 2:
            numeric_df = self.df[self.metric_cols].select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            'metric1': col1,
                            'metric2': col2,
                            'correlation': float(corr_value)
                        })
            
            summary['correlations'] = strong_correlations
        
        return summary
    
    def create_llm_prompt(self, summary: Dict[str, Any], analysis_type: str = "comprehensive") -> str:
        """Create optimized prompt for LLM analysis"""
        
        base_prompt = f"""You are an expert data analyst specializing in temporal data analysis and business insights. 

DATASET OVERVIEW:
- Total Records: {summary['dataset_info']['total_rows']:,}
- Time Span: {summary['dataset_info']['time_span_days']} days
- Metrics Analyzed: {len(self.metric_cols)} key metrics
- Data Quality: {100 - summary['dataset_info']['missing_data_pct']:.1f}% complete

STATISTICAL ANALYSIS:"""
        
        # Add metrics analysis
        for metric, stats in summary['metrics_analysis'].items():
            base_prompt += f"""

{metric.upper()}:
- Range: {stats['min']:.2f} to {stats['max']:.2f}
- Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}
- Trend: {'Upward' if stats.get('trend_correlation', 0) > 0.1 else 'Downward' if stats.get('trend_correlation', 0) < -0.1 else 'Stable'}
- Volatility: {'High' if stats['coefficient_variation'] > 0.3 else 'Moderate' if stats['coefficient_variation'] > 0.1 else 'Low'} (CV: {stats['coefficient_variation']:.2f})
- Outliers: {stats['outlier_count']} ({stats['outlier_percentage']:.1f}%)"""
        
        # Add correlations
        if 'correlations' in summary and summary['correlations']:
            base_prompt += "\n\nKEY CORRELATIONS:"
            for corr in summary['correlations'][:5]:
                strength = 'Very Strong' if abs(corr['correlation']) > 0.8 else 'Strong' if abs(corr['correlation']) > 0.6 else 'Moderate'
                direction = 'Positive' if corr['correlation'] > 0 else 'Negative'
                base_prompt += f"\n- {corr['metric1']} ↔ {corr['metric2']}: {strength} {direction} ({corr['correlation']:.3f})"
        
        # Analysis request based on type
        if analysis_type == "comprehensive":
            base_prompt += """

GENERATE ACTIONABLE BUSINESS INSIGHTS:

Please provide 8-10 high-value insights following this structure:

1. **TREND INSIGHTS** (2-3 insights):
   - Identify the most significant trends and their business implications
   - Quantify growth rates, acceleration/deceleration patterns
   - Predict potential future trajectories

2. **PATTERN INSIGHTS** (2-3 insights):
   - Discover seasonal patterns, cyclical behaviors
   - Identify recurring anomalies or systematic variations
   - Explain what drives these patterns

3. **RELATIONSHIP INSIGHTS** (2-3 insights):
   - Analyze causal relationships between metrics
   - Identify leading/lagging indicators
   - Explain correlation mechanisms

4. **OPPORTUNITY INSIGHTS** (2-3 insights):
   - Identify optimization opportunities
   - Highlight underperforming areas
   - Suggest data-driven improvements

FORMAT EACH INSIGHT AS:
🎯 **[INSIGHT TITLE]**: [Detailed explanation with specific numbers and percentages. Include why this matters and what actions could be taken.]

Focus on:
- Business impact and actionability
- Specific numerical evidence
- Root cause analysis
- Strategic recommendations
- Risk identification
"""
        
        elif analysis_type == "anomaly_focused":
            base_prompt += """

DEEP ANOMALY ANALYSIS:

Focus specifically on unusual patterns, outliers, and anomalies in the data. Provide insights about:

1. **ANOMALY PATTERNS**: What types of anomalies occur and when?
2. **ROOT CAUSES**: What might be driving these unusual events?
3. **BUSINESS IMPACT**: How do these anomalies affect overall performance?
4. **PREDICTION**: Are these anomalies predictable or random?
5. **MITIGATION**: How can similar issues be prevented or leveraged?

Provide 5-7 specific insights about anomalies and unusual patterns.
"""
        
        elif analysis_type == "predictive":
            base_prompt += """

PREDICTIVE INSIGHTS & FORECASTING:

Based on the historical patterns, provide insights about:

1. **FUTURE TRENDS**: Where are the metrics heading?
2. **RISK FACTORS**: What patterns suggest potential problems?
3. **GROWTH OPPORTUNITIES**: What trends indicate positive momentum?
4. **SEASONALITY FORECAST**: What patterns will likely repeat?
5. **VOLATILITY PREDICTION**: How stable will future performance be?

Provide 6-8 forward-looking insights with specific predictions.
"""
        
        return base_prompt
    
    def query_llm_api(self, prompt: str, provider: str = "huggingface", max_retries: int = 3) -> Optional[str]:
        """Query LLM API with fallback options"""
        
        if provider == "gemini":
            return self._query_gemini(prompt, max_retries)
        elif provider == "huggingface":
            return self._query_huggingface(prompt, max_retries)
        else:
            # Try all available providers
            for prov in ["huggingface"]:  # Start with most reliable free API
                try:
                    result = self.query_llm_api(prompt, prov, 1)
                    if result:
                        return result
                except:
                    continue
            return None
    
    def _query_huggingface(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Query Hugging Face Inference API (free tier)"""
        
        # Truncate prompt if too long
        if len(prompt) > 4000:
            prompt = prompt[:4000] + "...\n\nProvide insights based on the above data summary."
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_providers['huggingface']['url'],
                    headers=self.api_providers['huggingface']['headers'],
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '').strip()
                    elif isinstance(result, dict):
                        return result.get('generated_text', '').strip()
                
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    time.sleep(10 * (attempt + 1))
                    continue
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"HuggingFace API error: {str(e)}")
                time.sleep(2)
        
        return None
    
    def _query_gemini(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Query Google Gemini API (requires API key)"""
        
        # Note: Requires GOOGLE_API_KEY environment variable
        import os
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            return None
        
        url = f"{self.api_providers['gemini']['url']}?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "maxOutputTokens": 1024
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.api_providers['gemini']['headers'], json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'candidates' in result and len(result['candidates']) > 0:
                        content = result['candidates'][0].get('content', {})
                        if 'parts' in content and len(content['parts']) > 0:
                            return content['parts'][0].get('text', '').strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Gemini API error: {str(e)}")
                time.sleep(2)
        
        return None
    
    def generate_llm_insights(self, analysis_types: List[str] = None) -> List[Dict[str, Any]]:
        """Generate insights using LLM APIs"""
        
        if analysis_types is None:
            analysis_types = ["comprehensive", "anomaly_focused", "predictive"]
        
        all_insights = []
        
        # Generate statistical summary
        summary = self.generate_statistical_summary()
        
        for analysis_type in analysis_types:
            try:
                # Create specialized prompt
                prompt = self.create_llm_prompt(summary, analysis_type)
                
                # Query LLM
                llm_response = self.query_llm_api(prompt)
                
                if llm_response:
                    # Parse insights from response
                    insights = self._parse_llm_response(llm_response, analysis_type)
                    all_insights.extend(insights)
                else:
                    # Fallback to rule-based insights
                    fallback_insights = self._generate_fallback_insights(summary, analysis_type)
                    all_insights.extend(fallback_insights)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating {analysis_type} insights: {str(e)}")
                # Add fallback insights
                fallback_insights = self._generate_fallback_insights(summary, analysis_type)
                all_insights.extend(fallback_insights)
        
        return all_insights
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured insights"""
        insights = []
        
        # Split response into individual insights
        lines = response.split('\n')
        current_insight = ""
        
        for line in lines:
            line = line.strip()
            
            # Look for insight markers
            if line.startswith('🎯') or line.startswith('**') or line.startswith('#'):
                if current_insight:
                    # Process previous insight
                    parsed = self._parse_single_insight(current_insight, analysis_type)
                    if parsed:
                        insights.append(parsed)
                current_insight = line
            elif current_insight and line:
                current_insight += " " + line
        
        # Process last insight
        if current_insight:
            parsed = self._parse_single_insight(current_insight, analysis_type)
            if parsed:
                insights.append(parsed)
        
        return insights
    
    def _parse_single_insight(self, insight_text: str, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Parse a single insight from LLM text"""
        
        if len(insight_text.strip()) < 20:
            return None
        
        # Extract title and content
        title = ""
        content = insight_text
        
        # Look for title patterns
        if '**' in insight_text:
            parts = insight_text.split('**')
            if len(parts) >= 3:
                title = parts[1].strip()
                content = '**'.join(parts[2:]).strip()
        elif ':' in insight_text:
            parts = insight_text.split(':', 1)
            if len(parts) == 2:
                title = parts[0].strip().replace('🎯', '').replace('#', '').replace('*', '').strip()
                content = parts[1].strip()
        
        # Clean content
        content = content.replace('**', '').strip()
        if content.startswith(':'):
            content = content[1:].strip()
        
        # Determine importance based on content analysis
        importance = self._calculate_insight_importance(content)
        
        return {
            'type': 'llm_insight',
            'category': analysis_type,
            'title': title,
            'text': content,
            'importance': importance,
            'source': 'llm',
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_insight_importance(self, content: str) -> int:
        """Calculate importance score for an insight"""
        importance = 50  # Base score
        
        # Keywords that increase importance
        high_impact_keywords = [
            'significant', 'critical', 'major', 'strong', 'high', 'increase', 'decrease',
            'opportunity', 'risk', 'optimize', 'improve', 'growth', 'decline', 'anomaly',
            'pattern', 'trend', 'correlation', 'predict', 'forecast'
        ]
        
        content_lower = content.lower()
        
        for keyword in high_impact_keywords:
            if keyword in content_lower:
                importance += 10
        
        # Numbers and percentages increase importance
        if '%' in content or any(char.isdigit() for char in content):
            importance += 15
        
        # Length bonus (more detailed insights are often more valuable)
        if len(content) > 200:
            importance += 10
        
        return min(importance, 100)
    
    def _generate_fallback_insights(self, summary: Dict[str, Any], analysis_type: str) -> List[Dict[str, Any]]:
        """Generate fallback insights when LLM is unavailable"""
        
        insights = []
        
        if analysis_type == "comprehensive":
            # Generate rule-based insights from summary
            for metric, stats in summary['metrics_analysis'].items():
                
                # Trend insight
                if 'trend_correlation' in stats:
                    if abs(stats['trend_correlation']) > 0.3:
                        direction = "upward" if stats['trend_correlation'] > 0 else "downward"
                        strength = "strong" if abs(stats['trend_correlation']) > 0.6 else "moderate"
                        
                        insights.append({
                            'type': 'llm_insight',
                            'category': 'comprehensive',
                            'title': f"{metric} Trend Analysis",
                            'text': f"Shows a {strength} {direction} trend with correlation of {stats['trend_correlation']:.3f}. This indicates {'positive momentum' if direction == 'upward' else 'declining performance'} that requires {'investment to maintain growth' if direction == 'upward' else 'immediate attention to reverse the decline'}.",
                            'importance': int(abs(stats['trend_correlation']) * 80),
                            'source': 'rule_based',
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Volatility insight
                if stats['coefficient_variation'] > 0.3:
                    insights.append({
                        'type': 'llm_insight',
                        'category': 'comprehensive',
                        'title': f"{metric} Volatility Analysis",
                        'text': f"Exhibits high volatility (CV: {stats['coefficient_variation']:.2f}) with values ranging from {stats['min']:.2f} to {stats['max']:.2f}. This level of variation suggests external factors significantly impact this metric, requiring robust monitoring and flexible response strategies.",
                        'importance': int(stats['coefficient_variation'] * 60),
                        'source': 'rule_based',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Correlation insights
            if 'correlations' in summary:
                for corr in summary['correlations'][:3]:
                    strength = 'very strong' if abs(corr['correlation']) > 0.8 else 'strong'
                    direction = 'positive' if corr['correlation'] > 0 else 'negative'
                    
                    insights.append({
                        'type': 'llm_insight',
                        'category': 'comprehensive',
                        'title': f"{corr['metric1']} - {corr['metric2']} Relationship",
                        'text': f"Strong {direction} correlation ({corr['correlation']:.3f}) indicates these metrics move together. This relationship can be leveraged for {'mutual optimization' if direction == 'positive' else 'risk mitigation through diversification'}.",
                        'importance': int(abs(corr['correlation']) * 70),
                        'source': 'rule_based',
                        'timestamp': datetime.now().isoformat()
                    })
        
        return insights[:5]  # Limit fallback insights
    
    def generate_executive_summary(self, insights: List[Dict[str, Any]]) -> str:
        """Generate executive summary from insights"""
        
        if not insights:
            return "No insights available for executive summary."
        
        # Sort by importance
        sorted_insights = sorted(insights, key=lambda x: x.get('importance', 0), reverse=True)
        
        summary = f"""
# EXECUTIVE INSIGHTS SUMMARY
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Dataset Overview
- **Time Period**: {(self.df[self.time_col].max() - self.df[self.time_col].min()).days} days
- **Total Records**: {len(self.df):,}
- **Metrics Analyzed**: {len(self.metric_cols)} key performance indicators

## Top Strategic Insights

"""
        
        for i, insight in enumerate(sorted_insights[:5], 1):
            title = insight.get('title', f"Insight {i}")
            text = insight.get('text', '')
            importance = insight.get('importance', 0)
            
            summary += f"""
### {i}. {title}
**Priority**: {'🔴 High' if importance > 80 else '🟡 Medium' if importance > 60 else '🟢 Low'} ({importance}/100)

{text}

---
"""
        
        summary += f"""
## Analysis Methodology
- **Data Processing**: Statistical analysis of {len(self.df):,} records
- **Insight Generation**: {'LLM-powered' if any(i.get('source') == 'llm' for i in insights) else 'Rule-based'} analysis
- **Quality Score**: {len([i for i in insights if i.get('importance', 0) > 70])}/{len(insights)} high-confidence insights

## Recommendations
Based on the analysis, focus on the top 3 insights for immediate action. These represent the highest impact opportunities for your business.
"""
        
        return summary


class SmartInsightEngine:
    """Enhanced insight engine that combines rule-based and LLM insights"""
    
    def __init__(self, df: pd.DataFrame, time_column: str, metric_columns: List[str]):
        self.df = df
        self.time_col = time_column
        self.metric_cols = metric_columns
        
        # Initialize both generators
        try:
            from .insight_generator import InsightGenerator
            self.rule_based = InsightGenerator(df, time_column)
        except ImportError:
            self.rule_based = None
        
        self.llm_generator = LLMInsightGenerator(df, time_column, metric_columns)
    
    def generate_comprehensive_insights(self, use_llm: bool = True) -> Dict[str, Any]:
        """Generate comprehensive insights using both approaches"""
        
        results = {
            'rule_based_insights': [],
            'llm_insights': [],
            'combined_insights': [],
            'executive_summary': '',
            'generation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(self.df),
                'metrics_analyzed': len(self.metric_cols),
                'llm_enabled': use_llm
            }
        }
        
        # Generate rule-based insights
        if self.rule_based:
            try:
                rule_insights = self.rule_based.generate_all_insights(self.metric_cols)
                results['rule_based_insights'] = rule_insights
            except Exception as e:
                print(f"Error generating rule-based insights: {str(e)}")
        
        # Generate LLM insights
        if use_llm:
            try:
                llm_insights = self.llm_generator.generate_llm_insights()
                results['llm_insights'] = llm_insights
            except Exception as e:
                print(f"Error generating LLM insights: {str(e)}")
        
        # Combine and deduplicate
        all_insights = results['rule_based_insights'] + results['llm_insights']
        
        # Sort by importance and remove duplicates
        unique_insights = []
        seen_texts = set()
        
        for insight in sorted(all_insights, key=lambda x: x.get('importance', 0), reverse=True):
            text_key = insight.get('text', '')[:100].lower()  # First 100 chars for similarity
            if text_key not in seen_texts:
                unique_insights.append(insight)
                seen_texts.add(text_key)
        
        results['combined_insights'] = unique_insights[:15]  # Top 15 insights
        
        # Generate executive summary
        if results['combined_insights']:
            executive_summary = self.llm_generator.generate_executive_summary(results['combined_insights'])
            results['executive_summary'] = executive_summary
        
        return results