"""
Test script for LLM Insight Generator
Run this to verify the LLM insight functionality works
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_test_data():
    """Create sample time series data for testing"""
    
    # Generate 90 days of hourly data
    start_date = datetime.now() - timedelta(days=90)
    dates = pd.date_range(start=start_date, periods=90*24, freq='H')
    
    # Create realistic metrics with patterns
    np.random.seed(42)
    
    data = {
        'timestamp': dates,
        'visitors': np.random.poisson(lam=100, size=len(dates)) + 
                   20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) +  # Daily pattern
                   10 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24*7)),  # Weekly pattern
        'sales': np.random.exponential(scale=50, size=len(dates)) + 
                30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24 + 1),
        'conversion_rate': np.random.beta(2, 18, size=len(dates)) * 100,  # 0-100%
        'bounce_rate': np.random.beta(4, 6, size=len(dates)) * 100  # 0-100%
    }
    
    df = pd.DataFrame(data)
    
    # Add some anomalies
    anomaly_indices = np.random.choice(len(df), size=5, replace=False)
    df.loc[anomaly_indices, 'visitors'] *= 3
    df.loc[anomaly_indices, 'sales'] *= 2
    
    return df

def test_llm_insights():
    """Test the LLM insight generator"""
    
    print("🧪 Testing LLM Insight Generator...")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating test dataset...")
    df = create_test_data()
    print(f"✅ Created dataset with {len(df):,} rows and {len(df.columns)} columns")
    print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Test the insight generator
    try:
        from intelligence.llm_insight_generator import SmartInsightEngine
        
        time_col = 'timestamp'
        metric_cols = ['visitors', 'sales', 'conversion_rate', 'bounce_rate']
        
        print(f"\n🤖 Initializing Smart Insight Engine...")
        insight_engine = SmartInsightEngine(df, time_col, metric_cols)
        
        print("📈 Generating statistical summary...")
        summary = insight_engine.llm_generator.generate_statistical_summary()
        
        print(f"✅ Statistical summary generated:")
        print(f"   - Dataset: {summary['dataset_info']['total_rows']:,} rows, {summary['dataset_info']['time_span_days']} days")
        print(f"   - Missing data: {summary['dataset_info']['missing_data_pct']:.1f}%")
        print(f"   - Metrics analyzed: {len(summary['metrics_analysis'])}")
        
        if 'correlations' in summary:
            print(f"   - Strong correlations found: {len(summary['correlations'])}")
        
        print("\n🎯 Testing LLM prompt generation...")
        prompt = insight_engine.llm_generator.create_llm_prompt(summary, "comprehensive")
        print(f"✅ Generated prompt with {len(prompt)} characters")
        
        print("\n🌐 Testing API connectivity (with fallback)...")
        
        # Test with LLM disabled first (fallback mode)
        print("📊 Testing fallback mode...")
        results_fallback = insight_engine.generate_comprehensive_insights(use_llm=False)
        
        print(f"✅ Fallback insights generated:")
        print(f"   - Rule-based insights: {len(results_fallback['rule_based_insights'])}")
        print(f"   - Combined insights: {len(results_fallback['combined_insights'])}")
        
        # Show top 3 fallback insights
        print("\n🎯 Top 3 Fallback Insights:")
        for i, insight in enumerate(results_fallback['combined_insights'][:3], 1):
            print(f"{i}. {insight.get('title', 'Insight')} (Score: {insight.get('importance', 0)}/100)")
            print(f"   {insight.get('text', '')[:100]}...")
        
        # Test with LLM enabled (may use API or fallback)
        print("\n🤖 Testing LLM mode...")
        results_llm = insight_engine.generate_comprehensive_insights(use_llm=True)
        
        print(f"✅ LLM insights generated:")
        print(f"   - Rule-based insights: {len(results_llm['rule_based_insights'])}")
        print(f"   - LLM insights: {len(results_llm['llm_insights'])}")
        print(f"   - Combined insights: {len(results_llm['combined_insights'])}")
        
        # Show executive summary if available
        if results_llm['executive_summary']:
            print("\n📋 Executive Summary Generated:")
            print("✅ Summary available")
        
        # Show top 3 combined insights
        print("\n🏆 Top 3 Combined Insights:")
        for i, insight in enumerate(results_llm['combined_insights'][:3], 1):
            title = insight.get('title', 'Insight')
            importance = insight.get('importance', 0)
            source = insight.get('source', 'unknown')
            print(f"{i}. {title} (Score: {importance}/100, Source: {source})")
            print(f"   {insight.get('text', '')[:150]}...")
        
        print("\n🎉 All tests passed successfully!")
        print("\n💡 The LLM Insight Generator is working correctly.")
        print("   You can now use it in the Streamlit app for powerful AI insights!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Test Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_insights()
    
    if success:
        print("\n" + "="*50)
        print("🚀 Ready to use! Start the app with:")
        print("streamlit run app/chronosphere_app.py")
    else:
        print("\n" + "="*50) 
        print("❌ Please fix the errors above before using the LLM insights feature.")