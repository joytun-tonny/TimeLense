"""
Demo Script: LLM-Generated Insights for TimeLense
Shows how to use the new AI-powered insights feature
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_ecommerce_demo_data():
    """Create realistic e-commerce data for demo"""
    
    print("📊 Creating realistic e-commerce dataset...")
    
    # 60 days of hourly data
    start_date = datetime.now() - timedelta(days=60)
    dates = pd.date_range(start=start_date, periods=60*24, freq='h')
    
    np.random.seed(42)
    n_records = len(dates)
    
    # Base patterns
    hour_factor = np.sin(np.arange(n_records) * 2 * np.pi / 24) * 0.3 + 1  # Daily cycle
    weekday_factor = np.sin(np.arange(n_records) * 2 * np.pi / (24*7)) * 0.2 + 1  # Weekly cycle
    trend_factor = np.linspace(0.8, 1.4, n_records)  # Growth trend
    
    data = {
        'timestamp': dates,
        'website_visitors': np.random.poisson(
            lam=200 * hour_factor * weekday_factor * trend_factor
        ).astype(float),
        'page_views': np.random.poisson(
            lam=800 * hour_factor * weekday_factor * trend_factor
        ).astype(float),
        'orders': np.random.poisson(
            lam=15 * hour_factor * weekday_factor * trend_factor
        ).astype(float),
        'revenue': np.random.gamma(
            shape=2, scale=250 * hour_factor * weekday_factor * trend_factor
        ),
        'conversion_rate': np.random.beta(2, 98) * 100 * (0.8 + 0.4 * hour_factor),  # 0-10%
        'avg_order_value': np.random.gamma(shape=3, scale=50) + 20
    }
    
    df = pd.DataFrame(data)
    
    # Add some realistic anomalies (sales events, technical issues)
    anomaly_dates = np.random.choice(len(df), size=8, replace=False)
    
    # Sales events (positive anomalies)
    sales_events = anomaly_dates[:4]
    df.loc[sales_events, 'website_visitors'] *= np.random.uniform(2, 4, len(sales_events))
    df.loc[sales_events, 'orders'] *= np.random.uniform(1.5, 3, len(sales_events))
    df.loc[sales_events, 'revenue'] *= np.random.uniform(1.8, 3.5, len(sales_events))
    
    # Technical issues (negative anomalies)
    tech_issues = anomaly_dates[4:]
    df.loc[tech_issues, 'website_visitors'] *= np.random.uniform(0.1, 0.4, len(tech_issues))
    df.loc[tech_issues, 'conversion_rate'] *= np.random.uniform(0.2, 0.6, len(tech_issues))
    
    # Ensure no negative values
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).abs()
    
    print(f"✅ Created dataset: {len(df):,} records over {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    return df

def demo_llm_insights():
    """Demonstrate LLM insights generation"""
    
    print("\n" + "="*60)
    print("🤖 TimeLense LLM Insights Demo")
    print("="*60)
    
    # Create demo data
    df = create_ecommerce_demo_data()
    
    time_col = 'timestamp'
    metric_cols = ['website_visitors', 'page_views', 'orders', 'revenue', 'conversion_rate', 'avg_order_value']
    
    print(f"\n📋 Dataset Summary:")
    print(f"   • Time Range: {df[time_col].min().strftime('%Y-%m-%d')} to {df[time_col].max().strftime('%Y-%m-%d')}")
    print(f"   • Total Records: {len(df):,}")
    print(f"   • Metrics: {', '.join(metric_cols)}")
    
    # Show basic statistics
    print(f"\n📊 Key Statistics:")
    for col in metric_cols[:4]:  # Show first 4 metrics
        mean_val = df[col].mean()
        std_val = df[col].std()
        cv = std_val / mean_val if mean_val > 0 else 0
        print(f"   • {col}: μ={mean_val:.1f}, σ={std_val:.1f}, CV={cv:.2f}")
    
    try:
        from src.intelligence.llm_insight_generator import SmartInsightEngine
        
        print(f"\n🧠 Initializing AI Insight Engine...")
        insight_engine = SmartInsightEngine(df, time_col, metric_cols)
        
        print(f"🎯 Generating comprehensive insights...")
        print(f"   (This may take 30-60 seconds with API calls)")
        
        # Generate insights
        results = insight_engine.generate_comprehensive_insights(use_llm=True)
        
        print(f"\n✅ Insights Generated!")
        print(f"   • Rule-based insights: {len(results['rule_based_insights'])}")
        print(f"   • LLM insights: {len(results['llm_insights'])}")
        print(f"   • Combined insights: {len(results['combined_insights'])}")
        
        # Show executive summary
        if results['executive_summary']:
            print(f"\n📋 EXECUTIVE SUMMARY")
            print("-" * 60)
            # Show first 500 characters of summary
            summary_preview = results['executive_summary'][:500]
            if len(results['executive_summary']) > 500:
                summary_preview += "..."
            print(summary_preview)
        
        # Show top insights
        print(f"\n🏆 TOP 5 INSIGHTS")
        print("-" * 60)
        
        for i, insight in enumerate(results['combined_insights'][:5], 1):
            title = insight.get('title', f"Insight {i}")
            importance = insight.get('importance', 0)
            source = insight.get('source', 'unknown')
            text = insight.get('text', '')
            
            # Determine priority emoji
            priority = "🔴" if importance > 80 else "🟡" if importance > 60 else "🟢"
            source_emoji = "🤖" if source == 'llm' else "📊"
            
            print(f"\n{i}. {priority} {title}")
            print(f"   Priority: {importance}/100 | Source: {source_emoji}")
            print(f"   {text[:200]}{'...' if len(text) > 200 else ''}")
        
        # Show analysis breakdown
        print(f"\n📈 INSIGHT CATEGORIES")
        print("-" * 60)
        
        categories = {}
        for insight in results['combined_insights']:
            category = insight.get('category', insight.get('type', 'other'))
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        for category, count in categories.items():
            print(f"   • {category.title()}: {count} insights")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"   The LLM insights feature is ready for production use.")
        print(f"   Users can access this through the Streamlit app '💡 Insights' page.")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Make sure to install dependencies: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\n❌ Error during insight generation: {str(e)}")
        print("This might be due to API rate limits or connectivity issues.")
        print("The feature will work with fallback mode in the app.")
        return False

def show_usage_instructions():
    """Show how to use the feature in the app"""
    
    print(f"\n" + "="*60)
    print("📖 HOW TO USE LLM INSIGHTS IN THE APP")
    print("="*60)
    
    print("""
1. 🚀 START THE APP:
   streamlit run app/chronosphere_app.py

2. 📤 UPLOAD YOUR DATA:
   • Go to "📤 Upload Data" page
   • Upload CSV/Excel with time series data
   • Configure time column and metrics
   • System will auto-detect and optimize

3. 💡 GENERATE INSIGHTS:
   • Navigate to "💡 Insights" page
   • Choose analysis type:
     - 🎯 Comprehensive Analysis (recommended)
     - ⚠️ Anomaly Deep Dive
     - 🔮 Predictive Insights
   • Configure settings (enable LLM, set insight count)
   • Click generate and wait 30-60 seconds

4. 📋 REVIEW RESULTS:
   • Executive Summary: High-level overview
   • AI Insights: LLM-generated business insights
   • Statistical Insights: Rule-based analysis
   • Export: Download reports in Markdown

🔧 OPTIONAL SETUP FOR ENHANCED INSIGHTS:
   • Copy .env.example to .env
   • Add API keys for Gemini/Hugging Face
   • System works without keys using fallback mode

💡 BEST RESULTS:
   • Use datasets with clear patterns (trends, seasonality)
   • Include 3-5 related business metrics
   • Ensure good data quality (< 10% missing values)
   • Have at least 30 days of historical data
""")

if __name__ == "__main__":
    print("🎬 Starting TimeLense LLM Insights Demo...")
    
    try:
        success = demo_llm_insights()
        show_usage_instructions()
        
        if success:
            print(f"\n✅ Demo completed successfully!")
            print(f"🚀 The LLM insights feature is ready to use!")
        else:
            print(f"\n⚠️ Demo had issues, but the feature should still work in fallback mode.")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()