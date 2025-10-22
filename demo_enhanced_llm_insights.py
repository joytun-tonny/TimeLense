"""
🤖 ENHANCED LLM INSIGHTS DEMO FOR TIMELENSE
==========================================

This demo showcases the advanced AI-powered insights feature that generates
natural language business intelligence from temporal data.

Features Demonstrated:
- 🧠 LLM-powered pattern discovery
- 📊 Statistical analysis integration  
- 🎯 Business-focused recommendations
- 📈 Predictive insights
- 🔍 Anomaly explanations
- 📋 Executive summary generation

Run this script to see the insights system in action!
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_realistic_business_dataset():
    """Create a comprehensive business dataset with clear patterns"""
    
    print("📊 Generating realistic business dataset with intentional patterns...")
    
    # 90 days of hourly data
    start_date = datetime.now() - timedelta(days=90)
    dates = pd.date_range(start=start_date, periods=90*24, freq='h')
    
    np.random.seed(42)  # For reproducible results
    n_records = len(dates)
    
    # Create realistic business patterns
    
    # 1. Daily cycle (business hours effect)
    hour_factor = np.array([
        0.3 if h < 6 else    # Early morning (low activity)
        0.6 if h < 9 else    # Morning ramp-up
        1.0 if h < 12 else   # Morning peak
        1.2 if h < 14 else   # Lunch peak
        1.0 if h < 17 else   # Afternoon
        0.8 if h < 20 else   # Evening
        0.5 if h < 23 else   # Late evening
        0.3                  # Night
        for h in range(24)
    ])
    
    daily_pattern = np.tile(hour_factor, 90)
    
    # 2. Weekly cycle (weekday vs weekend)
    weekly_pattern = np.array([
        1.2 if d < 5 else 0.7  # Weekdays vs weekends
        for d in [date.weekday() for date in dates]
    ])
    
    # 3. Monthly growth trend
    growth_trend = np.linspace(1.0, 1.8, n_records)
    
    # 4. Seasonal effect (quarterly cycle)
    seasonal_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 90))
    
    # Base metrics with realistic correlations
    
    # Website Traffic
    base_traffic = 500 * daily_pattern * weekly_pattern * growth_trend * seasonal_pattern
    website_visitors = np.random.poisson(base_traffic).astype(float)
    
    # Page Views (correlated with visitors)
    pages_per_visitor = 2.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 7))
    page_views = website_visitors * pages_per_visitor + np.random.normal(0, 100, n_records)
    page_views = np.maximum(page_views, website_visitors)
    
    # Conversion funnel
    base_conversion_rate = 0.03 + 0.01 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 30))
    conversion_rate = base_conversion_rate + np.random.normal(0, 0.005, n_records)
    conversion_rate = np.clip(conversion_rate, 0.005, 0.08)
    
    orders = website_visitors * conversion_rate + np.random.poisson(2, n_records)
    
    # Revenue metrics
    base_order_value = 75 + 15 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 7))  # Weekly cycle
    avg_order_value = base_order_value + np.random.normal(0, 8, n_records)
    avg_order_value = np.maximum(avg_order_value, 25)
    
    revenue = orders * avg_order_value
    
    # Customer satisfaction (inverse of bounce rate)
    base_bounce_rate = 0.45 - 0.1 * np.sin(2 * np.pi * np.arange(n_records) / (24 * 30))
    bounce_rate = base_bounce_rate + np.random.normal(0, 0.05, n_records)
    bounce_rate = np.clip(bounce_rate, 0.2, 0.8)
    
    # Add realistic business anomalies
    
    # 1. Marketing campaigns (3-4 spikes)
    campaign_days = [15, 35, 65, 80]  # Day numbers for campaigns
    for day in campaign_days:
        start_idx = day * 24
        end_idx = min(start_idx + 48, n_records)  # 2-day campaigns
        
        if end_idx <= n_records:
            campaign_boost = np.random.uniform(1.8, 3.2)
            website_visitors[start_idx:end_idx] *= campaign_boost
            page_views[start_idx:end_idx] *= campaign_boost * 1.1
            orders[start_idx:end_idx] *= campaign_boost * 1.3  # Better conversion during campaigns
            revenue[start_idx:end_idx] *= campaign_boost * 1.3
    
    # 2. Technical issues (2-3 incidents)
    outage_days = [25, 55]
    for day in outage_days:
        incident_start = day * 24 + np.random.randint(8, 16)  # During business hours
        incident_duration = np.random.randint(2, 8)  # 2-8 hours
        end_idx = min(incident_start + incident_duration, n_records)
        
        if end_idx <= n_records:
            # Severe impact on metrics
            website_visitors[incident_start:end_idx] *= np.random.uniform(0.1, 0.3)
            conversion_rate[incident_start:end_idx] *= np.random.uniform(0.2, 0.5)
            bounce_rate[incident_start:end_idx] = np.minimum(bounce_rate[incident_start:end_idx] * 2, 0.9)
    
    # 3. Seasonal sale events
    sale_day = 50 * 24  # Mid-period sale
    if sale_day < n_records - 24:
        # 24-hour flash sale
        website_visitors[sale_day:sale_day+24] *= np.random.uniform(2.5, 4.0)
        orders[sale_day:sale_day+24] *= np.random.uniform(3.0, 5.0)
        avg_order_value[sale_day:sale_day+24] *= np.random.uniform(0.7, 0.9)  # Lower AOV due to discounts
        revenue[sale_day:sale_day+24] = orders[sale_day:sale_day+24] * avg_order_value[sale_day:sale_day+24]
    
    # Create final dataset
    df = pd.DataFrame({
        'timestamp': dates,
        'website_visitors': np.maximum(website_visitors, 10).astype(int),
        'page_views': np.maximum(page_views, website_visitors).astype(int),
        'orders': np.maximum(orders, 0).astype(int),
        'revenue': np.maximum(revenue, 0).round(2),
        'conversion_rate': np.clip(conversion_rate * 100, 0.1, 10.0).round(3),  # Convert to percentage
        'avg_order_value': np.maximum(avg_order_value, 20).round(2),
        'bounce_rate': np.clip(bounce_rate * 100, 10, 90).round(1)  # Convert to percentage
    })
    
    # Add some derived metrics for richer analysis
    df['revenue_per_visitor'] = (df['revenue'] / df['website_visitors']).round(2)
    df['pages_per_session'] = (df['page_views'] / df['website_visitors']).round(2)
    
    print(f"✅ Generated dataset with intentional patterns:")
    print(f"   • {len(df):,} hourly records over 90 days")
    print(f"   • Strong daily and weekly patterns")
    print(f"   • Growth trend with seasonal variations")
    print(f"   • {len(campaign_days)} marketing campaigns")
    print(f"   • {len(outage_days)} technical incidents")
    print(f"   • 1 major sales event")
    print(f"   • High correlation between metrics")
    
    return df

def demonstrate_llm_insights():
    """Run comprehensive LLM insights demonstration"""
    
    print("\n" + "="*70)
    print("🤖 TIMELENSE ENHANCED LLM INSIGHTS DEMONSTRATION")
    print("="*70)
    
    # Generate test data
    df = create_realistic_business_dataset()
    
    time_col = 'timestamp'
    metric_cols = ['website_visitors', 'page_views', 'orders', 'revenue', 
                   'conversion_rate', 'avg_order_value', 'bounce_rate', 
                   'revenue_per_visitor', 'pages_per_session']
    
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 50)
    print(f"Time Range: {df[time_col].min().strftime('%Y-%m-%d')} to {df[time_col].max().strftime('%Y-%m-%d')}")
    print(f"Total Records: {len(df):,}")
    print(f"Metrics: {len(metric_cols)}")
    print(f"Data Quality: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}% complete")
    
    # Show key statistics
    print(f"\n📈 KEY STATISTICS")
    print("-" * 50)
    for col in metric_cols[:6]:  # Show first 6 metrics
        mean_val = df[col].mean()
        std_val = df[col].std()
        trend_corr = np.corrcoef(range(len(df)), df[col])[0, 1]
        print(f"{col:20} | Mean: {mean_val:8.1f} | Std: {std_val:7.1f} | Trend: {trend_corr:+.3f}")
    
    try:
        from intelligence.llm_insight_generator import SmartInsightEngine
        
        print(f"\n🧠 INITIALIZING AI INSIGHT ENGINE...")
        print("-" * 50)
        
        insight_engine = SmartInsightEngine(df, time_col, metric_cols)
        
        print("✅ Engine initialized successfully")
        print("🔄 Generating statistical summary...")
        
        # Generate comprehensive analysis
        print(f"\n🎯 GENERATING COMPREHENSIVE INSIGHTS...")
        print("💡 This demonstrates all three analysis types:")
        print("   • Comprehensive: General business insights")
        print("   • Anomaly Focused: Deep dive on unusual patterns")  
        print("   • Predictive: Forward-looking analysis")
        print("\n⏳ Processing... (may take 30-90 seconds with API calls)")
        
        # Run all analysis types
        results = insight_engine.generate_comprehensive_insights(use_llm=True)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print("-" * 50)
        print(f"Total Insights Generated: {len(results.get('combined_insights', []))}")
        print(f"  • LLM-powered insights: {len(results.get('llm_insights', []))}")
        print(f"  • Statistical insights: {len(results.get('rule_based_insights', []))}")
        
        # Display Executive Summary
        if results.get('executive_summary'):
            print(f"\n📋 EXECUTIVE SUMMARY")
            print("=" * 70)
            print(results['executive_summary'])
        
        # Show top insights with enhanced formatting
        print(f"\n🏆 TOP INSIGHTS (Ranked by Business Impact)")
        print("=" * 70)
        
        combined_insights = results.get('combined_insights', [])
        
        if combined_insights:
            for i, insight in enumerate(combined_insights[:8], 1):  # Show top 8
                title = insight.get('title', f'Insight {i}')
                text = insight.get('text', 'No description available')
                importance = insight.get('importance', 0)
                source = insight.get('source', 'unknown')
                category = insight.get('category', 'general')
                
                # Priority indicators
                priority_emoji = "🔴" if importance > 80 else "🟡" if importance > 60 else "🟢"
                source_emoji = "🤖" if source == 'llm' else "📊"
                
                print(f"\n{i}. {priority_emoji} {title}")
                print(f"   Priority: {importance}/100 | Source: {source_emoji} {source.upper()} | Category: {category.upper()}")
                print(f"   {'-' * 65}")
                
                # Wrap text for better readability
                import textwrap
                wrapped_text = textwrap.fill(text, width=65, subsequent_indent='   ')
                print(f"   {wrapped_text}")
        
        # Analysis breakdown
        print(f"\n📊 INSIGHT CATEGORY BREAKDOWN")
        print("-" * 50)
        
        categories = {}
        sources = {}
        
        for insight in combined_insights:
            # Count by category
            category = insight.get('category', insight.get('type', 'other'))
            categories[category] = categories.get(category, 0) + 1
            
            # Count by source
            source = insight.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print("By Category:")
        for category, count in sorted(categories.items()):
            print(f"  • {category.replace('_', ' ').title():20}: {count} insights")
        
        print("\nBy Source:")
        for source, count in sorted(sources.items()):
            source_name = "AI-Generated" if source == 'llm' else "Statistical" if source == 'rule_based' else source.title()
            print(f"  • {source_name:20}: {count} insights")
        
        # Performance metrics
        metadata = results.get('generation_metadata', {})
        print(f"\n⚡ PERFORMANCE METRICS")
        print("-" * 50)
        print(f"Dataset Size: {metadata.get('total_records', 'N/A'):,} records")
        print(f"Metrics Analyzed: {metadata.get('metrics_analyzed', 'N/A')}")
        print(f"LLM Analysis: {'✅ Enabled' if metadata.get('llm_enabled', False) else '❌ Disabled'}")
        print(f"Generation Time: {metadata.get('timestamp', 'N/A')}")
        
        # Show sample API usage (if LLM insights were generated)
        llm_insights = results.get('llm_insights', [])
        if llm_insights:
            print(f"\n🌐 API INTEGRATION SUCCESS")
            print("-" * 50)
            print("✅ Successfully connected to LLM APIs")
            print(f"✅ Generated {len(llm_insights)} natural language insights")
            print("✅ Insights include business context and recommendations")
            print("✅ Results can be exported in multiple formats")
        else:
            print(f"\n⚠️ LLM API NOTICE")
            print("-" * 50)
            print("🔄 LLM insights used fallback mode")
            print("💡 To enable full LLM features:")
            print("   1. Copy .env.example to .env")
            print("   2. Add free API keys (Gemini/Hugging Face)")
            print("   3. Restart the application")
            print("📊 Statistical insights are still available and valuable!")
        
        # Export demonstration
        print(f"\n📤 EXPORT CAPABILITIES")
        print("-" * 50)
        
        # Generate sample exports
        export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON export
        json_export = {
            'metadata': metadata,
            'insights': combined_insights[:5],  # Top 5 for demo
            'export_info': {
                'generated_at': export_timestamp,
                'format': 'json',
                'purpose': 'api_integration'
            }
        }
        
        json_filename = f"demo_insights_{export_timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_export, f, indent=2, default=str)
        
        print(f"✅ JSON Export: {json_filename}")
        
        # Markdown report export
        markdown_report = f"""# TimeLense AI Insights Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- **Records**: {len(df):,}
- **Time Span**: {(df[time_col].max() - df[time_col].min()).days} days
- **Metrics**: {len(metric_cols)} key performance indicators
- **Data Quality**: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}% complete

## Executive Summary
{results.get('executive_summary', 'No executive summary available')}

## Top Business Insights
"""
        
        for i, insight in enumerate(combined_insights[:5], 1):
            importance = insight.get('importance', 0)
            title = insight.get('title', f'Insight {i}')
            text = insight.get('text', '')
            source = insight.get('source', 'unknown')
            
            priority = "HIGH" if importance > 80 else "MEDIUM" if importance > 60 else "LOW"
            
            markdown_report += f"""
### {i}. {title}
**Priority**: {priority} ({importance}/100) | **Source**: {source.upper()}

{text}

---
"""
        
        markdown_filename = f"demo_insights_report_{export_timestamp}.md"
        with open(markdown_filename, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        print(f"✅ Markdown Report: {markdown_filename}")
        
        # CSV export for spreadsheet analysis
        if combined_insights:
            insights_df = pd.DataFrame([
                {
                    'rank': i+1,
                    'title': insight.get('title', ''),
                    'importance': insight.get('importance', 0),
                    'source': insight.get('source', ''),
                    'category': insight.get('category', ''),
                    'text': insight.get('text', ''),
                    'timestamp': insight.get('timestamp', '')
                }
                for i, insight in enumerate(combined_insights[:10])
            ])
            
            csv_filename = f"demo_insights_table_{export_timestamp}.csv"
            insights_df.to_csv(csv_filename, index=False)
            print(f"✅ CSV Table: {csv_filename}")
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 70)
        print("✅ All features working correctly")
        print("🚀 Ready for production use in Streamlit app")
        print("📁 Export files generated for integration testing")
        
        # Usage instructions
        print(f"\n📖 NEXT STEPS - Using in the App:")
        print("-" * 50)
        print("1. 🚀 Launch: streamlit run app/chronosphere_app.py")
        print("2. 📤 Upload your CSV/Excel data with time series")
        print("3. 💡 Navigate to 'Insights' page")
        print("4. ⚙️ Configure analysis settings")
        print("5. 🤖 Click 'Generate AI Insights'")
        print("6. 📋 Review results and export reports")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ ERROR: Missing dependencies")
        print("-" * 50)
        print(f"Import error: {str(e)}")
        print("\n🔧 SOLUTION:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Ensure all modules are in src/ directory")
        print("3. Check Python path configuration")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: Insight generation failed")
        print("-" * 50)
        print(f"Error details: {str(e)}")
        print("\n🔧 POSSIBLE SOLUTIONS:")
        print("• Check internet connection for API access")
        print("• Verify data format and completeness")
        print("• Try running with smaller dataset")
        print("• Check API rate limits")
        print("\n💡 The app includes fallback modes for reliability")
        return False

def show_api_setup_guide():
    """Show how to set up free APIs for enhanced insights"""
    
    print(f"\n" + "="*70)
    print("🔑 FREE API SETUP GUIDE FOR ENHANCED INSIGHTS")
    print("="*70)
    
    print("""
🌟 RECOMMENDED: Google Gemini (Best Free Option)
═══════════════════════════════════════════════
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Create API Key" → "Create API key in new project"
4. Copy your API key
5. Add to .env file: GOOGLE_API_KEY=your_key_here

Benefits:
✅ Completely free with generous limits
✅ High-quality natural language generation
✅ Reliable uptime and fast responses
✅ Perfect for business insights

🤖 ALTERNATIVE: Hugging Face (Backup Option)
═══════════════════════════════════════════════
1. Go to: https://huggingface.co/settings/tokens
2. Sign up/login to Hugging Face
3. Click "New token" → Access Token
4. Give it a name and copy the token
5. Add to .env file: HUGGINGFACE_API_KEY=your_token_here

Benefits:
✅ Free tier available
✅ Access to multiple models
✅ Good for experimentation
✅ Open source community

💡 SETUP STEPS:
═══════════════
1. Copy .env.example to .env
2. Add your API key(s) to the .env file
3. Restart the Streamlit app
4. API keys are optional - the app works without them!

🔒 SECURITY NOTES:
══════════════════
• Never commit .env file to version control
• Keep API keys private and secure
• API keys are only used for insights generation
• No data is stored or sent to third parties permanently

⚠️ WITHOUT API KEYS:
════════════════════
The app still works great with:
• Statistical pattern analysis
• Rule-based insights
• All visualizations
• Data exploration features

API keys just add natural language explanations!
""")

if __name__ == "__main__":
    print("🎬 Starting Enhanced LLM Insights Demo...")
    print("This demo showcases the advanced AI capabilities of TimeLense")
    
    try:
        # Run the main demonstration
        success = demonstrate_llm_insights()
        
        # Show API setup guide
        show_api_setup_guide()
        
        if success:
            print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("The enhanced LLM insights feature is ready for use.")
            print("\nFiles generated in this demo can be used to test integrations.")
        else:
            print(f"\n⚠️ DEMO HAD ISSUES")
            print("The feature may still work in production with fallback modes.")
        
        print(f"\n🚀 Launch the app: streamlit run app/chronosphere_app.py")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Demo interrupted by user.")
        print("You can run this demo anytime to test the insights system.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("Check the error details above and requirements.")
        import traceback
        traceback.print_exc()