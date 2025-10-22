"""
Test Script for Advanced Visualizations
Quick verification that all components work together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_sample_generator():
    """Test sample dataset generation"""
    print("Testing sample dataset generation...")
    
    try:
        from utils.sample_generator import SampleDatasetGenerator
        
        # Test e-commerce data
        df = SampleDatasetGenerator.generate_ecommerce_data(n_days=7, freq='H')
        print(f"✅ E-commerce data: {df.shape} - Columns: {list(df.columns)}")
        
        # Test IoT data
        df = SampleDatasetGenerator.generate_iot_sensor_data(n_days=3)
        print(f"✅ IoT sensor data: {df.shape} - Columns: {list(df.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Sample generator error: {e}")
        return False

def test_smart_configurator():
    """Test smart dataset configurator"""
    print("\nTesting smart dataset configurator...")
    
    try:
        from utils.sample_generator import SampleDatasetGenerator
        from intelligence.smart_configurator import SmartDatasetConfigurator
        
        # Generate test data
        df = SampleDatasetGenerator.generate_ecommerce_data(n_days=5)
        time_col = 'timestamp'
        metric_cols = ['visitors', 'page_views', 'orders', 'revenue']
        
        # Test configurator
        configurator = SmartDatasetConfigurator(df, time_col, metric_cols)
        
        print(f"✅ Data frequency detected: {configurator.config['data_frequency']}")
        print(f"✅ Recommended visualizations: {len(configurator.config['recommended_visualizations'])}")
        print(f"✅ Optimal clusters: {configurator.config['optimal_clusters']}")
        
        # Test insights
        insights = configurator.get_insights()
        print(f"✅ Generated {len(insights)} insights")
        
        return True
    except Exception as e:
        print(f"❌ Smart configurator error: {e}")
        return False

def test_column_detector():
    """Test smart column detector"""
    print("\nTesting smart column detector...")
    
    try:
        from utils.sample_generator import SampleDatasetGenerator
        from intelligence.column_detector import SmartColumnDetector
        
        # Generate test data
        df = SampleDatasetGenerator.generate_financial_data(n_days=30)
        
        # Test detector
        detector = SmartColumnDetector(df)
        results = detector.detect_all()
        
        print(f"✅ Datetime columns detected: {len(results['datetime'])}")
        print(f"✅ Numeric columns detected: {len(results['numeric'])}")
        print(f"✅ Recommendations: {results['recommendations']}")
        
        return True
    except Exception as e:
        print(f"❌ Column detector error: {e}")
        return False

def test_visualization_generator():
    """Test advanced visualization generator"""
    print("\nTesting advanced visualization generator...")
    
    try:
        from utils.sample_generator import SampleDatasetGenerator
        from visualization.upload_viz_generator import UploadVizGenerator
        
        # Generate test data
        df = SampleDatasetGenerator.generate_website_analytics(n_days=7)
        time_col = 'timestamp'
        metric_cols = ['sessions', 'page_views', 'conversions', 'bounce_rate']
        
        # Test generator
        generator = UploadVizGenerator(df, time_col, metric_cols)
        
        # Test individual visualizations
        fig1 = generator.generate_multimetric_dashboard()
        print("✅ Multi-metric dashboard generated")
        
        fig2 = generator.generate_pattern_detection_radar()
        print("✅ Pattern detection radar generated")
        
        fig3 = generator.generate_anomaly_landscape(metric_cols[0])
        print("✅ Anomaly landscape generated")
        
        return True
    except Exception as e:
        print(f"❌ Visualization generator error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chronosphere_visualizer():
    """Test original chronosphere visualizer compatibility"""
    print("\nTesting chronosphere visualizer compatibility...")
    
    try:
        from utils.sample_generator import SampleDatasetGenerator
        from visualization.chronosphere_viz import ChronosphereVisualizer
        
        # Generate test data
        df = SampleDatasetGenerator.generate_iot_sensor_data(n_days=3)
        time_col = 'datetime'
        
        # Test visualizer
        visualizer = ChronosphereVisualizer(df, time_col)
        
        # Test basic visualizations
        fig1 = visualizer.temporal_heartbeat('temperature_c')
        print("✅ Temporal heartbeat generated")
        
        fig2 = visualizer.temporal_correlation_explorer('temperature_c', 'humidity_pct')
        print("✅ Correlation explorer generated")
        
        return True
    except Exception as e:
        print(f"❌ Chronosphere visualizer error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Advanced Visualization Tests\n")
    
    tests = [
        test_sample_generator,
        test_column_detector,
        test_smart_configurator,
        test_chronosphere_visualizer,
        test_visualization_generator
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Advanced visualizations are ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)