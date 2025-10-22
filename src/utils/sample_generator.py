"""
Sample Dataset Generator for Testing Advanced Visualizations
Creates realistic temporal datasets for demonstrating upload functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class SampleDatasetGenerator:
    """Generate sample datasets for testing upload visualizations"""
    
    @staticmethod
    def generate_ecommerce_data(n_days: int = 90, freq: str = 'H') -> pd.DataFrame:
        """Generate sample e-commerce metrics data"""
        
        # Date range
        start_date = datetime.now() - timedelta(days=n_days)
        end_date = datetime.now()
        
        if freq == 'H':
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        elif freq == 'D':
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        n_points = len(date_range)
        
        # Base patterns
        hour_pattern = np.sin(2 * np.pi * np.arange(n_points) / 24) * 0.3 + 1
        day_pattern = np.sin(2 * np.pi * np.arange(n_points) / (24 * 7)) * 0.2 + 1
        trend = np.linspace(1, 1.5, n_points)
        
        # Generate correlated metrics
        base_visitors = 1000 * hour_pattern * day_pattern * trend
        base_visitors += np.random.normal(0, 100, n_points)
        base_visitors = np.maximum(base_visitors, 50)
        
        page_views = base_visitors * (2.5 + np.random.normal(0, 0.3, n_points))
        page_views = np.maximum(page_views, base_visitors)
        
        conversion_rate = 0.03 + np.sin(2 * np.pi * np.arange(n_points) / (24 * 30)) * 0.01
        conversion_rate += np.random.normal(0, 0.005, n_points)
        conversion_rate = np.clip(conversion_rate, 0.005, 0.1)
        
        orders = base_visitors * conversion_rate
        orders += np.random.poisson(5, n_points)
        
        avg_order_value = 50 + np.sin(2 * np.pi * np.arange(n_points) / (24 * 7)) * 10
        avg_order_value += np.random.normal(0, 5, n_points)
        avg_order_value = np.maximum(avg_order_value, 20)
        
        revenue = orders * avg_order_value
        
        bounce_rate = 0.4 + np.sin(2 * np.pi * np.arange(n_points) / 24) * 0.1
        bounce_rate += np.random.normal(0, 0.05, n_points)
        bounce_rate = np.clip(bounce_rate, 0.1, 0.8)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_points, size=max(1, n_points // 100), replace=False)
        for idx in anomaly_indices:
            if idx < len(base_visitors):
                base_visitors[idx] *= np.random.uniform(0.3, 3.0)
                page_views[idx] *= np.random.uniform(0.3, 3.0)
                orders[idx] *= np.random.uniform(0.2, 4.0)
                revenue[idx] *= np.random.uniform(0.2, 4.0)
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'visitors': base_visitors.astype(int),
            'page_views': page_views.astype(int),
            'orders': orders.astype(int),
            'revenue': revenue.round(2),
            'conversion_rate': conversion_rate.round(4),
            'avg_order_value': avg_order_value.round(2),
            'bounce_rate': bounce_rate.round(3)
        })
        
        return df
    
    @staticmethod
    def generate_iot_sensor_data(n_days: int = 30, freq: str = '15T') -> pd.DataFrame:
        """Generate sample IoT sensor data"""
        
        start_date = datetime.now() - timedelta(days=n_days)
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        n_points = len(date_range)
        
        # Temperature with daily cycle and trend
        temp_base = 20 + 5 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 4))  # 4 readings per hour
        temp_trend = np.linspace(0, 3, n_points)  # Gradual warming
        temperature = temp_base + temp_trend + np.random.normal(0, 0.5, n_points)
        
        # Humidity inversely correlated with temperature
        humidity = 60 - (temperature - 20) * 2 + np.random.normal(0, 2, n_points)
        humidity = np.clip(humidity, 30, 90)
        
        # Pressure with weather patterns
        pressure_base = 1013 + np.sin(2 * np.pi * np.arange(n_points) / (24 * 4 * 3)) * 10
        pressure = pressure_base + np.random.normal(0, 2, n_points)
        
        # Light intensity with day/night cycle
        hour_of_day = (np.arange(n_points) * 15 / 60) % 24  # 15-minute intervals
        light_intensity = np.maximum(0, 800 * np.sin(np.pi * (hour_of_day - 6) / 12))
        light_intensity = np.where((hour_of_day < 6) | (hour_of_day > 18), 
                                 light_intensity * 0.1, light_intensity)
        light_intensity += np.random.normal(0, 20, n_points)
        light_intensity = np.maximum(light_intensity, 0)
        
        # Air quality index
        aqi_base = 50 + np.sin(2 * np.pi * np.arange(n_points) / (24 * 4 * 7)) * 20
        aqi = aqi_base + np.random.normal(0, 5, n_points)
        aqi = np.maximum(aqi, 10)
        
        # Add anomalies (sensor malfunctions)
        anomaly_indices = np.random.choice(n_points, size=max(1, n_points // 200), replace=False)
        for idx in anomaly_indices:
            if idx < len(temperature):
                temperature[idx] = np.random.uniform(-10, 50)
                humidity[idx] = np.random.uniform(0, 100)
                pressure[idx] = np.random.uniform(950, 1050)
        
        df = pd.DataFrame({
            'datetime': date_range,
            'temperature_c': temperature.round(1),
            'humidity_pct': humidity.round(1),
            'pressure_hpa': pressure.round(1),
            'light_intensity': light_intensity.round(0).astype(int),
            'air_quality_index': aqi.round(0).astype(int)
        })
        
        return df
    
    @staticmethod
    def generate_financial_data(n_days: int = 365, freq: str = 'D') -> pd.DataFrame:
        """Generate sample financial time series data"""
        
        start_date = datetime.now() - timedelta(days=n_days)
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        n_points = len(date_range)
        
        # Stock price with random walk and trend
        price_changes = np.random.normal(0.001, 0.02, n_points)  # Daily returns
        price_changes[0] = 0
        stock_price = 100 * np.exp(np.cumsum(price_changes))
        
        # Volume inversely correlated with price stability
        volatility = np.abs(price_changes)
        volume = 1000000 * (1 + volatility * 10) * np.random.lognormal(0, 0.3, n_points)
        volume = volume.astype(int)
        
        # Market cap
        shares_outstanding = 10000000
        market_cap = stock_price * shares_outstanding
        
        # RSI indicator
        def calculate_rsi(prices, window=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = pd.Series(gains).rolling(window=window).mean()
            avg_losses = pd.Series(losses).rolling(window=window).mean()
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50).values
        
        rsi = calculate_rsi(stock_price)
        
        # Moving averages
        ma_20 = pd.Series(stock_price).rolling(window=20).mean().fillna(stock_price[0])
        ma_50 = pd.Series(stock_price).rolling(window=50).mean().fillna(stock_price[0])
        
        # Beta (correlation with market)
        market_returns = np.random.normal(0.0005, 0.015, n_points)
        stock_returns = np.diff(np.log(stock_price))
        stock_returns = np.insert(stock_returns, 0, 0)
        
        # Calculate rolling beta
        beta = []
        for i in range(n_points):
            start_idx = max(0, i - 30)
            if i > 30:
                beta_val = np.corrcoef(stock_returns[start_idx:i], market_returns[start_idx:i])[0, 1]
                beta.append(beta_val if not np.isnan(beta_val) else 1.0)
            else:
                beta.append(1.0)
        
        df = pd.DataFrame({
            'date': date_range,
            'stock_price': stock_price.round(2),
            'volume': volume,
            'market_cap': market_cap.round(0).astype(int),
            'rsi': rsi.round(2),
            'ma_20': ma_20.round(2),
            'ma_50': ma_50.round(2),
            'beta': np.array(beta).round(3),
            'daily_return': (stock_returns * 100).round(3)
        })
        
        return df
    
    @staticmethod
    def generate_website_analytics(n_days: int = 60, freq: str = 'H') -> pd.DataFrame:
        """Generate sample website analytics data"""
        
        start_date = datetime.now() - timedelta(days=n_days)
        end_date = datetime.now()
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        n_points = len(date_range)
        
        # Base patterns
        hour_of_day = np.array([dt.hour for dt in date_range])
        day_of_week = np.array([dt.weekday() for dt in date_range])
        
        # Traffic patterns
        hourly_multiplier = np.where(
            (hour_of_day >= 9) & (hour_of_day <= 17), 
            1.5,  # Business hours
            np.where(
                (hour_of_day >= 19) & (hour_of_day <= 22),
                1.2,  # Evening peak
                0.6   # Off hours
            )
        )
        
        weekly_multiplier = np.where(
            day_of_week < 5, 1.0,  # Weekdays
            np.where(day_of_week == 5, 0.8, 0.6)  # Weekend
        )
        
        base_sessions = 500 * hourly_multiplier * weekly_multiplier
        base_sessions += np.random.normal(0, 50, n_points)
        base_sessions = np.maximum(base_sessions, 50)
        
        # Page views per session
        pages_per_session = 2.5 + np.random.normal(0, 0.5, n_points)
        pages_per_session = np.maximum(pages_per_session, 1.0)
        
        page_views = base_sessions * pages_per_session
        
        # Session duration (minutes)
        session_duration = 3.5 + np.random.exponential(2, n_points)
        session_duration = np.minimum(session_duration, 30)  # Cap at 30 minutes
        
        # Bounce rate
        bounce_rate = 0.45 + np.random.normal(0, 0.05, n_points)
        bounce_rate = np.clip(bounce_rate, 0.2, 0.8)
        
        # Conversion rate
        conversion_rate = 0.025 + np.random.normal(0, 0.005, n_points)
        conversion_rate = np.clip(conversion_rate, 0.005, 0.1)
        
        conversions = base_sessions * conversion_rate
        
        # Add some campaign spikes
        campaign_indices = np.random.choice(n_points, size=max(1, n_points // 50), replace=False)
        for idx in campaign_indices:
            if idx < len(base_sessions):
                multiplier = np.random.uniform(1.5, 3.0)
                base_sessions[idx] *= multiplier
                page_views[idx] *= multiplier
                conversions[idx] *= multiplier * 1.2  # Campaigns also improve conversion
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'sessions': base_sessions.astype(int),
            'page_views': page_views.astype(int),
            'avg_session_duration': session_duration.round(1),
            'bounce_rate': bounce_rate.round(3),
            'conversion_rate': conversion_rate.round(4),
            'conversions': conversions.astype(int),
            'pages_per_session': pages_per_session.round(2)
        })
        
        return df
    
    @staticmethod
    def get_sample_datasets() -> dict:
        """Get all available sample datasets"""
        return {
            'E-commerce Metrics (90 days, hourly)': SampleDatasetGenerator.generate_ecommerce_data(),
            'IoT Sensor Data (30 days, 15-min)': SampleDatasetGenerator.generate_iot_sensor_data(),
            'Financial Time Series (1 year, daily)': SampleDatasetGenerator.generate_financial_data(),
            'Website Analytics (60 days, hourly)': SampleDatasetGenerator.generate_website_analytics()
        }


def save_sample_datasets_to_csv():
    """Save sample datasets to CSV files for easy access"""
    datasets = SampleDatasetGenerator.get_sample_datasets()
    
    import os
    output_dir = "data/sample_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in datasets.items():
        filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('-', '_') + '.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {name} to {filepath}")


if __name__ == "__main__":
    # Generate and save sample datasets
    save_sample_datasets_to_csv()
    print("Sample datasets generated successfully!")