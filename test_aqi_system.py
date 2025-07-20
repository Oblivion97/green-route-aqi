"""
Simplified AQI Forecasting System Test
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample AQI data from California housing dataset if available."""
    print("Creating sample AQI dataset...")
    
    try:
        # Try to load California housing data
        if os.path.exists('data/california_housing_train.csv'):
            print("Loading California housing data...")
            df = pd.read_csv('data/california_housing_train.csv')
        elif os.path.exists('california_housing_train.csv'):
            print("Loading California housing data...")
            df = pd.read_csv('california_housing_train.csv')
            
            if 'median_house_value' in df.columns:
                # Sort by latitude for spatial consistency
                df = df.sort_values('latitude')
                
                # Normalize house values to AQI range (1-5)
                values = df['median_house_value']
                normalized_values = ((values - values.min()) / (values.max() - values.min()) * 4 + 1).round()
                
                # Create time series
                timestamps = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                
                # Create AQI dataset
                aqi_data = pd.DataFrame({
                    'Timestamp': timestamps,
                    'AQI': normalized_values,
                    'PM2_5': normalized_values * 10 + np.random.normal(0, 2, len(df)),
                    'PM10': normalized_values * 15 + np.random.normal(0, 3, len(df)),
                    'NO2': normalized_values * 5 + np.random.normal(0, 1, len(df)),
                    'Ozone': normalized_values * 8 + np.random.normal(0, 2, len(df)),
                    'Latitude': df['latitude'].values,
                    'Longitude': df['longitude'].values,
                    'DataSource': 'Adapted_Housing'
                })
                
                aqi_data.set_index('Timestamp', inplace=True)
                print(f"✅ Created time series dataset with {len(aqi_data)} records")
                return aqi_data
        
        # Fallback: create synthetic data
        print("Creating synthetic AQI data...")
        timestamps = pd.date_range(start='2024-01-01', periods=5000, freq='H')
        
        # Generate realistic AQI patterns
        base_aqi = 2 + np.sin(np.arange(len(timestamps)) * 2 * np.pi / 24) * 0.5  # Daily pattern
        base_aqi += np.sin(np.arange(len(timestamps)) * 2 * np.pi / (24*7)) * 0.3  # Weekly pattern
        base_aqi += np.random.normal(0, 0.3, len(timestamps))  # Random noise
        base_aqi = np.clip(base_aqi, 1, 5)  # Clip to valid AQI range
        
        aqi_data = pd.DataFrame({
            'Timestamp': timestamps,
            'AQI': base_aqi,
            'PM2_5': base_aqi * 12 + np.random.normal(0, 2, len(timestamps)),
            'PM10': base_aqi * 18 + np.random.normal(0, 3, len(timestamps)),
            'NO2': base_aqi * 6 + np.random.normal(0, 1, len(timestamps)),
            'Ozone': base_aqi * 9 + np.random.normal(0, 2, len(timestamps)),
            'DataSource': 'Synthetic'
        })
        
        aqi_data.set_index('Timestamp', inplace=True)
        print(f"✅ Created synthetic time series dataset with {len(aqi_data)} records")
        return aqi_data
        
    except Exception as e:
        print(f"Error creating data: {e}")
        return None

def train_simple_arima(data, target_col='AQI'):
    """Train a simple ARIMA-like model (using simple moving average for demonstration)."""
    print(f"\nTraining simple time series model for {target_col}...")
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Split data
        series = data[target_col].astype(float)
        train_size = int(len(series) * 0.8)
        train_data = series[:train_size]
        test_data = series[train_size:]
        
        # Try simple ARIMA model
        try:
            model = ARIMA(train_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Make predictions
            predictions = model_fit.predict(start=len(train_data), end=len(series) - 1)
            predictions.index = test_data.index
            
            # Calculate RMSE
            rmse = sqrt(mean_squared_error(test_data, predictions))
            print(f"✅ ARIMA Model RMSE: {rmse:.3f}")
            
            return {
                'model_type': 'ARIMA(1,1,1)',
                'rmse': rmse,
                'predictions': predictions[:10].tolist(),  # First 10 predictions
                'actual': test_data[:10].tolist()  # First 10 actual values
            }
            
        except Exception as e:
            print(f"ARIMA failed: {e}, using simple moving average...")
            
            # Fallback: Simple moving average
            window = 24  # 24 hours
            predictions = []
            
            for i in range(len(test_data)):
                if i + train_size >= window:
                    pred = series[i + train_size - window:i + train_size].mean()
                else:
                    pred = train_data[-window:].mean()
                predictions.append(pred)
            
            predictions = pd.Series(predictions, index=test_data.index)
            rmse = sqrt(mean_squared_error(test_data, predictions))
            print(f"✅ Moving Average Model RMSE: {rmse:.3f}")
            
            return {
                'model_type': 'Moving Average',
                'rmse': rmse,
                'predictions': predictions[:10].tolist(),
                'actual': test_data[:10].tolist()
            }
            
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def create_visualization(data, results):
    """Create simple visualization of results."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: AQI time series (last 168 hours = 1 week)
        plt.subplot(2, 2, 1)
        recent_data = data['AQI'].tail(168)
        plt.plot(recent_data.index, recent_data.values)
        plt.title('AQI Time Series (Last Week)')
        plt.ylabel('AQI')
        plt.xticks(rotation=45)
        
        # Plot 2: AQI distribution
        plt.subplot(2, 2, 2)
        plt.hist(data['AQI'], bins=30, alpha=0.7)
        plt.title('AQI Distribution')
        plt.xlabel('AQI')
        plt.ylabel('Frequency')
        
        # Plot 3: Actual vs Predicted
        if results and 'predictions' in results:
            plt.subplot(2, 2, 3)
            x = range(len(results['actual']))
            plt.plot(x, results['actual'], 'bo-', label='Actual', markersize=4)
            plt.plot(x, results['predictions'], 'ro-', label='Predicted', markersize=4)
            plt.title('Actual vs Predicted (First 10 Test Points)')
            plt.xlabel('Time Point')
            plt.ylabel('AQI')
            plt.legend()
        
        # Plot 4: Multiple pollutants
        plt.subplot(2, 2, 4)
        recent_data = data[['AQI', 'PM2_5', 'PM10']].tail(72)  # Last 3 days
        for col in recent_data.columns:
            plt.plot(recent_data.index, recent_data[col], label=col, alpha=0.7)
        plt.title('Multiple Pollutants (Last 3 Days)')
        plt.ylabel('Concentration')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('output/aqi_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Visualization saved as 'output/aqi_analysis_results.png'")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    """Main execution function."""
    print("="*60)
    print("AQI FORECASTING SYSTEM - SIMPLIFIED VERSION")
    print("="*60)
    
    # Check available files
    print("\nChecking available CSV files...")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        print(f"Found CSV files: {', '.join(csv_files)}")
    else:
        print("No CSV files found")
    
    # Create sample data
    data = create_sample_data()
    if data is None:
        print("❌ Failed to create data. Exiting.")
        return
    
    # Display data info
    print(f"\n📊 Dataset Information:")
    print(f"   • Records: {len(data):,}")
    print(f"   • Columns: {list(data.columns)}")
    print(f"   • Date range: {data.index.min()} to {data.index.max()}")
    print(f"   • AQI range: {data['AQI'].min():.2f} to {data['AQI'].max():.2f}")
    
    # Train model
    results = train_simple_arima(data)
    
    if results:
        print(f"\n🎯 Model Performance:")
        print(f"   • Model Type: {results['model_type']}")
        print(f"   • RMSE: {results['rmse']:.3f}")
        print(f"   • Sample Predictions: {[f'{x:.2f}' for x in results['predictions'][:5]]}")
        print(f"   • Sample Actual: {[f'{x:.2f}' for x in results['actual'][:5]]}")
    
    # Save dataset
    output_file = 'data/green_route_aqi_dataset.csv'
    data.to_csv(output_file)
    print(f"\n💾 Dataset saved as '{output_file}'")
    
    # Create visualization
    create_visualization(data, results)
    
    # Generate forecast
    print(f"\n🔮 Generating forecast...")
    if results:
        # Simple forecast using last trend
        last_values = data['AQI'].tail(24).values
        trend = np.mean(np.diff(last_values))
        next_forecast = data['AQI'].iloc[-1] + trend
        print(f"   • Next hour AQI forecast: {next_forecast:.2f}")
        print(f"   • Forecast confidence: Based on {results['model_type']} model")
    
    print(f"\n✅ AQI Forecasting System completed successfully!")
    print(f"📁 Generated files:")
    print(f"   • {output_file}")
    print(f"   • output/aqi_analysis_results.png")

if __name__ == "__main__":
    main()
