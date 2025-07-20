import pandas as pd
import os

# Load and analyze the generated dataset
df = pd.read_csv('green_route_aqi_dataset.csv')

print("🎉 AQI FORECASTING SYSTEM RESULTS")
print("=" * 50)

print(f"\n📊 Dataset Summary:")
print(f"   • Total records: {len(df):,}")
print(f"   • Columns: {list(df.columns)}")
print(f"   • Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

print(f"\n🌬️ Air Quality Metrics:")
aqi_stats = df['AQI'].describe()
print(f"   • AQI range: {aqi_stats['min']:.2f} - {aqi_stats['max']:.2f}")
print(f"   • Average AQI: {aqi_stats['mean']:.2f}")
print(f"   • Standard deviation: {aqi_stats['std']:.2f}")

print(f"\n💨 Pollutant Levels:")
for col in ['PM2_5', 'PM10', 'NO2', 'Ozone']:
    if col in df.columns:
        stats = df[col].describe()
        print(f"   • {col}: {stats['mean']:.2f} ± {stats['std']:.2f} µg/m³")

print(f"\n🗺️ Geographic Coverage:")
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    print(f"   • Latitude range: {df['Latitude'].min():.2f}° to {df['Latitude'].max():.2f}°")
    print(f"   • Longitude range: {df['Longitude'].min():.2f}° to {df['Longitude'].max():.2f}°")

print(f"\n📈 Data Source:")
if 'DataSource' in df.columns:
    sources = df['DataSource'].value_counts()
    for source, count in sources.items():
        print(f"   • {source}: {count:,} records")

print(f"\n📁 Generated Files:")
files = ['green_route_aqi_dataset.csv', 'aqi_analysis_results.png']
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024  # KB
        print(f"   • {file}: {size:.1f} KB")

print(f"\n✅ System Status: COMPLETED SUCCESSFULLY!")
print(f"   • AQI forecasting models can now be applied")
print(f"   • Dataset ready for time series analysis")
print(f"   • Visualization saved for review")
