# Green Route - Air Quality Aware Navigation System

🌱 **A real-time air quality-aware navigation system that provides optimal routing based on AQI forecasting**

## 📋 Overview

Green Route is an intelligent navigation system that incorporates air quality data to suggest the healthiest routes for travel. The system uses advanced time series forecasting models (ARIMA, LSTM, TCN) to predict air quality indices and optimize routing decisions.

## ✨ Features

- **Real-time AQI Data Collection**: Integration with OpenWeatherMap API
- **Advanced Forecasting Models**: ARIMA, LSTM, and Temporal Convolutional Networks
- **Synthetic Data Generation**: Fallback data creation for testing and development
- **Geographic-Aware Routing**: Location-based air quality analysis
- **Comprehensive Visualization**: Charts and graphs for data analysis
- **Multi-pollutant Tracking**: PM2.5, PM10, NO2, and Ozone monitoring

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package installer)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/green-route-aqi.git
cd green-route-aqi
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the simplified system:
```bash
python test_aqi_system.py
```

## 🔧 Usage

### Basic Usage

Run the simplified AQI forecasting system:

```python
from test_aqi_system import main
main()
```

### Advanced Usage

Use the full production system with all forecasting models:

```python
from green_route_aqi_forecasting import AQIForecastingSystem

# Initialize the system
system = AQIForecastingSystem()

# Load or generate data
data = system.load_or_generate_data()

# Train models
arima_results = system.train_arima_model(data)
lstm_results = system.train_lstm_model(data)
tcn_results = system.train_tcn_model(data)

# Generate forecasts
forecast = system.generate_forecast(data, hours_ahead=24)
```

## 📊 Data Sources

The system supports multiple data sources:

1. **OpenWeatherMap API**: Real-time and historical air quality data
2. **California Housing Dataset**: Adapted for AQI simulation
3. **Synthetic Generation**: Algorithm-generated realistic AQI patterns
4. **Custom CSV Files**: User-provided datasets

## 🤖 Models

### ARIMA (AutoRegressive Integrated Moving Average)
- Best for: Linear trends and seasonal patterns
- Parameters: Automatically optimized (p,d,q)
- Use case: Short-term forecasting with clear patterns

### LSTM (Long Short-Term Memory)
- Best for: Complex temporal dependencies
- Architecture: Configurable hidden layers and dropout
- Use case: Medium to long-term forecasting

### TCN (Temporal Convolutional Network)
- Best for: Parallel processing and long sequences
- Architecture: Dilated convolutions with residual connections
- Use case: Real-time applications requiring fast inference

## 📈 Sample Results

```
🎉 AQI FORECASTING SYSTEM RESULTS
==================================================

📊 Dataset Summary:
   • Total records: 17,000
   • Date range: 2024-01-01 00:00:00 to 2025-12-09 07:00:00
   • Geographic coverage: California region

🌬️ Air Quality Metrics:
   • AQI range: 1.00 - 5.00
   • Average AQI: 2.59 (Good air quality)
   • Standard deviation: 1.01

💨 Pollutant Levels:
   • PM2.5: 25.90 ± 10.32 µg/m³
   • PM10: 38.81 ± 15.40 µg/m³
   • NO2: 12.95 ± 5.16 µg/m³
   • Ozone: 20.71 ± 8.31 µg/m³
```

## 🔮 Future Enhancements

- [ ] Real-time GPS integration
- [ ] Mobile app development
- [ ] Multi-city support
- [ ] Weather pattern correlation
- [ ] Route optimization algorithms
- [ ] API endpoint development
- [ ] Machine learning model ensemble

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**H M Mahmudu Hasan**
- GitHub: [@oblivion97](https://github.com/your-username)
- Email: mahmudul.uiu041@gmail.com

## 🙏 Acknowledgments

- OpenWeatherMap for providing air quality APIs
- California Housing dataset contributors
- TensorFlow and Scikit-learn communities
- All contributors and testers

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/your-username/green-route-aqi/issues) page
2. Create a new issue if needed
3. Contact the maintainer

---
⭐ **Star this repository if you find it helpful!** ⭐
