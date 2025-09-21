# Stock Price Prediction with LSTM

## Overview
This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. The model uses time series analysis to forecast future stock movements.

## Features
- Real-time stock data fetching from Yahoo Finance
- LSTM-based deep learning model
- Technical indicators integration
- Future price prediction
- Performance visualization
- Model evaluation metrics

## Requirements
```
tensorflow>=2.13.0
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.1.87
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from stock_price_prediction import StockPredictor

# Initialize predictor
predictor = StockPredictor('AAPL', period='2y')

# Train model
predictor.fetch_data()
predictor.preprocess_data()
predictor.build_model()
predictor.train(epochs=50)

# Make predictions
predictions, actual = predictor.predict()
predictor.plot_results(predictions, actual)
```

### Advanced Usage
```python
# Custom parameters
predictor = StockPredictor('TSLA', period='5y')
predictor.preprocess_data(lookback=120)
predictor.build_model(lstm_units=100)
predictor.train(epochs=100, batch_size=64)

# Future predictions
future_prices = predictor.predict_future(days=60)
```

## Model Architecture
- **Input Layer**: 60-day lookback window
- **LSTM Layers**: 3 layers with 50 units each
- **Dropout**: 0.2 for regularization
- **Output Layer**: Single neuron for price prediction

## Data Processing
1. **Data Collection**: Yahoo Finance API
2. **Normalization**: MinMax scaling (0-1 range)
3. **Sequence Creation**: 60-day sliding windows
4. **Train/Test Split**: 80/20 ratio

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Visual comparison plots

## File Structure
```
stock_price_prediction/
├── stock_price_prediction.py
├── requirements.txt
├── README.md
├── models/
│   └── saved_models/
├── data/
│   └── processed/
└── results/
    └── plots/
```

## Performance Tips
1. **Lookback Window**: Experiment with 30-120 days
2. **LSTM Units**: Start with 50, increase for complex patterns
3. **Epochs**: Monitor validation loss to prevent overfitting
4. **Features**: Add volume, moving averages for better accuracy

## Limitations
- Past performance doesn't guarantee future results
- Market volatility can affect predictions
- External factors (news, events) not considered
- Requires sufficient historical data

## Real-World Applications
- **Trading Strategy**: Automated buy/sell signals
- **Risk Management**: Portfolio optimization
- **Market Analysis**: Trend identification
- **Investment Research**: Due diligence support

## Future Enhancements
- Multi-stock prediction
- Sentiment analysis integration
- Technical indicators (RSI, MACD)
- Real-time prediction API
- Web dashboard interface

## License
MIT License

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Disclaimer
This tool is for educational purposes only. Do not use for actual trading without proper risk assessment and professional advice.
