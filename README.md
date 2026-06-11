# Stock Price Predictor

A machine learning model that predicts stock prices based on historical data using linear regression and advanced techniques.

##  Overview

This project implements a stock price prediction system using historical stock data. The model uses linear regression as a baseline and can be extended with more advanced techniques like LSTM, Random Forest, or XGBoost.

##  Features

- Historical stock data analysis
- Linear regression-based price prediction
- Data visualization (price trends, predictions)
- Model performance evaluation
- Support for multiple advanced ML techniques

##  Requirements

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
yfinance
```

##  Installation

1. Clone the repository:
```bash
git clone https://github.com/verma-sankalp/stock-price-predictor.git
cd stock-price-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage with Linear Regression

```python
from stock_predictor import StockPricePredictor

# Initialize the predictor
predictor = StockPricePredictor()

# Fetch historical data for a stock (e.g., AAPL)
predictor.fetch_stock_data('AAPL', period='1y')

# Train the model
predictor.train_model()

# Predict future prices
predictions = predictor.predict_days(days=30)

# Visualize results
predictor.plot_predictions()
```

### Advanced Usage with LSTM

```python
# Initialize with LSTM model
predictor = StockPricePredictor(model_type='lstm')

# Fetch and prepare data
predictor.fetch_stock_data('GOOGL', period='2y')
predictor.prepare_lstm_data()

# Train LSTM model
predictor.train_lstm_model(epochs=100, batch_size=32)

# Predict and visualize
predictions = predictor.predict_lstm_days(days=30)
predictor.plot_lstm_predictions()
```

##  Project Structure

stock-price-predictor/
├── README.md
├── requirements.txt
├── stock_predictor.py
├── data/
│ └── .gitkeep
├── models/
│ └── .gitkeep
├── notebooks/
│ └── exploration.ipynb
└── tests/
└── test_stock_predictor.py


##  Implementation

See `stock_predictor.py` for the complete implementation of the StockPricePredictor class.

##  Model Evaluation

The model evaluates performance using:
- **Mean Squared Error (MSE)**: Lower values indicate better predictions
- **R² Score**: Closer to 1.0 indicates better fit
- **Visual Comparison**: Plot historical vs predicted prices

##  Testing

```bash
python tests/test_stock_predictor.py
```

##  Notes

- This model uses historical patterns and should not be used for actual trading decisions
- Stock markets are influenced by many unpredictable factors
- Always consult with financial professionals before making investment decisions
- The model can be improved with more features (news sentiment, economic indicators, etc.)

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


##  Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [TensorFlow](https://www.tensorflow.org/) for LSTM implementation

##  Contact

For questions or support, please open an issue in the repository.
or you can directly contact me on Email - sankalpverma2111@gmail.com
