import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

class StockPricePredictor:
    """
    Stock Price Predictor using Linear Regression and advanced ML techniques.
    """
    
    def __init__(self, model_type='linear'):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'linear' for Linear Regression, 'lstm' for LSTM, 
                       'random_forest' for Random Forest
        """
        self.model_type = model_type
        self.model = None
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = ['Open', 'High', 'Low', 'Volume']
        
    def fetch_stock_data(self, ticker, period='1y'):
        """
        Fetch historical stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1mo', '3mo', '1y', '5y')
        """
        stock = yf.Ticker(ticker)
        self.data = stock.history(period=period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Add date column
        self.data.reset_index(inplace=True)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        print(f"Successfully fetched {len(self.data)} records for {ticker}")
        return self.data
    
    def prepare_features(self):
        """Prepare features for modeling."""
        if self.data is None:
            raise ValueError("No data loaded. Fetch stock data first.")
        
        # Create time-based feature
        self.data['Day_Num'] = range(len(self.data))
        
        # Add technical indicators
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['Vol_Change'] = self.data['Volume'].pct_change()
        
        # Drop rows with NaN values
        self.data = self.data.dropna()
        
        return self.data
    
    def train_model(self):
        """Train the prediction model."""
        if self.data is None:
            raise ValueError("No data loaded. Fetch stock data first.")
        
        self.prepare_features()
        
        # Features and target
        X = self.data[['Day_Num', 'MA_5', 'MA_10', 'Vol_Change']].values
        y = self.data['Close'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if self.model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Linear Regression Results:")
            print(f"  MSE: {mse:.2f}")
            print(f"  R² Score: {r2:.2f}")
            
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Random Forest Results:")
            print(f"  MSE: {mse:.2f}")
            print(f"  R² Score: {r2:.2f}")
        
        return self.model
    
    def predict_days(self, days=30):
        """
        Predict stock prices for future days.
        
        Args:
            days: Number of days to predict
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Train first.")
        
        last_day = self.data['Day_Num'].max()
        last_ma5 = self.data['MA_5'].iloc[-1]
        last_ma10 = self.data['MA_10'].iloc[-1]
        last_vol = self.data['Vol_Change'].iloc[-1]
        
        predictions = []
        future_days = range(last_day + 1, last_day + days + 1)
        
        for day in future_days:
            # Simple assumption: features remain similar
            X_new = np.array([[day, last_ma5, last_ma10, last_vol]])
            pred = self.model.predict(X_new)[0]
            predictions.append(pred)
            
            # Update moving averages (simplified)
            last_ma5 = (last_ma5 * 4 + pred) / 5
            last_ma10 = (last_ma10 * 9 + pred) / 10
        
        prediction_df = pd.DataFrame({
            'Day': future_days,
            'Predicted_Close': predictions
        })
        
        return prediction_df
    
    def plot_predictions(self, days=30):
        """Plot historical data with predictions."""
        predictions = self.predict_days(days)
        
        plt.figure(figsize=(14, 7))
        
        # Historical data
        plt.plot(self.data['Date'], self.data['Close'], 
                label='Historical Price', color='blue', linewidth=2)
        
        # Predictions
        future_dates = pd.date_range(
            start=self.data['Date'].max() + pd.Day(1), 
            periods=days, 
            freq='D'
        )
        plt.plot(future_dates, predictions['Predicted_Close'], 
                label=f'Predicted ({days} days)', color='red', linewidth=2, linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Stock Price Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('stock_prediction_plot.png', dpi=300)
        plt.show()
        
        return predictions
    
    def train_lstm_model(self, epochs=100, batch_size=32):
        """Train LSTM model for sequence prediction."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        if self.data is None:
            raise ValueError("No data loaded.")
        
        self.prepare_features()
        
        # Prepare sequence data
        close_prices = self.data['Close'].values
        scaled_prices = self.scaler.fit_transform(close_prices.reshape(-1, 1))
        
        # Create sequences
        sequence_length = 60
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Train model
        print("Training LSTM model...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        
        self.model = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        print(f"LSTM Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R² Score: {r2:.2f}")
        
        return model
    
    def prepare_lstm_data(self):
        """Prepare data for LSTM modeling."""
        self.prepare_features()
        return self.data
    
    def predict_lstm_days(self, days=30):
        """Predict using LSTM model."""
        if self.model is None:
            raise ValueError("LSTM model not trained.")
        
        close_prices = self.data['Close'].values
        scaled_prices = self.scaler.fit_transform(close_prices.reshape(-1, 1))
        
        sequence_length = 60
        last_sequence = scaled_prices[-sequence_length:]
        
        predictions = []
        
        for _ in range(days):
            X_pred = np.reshape(last_sequence[-sequence_length:], (1, sequence_length, 1))
            pred = self.model.predict(X_pred)[0, 0]
            pred_inverse = self.scaler.inverse_transform([[pred]])[0, 0]
            predictions.append(pred_inverse)
            
            # Update sequence
            last_sequence = np.append(last_sequence, [[pred]], axis=0)
        
        prediction_df = pd.DataFrame({
            'Day': range(1, days + 1),
            'Predicted_Close': predictions
        })
        
        return prediction_df
    
    def plot_lstm_predictions(self, days=30):
        """Plot LSTM predictions."""
        predictions = self.predict_lstm_days(days)
        
        plt.figure(figsize=(14, 7))
        
        plt.plot(self.data['Date'], self.data['Close'], 
                label='Historical Price', color='blue', linewidth=2)
        
        future_dates = pd.date_range(
            start=self.data['Date'].max() + pd.Day(1), 
            periods=days, 
            freq='D'
        )
        plt.plot(future_dates, predictions['Predicted_Close'], 
                label=f'LSTM Predicted ({days} days)', color='green', linewidth=2, linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Stock Price Prediction - LSTM Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('lstm_stock_prediction.png', dpi=300)
        plt.show()
        
        return predictions


if __name__ == "__main__":
    # Example usage
    predictor = StockPricePredictor(model_type='linear')
    
    # Fetch data for Apple
    predictor.fetch_stock_data('AAPL', period='2y')
    
    # Train model
    predictor.train_model()
    
    # Predict and plot
    predictor.plot_predictions(days=30)
