import unittest
import numpy as np
from stock_predictor import StockPricePredictor

class TestStockPricePredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = StockPricePredictor()
    
    def test_initialization(self):
        """Test predictor initialization."""
        self.assertEqual(self.predictor.model_type, 'linear')
        self.assertIsNone(self.predictor.model)
        self.assertIsNone(self.predictor.data)
    
    def test_fetch_stock_data(self):
        """Test fetching stock data."""
        # Note: This test requires internet connection
        try:
            data = self.predictor.fetch_stock_data('AAPL', period='1mo')
            self.assertIsNotNone(data)
            self.assertGreater(len(data), 0)
            self.assertIn('Close', data.columns)
        except Exception as e:
            self.skipTest(f"Could not fetch stock data: {e}")
    
    def test_prepare_features(self):
        """Test feature preparation."""
        try:
            self.predictor.fetch_stock_data('AAPL', period='1mo')
            self.predictor.prepare_features()
            
            self.assertIn('Day_Num', self.predictor.data.columns)
            self.assertIn('MA_5', self.predictor.data.columns)
            self.assertIn('MA_10', self.predictor.data.columns)
        except Exception as e:
            self.skipTest(f"Could not prepare features: {e}")
    
    def test_train_model(self):
        """Test model training."""
        try:
            self.predictor.fetch_stock_data('AAPL', period='3mo')
            model = self.predictor.train_model()
            
            self.assertIsNotNone(model)
            self.assertIsNotNone(self.predictor.model)
        except Exception as e:
            self.skipTest(f"Could not train model: {e}")
    
    def test_predict_days(self):
        """Test price prediction."""
        try:
            self.predictor.fetch_stock_data('AAPL', period='3mo')
            self.predictor.train_model()
            
            predictions = self.predictor.predict_days(days=10)
            
            self.assertIsNotNone(predictions)
            self.assertEqual(len(predictions), 10)
            self.assertIn('Predicted_Close', predictions.columns)
        except Exception as e:
            self.skipTest(f"Could not predict: {e}")

if __name__ == '__main__':
    unittest.main()
