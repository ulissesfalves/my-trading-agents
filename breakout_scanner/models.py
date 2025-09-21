import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class BreakoutPredictor:
    """
    Handles training, prediction, and explanation of the breakout model.
    """
    def __init__(self, model_path='lightgbm_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def _prepare_features(self, df):
        """
        Engineers features from historical price data.
        """
        df['price_range_5d'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / df['Close']
        df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        df['volume_ratio_20d'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['rsi_14d'] = self._calculate_rsi(df['Close'], 14)
        df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        
        # Target variable: Did the price increase by 5% within the next 5 days?
        future_close = df['Close'].shift(-5)
        df['target'] = ((future_close - df['Close']) / df['Close']) > 0.05
        
        return df.dropna()

    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train(self, tickers, start_date, end_date):
        """
        Trains the LightGBM model on historical data for a list of tickers.
        """
        all_features = []
        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    features = self._prepare_features(data.copy())
                    all_features.append(features)
            except Exception as e:
                print(f"Could not download or process data for {ticker}: {e}")

        if not all_features:
            print("No data available for training.")
            return

        training_data = pd.concat(all_features)
        
        features = ['price_range_5d', 'volatility_20d', 'volume_ratio_20d', 'rsi_14d', 'macd']
        X = training_data[features]
        y = training_data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = lgb.LGBMClassifier(objective='binary', random_state=42)
        self.model.fit(X_train, y_train)

        print("Model evaluation on test set:")
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

        self.save_model()

    def predict(self, tickers, explain=False):
        """
        Predicts breakout probability for a list of candidate tickers.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train a model first.")

        predictions = []
        for ticker in tickers:
            try:
                data = yf.download(ticker, period='3mo', progress=False) # Get enough data for features
                if not data.empty:
                    features_df = self._prepare_features(data.copy()).tail(1)
                    if not features_df.empty:
                        feature_cols = ['price_range_5d', 'volatility_20d', 'volume_ratio_20d', 'rsi_14d', 'macd']
                        X_pred = features_df[feature_cols]
                        
                        probability = self.model.predict_proba(X_pred)[:, 1][0]
                        predictions.append({
                            'ticker': ticker,
                            'breakout_probability': probability,
                            'date': features_df.index[0].strftime('%Y-%m-%d')
                        })

                        if explain:
                            self.explain(X_pred, ticker)
            except Exception as e:
                print(f"Could not predict for {ticker}: {e}")
        
        return pd.DataFrame(predictions)

    def explain(self, X_instance, ticker):
        """
        Generates and saves a SHAP force plot for a single prediction.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_instance)
        
        # Use the correct index for binary classification shap values
        shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure()
        shap.summary_plot(shap_values_for_plot, X_instance, show=False)
        plt.title(f"SHAP Feature Importance for {ticker}")
        plt.tight_layout()
        plt.savefig(f'shap_outputs/{ticker}_shap_summary.png')
        plt.close()


    def save_model(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.model = None
