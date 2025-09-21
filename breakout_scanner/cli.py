import argparse
import pandas as pd
import os
from datetime import datetime

from .scanner import BreakoutScanner
from .models import BreakoutPredictor

def main():
    """
    Main function to run the CLI for breakout scanning, training, and prediction.
    """
    parser = argparse.ArgumentParser(
        description="Scan for stock breakouts, train a model, and predict breakout success."
    )
    parser.add_argument(
        '--market', type=str, default='sp500',
        help="The market to scan (e.g., 'sp500')."
    )
    parser.add_argument(
        '--train', action='store_true',
        help="Train the LightGBM model on historical data."
    )
    parser.add_argument(
        '--predict', action='store_true',
        help="Scan for new breakouts and predict their success."
    )
    parser.add_argument(
        '--explain', action='store_true', default=True,
        help="Generate SHAP plots for predictions."
    )
    parser.add_argument(
        '--start-date', type=str, default='2020-01-01',
        help="Start date for training data (YYYY-MM-DD)."
    )
    parser.add_argument(
        '--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
        help="End date for training data (YYYY-MM-DD)."
    )
    parser.add_argument(
        '--output-file', type=str, default='breakout_candidates.csv',
        help="Path to save the CSV output of candidates."
    )
    parser.add_argument(
        '--model-path', type=str, default='lightgbm_model.pkl',
        help="Path to save/load the trained model."
    )

    args = parser.parse_args()

    # Ensure output directories exist
    if not os.path.exists('shap_outputs'):
        os.makedirs('shap_outputs')

    scanner = BreakoutScanner(market=args.market)
    predictor = BreakoutPredictor(model_path=args.model_path)

    if args.train:
        print("Fetching historical data for training...")
        tickers = scanner.get_tickers()
        print(f"Found {len(tickers)} tickers for market '{args.market}'.")
        
        print("Training model...")
        predictor.train(
            tickers=tickers,
            start_date=args.start-date,
            end_date=args.end-date
        )
        print(f"Model trained and saved to {args.model_path}")

    if args.predict:
        if not os.path.exists(args.model_path):
            print("Error: Model file not found. Please train the model first using the --train flag.")
            return

        print("Scanning for breakout candidates...")
        candidates = scanner.scan()
        print(f"Found {len(candidates)} potential breakout candidates.")

        if not candidates:
            print("No breakout candidates found today.")
            return

        print("Predicting success probability for candidates...")
        predictions_df = predictor.predict(
            tickers=candidates,
            explain=args.explain
        )

        if not predictions_df.empty:
            # Sort by prediction score descending
            predictions_df = predictions_df.sort_values(by='breakout_probability', ascending=False)
            predictions_df.to_csv(args.output_file, index=False)
            print(f"Predictions saved to {args.output_file}")
            if args.explain:
                print("SHAP explanation plots saved to 'shap_outputs/' directory.")
        else:
            print("Could not generate predictions for any candidates.")

if __name__ == '__main__':
    main()
