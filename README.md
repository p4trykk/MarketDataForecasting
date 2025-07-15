# Jane Street Real-Time Market Data Forecasting

This project was developed as part of the [Jane Street Kaggle competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting), which challenges participants to forecast the profitability of trading decisions based on high-frequency market data. The competition reflects the real-world complexity of financial markets, including non-stationary time series, fat-tailed distributions, and frequent structural shifts.

## Project Overview

The goal was to predict a numeric responder (`responder_6`) indicating the profitability of a financial instrument at a given time, using anonymized historical market data. The dataset includes 79 features describing market behavior across multiple instruments and time periods.

Our model uses an **LSTM (Long Short-Term Memory)** neural network to capture temporal dependencies in market dynamics and forecast short-term profitability.

## Technologies and Tools

- Python
- TensorFlow / Keras
- Dask (for efficient data loading and filtering)
- NumPy, Pandas, Matplotlib
- Scikit-learn

## Data

- Source: [`/kaggle/input/jane-street-real-time-market-data-forecasting`](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- Format: `train.parquet`, `features.csv`
- Size: Large (batch processing used to manage memory)
- Preprocessing:
  - Filtering by `symbol_id` and `date_id`
  - Handling missing values (replaced with zero)
  - Sliding window generation (`TIME_STEPS=15`) for sequence modeling
  - Train/test split (80/20)

## Exploratory Analysis

- Correlation analysis across responder variables showed that `responder_6` is moderately correlated with other targets, especially `responder_3` and `responder_4`, suggesting some shared predictive patterns.
- No significant missing data after preprocessing.
- Visualization of missing values was attempted but not necessary post-cleaning.

## Model Architecture

```python
model = Sequential([
    LSTM(64, input_shape=(TIME_STEPS, 79), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])
```
- Input: sequences of shape (15, 79)

- LSTM Layer: captures time dependencies across market snapshots

- Dense Layers: project temporal embeddings to final prediction

- Loss Function: MSE

- Optimizer: Adam

- Callbacks: EarlyStopping, ReduceLROnPlateau


## Additional Notebooks

[`data_analysis_preprocessing.ipynb`](https://github.com/p4trykk/MarketDataForecasting/blob/main/JSR_projekt_analizaDanych_PatrykKlytta.ipynb): Initial data exploration, filtering, and feature inspection.

[`lstm_model_training.ipynb`](https://github.com/p4trykk/MarketDataForecasting/blob/main/JSR_projekt_model_PatrykKlytta.ipynb): Model building, training, evaluation.

## key Insight
- Financial market data presents unique challenges uncommon in typical ML datasets.
- Even moderately performing models can extract useful signals from complex, noisy environments.
- Techniques like batch loading and windowing are essential for working with high-frequency financial data.

## License

This project is licensed under the [MIT License](https://www.mit.edu/~amini/LICENSE.md).

