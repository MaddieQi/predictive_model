# Prophet Time Series Forecasting

This project implements an optimized Prophet-based forecasting system for sales and spend data. It provides a modular, object-oriented approach to time series forecasting with comprehensive feature engineering and evaluation capabilities.

## Features

- Advanced date feature generation including holidays and seasonal patterns
- Automated feature engineering for time series data
- Configurable Prophet model implementation
- Performance evaluation with MAPE (Mean Absolute Percentage Error)
- Visualization of forecasting results
- Support for both sales and spend forecasting

## Prerequisites

```python
pandas>=1.3.0
numpy>=1.20.0
prophet>=1.0
holidays>=0.14
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

## Test Data Format
### Data Fields

date: Date in MM/D/YYYY format
product: Product category identifier (BE, C, G, M, O)
lir_imp: Linear impressions
clicks: Number of clicks
spd: Daily spend value
order7d: 7-day rolling orders
sales7d: 7-day rolling sales value

## Key Components

### 1. ModelConfig
Configuration class that holds model parameters:
- List of regressors
- Forecast date range
- Test size for evaluation

### 2. DateFeatureGenerator
Handles all date-related feature generation:
- Promotional dates (Black Friday, Cyber Monday, Prime Days)
- Seasonal information
- Day of week encoding
- Holiday detection

### 3. TimeSeriesPreprocessor
Manages data preprocessing:
- Feature engineering
- Rolling statistics
- Date feature extraction
- Missing value handling

### 4. ProphetForecaster
Handles Prophet model operations:
- Model training
- Forecasting
- Performance evaluation
- Future data generation

## Usage Example

```python
from prophet_forecast import *

# Initialize components
date_generator = DateFeatureGenerator()
preprocessor = TimeSeriesPreprocessor(date_generator)

# Configure model
config = ModelConfig(
    regressors=['past_7_days_spd', 'day_of_week', 'year', 'month', 'day',
                'season', 'is_holiday', 'if_limit', 'spd_same_day_last_year'],
    forecast_start='2025-01-06',
    forecast_end='2025-01-19'
)

# Load and process data
df = pd.read_csv('timeseries.csv')
df_all = df.groupby(['date']).sum().reset_index()
processed_data = preprocessor.prepare_features(df_all, 'spd')

# Initialize forecaster
forecaster = ProphetForecaster(config)
prophet_data = forecaster.prepare_prophet_data(processed_data, 'spd')

# Train and evaluate
mape, _ = forecaster.train_and_evaluate(prophet_data)
forecast = forecaster.forecast(prophet_data)

print(f'Forecast MAPE: {mape:.2%}')
```

## Features Generated

The system generates the following features for each time series:

1. **Rolling Statistics**
   - Past 7 days total
   - Same day last year value

2. **Date Features**
   - Day of week (1-7)
   - Year
   - Month
   - Day
   - Season (1-4)

3. **Holiday Features**
   - Promotional events (Black Friday, Cyber Monday, Prime Days)
   - Federal holidays

## Model Evaluation
The system evaluates model performance using:

    - MAPE (Mean Absolute Percentage Error)
    - Train/test split with configurable test size
    - Visual inspection through plots

## Customization
You can customize the forecasting system by:

    - Modifying the ModelConfig parameters
    - Adding new features in TimeSeriesPreprocessor
    - Extending DateFeatureGenerator for additional date features
    - Adjusting Prophet parameters in ProphetForecaster
