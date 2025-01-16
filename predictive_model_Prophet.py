import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import holidays
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for Prophet model parameters"""
    regressors: List[str]
    forecast_start: str
    forecast_end: str
    test_size: int = 30

class DateFeatureGenerator:
    """Class to handle date-related feature generation"""
    def __init__(self):
        self.us_holidays = holidays.US()
        
    @staticmethod
    def get_nth_weekday(year: int, month: int, nth: int, weekday: int) -> datetime:
        """Get nth occurrence of weekday in given month/year"""
        first_day = datetime(year, month, 1)
        days_to_weekday = (weekday - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=days_to_weekday)
        return first_occurrence + timedelta(weeks=(nth - 1))
    
    def get_black_friday(self, year):
        """Get Black Friday date for a given year"""
        thanksgiving = self.get_nth_weekday(year, 11, 4, 3)  # 4th Thursday
        return thanksgiving + timedelta(days=1)
    
    def get_cyber_monday(self, year):
        """Get Cyber Monday date for a given year"""
        return self.get_black_friday(year) + timedelta(days=3)
    
    def get_promotional_dates(self, date: Union[str, datetime]) -> List[datetime]:
        """Get Black Friday, Cyber Monday and Prime Day dates"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        dates = []
        end_year = date.year
        
        # Black Friday & Cyber Monday dates
        for year in range(2022, end_year + 1):
            dates.append(self.get_black_friday(year))
            dates.append(self.get_cyber_monday(year))
        
        # Prime Day dates (2023-2024)
        prime_days = [
            (2023, 7, 2), (2023, 10, 2),
            (2024, 7, 3), (2024, 10, 2)
        ]
        
        for year, month, week in prime_days:
            dates.extend([
                self.get_nth_weekday(year, month, week, 1),
                self.get_nth_weekday(year, month, week, 2)
            ])
            
        return dates
    
    @staticmethod
    def get_season(date: datetime) -> int:
        """Get season number (1-4) for given date"""
        month = date.month
        season_mapping = {
            (12, 1, 2): 4,  # Winter
            (3, 4, 5): 1,   # Spring
            (6, 7, 8): 2,   # Summer
            (9, 10, 11): 3  # Fall
        }
        return next(value for months, value in season_mapping.items() if month in months)
    
    @staticmethod
    def get_weekday(date: Union[str, datetime]) -> int:
        """Get weekday number (1-7) for given date"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        weekday_mapping = {
            'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
            'Friday': 5, 'Saturday': 6, 'Sunday': 7
        }
        return weekday_mapping[date.strftime('%A')]

class TimeSeriesPreprocessor:
    """Class to handle time series data preprocessing"""
    def __init__(self, date_generator: DateFeatureGenerator):
        self.date_generator = date_generator
        
    def prepare_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare features for Prophet model"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Add rolling features
        df[f'past_7_days_{target_col}'] = df[target_col].rolling(window=7).sum().shift(1)
        df[f'{target_col}_same_day_last_year'] = df[target_col].shift(365)
        
        # Add date features
        df['day_of_week'] = df['date'].apply(self.date_generator.get_weekday)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['season'] = df['date'].apply(self.date_generator.get_season)
        
        # Add holiday features
        df['is_holiday'] = df['date'].apply(lambda x: x in self.date_generator.get_promotional_dates(x))
        df['if_limit'] = df['date'].apply(lambda x: x in self.date_generator.us_holidays)
        
        return df.dropna()

class ProphetForecaster:
    """Class to handle Prophet model training and forecasting"""
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def _create_model(self) -> Prophet:
        """Create a new Prophet model instance with configured regressors"""
        model = Prophet()
        for regressor in self.config.regressors:
            model.add_regressor(regressor)
        return model
    
    def prepare_prophet_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare data in Prophet format"""
        prophet_df = df.rename(columns={'date': 'ds', target_col: 'y'})
        
        # Ensure all required regressors are present
        missing_regressors = set(self.config.regressors) - set(prophet_df.columns)
        if missing_regressors:
            raise ValueError(f"Missing required regressors: {missing_regressors}")
            
        return prophet_df
    
    def create_future_df(self, last_data: pd.DataFrame) -> pd.DataFrame:
        """Create future dataframe for forecasting"""
        future_dates = pd.date_range(
            start=self.config.forecast_start,
            end=self.config.forecast_end,
            freq='D'
        )
        future_df = pd.DataFrame(future_dates, columns=['ds'])
        
        # Copy last values for regressors
        for regressor in self.config.regressors:
            future_df[regressor] = last_data[regressor].iloc[-1]
            
        return future_df
    
    def train_and_evaluate(self, df: pd.DataFrame) -> tuple:
        """Train model and evaluate performance"""
        train_df = df[:-self.config.test_size]
        test_df = df[-self.config.test_size:]
        
        # Create new model instance for training
        model = self._create_model()
        model.fit(train_df)
        
        future_df = test_df.drop(columns=['y'])
        forecast = model.predict(future_df)
        
        mape = mean_absolute_percentage_error(test_df['y'], forecast['yhat'])
        return mape, forecast
    
    def forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate forecast"""
        model = self._create_model()
        model.fit(df)
        future_df = self.create_future_df(df)
        return model.predict(future_df)

def main():
    # Load and preprocess data
    print("Loading data from timeseries.csv...")
    df = pd.read_csv('timeseries.csv')
    
    # Convert date format and aggregate data
    df['date'] = pd.to_datetime(df['date'])
    df_all = df.groupby(['date']).agg({
        'imp': 'sum',
        'clicks': 'sum',
        'spd': 'sum',
        'order7d': 'sum',
        'sales7d': 'sum'
    }).reset_index()
    
    # Initialize components
    date_generator = DateFeatureGenerator()
    preprocessor = TimeSeriesPreprocessor(date_generator)
    
    # Configure models
    spd_config = ModelConfig(
        regressors=['past_7_days_spd', 'day_of_week', 'year', 'month', 'day',
                   'season', 'is_holiday', 'if_limit', 'spd_same_day_last_year'],
        forecast_start='2025-01-06',
        forecast_end='2025-01-19'
    )
    
    sales_config = ModelConfig(
        regressors=['past_7_days_sales7d', 'day_of_week', 'year', 'month', 'day',
                   'season', 'is_holiday', 'if_limit', 'sales7d_same_day_last_year'],
        forecast_start='2025-01-06',
        forecast_end='2025-01-19'
    )
    
    # Process and forecast spend data
    print("\nProcessing spend data...")
    spd_data = preprocessor.prepare_features(df_all, 'spd')
    spd_prophet_data = ProphetForecaster(spd_config).prepare_prophet_data(spd_data, 'spd')
    spd_forecaster = ProphetForecaster(spd_config)
    spd_mape, spd_test_forecast = spd_forecaster.train_and_evaluate(spd_prophet_data)
    spd_forecast = spd_forecaster.forecast(spd_prophet_data)
    
    # Process and forecast sales data
    print("\nProcessing sales data...")
    sales_data = preprocessor.prepare_features(df_all, 'sales7d')
    sales_prophet_data = ProphetForecaster(sales_config).prepare_prophet_data(sales_data, 'sales7d')
    sales_forecaster = ProphetForecaster(sales_config)
    sales_mape, sales_test_forecast = sales_forecaster.train_and_evaluate(sales_prophet_data)
    sales_forecast = sales_forecaster.forecast(sales_prophet_data)
    
    # Print results
    print(f'\nSpend Forecast MAPE: {spd_mape:.2%}')
    print(f'Sales Forecast MAPE: {sales_mape:.2%}')
    
    # Save forecasts
    spd_forecast[['ds', 'yhat']].to_csv('spd_forecast.csv', index=False)
    sales_forecast[['ds', 'yhat']].to_csv('sales_forecast.csv', index=False)
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(spd_prophet_data['ds'], spd_prophet_data['y'], label='Actual SPD')
    plt.plot(spd_forecast['ds'], spd_forecast['yhat'], label='Forecast SPD')
    plt.title('SPD Forecast')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(sales_prophet_data['ds'], sales_prophet_data['y'], label='Actual Sales')
    plt.plot(sales_forecast['ds'], sales_forecast['yhat'], label='Forecast Sales')
    plt.title('Sales Forecast')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('forecast_plots.png')
    plt.close()
    
    print("\nForecasts have been saved to 'spd_forecast.csv' and 'sales_forecast.csv'")
    print("Plots have been saved to 'forecast_plots.png'")
    
    return spd_forecast, sales_forecast, df_all

if __name__ == "__main__":
    spd_forecast, sales_forecast, df_all = main()
