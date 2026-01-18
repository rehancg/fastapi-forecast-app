import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# Time series forecasting libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    THETA_AVAILABLE = True
except ImportError:
    THETA_AVAILABLE = False
    ThetaModel = None
import pmdarima as pm
from prophet import Prophet
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(actual, predicted))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return float('inf')
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def naive_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Naive forecast: last value"""
    last_value = data.iloc[-1]
    forecast = np.full(forecast_steps, last_value)
    metadata = {
        "model": "Naive",
        "method": "Last value",
        "last_value": float(last_value)
    }
    return forecast, metadata


def seasonal_naive_forecast(data: pd.Series, forecast_steps: int, seasonality: int = None) -> Tuple[np.ndarray, Dict]:
    """Seasonal Naive forecast: last season's values"""
    if seasonality is None:
        # Auto-detect seasonality (try common periods)
        if len(data) >= 12:
            seasonality = 12
        elif len(data) >= 4:
            seasonality = 4
        else:
            seasonality = 1
    
    if len(data) < seasonality:
        seasonality = len(data)
    
    last_season = data.iloc[-seasonality:].values
    repetitions = (forecast_steps // seasonality) + 1
    forecast = np.tile(last_season, repetitions)[:forecast_steps]
    
    metadata = {
        "model": "Seasonal Naive",
        "seasonality": seasonality,
        "method": "Last season repetition"
    }
    return forecast, metadata


def moving_average_forecast(data: pd.Series, forecast_steps: int, window: int = None) -> Tuple[np.ndarray, Dict]:
    """Moving Average forecast"""
    if window is None:
        window = min(3, len(data) // 2)
    if window > len(data):
        window = len(data)
    
    ma_value = data.iloc[-window:].mean()
    forecast = np.full(forecast_steps, ma_value)
    
    metadata = {
        "model": "Moving Average",
        "window": window,
        "ma_value": float(ma_value)
    }
    return forecast, metadata


def ses_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Simple Exponential Smoothing"""
    try:
        model = SimpleExpSmoothing(data).fit(optimized=True)
        forecast = model.forecast(forecast_steps)
        metadata = {
            "model": "SES (Simple Exponential Smoothing)",
            "alpha": float(model.params['smoothing_level']),
            "aic": float(model.aic) if hasattr(model, 'aic') else None
        }
        return forecast.values, metadata
    except Exception as e:
        # Fallback to naive
        forecast, _ = naive_forecast(data, forecast_steps)
        metadata = {
            "model": "SES (Simple Exponential Smoothing)",
            "error": str(e),
            "fallback": "Naive"
        }
        return forecast, metadata


def holt_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Holt's linear trend method"""
    try:
        model = ExponentialSmoothing(data, trend='add', seasonal=None).fit(optimized=True)
        forecast = model.forecast(forecast_steps)
        metadata = {
            "model": "Holt",
            "alpha": float(model.params['smoothing_level']),
            "beta": float(model.params['smoothing_trend']),
            "aic": float(model.aic) if hasattr(model, 'aic') else None
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = naive_forecast(data, forecast_steps)
        metadata = {
            "model": "Holt",
            "error": str(e),
            "fallback": "Naive"
        }
        return forecast, metadata


def holt_winters_additive_forecast(data: pd.Series, forecast_steps: int, seasonality: int = None) -> Tuple[np.ndarray, Dict]:
    """Holt-Winters Additive"""
    if seasonality is None:
        if len(data) >= 12:
            seasonality = 12
        elif len(data) >= 4:
            seasonality = 4
        else:
            seasonality = 2
    
    if len(data) < seasonality * 2:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Holt-Winters (Additive)",
            "error": "Insufficient data for seasonality",
            "fallback": "Holt"
        }
        return forecast, metadata
    
    try:
        model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=seasonality).fit(optimized=True)
        forecast = model.forecast(forecast_steps)
        metadata = {
            "model": "Holt-Winters (Additive)",
            "seasonality": seasonality,
            "alpha": float(model.params['smoothing_level']),
            "beta": float(model.params['smoothing_trend']),
            "gamma": float(model.params['smoothing_seasonal']),
            "aic": float(model.aic) if hasattr(model, 'aic') else None
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Holt-Winters (Additive)",
            "error": str(e),
            "fallback": "Holt"
        }
        return forecast, metadata


def holt_winters_multiplicative_forecast(data: pd.Series, forecast_steps: int, seasonality: int = None) -> Tuple[np.ndarray, Dict]:
    """Holt-Winters Multiplicative"""
    if seasonality is None:
        if len(data) >= 12:
            seasonality = 12
        elif len(data) >= 4:
            seasonality = 4
        else:
            seasonality = 2
    
    if len(data) < seasonality * 2:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Holt-Winters (Multiplicative)",
            "error": "Insufficient data for seasonality",
            "fallback": "Holt"
        }
        return forecast, metadata
    
    try:
        model = ExponentialSmoothing(data, trend='add', seasonal='mul', seasonal_periods=seasonality).fit(optimized=True)
        forecast = model.forecast(forecast_steps)
        metadata = {
            "model": "Holt-Winters (Multiplicative)",
            "seasonality": seasonality,
            "alpha": float(model.params['smoothing_level']),
            "beta": float(model.params['smoothing_trend']),
            "gamma": float(model.params['smoothing_seasonal']),
            "aic": float(model.aic) if hasattr(model, 'aic') else None
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Holt-Winters (Multiplicative)",
            "error": str(e),
            "fallback": "Holt"
        }
        return forecast, metadata


def arima_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """ARIMA with AIC-based order selection - optimized for speed"""
    try:
        # Reduced search space for faster execution
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_model = None
        
        # Optimized: Try common orders first, then limited grid
        common_orders = [(1, 1, 1), (2, 1, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
        for order in common_orders:
            try:
                model = ARIMA(data, order=order).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = order
                    best_model = model
            except:
                continue
        
        # If common orders work, skip grid search for speed
        if best_model is not None and best_aic < np.inf:
            forecast = best_model.forecast(forecast_steps)
            metadata = {
                "model": "ARIMA",
                "order": best_order,
                "aic": float(best_aic)
            }
            return forecast.values, metadata
        
        # Limited grid search only if needed
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    if (p, d, q) in common_orders:
                        continue
                    try:
                        model = ARIMA(data, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                            best_model = model
                    except:
                        continue
        
        if best_model is None:
            raise Exception("No valid ARIMA model found")
        
        forecast = best_model.forecast(forecast_steps)
        metadata = {
            "model": "ARIMA",
            "order": best_order,
            "aic": float(best_aic)
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = naive_forecast(data, forecast_steps)
        metadata = {
            "model": "ARIMA",
            "error": str(e),
            "fallback": "Naive"
        }
        return forecast, metadata


def sarima_forecast(data: pd.Series, forecast_steps: int, seasonality: int = None) -> Tuple[np.ndarray, Dict]:
    """SARIMA with AIC-based order selection"""
    if seasonality is None:
        if len(data) >= 12:
            seasonality = 12
        elif len(data) >= 4:
            seasonality = 4
        else:
            seasonality = 2
    
    if len(data) < seasonality * 2:
        forecast, _ = arima_forecast(data, forecast_steps)
        metadata = {
            "model": "SARIMA",
            "error": "Insufficient data for seasonality",
            "fallback": "ARIMA"
        }
        return forecast, metadata
    
    try:
        # Optimized: Try common SARIMA orders first
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, seasonality)
        best_model = None
        
        # Common SARIMA orders - try these first
        common_orders = [
            ((1, 1, 1), (1, 1, 1)),
            ((1, 1, 0), (1, 1, 0)),
            ((0, 1, 1), (0, 1, 1)),
            ((1, 1, 1), (0, 1, 1)),
        ]
        
        for (p, d, q), (P, D, Q) in common_orders:
            try:
                model = SARIMAX(data, order=(p, d, q), 
                               seasonal_order=(P, D, Q, seasonality)).fit(disp=False, maxiter=50)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_seasonal_order = (P, D, Q, seasonality)
                    best_model = model
            except:
                continue
        
        # If common orders work, skip extensive grid search
        if best_model is not None and best_aic < np.inf:
            forecast = best_model.forecast(forecast_steps)
            metadata = {
                "model": "SARIMA",
                "order": best_order,
                "seasonal_order": best_seasonal_order,
                "aic": float(best_aic)
            }
            return forecast.values, metadata
        
        # Limited grid search only if needed (reduced space)
        for p in range(0, 2):
            for d in range(0, 2):
                for q in range(0, 2):
                    for P in range(0, 2):
                        for D in range(0, 2):
                            for Q in range(0, 2):
                                if ((p, d, q), (P, D, Q)) in common_orders:
                                    continue
                                try:
                                    model = SARIMAX(data, order=(p, d, q), 
                                                   seasonal_order=(P, D, Q, seasonality)).fit(disp=False, maxiter=50)
                                    if model.aic < best_aic:
                                        best_aic = model.aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = (P, D, Q, seasonality)
                                        best_model = model
                                except:
                                    continue
        
        if best_model is None:
            raise Exception("No valid SARIMA model found")
        
        forecast = best_model.forecast(forecast_steps)
        metadata = {
            "model": "SARIMA",
            "order": best_order,
            "seasonal_order": best_seasonal_order,
            "aic": float(best_aic)
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = arima_forecast(data, forecast_steps)
        metadata = {
            "model": "SARIMA",
            "error": str(e),
            "fallback": "ARIMA"
        }
        return forecast, metadata


def auto_arima_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Auto ARIMA using pmdarima - optimized for speed"""
    try:
        model = pm.auto_arima(data, seasonal=False, stepwise=True, 
                             suppress_warnings=True, error_action='ignore',
                             max_p=2, max_d=2, max_q=2,  # Reduced search space
                             maxiter=50,  # Limit iterations
                             n_jobs=1)  # Single thread for stability
        forecast = model.predict(forecast_steps)
        metadata = {
            "model": "AUTO ARIMA",
            "order": model.order,
            "aic": float(model.aic())
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = arima_forecast(data, forecast_steps)
        metadata = {
            "model": "AUTO ARIMA",
            "error": str(e),
            "fallback": "ARIMA"
        }
        return forecast, metadata


def ets_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """ETS(A,A,A) - Additive error, Additive trend, Additive seasonality"""
    try:
        # ETSModel with (A,A,A) specification
        if len(data) >= 12:
            seasonality = 12
        elif len(data) >= 4:
            seasonality = 4
        else:
            # No seasonality
            model = ETSModel(data, error='add', trend='add', seasonal=None).fit()
            forecast = model.forecast(forecast_steps)
            metadata = {
                "model": "ETS(A,A,A)",
                "error_type": "additive",
                "trend_type": "additive",
                "seasonal_type": None,
                "aic": float(model.aic) if hasattr(model, 'aic') else None
            }
            return forecast.values, metadata
        
        if len(data) < seasonality * 2:
            model = ETSModel(data, error='add', trend='add', seasonal=None).fit()
            forecast = model.forecast(forecast_steps)
            metadata = {
                "model": "ETS(A,A,A)",
                "error_type": "additive",
                "trend_type": "additive",
                "seasonal_type": None,
                "aic": float(model.aic) if hasattr(model, 'aic') else None
            }
            return forecast.values, metadata
        
        model = ETSModel(data, error='add', trend='add', seasonal='add', 
                        seasonal_periods=seasonality).fit()
        forecast = model.forecast(forecast_steps)
        metadata = {
            "model": "ETS(A,A,A)",
            "error_type": "additive",
            "trend_type": "additive",
            "seasonal_type": "additive",
            "seasonality": seasonality,
            "aic": float(model.aic) if hasattr(model, 'aic') else None
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "ETS(A,A,A)",
            "error": str(e),
            "fallback": "Holt"
        }
        return forecast, metadata


def theta_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Theta model (Holt Proxy)"""
    if not THETA_AVAILABLE:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Theta (Holt Proxy)",
            "error": "ThetaModel not available in this statsmodels version",
            "fallback": "Holt"
        }
        return forecast, metadata
    
    try:
        model = ThetaModel(data, period=None).fit()
        forecast = model.forecast(forecast_steps)
        metadata = {
            "model": "Theta (Holt Proxy)",
            "period": model.period if hasattr(model, 'period') else None,
            "theta": float(model.theta) if hasattr(model, 'theta') else None
        }
        return forecast.values, metadata
    except Exception as e:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Theta (Holt Proxy)",
            "error": str(e),
            "fallback": "Holt"
        }
        return forecast, metadata


def prophet_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Prophet forecast"""
    try:
        # Prophet requires DataFrame with 'ds' and 'y' columns
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='D'),
            'y': data.values
        })
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=forecast_steps)
        forecast_df = model.predict(future)
        
        forecast = forecast_df['yhat'].tail(forecast_steps).values
        
        metadata = {
            "model": "Prophet",
            "components": {
                "trend": True,
                "yearly_seasonality": True,
                "weekly_seasonality": True
            }
        }
        return forecast, metadata
    except Exception as e:
        forecast, _ = holt_forecast(data, forecast_steps)
        metadata = {
            "model": "Prophet",
            "error": str(e),
            "fallback": "Holt"
        }
        return forecast, metadata


def croston_forecast(data: pd.Series, forecast_steps: int) -> Tuple[np.ndarray, Dict]:
    """Croston's method for intermittent demand"""
    try:
        # Croston's method implementation
        # This is a simplified version
        non_zero = data[data > 0]
        if len(non_zero) == 0:
            forecast = np.zeros(forecast_steps)
            metadata = {
                "model": "Croston",
                "error": "No non-zero values",
                "forecast": "zero"
            }
            return forecast, metadata
        
        # Calculate demand intervals and sizes
        demand_indices = data[data > 0].index
        intervals = []
        for i in range(1, len(demand_indices)):
            intervals.append(demand_indices[i] - demand_indices[i-1])
        
        if len(intervals) == 0:
            avg_interval = len(data)
        else:
            avg_interval = np.mean(intervals)
        
        avg_demand = non_zero.mean()
        
        # Forecast
        forecast_value = avg_demand / max(avg_interval, 1) if avg_interval > 0 else 0
        forecast = np.full(forecast_steps, forecast_value)
        
        metadata = {
            "model": "Croston",
            "average_demand": float(avg_demand),
            "average_interval": float(avg_interval),
            "forecast_rate": float(forecast_value)
        }
        return forecast, metadata
    except Exception as e:
        forecast, _ = naive_forecast(data, forecast_steps)
        metadata = {
            "model": "Croston",
            "error": str(e),
            "fallback": "Naive"
        }
        return forecast, metadata


def _evaluate_single_model(model_name: str, forecast_func, data: pd.Series, train_data: pd.Series, 
                          test_data: pd.Series, forecast_steps: int, validation_mode: bool) -> Dict:
    """Evaluate a single model - used for parallel processing"""
    try:
        # Generate forecast for validation period
        if validation_mode and len(test_data) > 0:
            val_forecast, metadata = forecast_func(train_data, len(test_data))
            # Calculate metrics
            rmse = calculate_rmse(test_data.values, val_forecast)
            mape = calculate_mape(test_data.values, val_forecast)
        else:
            # No validation, set metrics to None
            metadata = {}
            rmse = None
            mape = None
        
        # Generate final forecast using all data
        final_forecast, final_metadata = forecast_func(data, forecast_steps)
        # Merge metadata
        if metadata:
            final_metadata.update(metadata)
        
        result = {
            "model_name": model_name,
            "metadata": final_metadata,
            "rmse": float(rmse) if rmse is not None and not np.isinf(rmse) else None,
            "mape": float(mape) if mape is not None and not np.isinf(mape) else None,
            "forecast": final_forecast.tolist() if isinstance(final_forecast, np.ndarray) else list(final_forecast)
        }
        return result
        
    except Exception as e:
        # If model fails completely, still add it with error
        return {
            "model_name": model_name,
            "metadata": {"error": str(e)},
            "rmse": None,
            "mape": None,
            "forecast": None
        }


def evaluate_models(data: pd.Series, forecast_steps: int = 12, models_to_run: List[str] = None, 
                   skip_validation: bool = False, parallel: bool = True) -> List[Dict]:
    """Evaluate forecasting models - runs all models in parallel for speed
    
    Args:
        data: Time series data
        forecast_steps: Number of steps to forecast
        models_to_run: List of model names to run. If None, runs all models.
        skip_validation: If True, skip validation step (faster but no RMSE/MAPE)
        parallel: If True, run models in parallel (default: True)
    """
    results = []
    
    # Setup validation split
    if skip_validation or len(data) < 20:
        validation_mode = False
        train_data = data
        test_data = pd.Series(dtype=float)
    else:
        # Use 70/30 split for faster validation
        split_idx = int(len(data) * 0.7)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        validation_mode = len(test_data) > 0
    
    # List of all forecast functions
    all_forecast_functions = [
        ("Naive", naive_forecast),
        ("Seasonal Naive", seasonal_naive_forecast),
        ("Moving Average", moving_average_forecast),
        ("SES", ses_forecast),
        ("Holt", holt_forecast),
        ("Holt-Winters (Additive)", holt_winters_additive_forecast),
        ("Holt-Winters (Multiplicative)", holt_winters_multiplicative_forecast),
        ("ARIMA", arima_forecast),
        ("SARIMA", sarima_forecast),
        ("AUTO ARIMA", auto_arima_forecast),
        ("ETS(A,A,A)", ets_forecast),
        ("Theta", theta_forecast),
        ("Prophet", prophet_forecast),
        ("Croston", croston_forecast),
    ]
    
    # Filter models if specified
    if models_to_run:
        forecast_functions = [(name, func) for name, func in all_forecast_functions if name in models_to_run]
    else:
        forecast_functions = all_forecast_functions
    
    # Run models in parallel for speed
    if parallel and len(forecast_functions) > 1:
        with ThreadPoolExecutor(max_workers=min(len(forecast_functions), 6)) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(_evaluate_single_model, model_name, forecast_func, 
                               data, train_data, test_data, forecast_steps, validation_mode): model_name
                for model_name, forecast_func in forecast_functions
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    model_name = future_to_model[future]
                    results.append({
                        "model_name": model_name,
                        "metadata": {"error": str(e)},
                        "rmse": None,
                        "mape": None,
                        "forecast": None
                    })
    else:
        # Sequential execution (fallback)
        for model_name, forecast_func in forecast_functions:
            result = _evaluate_single_model(model_name, forecast_func, data, 
                                          train_data, test_data, forecast_steps, validation_mode)
            results.append(result)
    
    return results


def select_best_model(results: List[Dict]) -> Dict:
    """Select best model based on RMSE (primary) and MAPE (tie-breaker)"""
    # Filter out models with None or invalid metrics
    valid_results = [r for r in results if r['rmse'] is not None and not np.isinf(r['rmse'])]
    
    if not valid_results:
        # If no valid results, return first model
        return results[0] if results else None
    
    # Sort by RMSE (ascending), then by MAPE (ascending)
    valid_results.sort(key=lambda x: (x['rmse'], x['mape'] if x['mape'] is not None else float('inf')))
    
    return valid_results[0]



