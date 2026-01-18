from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json

# Import all the forecasting functions from lambda_function
from forecasting_models import (
    evaluate_models,
    select_best_model
)
import pandas as pd

app = FastAPI(title="Time Series Forecasting API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataPoint(BaseModel):
    date: str
    value: float


class ForecastRequest(BaseModel):
    data: List[DataPoint]
    forecast_steps: int = 1
    models: Optional[List[str]] = None


@app.get("/")
def read_root():
    return {
        "message": "Time Series Forecasting API",
        "endpoint": "/forecast",
        "method": "POST"
    }


@app.post("/forecast")
def forecast(request: ForecastRequest):
    """
    Forecast time series data using multiple models
    
    - **data**: List of data points with date and value
    - **forecast_steps**: Number of steps to forecast (default: 1)
    - **models**: Optional list of model names to run. If not provided, runs all models.
    """
    try:
        # Convert to DataFrame
        data_list = [{"date": dp.date, "value": dp.value} for dp in request.data]
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.set_index('date')
        
        if 'value' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail='Missing "value" column in data'
            )
        
        data_series = df['value']
        
        if len(data_series) < 2:
            raise HTTPException(
                status_code=400,
                detail='Insufficient data. Need at least 2 data points.'
            )
        
        # Evaluate models (limit to faster models if not specified to avoid timeout)
        models_to_run = request.models
        if models_to_run is None and len(data_series) > 50:
            # For larger datasets, run faster models by default
            models_to_run = ["Naive", "Seasonal Naive", "Moving Average", "SES", "Holt", "AUTO ARIMA"]
        
        # Evaluate all models
        all_results = evaluate_models(data_series, request.forecast_steps, models_to_run)
        
        # Select best model
        best_model = select_best_model(all_results)
        
        return {
            'all_models': all_results,
            'best_model': {
                'model_name': best_model['model_name'],
                'metadata': best_model['metadata'],
                'rmse': best_model['rmse'],
                'mape': best_model['mape'],
                'forecast': best_model['forecast']
            },
            'forecast_steps': request.forecast_steps,
            'data_points': len(data_series)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

