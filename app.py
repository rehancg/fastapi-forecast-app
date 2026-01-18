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
    forecast_steps: int = 12  # Default to 12 months
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
    - **forecast_steps**: Number of steps to forecast (default: 12 months)
    - **models**: Optional list of model names to run. If not provided, runs all models.
    
    Returns:
    - **all_models**: List of all models with metadata (RMSE, MAPE) - only best model includes forecast
    - **best_model**: The selected best model with full forecast (lowest RMSE, MAPE as tie-breaker)
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
        
        # Evaluate all models - parallel processing makes it fast
        models_to_run = request.models
        # Always run validation to properly compare models (parallel makes it fast)
        skip_validation = False
        
        # If no models specified, run all models (parallel processing handles speed)
        # Evaluate all models in parallel
        all_results = evaluate_models(data_series, request.forecast_steps, models_to_run, 
                                     skip_validation=skip_validation, parallel=True)
        
        # Select best model
        best_model = select_best_model(all_results)
        
        # Prepare response: only include forecast for best model, metadata for others
        all_models_metadata = []
        for model in all_results:
            model_data = {
                'model_name': model['model_name'],
                'metadata': model['metadata'],
                'rmse': model['rmse'],
                'mape': model['mape']
            }
            # Only include forecast for the best model
            if model['model_name'] == best_model['model_name']:
                model_data['forecast'] = model['forecast']
            all_models_metadata.append(model_data)
        
        return {
            'all_models': all_models_metadata,
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
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

