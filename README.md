# Time Series Forecasting API

FastAPI application for time series forecasting with multiple models.

## Local Development

```bash
# Install dependencies
# For Python 3.11 (matches deployment):
pip install -r requirements.txt

# For Python 3.13+ (local development):
pip install -r requirements-dev.txt

# Run the app
uvicorn app:app --reload

# Or
python app.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deploy to Render (Free)

1. Push code to GitHub
2. Go to https://render.com
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Settings:
   - Name: `forecasting-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
6. Click "Create Web Service"

## Deploy to Railway (Free)

1. Push code to GitHub
2. Go to https://railway.app
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Railway will auto-detect Python and use the Procfile

## API Usage

```bash
curl -X POST https://your-app.onrender.com/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"date": "2020-01-01", "value": 100},
      {"date": "2020-01-02", "value": 105},
      {"date": "2020-01-03", "value": 110}
    ],
    "forecast_steps": 5
  }'
```

