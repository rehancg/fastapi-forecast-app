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

### Option 1: Using render.yaml (Recommended)

1. **Push code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Render:**
   - Go to https://render.com
   - Sign up/login with GitHub
   - Click "New +" → "Blueprint"
   - Connect your GitHub repo
   - Render will auto-detect `render.yaml` and deploy

### Option 2: Manual Setup

1. **Push code to GitHub** (same as above)

2. **Create Web Service on Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Select your repo

3. **Configure Settings:**
   - **Name**: `forecasting-api`
   - **Environment**: `Python 3`
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or `.` if needed)
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: `3.11.0` (or leave auto)

4. **Click "Create Web Service"**

### Build & Deploy Commands:

**Build Command:**
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Note:** Render automatically sets the `$PORT` environment variable, so use it in the start command.

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

