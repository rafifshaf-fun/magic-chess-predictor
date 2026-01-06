# Magic Chess Predictor

## Overview
A web application for predicting opponents in Magic Chess games. It uses a Flask backend that serves a static HTML frontend and proxies predictions to an ML server.

## Project Structure
- `app.py` - Flask backend with API endpoints and static file serving
- `index.html` - Frontend UI for the prediction interface
- `requirements.txt` - Python dependencies
- `Match *.csv` - Match data files

## Technology Stack
- **Backend**: Python 3.11, Flask, Gunicorn
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **External**: Proxies to ML server at ML_SERVER_URL environment variable

## API Endpoints
- `GET /` - Serves the main HTML page
- `GET /api/session-stats` - Returns session statistics
- `POST /api/toggle-death` - Toggles player elimination status
- `POST /api/predict` - Forwards prediction requests to ML server
- `POST /api/log-feedback` - Logs user feedback

## Running the Application
The application runs on port 5000 using gunicorn:
```
gunicorn --bind 0.0.0.0:5000 --reuse-port app:app
```

## Environment Variables
- `ML_SERVER_URL` - URL of the ML prediction server (defaults to https://magic-chess-ml.onrender.com)
