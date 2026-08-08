# Cleaning Transaction Data GCP Function

A Google Cloud Function that fetches SMS transaction data from Supabase, processes it using the Gemini API to extract transaction details (amount, transaction type, merchant, category), and saves the parsed transaction data back to Supabase.

## Setup & Configuration

### Prerequisites
- Python 3.10+ (compatible with Python 3.13)
- A Supabase Project with `sms_messages` and `analyzed_transactions` tables.
- A Gemini API Key from Google AI Studio.

### Local Configuration
Create a `.env` file in the root directory:
```env
SUPABASE_URL="https://your-supabase-project-url.supabase.co"
SUPABASE_KEY="your-supabase-anon-key"
GEMINI_API_KEY="your-gemini-api-key"
```

## Running Locally

### CLI Runner
You can run the script as a command-line tool to analyze SMS messages:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the analyzer for a specific date range
python main.py --start-date 2026-08-01 --end-date 2026-08-08
```

### Local Functions Framework (HTTP Server)
To run the function locally using the functions framework:
```bash
# Install functions framework
pip install functions-framework

# Run the function local server
functions-framework --target=analyze_spending --debug
```
You can then trigger the endpoint:
- **GET**: `http://localhost:8080/?start_date=2026-08-01&end_date=2026-08-08`
- **POST**: `http://localhost:8080/` with JSON body:
  ```json
  {
    "start_date": "2026-08-01",
    "end_date": "2026-08-08",
    "delay": 4.0,
    "model": "gemini-2.0-flash"
  }
  ```

## Deploying to Google Cloud Functions

To deploy this function to GCP, use the following `gcloud` command:
```bash
gcloud functions deploy analyze-spending \
  --runtime=python310 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point=analyze_spending \
  --set-env-vars SUPABASE_URL="your-supabase-url",SUPABASE_KEY="your-supabase-key",GEMINI_API_KEY="your-gemini-api-key"
```
*(Make sure to adjust runtime and environment variables accordingly.)*
