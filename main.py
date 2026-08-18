import os
import sys
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from supabase import create_client, Client
from dotenv import load_dotenv
import functions_framework
from flask import jsonify

# Load environment variables from .env file for local development
load_dotenv()

# --- Setup configuration from environment variables ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize API Clients lazily
supabase: Client = None
ai_client: genai.Client = None

def init_clients():
    global supabase, ai_client
    if not supabase and SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    if not ai_client and GEMINI_API_KEY:
        # Initialize Google GenAI client
        ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Define Pydantic schema for Structured Output from Gemini
class TransactionAnalysis(BaseModel):
    is_transaction: bool = Field(description="True if this message represents a transaction where money entered or left the user's account, False otherwise.")
    amount: Optional[float] = Field(None, description="The transaction amount, or null if not a transaction.")
    transaction_type: Optional[str] = Field(None, description="Must be 'DEBIT' (money leaving account) or 'CREDIT' (money entering account), or null.")
    merchant: Optional[str] = Field(None, description="The merchant name, bank name, or person involved in the transaction, or null.")
    category: Optional[str] = Field(None, description="Category of spend: Food, Utilities, Shopping, Travel, Entertainment, Transfer, Investment, Income, Other, or null.")

@functions_framework.http
def analyze_spending(request):
    """HTTP Cloud Function Entrypoint."""
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # 1. Check Server Configurations
    if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
        return (jsonify({"error": "Missing environment configurations (SUPABASE_URL, SUPABASE_KEY, or GEMINI_API_KEY)."}), 500, headers)
        
    init_clients()
    
    # 2. Parse request variables
    request_json = request.get_json(silent=True) or {}
    
    # Support both nesting under a "msg" object or directly in the root JSON
    msg_data = request_json.get("msg") if "msg" in request_json else request_json
    
    msg_id = msg_data.get("id") or msg_data.get("msg_id")
    sender = msg_data.get("sender")
    message_body = msg_data.get("message_body") or msg_data.get("body")
    
    if not msg_id or not message_body:
        return (jsonify({"error": "Missing required fields: msg_id (or id) and message_body (or body) are required."}), 400, headers)

    # 3. Formulate Prompt for Gemini
    prompt = (
        "Analyze the following SMS message and extract structured transaction details.\n"
        "Determine if it represents an actual financial transaction (money entering or leaving the user's account).\n\n"
        "CRITICAL DEFINITION:\n"
        "A transaction means there has been an ACTUAL movement of money into or out of the user's account.\n\n"
        "EXAMPLES OF ACTUAL TRANSACTIONS (is_transaction = True):\n"
        "- Bank account debited / debited Rs. X\n"
        "- Bank account credited / credited Rs. X\n"
        "- UPI payment completed / sent / paid / transferred to someone\n"
        "- Card purchase / spent on card\n"
        "- ATM withdrawal\n"
        "- Money transferred to another person (DEBIT)\n"
        "- Money received from another person (CREDIT)\n"
        "- Refund actually credited back to the account (CREDIT)\n\n"
        "EXAMPLES OF NON-TRANSACTIONS (is_transaction = False):\n"
        "- One-Time Password (OTP) / login verification codes\n"
        "- Failed, declined, or blocked transaction notifications (no money actually moved)\n"
        "- Payment reminders / upcoming bill-due reminders (money has not moved yet)\n"
        "- Promotional offers, discount codes, or marketing messages\n"
        "- Balance updates that only display current balance without an associated transaction\n"
        "- Card expiry, renewal, or dispatch notifications\n"
        "- Login, security, or password change alerts\n"
        "- Generic service notifications from banks (e.g., 'maintain minimum balance')\n"
        "- Messages mentioning 'UPI', 'payment', or 'transfer' but not representing actual money movement (e.g., promotional or informational messages)\n\n"
        "TRANSACTION TYPE RULES:\n"
        "- DEBIT: Money went out of the user's account.\n"
        "- CREDIT: Money came into the user's account.\n\n"
        f"Sender: {sender}\n"
        f"Message: {message_body}"
    )

    # 4. Invoke LLM for analysis
    try:
        response = ai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TransactionAnalysis,
                temperature=0.1
            )
        )
        
        # Parse the structured output
        if response.parsed:
            analysis = response.parsed
        else:
            analysis = TransactionAnalysis.model_validate_json(response.text)
            
    except Exception as e:
        return (jsonify({"error": f"LLM analysis failed: {str(e)}"}), 500, headers)

    # 5. Conditionally Save to Database
    if analysis.is_transaction:
        try:
            db_response = supabase.table("analyzed_transactions").upsert({
                "msg_id": msg_id,
                "is_transaction": True,
                "amount": analysis.amount,
                "transaction_type": analysis.transaction_type,
                "merchant": analysis.merchant,
                "category": analysis.category
            }).execute()
            
            return (jsonify({
                "status": "inserted",
                "message": "Transaction parsed and inserted successfully.",
                "data": {
                    "msg_id": msg_id,
                    "amount": analysis.amount,
                    "transaction_type": analysis.transaction_type,
                    "merchant": analysis.merchant,
                    "category": analysis.category
                }
            }), 200, headers)
            
        except Exception as e:
            return (jsonify({"error": f"Database insertion failed: {str(e)}"}), 500, headers)
    else:
        # Ignore non-transactions and do not insert into db
        return (jsonify({
            "status": "ignored",
            "message": "Message is not a credit or debit transaction. Ignored successfully."
        }), 200, headers)

if __name__ == "__main__":
    # Local CLI/testing runner
    print("=" * 60)
    print("Running GCP Function locally...")
    print("=" * 60)
    
    if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
        print("Missing Environment Variables!")
        print("Please set the credentials in a local .env file in the workspace directory:")
        print("  SUPABASE_URL=\"your_supabase_project_url\"")
        print("  SUPABASE_KEY=\"your_supabase_anon_key\"")
        print("  GEMINI_API_KEY=\"your_gemini_api_key\"")
        sys.exit(1)
        
    init_clients()
    print("[+] Clients initialized successfully.")
    
    # Simple interactive test for developer
    import json
    test_body = {
        "id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "sender": "HDFCBK",
        "message_body": "Alert: Rs. 150.00 spent on HDFC Card XX1234 at Amazon on 2026-08-18."
    }
    print(f"[*] Simulating local run with message: {json.dumps(test_body, indent=2)}")
    print("Note: To run HTTP server locally, execute: functions-framework --target=analyze_spending")
