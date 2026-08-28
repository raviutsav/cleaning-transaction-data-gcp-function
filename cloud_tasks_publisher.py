import os
import sys
import json
import time
import uuid
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from supabase import create_client, Client

# Load environment variables from .env
load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")  # Lite model for fast/efficient extraction

# Clients
supabase: Client = None
ai_client: genai.Client = None

def init_clients():
    """Initializes the API clients if they haven't been initialized yet."""
    global supabase, ai_client
    if not supabase:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Supabase client: {e}") from e
            
    if not ai_client:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        try:
            ai_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini AI client: {e}") from e

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
class TransactionAnalysis(BaseModel):
    """Pydantic schema for structured output from Gemini."""
    is_transaction: bool = Field(
        description="True if this message represents an actual financial transaction where money entered or left the account, False otherwise."
    )
    amount: Optional[float] = Field(
        None, 
        description="The transaction amount, or null if not a transaction."
    )
    transaction_type: Optional[str] = Field(
        None, 
        description="Must be 'DEBIT' (money leaving account) or 'CREDIT' (money entering account), or null."
    )
    merchant: Optional[str] = Field(
        None, 
        description="The merchant name, bank name, or person involved in the transaction, or null."
    )
    category: Optional[str] = Field(
        None, 
        description="Category of spend: Food, Utilities, Shopping, Travel, Entertainment, Transfer, Investment, Income, Other, or null."
    )

# ==============================================================================
# RETRY HELPERS (ROBUSTNESS)
# ==============================================================================
def generate_content_with_retry(
    client: genai.Client, 
    model: str, 
    contents: str, 
    config: types.GenerateContentConfig, 
    retries: int = 4,
    initial_delay: float = 2.0
) -> types.GenerateContentResponse:
    """Helper to generate content with automatic retries for rate limits (429) and transient server errors."""
    delay = initial_delay
    for i in range(retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_msg = str(e).lower()
            
            # Check if the error is rate limit (429) or a transient server error (5xx, timeouts, etc.)
            is_rate_limit = "429" in err_msg or "resource_exhausted" in err_msg
            is_transient = any(
                phrase in err_msg
                for phrase in ["500", "502", "503", "504", "internal server error", "service unavailable", "timeout", "deadline", "connection"]
            )
            
            if (is_rate_limit or is_transient) and i < retries - 1:
                # If the API specifies a specific retry wait time, use it
                if "retry in" in err_msg:
                    try:
                        parts = err_msg.split("retry in")
                        subpart = parts[1].strip().split("s")[0].strip()
                        wait_time = float(subpart) + 1.5  # Add a tiny safety buffer
                    except Exception:
                        wait_time = delay
                else:
                    # Exponential backoff for other transient errors
                    wait_time = delay
                    delay *= 2
                
                print(f"[!] Gemini API error (Rate Limit/Transient): {e}. Waiting {wait_time:.2f} seconds before retrying (Attempt {i+1}/{retries})...")
                time.sleep(wait_time)
            else:
                # Re-raise any non-transient exceptions or when retries are exhausted
                raise e

    # Last attempt before failure
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

def supabase_execute_with_retry(query_builder_fn, retries: int = 3, initial_delay: float = 1.0):
    """Executes a Supabase query builder function with automatic retries for transient errors."""
    delay = initial_delay
    for i in range(retries):
        try:
            return query_builder_fn()
        except Exception as e:
            err_msg = str(e).lower()
            # Retry on connection issues, timeouts, or 5xx server errors
            is_transient = any(
                phrase in err_msg
                for phrase in ["timeout", "connection", "500", "502", "503", "504", "handshake", "socket"]
            )
            if is_transient and i < retries - 1:
                print(f"[!] Supabase transient error: {e}. Retrying in {delay:.2f} seconds (Attempt {i+1}/{retries})...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e

# ==============================================================================
# PIPELINE IMPLEMENTATION
# ==============================================================================
def clean_and_insert_sms(sms_row: dict) -> dict:
    """
    Accepts one row of sms_messages table in JSON form (dict),
    uses Gemini to clean it (extracting transaction details if relevant),
    and inserts the cleaned record into the analyzed_transactions table in Supabase.

    Args:
        sms_row (dict): A dictionary representing one row of the sms_messages table.
                        Must contain 'id', 'sender', and 'message_body'.

    Returns:
        dict: A status dict showing if the row was inserted or ignored, along with the data.
              This function is wrapped in a top-level try-except block so it will not crash
              the host loop.
    """
    try:
        # 1. Null checks & Type checks on inputs
        if sms_row is None:
            raise ValueError("Input SMS row cannot be None.")
        
        if not isinstance(sms_row, dict):
            raise ValueError(f"Input SMS row must be a dictionary. Received type: {type(sms_row).__name__}")

        msg_id = sms_row.get("id")
        sender = sms_row.get("sender")
        message_body = sms_row.get("message_body")

        # Convert fields to string safely, checking for null values
        sender_str = str(sender).strip() if sender is not None else ""
        message_body_str = str(message_body).strip() if message_body is not None else ""

        if msg_id is None:
            raise ValueError("Required field 'id' is missing or Null.")
        
        # Verify msg_id is a valid UUID format
        msg_id_str = str(msg_id).strip()
        try:
            uuid.UUID(msg_id_str)
        except ValueError as e:
            raise ValueError(f"Field 'id' ({msg_id_str}) is not a valid UUID format: {e}") from e

        # Handle empty message body gracefully without making API calls
        if not message_body_str:
            return {
                "status": "ignored",
                "message": "Message body is empty or whitespace. Ignored.",
                "data": None
            }

        # 2. Check and initialize API Clients
        init_clients()
        if not supabase:
            raise RuntimeError("Supabase client is not initialized.")
        if not ai_client:
            raise RuntimeError("Gemini client is not initialized.")

        # 3. Prompt Gemini for structured analysis
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
            "- Balance updates that only display current balance without an associated transaction\n\n"
            "TRANSACTION TYPE RULES:\n"
            "- DEBIT: Money went out of the user's account.\n"
            "- CREDIT: Money came into the user's account.\n\n"
            f"Sender: {sender_str}\n"
            f"Message: {message_body_str}"
        )

        # Call Gemini via retry helper
        response = generate_content_with_retry(
            client=ai_client,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TransactionAnalysis,
                temperature=0.1
            )
        )

        if not response:
            raise RuntimeError("Received a null response object from Gemini API.")

        # 4. Parse Gemini structured response safely
        try:
            if response.parsed:
                analysis = response.parsed
            elif response.text:
                analysis = TransactionAnalysis.model_validate_json(response.text)
            else:
                raise ValueError("Gemini response did not contain parsed data or raw text.")
        except Exception as parse_err:
            raw_text = getattr(response, "text", "No raw text available")
            print(f"[-] Failed to parse Gemini response as JSON. Raw output: {raw_text}", file=sys.stderr)
            raise ValueError(f"Failed to parse transaction analysis from Gemini response: {parse_err}") from parse_err

        # 5. Insert only if it's a transaction
        is_tx = getattr(analysis, "is_transaction", False)
        if is_tx:
            # Normalize transaction type to uppercase string or None
            tx_type = analysis.transaction_type
            if tx_type and isinstance(tx_type, str):
                tx_type = tx_type.strip().upper()
                
            transaction_data = {
                "msg_id": msg_id_str,
                "amount": analysis.amount,
                "transaction_type": tx_type,
                "merchant": str(analysis.merchant).strip() if analysis.merchant else None,
                "category": str(analysis.category).strip() if analysis.category else None
            }
            
            try:
                # Insert/Upsert the cleaned data into public.analyzed_transactions
                supabase_execute_with_retry(
                    lambda: supabase.table("analyzed_transactions").upsert(transaction_data).execute()
                )
            except Exception as db_err:
                raise RuntimeError(f"Supabase upsert failed for msg_id {msg_id_str}: {db_err}") from db_err
            
            return {
                "status": "inserted",
                "message": "Transaction analyzed and inserted into database.",
                "data": transaction_data
            }
        else:
            return {
                "status": "ignored",
                "message": "Message is not a financial transaction. Ignored.",
                "data": None
            }

    except Exception as e:
        # Mission critical protection: return status="failed" with details
        # rather than letting the exception crash the entire process.
        print(f"[!] Critical Error in clean_and_insert_sms: {e}", file=sys.stderr)
        return {
            "status": "failed",
            "message": "An error occurred during pipeline execution.",
            "error": str(e),
            "data": None
        }

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Fetching and processing 1 row from sms_messages table...")
    print("=" * 60)

    try:
        init_clients()
        if not supabase:
            raise ValueError("Supabase client is not initialized. Please configure SUPABASE_URL and SUPABASE_KEY.")

        # Fetch exactly 1 row from public.sms_messages
        response = supabase_execute_with_retry(
            lambda: supabase.table("sms_messages").select("*").limit(1).execute()
        )
        
        if response.data and len(response.data) > 0:
            sms_row = response.data[0]
            print(f"[+] Successfully fetched row from Supabase:")
            print(json.dumps(sms_row, indent=2))
            print("\n[*] Processing row through the cleaning pipeline...")
            
            result = clean_and_insert_sms(sms_row)
            print(f"\nPipeline Result:\n{json.dumps(result, indent=2)}")
            if result.get("status") == "failed":
                sys.exit(1)
        else:
            print("[-] No rows found in the 'sms_messages' table.")
            
    except Exception as e:
        print(f"\n[!] Pipeline error occurred: {e}", file=sys.stderr)
        sys.exit(1)
