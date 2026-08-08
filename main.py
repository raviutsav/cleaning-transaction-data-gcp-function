import os
import sys
import datetime
import time
import argparse
import re
from typing import Optional, List
from supabase import create_client, Client
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import functions_framework
from flask import jsonify

# Load environment variables from .env file for local development
load_dotenv()

# --- Setup configuration from environment variables ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def print_help_and_exit():
    print("=" * 60)
    print("Missing Environment Variables!")
    print("=" * 60)
    print("Please set the credentials in a local .env file in the workspace directory:")
    print("  SUPABASE_URL=\"your_supabase_project_url\"")
    print("  SUPABASE_KEY=\"your_supabase_anon_key\"")
    print("  GEMINI_API_KEY=\"your_gemini_api_key\"")
    print("=" * 60)
    sys.exit(1)

# Initialize API Clients lazily
supabase: Optional[Client] = None
ai_client: Optional[genai.Client] = None

def init_clients():
    global supabase, ai_client
    if not supabase and SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    if not ai_client and GEMINI_API_KEY:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)

table_name = "sms_messages"

# Define Pydantic models for Structured Output
class TransactionItem(BaseModel):
    msg_id: str = Field(description="The unique database ID of the message")
    is_transaction: bool = Field(description="True if this message is a financial transaction (Debit/Credit/Transfer), False otherwise")
    amount: Optional[float] = Field(None, description="The transaction amount (as a number), or null if not a transaction")
    transaction_type: Optional[str] = Field(None, description="Must be 'DEBIT' (for spending/outgoing money) or 'CREDIT' (for incoming money/earnings), or null if not a transaction")
    merchant: Optional[str] = Field(None, description="The merchant name, bank name, or person involved in the transaction, or null if not a transaction")
    category: Optional[str] = Field(None, description="Category of spending: Food, Utilities, Shopping, Travel, Entertainment, Transfer, Investment, Income, Other, or null")

class TransactionBatchResponse(BaseModel):
    transactions: List[TransactionItem]

def is_potential_transaction(message_body: str) -> bool:
    if not message_body:
        return False
    body_lower = message_body.lower()
    
    # 1. Check for common transaction keywords
    keywords = [
        "debited", "credited", "spent", "withdrawn", "received", "sent", 
        "transfer", "paytm", "gpay", "phonepe", "upi", "txn", "transaction", 
        "payment", "purchased", "shoppe", "refund", "remittance"
    ]
    if any(kw in body_lower for kw in keywords):
        return True
        
    # 2. Check for currency indicators followed by numbers, e.g., Rs. 500, Rs 500, INR 500
    currency_patterns = [
        r"(?:rs\.?|inr|usd|eur|gbp)\s*\d+",
        r"\d+\s*(?:rs\.?|inr|usd|eur|gbp)"
    ]
    for pattern in currency_patterns:
        if re.search(pattern, body_lower):
            return True
            
    return False

def fetch_sms_data(start_date: Optional[str] = None, end_date: Optional[str] = None):
    print(f"[*] Fetching SMS records from table: '{table_name}'...")
    try:
        # Load sms messages and join with analyzed_transactions in a single query
        query = supabase.table(table_name).select("*, analyzed_transactions(*)")
        
        start_timestamp = None
        end_timestamp = None
        
        if start_date:
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            start_timestamp = datetime.datetime.combine(start_dt, datetime.time.min).isoformat() + "Z"
            
        if end_date:
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            end_timestamp = datetime.datetime.combine(end_dt, datetime.time.max).isoformat() + "Z"
            
        if start_timestamp:
            query = query.gte("received_timestamp", start_timestamp)
        if end_timestamp:
            query = query.lte("received_timestamp", end_timestamp)
            
        response = query.execute()
        data = response.data
        if not data:
            return [], [], []
            
        to_be_analyzed = []
        already_analyzed = []
        local_non_transactions = []
        
        for row in data:
            analysis_list = row.get("analyzed_transactions", [])
            if analysis_list:
                # Retrieve analyzed cached result (always a transaction)
                analysis = analysis_list[0] if isinstance(analysis_list, list) else analysis_list
                already_analyzed.append({
                    "id": row["id"],
                    "sender": row["sender"],
                    "message_body": row["message_body"],
                    "received_timestamp": row["received_timestamp"],
                    "device_name": row["device_name"],
                    "created_at": row["created_at"],
                    "is_transaction": True,
                    "amount": analysis["amount"],
                    "transaction_type": analysis["transaction_type"],
                    "merchant": analysis["merchant"],
                    "category": analysis["category"]
                })
            else:
                body = row.get("message_body", "")
                if is_potential_transaction(body):
                    to_be_analyzed.append({
                        "id": row["id"],
                        "sender": row["sender"],
                        "message_body": row["message_body"],
                        "received_timestamp": row["received_timestamp"],
                        "device_name": row["device_name"],
                        "created_at": row["created_at"]
                    })
                else:
                    local_non_transactions.append({
                        "id": row["id"],
                        "sender": row["sender"],
                        "message_body": row["message_body"],
                        "received_timestamp": row["received_timestamp"],
                        "device_name": row["device_name"],
                        "created_at": row["created_at"],
                        "is_transaction": False,
                        "amount": None,
                        "transaction_type": None,
                        "merchant": None,
                        "category": None
                    })
                
        print(f"[+] Loaded {len(data)} messages successfully.")
        print(f"    - Cached Transactions: {len(already_analyzed)}")
        print(f"    - Potential Transactions (LLM): {len(to_be_analyzed)}")
        print(f"    - Local Non-Transactions (Skipped): {len(local_non_transactions)}")
        return to_be_analyzed, already_analyzed, local_non_transactions
    except Exception as e:
        print(f"[!] Error fetching data from Supabase: {e}")
        raise e

def save_analyzed_transactions(newly_analyzed: List[dict]):
    if not newly_analyzed:
        return
        
    payload = []
    for r in newly_analyzed:
        if r.get("is_transaction") == True:
            payload.append({
                "msg_id": r["id"],
                "is_transaction": True,
                "amount": r["amount"],
                "transaction_type": r["transaction_type"],
                "merchant": r["merchant"],
                "category": r["category"]
            })
            
    if not payload:
        return
        
    print(f"[*] Saving {len(payload)} newly analyzed transactions to Supabase...")
    try:
        # Write to supabase in chunks of 100
        chunk_size = 100
        for i in range(0, len(payload), chunk_size):
            chunk = payload[i:i+chunk_size]
            supabase.table("analyzed_transactions").upsert(chunk).execute()
        print("[+] Saved successfully.")
    except Exception as e:
        print(f"[!] Error saving analyzed transactions to Supabase: {e}")
        raise e

def analyze_messages(to_be_analyzed: List[dict], model: str = "gemini-3.1-flash-lite", delay: float =5.0) -> List[dict]:
    if not to_be_analyzed:
        return []
        
    print(f"[*] Processing {len(to_be_analyzed)} messages with Gemini API ({model}) for transaction details...")
    
    # We will process in batches of 25 messages to stay within token limits and optimize network requests
    batch_size = 25
    newly_analyzed = []
    original_map = {row["id"]: row for row in to_be_analyzed}
    
    for i in range(0, len(to_be_analyzed), batch_size):
        chunk = to_be_analyzed[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(to_be_analyzed) + batch_size - 1) // batch_size
        print(f"    - Processing batch {batch_num}/{total_batches} ({len(chunk)} messages)...")
        
        # Apply delay between batches to prevent rate limiting
        if i > 0 and delay > 0:
            print(f"      [~] Sleeping for {delay} seconds between LLM calls to prevent rate limiting...")
            time.sleep(delay)
            
        # Format the batch for the LLM
        messages_text = ""
        for row in chunk:
            messages_text += f"ID: {row['id']}\nSender: {row['sender']}\nMessage: {row['message_body']}\n{'-'*40}\n"
            
        prompt = (
            "Analyze the following SMS messages and extract structured transaction details. "
            "For each message, determine if it represents a financial transaction (debit, credit, or money transfer). "
            "Examine message bodies carefully, ignoring OTPs, spam, updates, or other notifications.\n\n"
            f"{messages_text}"
        )
        
        max_retries = 5
        base_backoff = 5.0
        batch_success = False
        batch_result = None
        
        for attempt in range(max_retries):
            try:
                response = ai_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TransactionBatchResponse,
                        temperature=0.1
                    )
                )
                
                if response.parsed:
                    batch_result = response.parsed
                else:
                    batch_result = TransactionBatchResponse.model_validate_json(response.text)
                batch_success = True
                break
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or "rate limit" in error_str.lower()
                
                if is_rate_limit:
                    match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", error_str)
                    if match:
                        sleep_time = float(match.group(1)) + 2.0
                        print(f"    [!] Rate limit: API requested waiting {sleep_time:.2f} seconds. Sleeping...")
                    else:
                        sleep_time = base_backoff * (2 ** attempt)
                        print(f"    [!] Rate limit: Retrying in {sleep_time} seconds (attempt {attempt + 1}/{max_retries})...")
                    
                    time.sleep(sleep_time)
                else:
                    print(f"    [!] Non-rate-limit error: {e}.")
                    break
                    
        if not batch_success:
            print(f"\n[!] Critical: Batch {batch_num} failed repeatedly due to rate limit/errors.")
            print("[*] Progress saved up to this point. Exiting to allow future resumption.")
            break
            
        batch_mapped = []
        for item in batch_result.transactions:
            orig = original_map.get(item.msg_id, {})
            batch_mapped.append({
                "id": item.msg_id,
                "sender": orig.get("sender", ""),
                "message_body": orig.get("message_body", ""),
                "received_timestamp": orig.get("received_timestamp", ""),
                "device_name": orig.get("device_name"),
                "created_at": orig.get("created_at"),
                "is_transaction": item.is_transaction,
                "amount": item.amount,
                "transaction_type": item.transaction_type,
                "merchant": item.merchant,
                "category": item.category
            })
            
        save_analyzed_transactions(batch_mapped)
        newly_analyzed.extend(batch_mapped)
        
    return newly_analyzed

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
    request_json = request.get_json(silent=True)
    request_args = request.args
    
    start_date = None
    end_date = None
    model = "gemini-3.1-flash"
    delay = 5.0
    
    if request_json:
        start_date = request_json.get("start_date")
        end_date = request_json.get("end_date")
        model = request_json.get("model", model)
        delay_val = request_json.get("delay", delay)
        try:
            delay = float(delay_val)
        except (ValueError, TypeError):
            pass
    elif request_args:
        start_date = request_args.get("start_date")
        end_date = request_args.get("end_date")
        model = request_args.get("model", model)
        delay_val = request_args.get("delay", delay)
        try:
            delay = float(delay_val)
        except (ValueError, TypeError):
            pass
            
    # 3. Validate Date Range
    for date_str, name in [(start_date, "start_date"), (end_date, "end_date")]:
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return (jsonify({"error": f"Invalid format for {name}. Please use YYYY-MM-DD."}), 400, headers)
                
    if start_date and end_date:
        if start_date > end_date:
            return (jsonify({"error": "Start date cannot be after end date."}), 400, headers)
            
    # 4. Fetch SMS messages
    try:
        to_be_analyzed, already_analyzed, local_non_transactions = fetch_sms_data(start_date=start_date, end_date=end_date)
    except Exception as e:
        return (jsonify({"error": f"Database fetch error: {str(e)}"}), 500, headers)
        
    # 5. Process newly fetched unanalyzed transactions
    newly_analyzed = []
    if to_be_analyzed:
        try:
            newly_analyzed = analyze_messages(to_be_analyzed, model=model, delay=delay)
        except Exception as e:
            return (jsonify({"error": f"LLM processing error: {str(e)}"}), 500, headers)
            
    summary = {
        "status": "success",
        "loaded_messages": len(to_be_analyzed) + len(already_analyzed) + len(local_non_transactions),
        "cached_transactions": len(already_analyzed),
        "potential_transactions_processed": len(to_be_analyzed),
        "newly_confirmed_transactions": sum(1 for tx in newly_analyzed if tx.get("is_transaction") == True),
        "local_non_transactions_skipped": len(local_non_transactions)
    }
    return (jsonify(summary), 200, headers)

if __name__ == "__main__":
    # Local CLI runner
    parser = argparse.ArgumentParser(description="Analyze SMS messages for spending insights using Gemini API.")
    parser.add_argument(
        "-s", "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (inclusive). If omitted but end-date is present, retrieves all messages before end-date."
    )
    parser.add_argument(
        "-e", "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (inclusive). If omitted but start-date is present, retrieves all messages after start-date."
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=4.0,
        help="Delay in seconds between LLM calls to prevent rate limiting (default: 4.0)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-3.1-flash",
        help="Gemini model to use for analysis (default: gemini-3.1-flash)."
    )
    args = parser.parse_args()
    
    # Validate date input
    for date_str, name in [(args.start_date, "start-date"), (args.end_date, "end-date")]:
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                print(f"[!] Error: Invalid format for {name}. Please use YYYY-MM-DD.")
                sys.exit(1)
                
    if args.start_date and args.end_date:
        if args.start_date > args.end_date:
            print("[!] Error: Start date cannot be after end date.")
            sys.exit(1)
            
    # Validate local credentials
    if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_API_KEY:
        print_help_and_exit()
        
    init_clients()
    
    try:
        to_be_analyzed, already_analyzed, local_non_transactions = fetch_sms_data(start_date=args.start_date, end_date=args.end_date)
        
        if not to_be_analyzed and not already_analyzed and not local_non_transactions:
            print("[*] No messages found for the specified range.")
            sys.exit(0)
            
        newly_analyzed = []
        if to_be_analyzed:
            newly_analyzed = analyze_messages(to_be_analyzed, model=args.model, delay=args.delay)
        else:
            print("[*] No new potential transactions to analyze.")
            
        print("[+] Analysis complete. Confirmed transactions successfully saved to Supabase.")
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)
