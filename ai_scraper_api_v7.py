from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import uuid
import os
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from scrapegraphai.graphs import SmartScraperGraph
import asyncio
import concurrent.futures
import json
import openai
from openai import OpenAI

# FastAPI app instance
app = FastAPI(title="AI Web Scraper Chat API", description="AI-powered web scraper with conversational interface")

# In-memory session storage (use Redis in production)
sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TIMEOUT = timedelta(hours=24)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    openai_api_key: Optional[str] = None

# Legacy model for backward compatibility
class ScrapeRequest(BaseModel):
    prompt: str
    base_url: str
    openai_api_key: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    results: List[Dict[str, Any]]
    session_id: str
    status: str
    pages_scraped: Optional[int] = None
    total_records: Optional[int] = None
    csv_file: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    status: str
    total_records: int
    pages_scraped: int
    conversation_history: List[Dict[str, str]]
    csv_file: Optional[str] = None

# Session management
def create_session() -> str:
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "status": "created",
        "data": [],
        "total_records": 0,
        "pages_scraped": 0,
        "config": None,
        "conversation_history": [],
        "last_scrape_info": None,
        "csv_file": None
    }
    return session_id

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Check if session expired
    if datetime.now() - session["last_activity"] > SESSION_TIMEOUT:
        if session["csv_file"] and os.path.exists(session["csv_file"]):
            os.remove(session["csv_file"])
        del sessions[session_id]
        raise HTTPException(status_code=410, detail="Session expired")
    
    # Load data from CSV if exists
    if session["csv_file"] and os.path.exists(session["csv_file"]):
        try:
            df = pd.read_csv(session["csv_file"])
            session["data"] = df.to_dict('records')
            session["total_records"] = len(session["data"])
        except Exception as e:
            print(f"Error loading CSV: {e}")
            session["data"] = []
            session["total_records"] = 0
    
    # Update last activity
    session["last_activity"] = datetime.now()
    return session

def cleanup_expired_sessions():
    """Remove expired sessions and their CSV files"""
    expired_sessions = []
    for session_id, session_data in sessions.items():
        if datetime.now() - session_data["last_activity"] > SESSION_TIMEOUT:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        if sessions[session_id]["csv_file"] and os.path.exists(sessions[session_id]["csv_file"]):
            os.remove(sessions[session_id]["csv_file"])
        del sessions[session_id]

def replace_empty_values(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace empty, null, or None values with '-'"""
    if not data:
        return []
    
    cleaned_data = []
    for item in data:
        if isinstance(item, dict):
            cleaned_item = {}
            for key, value in item.items():
                if value is None or value == "" or value == [] or value == {}:
                    cleaned_item[key] = "-"
                elif isinstance(value, str) and value.strip() == "":
                    cleaned_item[key] = "-"
                else:
                    cleaned_item[key] = value
            cleaned_data.append(cleaned_item)
        else:
            cleaned_data.append(item if item is not None else "-")
    
    return cleaned_data

def save_to_csv(session_id: str, data: List[Dict[str, Any]]) -> str:
    """Save data to CSV and return filename"""
    if not data:
        return None
    cleaned_data = replace_empty_values(data)
    filename = f"scraped_data_{session_id}.csv"
    df = pd.DataFrame(cleaned_data)
    df.to_csv(filename, index=False)
    return filename

# Helper functions
def extract_max_pages(prompt: str) -> int:
    """Extract maximum pages to scrape from prompt"""
    match = re.search(r'page\s*1\s*(?:to|until|through|until page)?\s*(\d+)', prompt.lower())
    return int(match.group(1)) if match else 1

def extract_url_from_prompt(prompt: str) -> Optional[str]:
    """Extract URL from prompt if present"""
    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, prompt)
    return match.group(0) if match else None

def is_scraping_request(prompt: str) -> bool:
    """Determine if the prompt is asking for web scraping"""
    scraping_keywords = [
        'scrape', 'extract', 'get data from', 'fetch from', 'crawl',
        'get information from', 'retrieve from', 'collect from',
        'page 1', 'from page', 'website', 'url', 'http'
    ]
    return any(keyword in prompt.lower() for keyword in scraping_keywords)

def is_filter_request(prompt: str) -> bool:
    """Determine if the prompt is asking to filter existing data"""
    filter_keywords = [
        'filter', 'where', 'only show', 'only get', 'higher than',
        'lower than', 'greater than', 'less than', 'equal to',
        'contains', 'includes', 'exclude', 'remove', 'sort by'
    ]
    return any(keyword in prompt.lower() for keyword in filter_keywords)

async def analyze_with_ai(message: str, data: List[Dict[str, Any]], api_key: str) -> tuple:
    """Use OpenAI to analyze and transform data based on natural language"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare data sample for AI analysis
        data_sample = data[:3] if len(data) > 3 else data
        data_structure = str(data_sample)
        
        # Create prompt for AI
        system_prompt = f"""You are a data analyst AI. You have access to scraped data with this structure:
{data_structure}

The user wants to: {message}

You MUST respond with ONLY a valid JSON object containing:
1. "action" - what type of operation this is (filter, transform, calculate, analyze)
2. "explanation" - explain what you're doing in one sentence
3. "python_code" - Python code to perform the operation on the data
4. "requires_execution" - true if code needs to be executed, false if just explanation

Rules for python_code:
- The data variable is called 'data' and is a list of dictionaries
- For filtering/finding items, store result in 'result' variable
- For transformations, modify 'data' directly or create 'result' with new data
- Handle string parsing carefully for pay/salary fields
- Use regex (re module is available) for complex string operations
- NO markdown formatting, just raw Python code
- End result should be stored in 'result' variable

Example for wage conversion:
{{"action": "transform", "explanation": "Converting daily wages to hourly by dividing by 8", "python_code": "import re\\nresult = []\\nfor job in data:\\n    new_job = job.copy()\\n    if 'per day' in str(job.get('pay', '')):\\n        daily_amount = re.findall(r'\\\\\\$([\\\\d,]+)', str(job['pay']))\\n        if daily_amount:\\n            hourly = int(daily_amount[0].replace(',', '')) // 8\\n            new_job['pay'] = f'${{hourly}} per hour'\\n    result.append(new_job)", "requires_execution": true}}

Your response must be ONLY valid JSON with no extra text."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0
        )
        
        ai_response = response.choices[0].message.content.strip()
        print(f"Raw AI response: {ai_response}")  # Debug log
        
        # Try parsing JSON directly
        try:
            analysis = json.loads(ai_response)
            print(f"Successfully parsed JSON: {analysis}")
            return analysis, None
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Try to extract JSON from potential malformed response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    # Clean up escaped newlines and quotes
                    json_str = json_str.replace('\n', '\\n').replace('\r', '')
                    analysis = json.loads(json_str)
                    print(f"Successfully parsed cleaned JSON: {analysis}")
                    return analysis, None
                except json.JSONDecodeError as e2:
                    print(f"Cleaned JSON parsing error: {e2}")
            
            # Fallback to explanation mode if JSON parsing fails
            return {
                "action": "explanation",
                "explanation": "Failed to parse AI response, please try rephrasing your request",
                "python_code": None,
                "requires_execution": False
            }, f"JSON parsing failed: {str(e)}"
            
    except Exception as e:
        print(f"Error in analyze_with_ai: {e}")
        return {
            "action": "explanation",
            "explanation": "An error occurred while processing the request",
            "python_code": None,
            "requires_execution": False
        }, str(e)
    
    
def execute_ai_code(code: str, data: List[Dict[str, Any]]) -> tuple:
    """Safely execute AI-generated code on data"""
    try:
        # Clean the code
        cleaned_code = code.strip()
        if cleaned_code.startswith("```python"):
            cleaned_code = cleaned_code[9:]
        if cleaned_code.startswith("```"):
            cleaned_code = cleaned_code[3:]
        if cleaned_code.endswith("```"):
            cleaned_code = cleaned_code[:-3]
        
        cleaned_code = cleaned_code.strip()
        
        # Remove import statements since we'll pre-import modules
        lines = cleaned_code.split('\n')
        code_lines = []
        for line in lines:
            if not line.strip().startswith('import '):
                code_lines.append(line)
        cleaned_code = '\n'.join(code_lines)
        
        print(f"Executing code: {cleaned_code}")
        
        # Safe execution environment with pre-imported modules
        safe_globals = {
            "__builtins__": {},
            "len": len, "str": str, "int": int, "float": float,
            "round": round, "sum": sum, "max": max, "min": min,
            "sorted": sorted, "list": list, "dict": dict,
            "range": range, "enumerate": enumerate, "zip": zip,
            "any": any, "all": all, "print": print,
            "re": __import__('re'),  # Pre-import re module
            "math": __import__('math'),  # Pre-import math module
            "datetime": __import__('datetime')  # Pre-import datetime module
        }
        
        safe_locals = {"data": data, "result": None}
        
        # Execute the code
        exec(cleaned_code, safe_globals, safe_locals)
        
        # Get result
        result = safe_locals.get("result") or safe_locals.get("data")
        
        # Ensure result is a list
        if result and not isinstance(result, list):
            result = [result]
        
        return result or data, None
        
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        print(error_msg)
        return data, error_msg

def fallback_highest_pay(data: List[Dict[str, Any]]) -> tuple:
    """Fallback function to find highest pay manually"""
    try:
        highest_job = None
        highest_amount = 0
        
        for job in data:
            pay_str = str(job.get('pay', '0'))
            numbers = re.findall(r'[\d,]+', pay_str.replace('$', ''))
            if numbers:
                amount = int(numbers[0].replace(',', ''))
                if 'per day' in pay_str.lower():
                    amount = amount / 8
                elif 'per year' in pay_str.lower():
                    amount = amount / (52 * 40)
                
                if amount > highest_amount:
                    highest_amount = amount
                    highest_job = job
        
        return [highest_job] if highest_job else [], None
    except Exception as e:
        return [], str(e)

def fallback_convert_wages(data: List[Dict[str, Any]]) -> tuple:
    """Fallback function to convert daily wages to hourly"""
    try:
        result = []
        for job in data:
            new_job = job.copy()
            pay_str = str(job.get('pay', ''))
            
            if 'per day' in pay_str.lower():
                numbers = re.findall(r'[\d,]+', pay_str.replace('$', ''))
                if numbers:
                    daily_amount = int(numbers[0].replace(',', ''))
                    hourly_amount = int(daily_amount / 8)
                    new_job['pay'] = f"${hourly_amount} per hour"
            
            result.append(new_job)
        
        return result, None
    except Exception as e:
        return data, str(e)

def filter_data(data: List[Dict[str, Any]], filter_prompt: str) -> List[Dict[str, Any]]:
    """Simple filter function - kept for backward compatibility"""
    if not data:
        return []
    
    filtered_data = []
    filter_lower = filter_prompt.lower()
    
    try:
        # Handle pay/salary filtering
        if 'pay' in filter_lower or 'salary' in filter_lower:
            amount_match = re.search(r'[\$]?(\d+(?:,\d+)*)', filter_prompt)
            if amount_match:
                amount = int(amount_match.group(1).replace(',', ''))
                
                for item in data:
                    for key, value in item.items():
                        if 'pay' in key.lower() or 'salary' in key.lower():
                            if isinstance(value, str):
                                value_match = re.search(r'[\$]?(\d+(?:,\d+)*)', value)
                                if value_match:
                                    item_amount = int(value_match.group(1).replace(',', ''))
                                    if 'higher than' in filter_lower and item_amount > amount:
                                        filtered_data.append(item)
                                        break
                                    elif 'lower than' in filter_lower and item_amount < amount:
                                        filtered_data.append(item)
                                        break
        
        # Handle text-based filtering
        elif 'contains' in filter_lower or 'includes' in filter_lower:
            search_term = re.search(r'contains?\s+["\']?([^"\']+)["\']?', filter_lower)
            if not search_term:
                search_term = re.search(r'includes?\s+["\']?([^"\']+)["\']?', filter_lower)
            
            if search_term:
                term = search_term.group(1).strip()
                for item in data:
                    for value in item.values():
                        if isinstance(value, str) and term.lower() in value.lower():
                            filtered_data.append(item)
                            break
    
    except Exception as e:
        print(f"Error filtering data: {e}")
        return data
    
    return filtered_data if filtered_data else data

def get_llm_config(api_key: str) -> Dict[str, Any]:
    """Get LLM configuration"""
    return {
        "llm": {
            "model": "gpt-4",
            "api_key": api_key,
            "temperature": 0
        }
    }

async def perform_scraping(session: Dict[str, Any], prompt: str, url: str, session_id: str) -> tuple:
    """Perform the actual scraping operation"""
    max_pages = extract_max_pages(prompt)
    all_results = []
    pages_scraped = 0
    
    for page in range(1, max_pages + 1):
        page_url = re.sub(r'page=\d+', f'page={page}', url)
        
        try:
            def run_scraper():
                graph = SmartScraperGraph(
                    prompt=prompt,
                    source=page_url,
                    config=session["config"]
                )
                return graph.run()
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, run_scraper)
            
            if not result or "content" not in result:
                break
            
            all_results.extend(result["content"])
            pages_scraped += 1
            await asyncio.sleep(2)  # Polite scraping
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    # Save to CSV
    csv_file = save_to_csv(session_id, all_results)
    session["csv_file"] = csv_file
    
    return all_results, pages_scraped

# API Endpoints
@app.post("/start-session")
async def start_session():
    """Create a new AI chat session"""
    session_id = create_session()
    return {
        "text": "New AI chat session created! You can now ask me to scrape websites or filter existing data.",
        "results": [],
        "session_id": session_id,
        "status": "created",
        "csv_file": None
    }

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: ChatRequest):
    """Main chat endpoint for AI conversation"""
    cleanup_expired_sessions()
    session = get_session(session_id)
    
    message = request.message
    api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
    
    # Add message to conversation history
    session["conversation_history"].append({
        "role": "user",
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
    
    if not api_key:
        raise HTTPException(
            status_code=400, 
            detail="OpenAI API key required. Provide in request or set OPENAI_API_KEY environment variable"
        )
    
    session["config"] = get_llm_config(api_key)
    
    try:
        # Determine the type of request
        if is_scraping_request(message):
            # Handle scraping request
            url = extract_url_from_prompt(message)
            if not url:
                response_text = "I need a URL to scrape data. Please provide a URL in your message."
                response_data = []
                status = "error"
                csv_file = session["csv_file"]
            else:
                session["status"] = "scraping"
                
                all_results, pages_scraped = await perform_scraping(session, message, url, session_id)
                all_results = replace_empty_values(all_results)
                
                session["data"] = all_results
                session["total_records"] = len(all_results)
                session["pages_scraped"] = pages_scraped
                session["status"] = "completed" if all_results else "no_data"
                
                response_text = f"Successfully scraped {len(all_results)} records from {pages_scraped} pages and saved to CSV. You can now ask me to filter or analyze this data!"
                response_data = all_results
                status = session["status"]
                csv_file = session["csv_file"]
        
        elif session["data"]:
            # Use AI to analyze and process the request
            analysis, error = await analyze_with_ai(message, session["data"], api_key)
            
            if error:
                response_text = f"Sorry, I couldn't process your request: {error}"
                response_data = []
                status = "error"
                csv_file = session["csv_file"]
            elif analysis:
                print(f"Analysis result: {analysis}")  # Debug log
                
                if analysis.get("requires_execution", False) and analysis.get("python_code"):
                    print(f"Executing AI code...")  # Debug log
                    # Execute AI-generated code
                    processed_data, exec_error = execute_ai_code(analysis["python_code"], session["data"])
                    
                    if exec_error:
                        print(f"AI code failed: {exec_error}")  # Debug log
                        # Try fallback logic for common requests
                        if "highest pay" in message.lower():
                            processed_data, fallback_error = fallback_highest_pay(session["data"])
                            if not fallback_error:
                                response_text = f"✅ Found the job with highest pay"
                                status = "filter"
                            else:
                                response_text = f"Error: {fallback_error}"
                                status = "error"
                        elif "convert" in message.lower() and ("day" in message.lower() or "hour" in message.lower()):
                            processed_data, fallback_error = fallback_convert_wages(session["data"])
                            if not fallback_error:
                                response_text = f"✅ Successfully converted daily wages to hourly rates"
                                status = "transform"
                            else:
                                response_text = f"Error: {fallback_error}"
                                status = "error"
                        else:
                            response_text = f"Error executing analysis: {exec_error}"
                            processed_data = []
                            status = "error"
                    else:
                        print(f"AI code executed successfully")  # Debug log
                        # AI code executed successfully
                        processed_data = replace_empty_values(processed_data)
                        response_text = f"✅ {analysis.get('explanation', 'Analysis completed')}"
                        status = analysis.get("action", "completed")
                        
                        # Update session data but do not save to CSV
                        if analysis.get("action") in ["transform", "calculate"]:
                            session["data"] = processed_data
                            session["total_records"] = len(processed_data)
                    
                    response_data = processed_data
                    csv_file = session["csv_file"]
                else:
                    print(f"No execution required or no code provided")  # Debug log
                    # Just explanation, no code execution
                    response_text = analysis.get("explanation", "Analysis completed")
                    response_data = session["data"][:10]
                    status = "explanation"
                    csv_file = session["csv_file"]
            else:
                print(f"No analysis returned")  # Debug log
                # Fallback to simple filtering
                if is_filter_request(message):
                    filtered_data = filter_data(session["data"], message)
                    filtered_data = replace_empty_values(filtered_data)
                    response_text = f"Filtered data. Found {len(filtered_data)} records out of {len(session['data'])} total."
                    response_data = filtered_data
                    status = "filtered"
                    csv_file = session["csv_file"]
                else:
                    response_text = f"I have {len(session['data'])} records in CSV. Ask me to filter, analyze, or scrape new data!"
                    response_data = session["data"][:10]
                    status = "info"
                    csv_file = session["csv_file"]
        
        else:
            response_text = "Hi! I'm an AI web scraper. Ask me to scrape websites or analyze data!"
            response_data = []
            status = "ready"
            csv_file = session["csv_file"]
        
        # Add response to conversation history
        session["conversation_history"].append({
            "role": "assistant",
            "message": response_text,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(response_data)
        })
        
        return ChatResponse(
            text=response_text,
            results=response_data,
            session_id=session_id,
            status=status,
            pages_scraped=session.get("pages_scraped"),
            total_records=session.get("total_records"),
            csv_file=session["csv_file"]
        )
        
    except Exception as e:
        session["status"] = "error"
        error_text = f"Sorry, I encountered an error: {str(e)}"
        
        session["conversation_history"].append({
            "role": "assistant",
            "message": error_text,
            "timestamp": datetime.now().isoformat(),
            "error": True
        })
        
        return ChatResponse(
            text=error_text,
            results=[],
            session_id=session_id,
            status="error",
            csv_file=session["csv_file"]
        )

@app.post("/scrape/{session_id}")
async def scrape_data(session_id: str, request: Union[ChatRequest, ScrapeRequest]):
    """Legacy scrape endpoint that also supports chat format"""
    cleanup_expired_sessions()
    session = get_session(session_id)
    
    if hasattr(request, 'message'):
        message = request.message
        api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
    else:
        message = f"{request.prompt} from {request.base_url}"
        api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
    
    chat_request = ChatRequest(message=message, openai_api_key=api_key)
    return await chat(session_id, chat_request)

@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information and conversation history"""
    cleanup_expired_sessions()
    session = get_session(session_id)
    
    return SessionInfo(
        session_id=session_id,
        created_at=session["created_at"].isoformat(),
        last_activity=session["last_activity"].isoformat(),
        status=session["status"],
        total_records=session["total_records"],
        pages_scraped=session["pages_scraped"],
        conversation_history=session["conversation_history"],
        csv_file=session["csv_file"]
    )

@app.get("/session/{session_id}/data")
async def get_session_data(session_id: str, limit: Optional[int] = None):
    """Get all scraped data from session"""
    cleanup_expired_sessions()
    session = get_session(session_id)
    
    data = replace_empty_values(session["data"])
    if limit:
        data = data[:limit]
    
    return {
        "text": f"Retrieved {len(data)} records from session",
        "results": data,
        "session_id": session_id,
        "status": session["status"],
        "total_records": session["total_records"],
        "csv_file": session["csv_file"]
    }

@app.get("/session/{session_id}/download-csv")
async def download_csv(session_id: str):
    """Download scraped data as CSV"""
    cleanup_expired_sessions()
    session = get_session(session_id)
    
    if not session["data"] or not session["csv_file"] or not os.path.exists(session["csv_file"]):
        return {
            "text": "No data available for download",
            "results": [],
            "session_id": session_id,
            "status": "no_data",
            "csv_file": session["csv_file"]
        }
    
    cleaned_data = replace_empty_values(session["data"])
    filename = session["csv_file"]
    
    return {
        "text": f"Data available in {filename}",
        "results": [{"filename": filename, "total_records": len(cleaned_data)}],
        "session_id": session_id,
        "status": "downloaded",
        "csv_file": filename
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its CSV file"""
    if session_id not in sessions:
        return {
            "text": "Session not found",
            "results": [],
            "session_id": session_id,
            "status": "not_found"
        }
    
    if sessions[session_id]["csv_file"] and os.path.exists(sessions[session_id]["csv_file"]):
        os.remove(sessions[session_id]["csv_file"])
    
    del sessions[session_id]
    return {
        "text": f"Session {session_id} deleted successfully",
        "results": [],
        "session_id": session_id,
        "status": "deleted"
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    cleanup_expired_sessions()
    
    session_list = []
    for session_id, session_data in sessions.items():
        session_list.append({
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "last_activity": session_data["last_activity"].isoformat(),
            "status": session_data["status"],
            "total_records": session_data["total_records"],
            "pages_scraped": session_data["pages_scraped"],
            "conversation_length": len(session_data["conversation_history"]),
            "csv_file": session_data["csv_file"]
        })
    
    return {
        "text": f"Found {len(session_list)} active sessions",
        "results": session_list,
        "status": "success"
    }

@app.get("/")
async def root():
    """API information"""
    return {
        "text": "AI Web Scraper Chat API - Start a conversation to scrape and analyze web data!",
        "results": [{
            "endpoints": {
                "POST /start-session": "Create a new AI chat session",
                "POST /chat/{session_id}": "Chat with AI to scrape websites or filter data",
                "POST /scrape/{session_id}": "Legacy scrape endpoint (also supports chat)",
                "GET /session/{session_id}": "Get session information and conversation history",
                "GET /session/{session_id}/data": "Get all scraped data",
                "GET /session/{session_id}/download-csv": "Download data as CSV",
                "DELETE /session/{session_id}": "Delete session",
                "GET /sessions": "List all active sessions"
            }
        }],
        "status": "ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)