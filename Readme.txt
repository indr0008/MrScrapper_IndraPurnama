AI Web Scraper Chat API
This is a FastAPI-based web scraper with a conversational interface powered by OpenAI and ScrapeGraphAI. It allows users to scrape web data (e.g., job listings) and analyze or filter the results using natural language queries. Scraped data is saved to a CSV file during scraping requests, and subsequent chat requests read from this CSV without modifying it.
Features

Conversational Interface: Interact with the API using natural language to scrape websites or filter/analyze data.
Session Management: Maintains user sessions with a 24-hour timeout, storing scraped data in memory and CSV.
CSV Persistence: Scraped data is saved to a CSV file only during scraping requests. Subsequent chats (e.g., filtering) read from the CSV without overwriting it.
Endpoints: Start sessions, scrape websites, filter/analyze data, download CSV files, and manage sessions.
Error Handling: Robust error handling for invalid inputs, expired sessions, and scraping failures.

Prerequisites

Python 3.8+
An OpenAI API key (can be provided via environment variable or in API requests)
Required Python packages: fastapi, uvicorn, scrapegraphai, pandas, openai

Installation

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install fastapi uvicorn scrapegraphai pandas openai


Set Up Environment Variables (optional):

Create a .env file or set the OPENAI_API_KEY environment variable:export OPENAI_API_KEY="your-openai-api-key"  # On Windows: set OPENAI_API_KEY=your-openai-api-key


Alternatively, provide the API key in each request.


Ensure Write Permissions:

The application saves CSV files in the same directory as the script. Ensure the directory has write permissions.



Running the Application

Start the FastAPI Server:
python app.py

The server will run on http://localhost:8000. Use uvicorn app:app --host 0.0.0.0 --port 8000 for production.

Access the API Documentation:

Open http://localhost:8000/docs in a browser to view the interactive Swagger UI for testing endpoints.



Using the API
The API uses a session-based approach. Start a session, then use the chat endpoint to scrape or analyze data. Below are the key endpoints and example requests.
1. Start a New Session

Endpoint: POST /start-session
Description: Creates a new session and returns a session ID.
Example Request (using curl or Postman):curl -X POST http://localhost:8000/start-session


Example Response:{
  "text": "New AI chat session created! You can now ask me to scrape websites or filter existing data.",
  "results": [],
  "session_id": "c70e0148-aaf8-4ea5-b6ad-313bba398361",
  "status": "created",
  "csv_file": null
}


Note: Save the session_id for subsequent requests.

2. Chat with the API

Endpoint: POST /chat/{session_id}

Description: Send natural language queries to scrape websites or filter/analyze data. Scraping requests save data to a CSV file; other requests read from the CSV without modifying it.

Request Body:
{
  "message": "your natural language query",
  "openai_api_key": "your-openai-api-key"  // Optional if set in environment
}


Example 1: Scraping Request:
curl -X POST http://localhost:8000/chat/c70e0148-aaf8-4ea5-b6ad-313bba398361 \
-H "Content-Type: application/json" \
-d '{"message": "get the role, location, and pay of jobs from https://medrecruit.medworld.com/jobs/list?location=New+South+Wales&page=1 from page 1", "openai_api_key": "sk-..."}'


Response:{
  "text": "Successfully scraped 10 records from 1 pages and saved to CSV. You can now ask me to filter or analyze this data!",
  "results": [/* job objects with role, location, pay */],
  "session_id": "c70e0148-aaf8-4ea5-b6ad-313bba398361",
  "status": "completed",
  "pages_scraped": 1,
  "total_records": 10,
  "csv_file": "scraped_data_c70e0148-aaf8-4ea5-b6ad-313bba398361.csv"
}


Note: This creates a CSV file (scraped_data_{session_id}.csv) in the script's directory.


Example 2: Filtering Request:
curl -X POST http://localhost:8000/chat/c70e0148-aaf8-4ea5-b6ad-313bba398361 \
-H "Content-Type: application/json" \
-d '{"message": "filter jobs with pay higher than $100", "openai_api_key": "sk-..."}'


Response:{
  "text": "Filtered data. Found 3 records out of 10 total.",
  "results": [/* filtered job objects */],
  "session_id": "c70e0148-aaf8-4ea5-b6ad-313bba398361",
  "status": "filtered",
  "pages_scraped": 1,
  "total_records": 10,
  "csv_file": "scraped_data_c70e0148-aaf8-4ea5-b6ad-313bba398361.csv"
}


Note: This reads from the existing CSV without modifying it.



3. Download CSV

Endpoint: GET /session/{session_id}/download-csv
Description: Returns information about the CSV file created during a scraping request.
Example Request:curl http://localhost:8000/session/c70e0148-aaf8-4ea5-b6ad-313bba398361/download-csv


Example Response:{
  "text": "Data available in scraped_data_c70e0148-aaf8-4ea5-b6ad-313bba398361.csv",
  "results": [{"filename": "scraped_data_c70e0148-aaf8-4ea5-b6ad-313bba398361.csv", "total_records": 10}],
  "session_id": "c70e0148-aaf8-4ea5-b6ad-313bba398361",
  "status": "downloaded",
  "csv_file": "scraped_data_c70e0148-aaf8-4ea5-b6ad-313bba398361.csv"
}


Note: The CSV file is located in the script's directory. You can manually retrieve it or configure your client to download it.

4. Other Endpoints

Get Session Info: GET /session/{session_id} - View session details and conversation history.
Get Session Data: GET /session/{session_id}/data?limit=N - Retrieve all scraped data (optionally limited to N records).
Delete Session: DELETE /session/{session_id} - Delete a session and its CSV file.
List Sessions: GET /sessions - List all active sessions.
Root: GET / - View API information and available endpoints.

CSV Handling

Generation: A CSV file (scraped_data_{session_id}.csv) is created only when a scraping request is made (e.g., asking to scrape a website).
Read-Only for Subsequent Chats: Filtering or analysis requests (e.g., "filter jobs with pay higher than $100") read from the CSV without modifying it. Results are processed in memory and returned in the response.
Location: CSV files are saved in the same directory as app.py. Ensure write permissions.
Cleanup: CSV files are deleted when the session expires (after 24 hours) or when the session is explicitly deleted via DELETE /session/{session_id}.

Example Workflow

Start a session:
curl -X POST http://localhost:8000/start-session

Get the session_id (e.g., c70e0148-aaf8-4ea5-b6ad-313bba398361).

Scrape job listings:
curl -X POST http://localhost:8000/chat/c70e0148-aaf8-4ea5-b6ad-313bba398361 \
-H "Content-Type: application/json" \
-d '{"message": "get the role, location, and pay of jobs from https://medrecruit.medworld.com/jobs/list?location=New+South+Wales&page=1 from page 1", "openai_api_key": "sk-..."}'

This saves the scraped data to scraped_data_c70e0148-aaf8-4ea5-b6ad-313bba398361.csv.

Filter the data:
curl -X POST http://localhost:8000/chat/c70e0148-aaf8-4ea5-b6ad-313bba398361 \
-H "Content-Type: application/json" \
-d '{"message": "filter jobs with pay higher than $100", "openai_api_key": "sk-..."}'

This reads from the CSV, filters the data in memory, and returns the results without altering the CSV.

Download the CSV:
curl http://localhost:8000/session/c70e0148-aaf8-4ea5-b6ad-313bba398361/download-csv

Retrieve the CSV file from the script's directory.

Delete the session (optional):
curl -X DELETE http://localhost:8000/session/c70e0148-aaf8-4ea5-b6ad-313bba398361

This deletes the session and its CSV file.


Troubleshooting

Error: "Session not found": Ensure the session_id is correct and the session hasn’t expired (24-hour timeout).
Error: "OpenAI API key required": Provide a valid OpenAI API key in the request or set the OPENAI_API_KEY environment variable.
No data in response: Check if the website is accessible and allows scraping. Some websites may block scraping attempts.
CSV not found: Ensure the script has write permissions in its directory. Check terminal logs for errors during CSV creation.
Filtering returns unexpected results: Rephrase the query for clarity (e.g., "show jobs with salary above $100 per hour"). Check the CSV to verify the data structure.
Terminal logs: Run the server in a terminal to view detailed logs for debugging.

Notes

Dependencies: Ensure all required packages are installed. Use pip show <package-name> to verify.
OpenAI API Key: Obtain a key from OpenAI. Keep it secure and avoid hardcoding it in the script.
Scraping Ethics: Respect website terms of service and robots.txt. Include delays (2 seconds between page requests) to avoid overloading servers.
Production: For production, use a proper WSGI server (e.g., Gunicorn) and consider a database (e.g., Redis) instead of in-memory session storage.
CSV Persistence: The CSV is read-only for non-scraping requests, ensuring the original scraped data remains unchanged.

Support
For issues or questions, contact the developer or open an issue in the repository. Provide terminal logs and the exact request/response details for faster resolution.