#!/usr/bin/env python3
"""
ZenDriver MCP Server

A Model Context Protocol server that provides web scraping capabilities
using ZenDriver's undetectable browser automation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import csv
import os
from collections import defaultdict
from bs4 import BeautifulSoup
import time

try:
    import zendriver as zd
except ImportError:
    raise ImportError("zendriver is required. Install with: pip install zendriver")
try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("beautifulsoup4 is required. Install with: pip install beautifulsoup4")
try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("openai is required. Install with: pip install openai")

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zendriver-mcp")

class ZenDriverManager:
    def __init__(self):
        self.browser = None
        self.pages = {}  # Store active pages by ID
        self.output_dir = Path(r"C:\Users\purna\Downloads\MrScrapper\Claude\Github_MrScrapper\MrScrapper_IndraPurnama\zendriver_mcp")
        self.output_dir.mkdir(exist_ok=True)
        self.openai_client = AsyncOpenAI(api_key="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")
        
    async def _ensure_browser(self):
        """Ensure browser is started"""
        if self.browser is None:
            logger.info("Starting ZenDriver browser...")
            self.browser = await zd.start()
            
    async def _get_page(self, url: str):
        """Get or create a page for the given URL"""
        await self._ensure_browser()
        
        # Use URL as page key for simplicity
        page_key = url
        
        if page_key not in self.pages:
            logger.info(f"Creating new page for {url}")
            page = await self.browser.get(url)
            self.pages[page_key] = page
        else:
            page = self.pages[page_key]
            # Navigate to URL if it's different
            if page.url != url:
                await page.get(url)
                
        return self.pages[page_key]

    async def scrape_website(self, url: str, wait_for: str = None, 
                             wait_timeout: float = 10, execute_js: str = None) -> Dict[str, str]:
        """Scrape full HTML from a website, analyze saved HTML for unique selectors, and save results"""
        page = await self._get_page(url)
        
        # Wait for element if specified
        if wait_for:
            try:
                if wait_for.startswith('#') or wait_for.startswith('.') or ' ' in wait_for:
                    # CSS selector
                    await page.select(wait_for, timeout=wait_timeout)
                else:
                    # Text content
                    await page.find(wait_for, timeout=wait_timeout)
            except Exception as e:
                logger.warning(f"Wait condition not met: {e}")
        
        # Execute JavaScript if provided
        if execute_js:
            try:
                await page.evaluate(execute_js)
            except Exception as e:
                logger.warning(f"JavaScript execution failed: {e}")
        
        # Extract full HTML content
        content = []
        full_html = ""
        try:
            full_html = await page.get_content()
            content.append(full_html)
        except Exception as e:
            logger.warning(f"Error getting full page content: {e}")
            full_html = f"Error getting page content: {e}"
            content.append(full_html)
        
        # Generate filenames based on timestamp
        timestamp = int(time.time())
        html_filename = f"scrape_{timestamp}.html"
        csv_filename = f"scrape_{timestamp}.csv"
        selectors_csv_filename = f"selectors_{timestamp}.csv"
        html_filepath = self.output_dir / html_filename
        csv_filepath = self.output_dir / csv_filename
        selectors_csv_filepath = self.output_dir / selectors_csv_filename
        
        # Save HTML content
        try:
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(full_html)
            logger.info(f"Saved HTML content to {html_filepath}")
        except Exception as e:
            logger.error(f"Error saving HTML file: {e}")
            content.append(f"Error saving HTML file: {e}")
        
        # Save content as CSV
        try:
            with open(csv_filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['URL', 'Content'])
                writer.writerow([url, full_html])
            logger.info(f"Saved CSV content to {csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving CSV file: {e}")
            content.append(f"Error saving CSV file: {e}")
        
        # Analyze saved HTML for unique selectors using BeautifulSoup
        selector_examples = defaultdict(list)
        try:
            # Read the saved CSV to get the HTML content
            with open(csv_filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) < 2:
                    logger.warning("CSV file is empty or missing content")
                    content.append("Error analyzing DOM: CSV file is empty or missing content")
                else:
                    # Get HTML from the second row (first row is headers)
                    html_content = rows[1][1]  # Content column
                    soup = BeautifulSoup(html_content, 'html.parser')
                    # Get all elements
                    elements = soup.find_all(True)  # Find all tags
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if text:
                            # Construct concise CSS selector (tag#id.class format)
                            selector = elem.name
                            if elem.get('id'):
                                selector += f"#{elem.get('id')}"
                            if elem.get('class'):
                                classes = '.'.join(elem.get('class')).replace(' ', '.')
                                if classes:
                                    selector += f".{classes}"
                            if selector and len(selector_examples[selector]) < 2:
                                selector_examples[selector].append(text)
        except Exception as e:
            logger.warning(f"Error analyzing HTML for selectors: {e}")
            content.append(f"Error analyzing HTML for selectors: {str(e)}")
        
        # Save unique selectors with up to two examples each in a separate CSV
        if selector_examples:
            try:
                with open(selectors_csv_filepath, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Selector', 'Example1', 'Example2'])
                    for sel, examples in selector_examples.items():
                        example1 = examples[0] if examples else ""
                        example2 = examples[1] if len(examples) > 1 else ""
                        writer.writerow([sel, example1, example2])
                logger.info(f"Saved selectors CSV to {selectors_csv_filepath}")
            except Exception as e:
                logger.error(f"Error saving selectors CSV file: {e}")
                content.append(f"Error saving selectors CSV file: {e}")
        
        result = "\n".join(content) if content else "No content extracted"
        return {
            "result": result,
            "selectors_csv": str(selectors_csv_filepath)
        }

    async def select_selectors(self, query: str, selectors_csv: str = None) -> str:
        """Use OpenAI API to select appropriate CSS selectors based on user query"""
        # Find the latest selectors CSV if not provided
        if not selectors_csv:
            selectors_files = list(self.output_dir.glob("selectors_*.csv"))
            if not selectors_files:
                return "Error: No selectors CSV files found in output directory"
            selectors_csv = str(max(selectors_files, key=os.path.getmtime))
            logger.info(f"Using latest selectors CSV: {selectors_csv}")
        else:
            if not os.path.exists(selectors_csv):
                return f"Error: Selectors CSV file not found: {selectors_csv}"
        
        # Read selectors and examples from CSV
        selectors_data = []
        try:
            with open(selectors_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    selectors_data.append({
                        "selector": row["Selector"],
                        "example1": row["Example1"],
                        "example2": row["Example2"]
                    })
        except Exception as e:
            logger.error(f"Error reading selectors CSS: {e}")
            return f"Error reading selectors CSV: {e}"
        
        if not selectors_data:
            return "Error: No selectors found in CSV file"
        
        # Prepare prompt for OpenAI
        prompt = f"""
You are a web scraping expert. Given a user query and a list of CSS selectors with example contents from a webpage, identify the most appropriate selectors that match the user's request. Return only the selectors (one per line) that are relevant. Prefer specific selectors like 'span.text' or 'small.author' for nested structures.

User Query: {query}

Selectors and Examples:
{json.dumps(selectors_data, indent=2)}

For example, if the query is "authors and quotes from quotes.toscrape.com", select selectors like "span.text" for quotes and "small.author" for authors if their examples match quote texts and author names.

Output only the selectors, one per line.
"""
        
        # Call OpenAI API
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a web scraping assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            selected_selectors = response.choices[0].message.content.strip().split("\n")
            selected_selectors = [s.strip() for s in selected_selectors if s.strip()]
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error calling OpenAI API: {e}"
        
        if not selected_selectors:
            return "No selectors matched the query"
        
        # Save selected selectors to a new CSV
        timestamp = int(time.time())
        selected_csv_filename = f"selected_selectors_{timestamp}.csv"
        selected_csv_filepath = self.output_dir / selected_csv_filename
        try:
            with open(selected_csv_filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Selector', 'Example1', 'Example2', 'Query'])
                for selector in selected_selectors:
                    # Find the original selector data
                    for data in selectors_data:
                        if data["selector"] == selector or selector.startswith(data["selector"]):
                            writer.writerow([selector, data["example1"], data["example2"], query])
                            break
                    else:
                        writer.writerow([selector, "", "", query])
            logger.info(f"Saved selected selectors CSV to {selected_csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving selected selectors CSV: {e}")
            return f"Error saving selected selectors CSV: {e}"
        
        return f"Selected selectors saved to {selected_csv_filepath}\n" + "\n".join(selected_selectors)

    async def scrape_with_selected_selectors(self, url: str, selected_csv: str = None, wait_for: str = None, 
                                            wait_timeout: float = 30) -> str:
        """Scrape webpage content using selectors from selected_selectors CSV and save results"""
        # Find the latest selected selectors CSV if not provided
        if not selected_csv:
            selected_files = list(self.output_dir.glob("selected_selectors_*.csv"))
            if not selected_files:
                return "Error: No selected selectors CSV files found in output directory"
            selected_csv = str(max(selected_files, key=os.path.getmtime))
            logger.info(f"Using latest selected selectors CSV: {selected_csv}")
        else:
            if not os.path.exists(selected_csv):
                return f"Error: Selected selectors CSV file not found: {selected_csv}"
        
        # Read selectors and query from CSV
        selectors_data = []
        try:
            with open(selected_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    selectors_data.append({
                        "selector": row["Selector"],
                        "query": row["Query"]
                    })
        except Exception as e:
            logger.error(f"Error reading selected selectors CSV: {e}")
            return f"Error reading selected selectors CSV: {e}"
        
        if not selectors_data:
            return "Error: No selectors found in selected selectors CSV"
        
        # Navigate to the webpage
        page = await self._get_page(url)
        
        # Wait for element if specified, with increased timeout
        if wait_for:
            try:
                if wait_for.startswith('#') or wait_for.startswith('.') or ' ' in wait_for:
                    await page.select(wait_for, timeout=wait_timeout)
                else:
                    await page.find(wait_for, timeout=wait_timeout)
            except Exception as e:
                logger.warning(f"Wait condition not met: {e}")
        
        # Wait for page to be fully loaded
        try:
            await page.wait_for_load_state('networkidle', timeout=wait_timeout * 1000)
        except Exception as e:
            logger.warning(f"Error waiting for page load: {e}")
        
        # Scrape content for each selector
        scraped_data = []
        for data in selectors_data:
            selector = data["selector"]
            query = data["query"]
            try:
                # Validate selector with a single select
                element = await page.select(selector, timeout=5000)
                if element is None:
                    logger.warning(f"No elements found for selector: {selector}")
                    # Try JavaScript fallback
                    js_code = f"""
                    (function() {{
                        const elements = document.querySelectorAll('{selector}');
                        const results = [];
                        elements.forEach(elem => {{
                            const text = elem.innerText || '';
                            if (text.trim()) results.push(text.trim());
                        }});
                        return results;
                    }})();
                    """
                    try:
                        js_results = await page.evaluate(js_code)
                        if js_results:
                            for text in js_results:
                                scraped_data.append({"selector": selector, "content": text, "query": query})
                        else:
                            scraped_data.append({"selector": selector, "content": "No elements found", "query": query})
                    except Exception as e:
                        logger.warning(f"JavaScript fallback failed for selector {selector}: {e}")
                        scraped_data.append({"selector": selector, "content": f"Error: {str(e)}", "query": query})
                    continue
                
                elements = await page.select_all(selector)
                if elements is None:
                    logger.warning(f"No elements found for selector: {selector}")
                    scraped_data.append({"selector": selector, "content": "No elements found", "query": query})
                    continue
                
                for elem in elements:
                    if elem is None:
                        logger.warning(f"Encountered None element for selector: {selector}")
                        continue
                    try:
                        text = await elem.get_text()
                        if text and text.strip():
                            scraped_data.append({"selector": selector, "content": text.strip(), "query": query})
                    except Exception as e:
                        logger.warning(f"Error getting text for element with selector {selector}: {e}")
                        # Fallback to JavaScript for this element
                        js_code = f"""
                        (function() {{
                            const elements = document.querySelectorAll('{selector}');
                            const results = [];
                            elements.forEach(elem => {{
                                const text = elem.innerText || '';
                                if (text.trim()) results.push(text.trim());
                            }});
                            return results;
                        }})();
                        """
                        try:
                            js_results = await page.evaluate(js_code)
                            if js_results:
                                for text in js_results:
                                    scraped_data.append({"selector": selector, "content": text, "query": query})
                            else:
                                scraped_data.append({"selector": selector, "content": f"Error: {str(e)}", "query": query})
                        except Exception as js_e:
                            logger.warning(f"JavaScript fallback failed for selector {selector}: {js_e}")
                            scraped_data.append({"selector": selector, "content": f"Error: {str(e)}", "query": query})
            except Exception as e:
                logger.error(f"Error scraping with selector {selector}: {e}")
                scraped_data.append({"selector": selector, "content": f"Error: {str(e)}", "query": query})
        
        if not scraped_data:
            return "No content scraped for the provided selectors"
        
        # Save scraped data to a new CSV
        timestamp = int(time.time())
        scraped_csv_filename = f"scraped_data_{timestamp}.csv"
        scraped_csv_filepath = self.output_dir / scraped_csv_filename
        try:
            with open(scraped_csv_filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Selector', 'Content', 'Query'])
                for item in scraped_data:
                    writer.writerow([item["selector"], item["content"], item["query"]])
            logger.info(f"Saved scraped data CSV to {scraped_csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving scraped data CSV: {e}")
            return f"Error saving scraped data CSV: {e}"
        
        result = f"Scraped data saved to {scraped_csv_filepath}\n" + "\n".join(
            [f"{item['selector']}: {item['content']}" for item in scraped_data]
        )
        return result

    async def execute_javascript(self, url: str, code: str, wait_for_result: bool = True) -> str:
        """Execute JavaScript on a webpage"""
        page = await self._get_page(url)
        
        try:
            # Wrap code in a function to avoid illegal return statements
            wrapped_code = f"(function() {{ {code} }})()"
            if wait_for_result:
                result = await page.evaluate(wrapped_code)
                return f"JavaScript result: {json.dumps(result)}"
            else:
                await page.evaluate(wrapped_code)
                return "JavaScript executed successfully"
        except Exception as e:
            return f"Error executing JavaScript: {e}"

    async def click_element(self, url: str, selector: str = None, text: str = None, 
                            wait_after_click: float = 1) -> str:
        """Click on an element"""
        page = await self._get_page(url)
        
        try:
            if selector:
                element = await page.select(selector)
            elif text:
                element = await page.find(text)
            else:
                return "Error: Must provide either 'selector' or 'text' parameter"
            
            await element.click()
            await asyncio.sleep(wait_after_click)
            
            return "Element clicked successfully"
            
        except Exception as e:
            return f"Error clicking element: {e}"

    async def fill_form(self, url: str, fields: Dict[str, str], submit: bool = False, 
                        submit_selector: str = None) -> str:
        """Fill out form fields"""
        page = await self._get_page(url)
        
        try:
            results = []
            
            for field_selector, value in fields.items():
                try:
                    element = await page.select(field_selector)
                    await element.send_keys(value)
                    results.append(f"Filled field '{field_selector}' with '{value}'")
                except Exception as e:
                    results.append(f"Error filling field '{field_selector}': {e}")
            
            if submit:
                try:
                    if submit_selector:
                        submit_btn = await page.select(submit_selector)
                    else:
                        # Try to find submit button
                        submit_btn = await page.select('input[type="submit"], button[type="submit"], button:contains("Submit")')
                    
                    await submit_btn.click()
                    results.append("Form submitted successfully")
                except Exception as e:
                    results.append(f"Error submitting form: {e}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error filling form: {e}"

    async def cleanup(self):
        """Clean up resources"""
        if self.browser:
            try:
                await self.browser.stop()
            except Exception as e:
                logger.error(f"Error stopping browser: {e}")
            self.browser = None
        self.pages.clear()

# Create the MCP server
server = Server("zendriver-scraper")
zendriver_manager = ZenDriverManager()

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools"""
    return [
        types.Tool(
            name="scrape_website",
            description="Scrape full HTML from a website and extract unique selectors from saved HTML",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "wait_for": {"type": "string", "description": "CSS selector or text to wait for before scraping (optional)"},
                    "wait_timeout": {"type": "number", "description": "Timeout in seconds to wait for elements (default: 10)", "default": 10},
                    "execute_js": {"type": "string", "description": "JavaScript code to execute before scraping (optional)"}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="select_selectors",
            description="Use OpenAI API to select appropriate CSS selectors based on a user query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query describing desired content (e.g., 'authors and quotes from quotes.toscrape.com')"},
                    "selectors_csv": {"type": "string", "description": "Path to selectors CSV file (optional, uses latest if not provided)"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="scrape_with_selected_selectors",
            description="Scrape webpage content using selectors from a selected_selectors CSV file",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"},
                    "selected_csv": {"type": "string", "description": "Path to selected selectors CSV file (optional, uses latest if not provided)"},
                    "wait_for": {"type": "string", "description": "CSS selector or text to wait for before scraping (optional)"},
                    "wait_timeout": {"type": "number", "description": "Timeout in seconds to wait for elements (default: 30)", "default": 30}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="execute_javascript",
            description="Execute JavaScript code on a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to execute JavaScript on"},
                    "code": {"type": "string", "description": "JavaScript code to execute"},
                    "wait_for_result": {"type": "boolean", "description": "Wait for and return the result (default: true)", "default": True}
                },
                "required": ["url", "code"]
            }
        ),
        types.Tool(
            name="click_element",
            description="Click on an element on a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the page"},
                    "selector": {"type": "string", "description": "CSS selector of element to click (optional)"},
                    "text": {"type": "string", "description": "Text of element to click (optional)"},
                    "wait_after_click": {"type": "number", "description": "Seconds to wait after clicking (default: 1)", "default": 1}
                },
                "required": ["url"]
            }
        ),
        types.Tool(
            name="fill_form",
            description="Fill out form fields on a webpage",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the page with the form"},
                    "fields": {"type": "object", "description": "Dictionary of field selectors/names and their values"},
                    "submit": {"type": "boolean", "description": "Whether to submit the form after filling (default: false)", "default": False},
                    "submit_selector": {"type": "string", "description": "CSS selector for submit button (optional)"}
                },
                "required": ["url", "fields"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[types.TextContent, types.ImageContent]]:
    """Handle tool calls"""
    try:
        if name == "scrape_website":
            result_dict = await zendriver_manager.scrape_website(**arguments)
            return [types.TextContent(type="text", text=result_dict["result"])]
        
        elif name == "select_selectors":
            result = await zendriver_manager.select_selectors(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "scrape_with_selected_selectors":
            result = await zendriver_manager.scrape_with_selected_selectors(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "execute_javascript":
            result = await zendriver_manager.execute_javascript(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "click_element":
            result = await zendriver_manager.click_element(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        elif name == "fill_form":
            result = await zendriver_manager.fill_form(**arguments)
            return [types.TextContent(type="text", text=result)]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main entry point for the MCP server"""
    try:
        # Run the server using stdio transport
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="zendriver-scraper",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        await zendriver_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())