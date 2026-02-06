# Autonomous Web Agent (Gemini + Playwright)

An autonomous AI agent capable of navigating websites, clicking elements, filling forms, and self-correcting errors using **Gemini 2.5 Flash** and **Playwright**.

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.11+
* Google Gemini API Key

### 2. Installation
Run the following commands in your terminal to set up the environment:

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install Python dependencies
pip install langchain langchain-google-genai langchain-community langgraph playwright beautifulsoup4

# Install browser binaries for Playwright
playwright install
```

### 3. Configuration
Create a file named `config.py` in the project root and add your key:

```python
# config.py
API_KEY = "your_google_api_key_here"
```

## 🏃‍♂️ Usage

Run the agent:

```bash
python web-agent.py
```

**Example Commands:**
* "Navigate to https://www.facemash.in"
* "Login with username 'test' and password 'test'"
* "Find the Logout button and click it"

## ⚠️ Notes
* **Crash-Safe:** The agent uses custom robust tools. If a selector fails, it will report the error instead of crashing.
* **Async Mode:** The browser runs asynchronously. Do not manually close the browser window while the agent is thinking.
