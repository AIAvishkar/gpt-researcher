# Usage Guide

## Setup Procedure

* Setup virtual env

  ```
  python3 -m venv venv
  ```

* Activate virtual env

  ```
  source venv/bin/activate
  ```

* Install packages
  
  ```
  pip install -r requirements.txt
  ```

* Create an `.env` file with following content
  ```
  TAVILY_API_KEY=<tavily_key>
  OPENAI_API_KEY=<open_api_key>
  ```

* Run server

  ```
  python main.py
  ```

## Testing

* Do an API call to trigger research

  ```
  curl -X POST http://localhost:8001/research   -H "Content-Type: application/json"   -d '{"query": "What is quantum computing?", "tone": "objective"}'
  ```

* Do an API call to check status of the research

  ```
  curl -X GET http://localhost:8001/research/342f4307-116d-4742-b169-baccb1e8d18d/status
  ```