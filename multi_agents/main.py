from asyncio.log import logger
import datetime
from enum import Enum
from dotenv import load_dotenv
import sys
import os
import uuid
import re

# from .utils import sanitize_filename
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_agents.agents import ChiefEditorAgent
import asyncio
import json
from gpt_researcher.utils.enum import Tone
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

# Create a dictionary to store task status
research_tasks = {}

# Run with LangSmith if API key is set
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

app = FastAPI(title="GPT Researcher API")

# Create a Pydantic-compatible enum
class ToneEnum(str, Enum):
    OBJECTIVE = "objective"
    BALANCED = "balanced"
    CRITICAL = "critical"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"

class TaskConfig(BaseModel):
    query: str
    max_sections: int = 3
    publish_formats: dict = {
        "markdown": True,
        "pdf": True,
        "docx": True
    }
    include_human_feedback: bool = False
    follow_guidelines: bool = False
    model: str = "gpt-4o"
    guidelines: list = [
        "The report MUST be written in APA format",
        "Each sub section MUST include supporting sources using hyperlinks. If none exist, erase the sub section or rewrite it to be a part of the previous section"
    ]
    verbose: bool = True


class ResearchRequest(TaskConfig):
    tone: ToneEnum = ToneEnum.OBJECTIVE
    output_directory_path: str = "./outputs/"

    def get_tone(self) -> Tone:
        return Tone[self.tone.upper()]
    
    def to_task_config(self) -> dict:
        # Convert ResearchRequest to TaskConfig format
        # Exclude the additional fields that are specific to ResearchRequest
        return self.model_dump(exclude={'tone', 'output_directory_path'})

def open_task(request: ResearchRequest):
    task: TaskConfig = request.to_task_config()
    logging.debug("task - ", task)

    # Override model with STRATEGIC_LLM if defined in environment
    strategic_llm = os.environ.get("STRATEGIC_LLM")
    if strategic_llm and ":" in strategic_llm:
        # Extract the model name (part after the first colon)
        model_name = strategic_llm.split(":", 1)[1]
        task["model"] = model_name
    elif strategic_llm:
        task["model"] = strategic_llm

    return task

async def execute_research_task(task: dict, tone: ToneEnum, output_directory_path: str, task_id: uuid.UUID):
    try:
        chief_editor = ChiefEditorAgent(task=task, websocket=None, stream_output=None, tone=tone, headers=None, output_directory_path=output_directory_path, task_id=task_id) # output_directory_path=output_directory_path, task_id=task_id
        research_tasks[task_id] = {"status": "running", "started_at": datetime.datetime.now()}
        
        await chief_editor.run_research_task()
        
        research_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.datetime.now()
        })
        logging.info(f"Research completed successfully for task {task_id}")
        
    except Exception as e:
        research_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.datetime.now()
        })
        logging.error(f"Research failed for task {task_id}: {str(e)}")

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a given filename by replacing characters that are invalid 
    in Windows file paths with an underscore ('_').

    This function ensures that the filename is compatible with all 
    operating systems by removing or replacing characters that are 
    not allowed in Windows file paths. Specifically, it replaces 
    the following characters: < > : " / \\ | ? *

    Parameters:
    filename (str): The original filename to be sanitized.

    Returns:
    str: The sanitized filename with invalid characters replaced by an underscore.
    
    Examples:
    >>> sanitize_filename('invalid:file/name*example?.txt')
    'invalid_file_name_example_.txt'
    
    >>> sanitize_filename('valid_filename.txt')
    'valid_filename.txt'
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


@app.post("/research")
async def create_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    try:
        task_id = uuid.uuid4()
        task = open_task(request)
        
        # Add task to background tasks
        background_tasks.add_task(
            execute_research_task,
            task=task,
            tone=request.tone,
            output_directory_path=request.output_directory_path,
            task_id=task_id
        )

        documents_dir_path = request.output_directory_path + \
            sanitize_filename(
                f"run_{task_id}")
        
        return {
            "status": "accepted",
            "task_id": str(task_id),
            "message": "Research task started successfully",
            "documents_dir_path": documents_dir_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/{task_id}/status")
async def get_research_status(task_id: str):
    try:
        task_uuid = uuid.UUID(task_id)
        if task_uuid not in research_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task_id,
            **research_tasks[task_uuid]
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")


# @app.websocket("/ws/research")
# async def websocket_research(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         data = await websocket.receive_json()
#         research_report = await run_research_task(
#             query=data["query"],
#             websocket=websocket,
#             stream_output=True,
#             tone=Tone(data.get("tone", Tone.Objective))
#         )
#         await websocket.send_json({"status": "complete", "report": research_report})
#     except Exception as e:
#         await websocket.send_json({"status": "error", "message": str(e)})
#     finally:
#         await websocket.close()

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    
    # Get host and port from environment variables with defaults
    HOST = os.getenv("GPT_RESEARCHER_HOST", "0.0.0.0")
    PORT = int(os.getenv("GPT_RESEARCHER_PORT", "8001"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    
    # Log the configuration
    logging.info(f"Starting server on {HOST}:{PORT}")
    
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        log_level=LOG_LEVEL
    )