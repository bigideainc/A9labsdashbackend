from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect,UploadFile, Form,File
from typing import List, Dict, Any, AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,JSONResponse
from firebase_admin import firestore, credentials
import firebase_admin
from datasets import DatasetDict
import pandas as pd
import os
import wandb
import json
import asyncio
import numpy as np  
import math
from datetime import datetime
import uuid
from utils.utils import transform_csv_to_hf_dataset

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# W&B API Key
WANDB_API_KEY = "650810c567842db08fc2707d0668dc568cad00b4"
HF_TOKEN="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"
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN
if not firebase_admin._apps:
    cred = credentials.Certificate("creds.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def generate_job_id():
    return str(uuid.uuid4())

def store_job_in_db(job_data):
    try:
        db.collection("a9jobs").document(job_data["job_id"]).set(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing job: {str(e)}")

def sanitize_value(value):
    """
    Sanitize numeric values, converting NaN and None to None
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# Dependency to initialize W&B API once
def get_wandb_api() -> wandb.Api:
    return wandb.Api()


# Fetch all projects
@app.post("/projects", response_model=List[str])
async def list_projects(api: wandb.Api = Depends(get_wandb_api)):
    """
    Endpoint to list all W&B projects available to the user.
    """
    try:
        projects = [project.name for project in api.projects()]
        return projects
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")


@app.websocket("/ws/projects/active")
async def websocket_active_projects(websocket: WebSocket, api: wandb.Api = Depends(get_wandb_api)):
    """
    WebSocket endpoint to continuously stream updates of active/running W&B projects with metadata.
    """
    await websocket.accept()  # Accept the WebSocket connection
    try:
        while True:  # Continuously fetch and send data
            active_projects = []
            try:
                for project in api.projects():
                    # Handle created_at
                    created_at = project.created_at
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.fromisoformat(created_at.replace("Z", ""))
                        except ValueError:
                            created_at = None

                    project_data = {
                        "name": project.name,
                        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else "N/A",
                        "active_runs": []
                    }

                    # Fetch runs for the project
                    runs = api.runs(project.name)
                    for run in runs:
                        if run.state == "killed" or run.state == "running":
                            run_started = run.created_at
                            if isinstance(run_started, str):
                                try:
                                    run_started = datetime.fromisoformat(run_started.replace("Z", ""))
                                except ValueError:
                                    run_started = None

                            if run_started:
                                duration = datetime.now() - run_started
                            else:
                                duration = "N/A"

                            project_data["active_runs"].append({
                                "run_name": run.name,
                                "started_at": run_started.isoformat() if isinstance(run_started, datetime) else "N/A",
                                "duration": str(duration) if duration != "N/A" else "N/A",
                                "state": run.state
                            })

                    if project_data["active_runs"]:
                        active_projects.append(project_data)

                # Send the data to the WebSocket client
                await websocket.send_text(json.dumps({"active_projects": active_projects}))

                # Wait for a few seconds before fetching updates again
                await asyncio.sleep(5)

            except Exception as fetch_error:
                print(f"Error fetching active projects: {fetch_error}")
                await websocket.send_text(json.dumps({"error": "Failed to fetch active projects."}))
                break

    except WebSocketDisconnect:
        print("WebSocket connection disconnected normally.")
    except Exception as websocket_error:
        print(f"WebSocket error: {websocket_error}")
    finally:
        try:
            await websocket.close(code=1000)  # Close the WebSocket gracefully
        except Exception as close_error:
            print(f"Error closing WebSocket: {close_error}")



# Real-time plot graphs for active/running jobs in a project
@app.websocket("/ws/projects/{project_name}/plots")
async def websocket_endpoint(websocket: WebSocket, project_name: str, api: wandb.Api = Depends(get_wandb_api)):
    await websocket.accept()
    try:
        runs = [run for run in api.runs(project_name) if run.state in ['running', 'killed']]

        if not runs:
            await websocket.send_text(json.dumps({"error": "No active jobs found for this project"}))
            return

        while True:
            try:
                plot_data = []
                leaderboard_data = []

                for run in runs:
                    try:
                        # Fetch recent history for plotting
                        history = run.history(samples=100)
                        
                        if not history.empty:
                            for _, row in history.iterrows():
                                row_dict = row.to_dict()
                                row_dict["miner"] = run.name

                                # Sanitize and filter relevant metrics
                                sanitized_row = {
                                    k: sanitize_value(v)
                                    for k, v in row_dict.items()
                                    if k in [
                                        'train/loss', 'train/epoch', 
                                        'train/global_step', 'train/learning_rate', 
                                        'miner'
                                    ]
                                }
                                
                                # Only add row if it has valid data
                                if any(sanitized_row.values()):
                                    plot_data.append(sanitized_row)

                        # Add summary data for leaderboard
                        summary = run.summary
                        leaderboard_entry = {
                            "miner": run.name,
                            "final_loss": sanitize_value(summary.get("final_loss")),
                            "train_loss": sanitize_value(summary.get("train_loss")),
                            "learning_rate": sanitize_value(summary.get("train/learning_rate")),
                            "steps_completed": int(history["_step"].max()) if "_step" in history else 0,
                        }
                        
                        # Only add entry if it has meaningful data
                        if any(v is not None for v in leaderboard_entry.values()):
                            leaderboard_data.append(leaderboard_entry)

                    except Exception as e:
                        print(f"Error fetching data for run {run.name}: {e}")

                # Sort leaderboard data by `final_loss` or `train_loss`
                leaderboard_data = sorted(
                    leaderboard_data,
                    key=lambda x: x["final_loss"] or x["train_loss"] or float("inf"),
                )

                # Send data over WebSocket
                await websocket.send_text(json.dumps({
                    "plot_data": plot_data,
                    "leaderboard": leaderboard_data
                }))
                
                # Wait for 5 seconds before next update
                await asyncio.sleep(5)

            except Exception as inner_e:
                print(f"Inner loop error: {inner_e}")
                break

    except WebSocketDisconnect:
        print("WebSocket connection disconnected normally")
    except Exception as e:
        print(f"Outer WebSocket connection error: {e}")
    finally:
        try:
            await websocket.close(code=1000)
        except Exception as close_e:
            print(f"Error closing WebSocket: {close_e}")


# Display detailed leaderboard for all completed projects
@app.post("/projects/completed")
async def completed_projects_leaderboard(api: wandb.Api = Depends(get_wandb_api)):
    """
    Endpoint to display all completed projects and their details:
    - Datasets used
    - Best miner
    - Leaderboard with rank, miner, and training loss
    """
    try:
        all_projects = api.projects()
        completed_projects = []

        for project in all_projects:
            project_name = project.name
            runs = [run for run in api.runs(project_name) if run.state == "finished"]
            
            if not runs:
                continue  # Skip projects with no completed runs
            
            datasets = set()
            leaderboard = []
            best_miner = None
            best_train_loss = float('inf')

            for run in runs:
                summary = run.summary

                # Collect dataset information
                dataset_name = summary.get("dataset", "Unknown")
                datasets.add(dataset_name)

                # Leaderboard entry
                train_loss = sanitize_value(summary.get("train_loss"))
                entry = {
                    "miner": run.name,
                    "train_loss": train_loss
                }
                leaderboard.append(entry)

                # Determine the best miner by lowest training loss
                if train_loss is not None and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_miner = run.name
            # Sort leaderboard by training loss
            leaderboard = sorted(leaderboard, key=lambda x: x["train_loss"] if x["train_loss"] is not None else float('inf'))

            # Add rank to leaderboard entries
            ranked_leaderboard = [
                {"rank": index + 1, "miner": entry["miner"], "train_loss": entry["train_loss"]}
                for index, entry in enumerate(leaderboard)
            ]

            completed_projects.append({
                "project_name": project_name,
                "datasets_used": list(datasets),
                "best_miner": best_miner,
                "leaderboard": ranked_leaderboard
            })

        return {"completed_projects": completed_projects}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching completed projects: {str(e)}")

# Endpoint: Create a training job
@app.post("/jobs/create")
async def create_job(
    user_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Create a new training job by uploading a dataset and transforming it into a Hugging Face dataset.
    """
    try:
        # Validate uploaded file
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
        # Save the uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Generate unique job ID and process dataset
        job_id = generate_job_id()
        dataset_id = f"Tobius/{job_id}"
        
        # Use the transform_csv_to_hf_dataset function (assume it’s imported)
        transform_csv_to_hf_dataset(file_path, repo_name=dataset_id)
        
        # Prepare job data
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "title": title,
            "description": description,
            "date_created": datetime.utcnow().isoformat(),
            "status": "open",
            "dataset_id": dataset_id
        }
        
        # Store job data in Firestore
        store_job_in_db(job_data)
        
        # Cleanup temporary file
        os.remove(file_path)

        return JSONResponse(content={"job_id": job_id, "dataset_id": dataset_id}, status_code=201)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


# WebSocket: Retrieve open jobs
@app.websocket("/ws/jobs/open")
async def open_jobs_websocket(websocket: WebSocket):
    """
    WebSocket endpoint to continuously stream open jobs with metadata.
    """
    await websocket.accept()
    try:
        while True:
            try:
                # Fetch open jobs from Firestore
                jobs_ref = db.collection("a9jobs").where("status", "==", "open")
                open_jobs = [job.to_dict() for job in jobs_ref.stream()]
                
                # Send open jobs over WebSocket
                await websocket.send_text(json.dumps({"open_jobs": open_jobs}))
                
                # Wait for a few seconds before next update
                await asyncio.sleep(5)
            
            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
                break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        try:
            await websocket.close()
        except Exception as close_e:
            print(f"Error closing WebSocket: {close_e}")

# Endpoint: Update a job
@app.post("/jobs/update")
async def update_job(job_id: str = Form(...), status: str = Form(...)):
    """
    Update the status of a job using job_id as a field in Firestore documents.
    """
    try:
        # Query Firestore to find the document with the matching job_id field
        jobs_collection = db.collection("a9jobs")
        query = jobs_collection.where("job_id", "==", job_id).limit(1)
        docs = query.stream()

        # Extract the document to update
        job_doc = next(docs, None)
        if not job_doc:
            raise HTTPException(status_code=404, detail=f"Job with job_id {job_id} not found.")

        # Update the status field in the document
        job_doc.reference.update({"status": status})

        return JSONResponse(
            content={"message": f"Job {job_id} status updated successfully."},
            status_code=200,
        )

    except Exception as e:
        print(f"Error updating job with job_id {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating job: {str(e)}")

# Endpoint: Delete a job
@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from the database.
    """
    try:
        # Fetch job from Firestore
        job_ref = db.collection("a9jobs").document(job_id)
        job = job_ref.get()
        if not job.exists:
            raise HTTPException(status_code=404, detail="Job not found.")
        job_ref.delete()
        return JSONResponse(content={"message": f"Job {job_id} deleted successfully."}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True, log_level="error")
