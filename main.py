from fastapi import FastAPI, HTTPException, Depends, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from firebase_admin import firestore, credentials
import firebase_admin
import os
import math
import wandb
import pandas as pd
import json
from datetime import datetime
import uuid
from utils.utils import transform_csv_to_hf_dataset,calculate_system_requirements
from urllib.parse import urlparse
import paramiko
import logging
import uvicorn
import ngrok
import time
import asyncio


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
HF_TOKEN = "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"
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

# ngrok_token = os.getenv("NGROK_TOKEN")
# if not ngrok_token:
#     raise ValueError("The NGROK_TOKEN environment variable is not set.")

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not firebase_admin._apps:
    cred = credentials.Certificate("creds.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

def generate_job_id():
    return str(uuid.uuid4())

def sanitize_value(value):
    """Sanitize numeric values for JSON compatibility."""
    if value is None:
        return None
    try:
        sanitized = float(value)
        if math.isnan(sanitized) or math.isinf(sanitized):
            return None  # Replace invalid values with None
        return sanitized
    except (ValueError, TypeError):
        return None

def sanitize_response(data):
    """Recursively sanitize all values in the response."""
    if isinstance(data, dict):
        return {key: sanitize_response(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_response(item) for item in data]
    elif isinstance(data, (int, float)):
        return sanitize_value(data)
    return data

def log_debug_data(data, filename="debug_data.json"):
    """Log problematic data to a file for debugging."""
    try:
        with open(filename, "w") as debug_file:
            json.dump(data, debug_file, indent=4, default=str)
    except Exception as e:
        print(f"Failed to log debug data: {e}")
# Dependency to initialize W&B API
def get_wandb_api() -> wandb.Api:
    return wandb.Api()


def store_job_in_db(job_data):
    try:
        db.collection("a9jobs").document(job_data["job_id"]).set(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing job: {str(e)}")

@app.post("/projects")
async def list_projects(api: wandb.Api = Depends(get_wandb_api)):
    try:
        projects = [project.name for project in api.projects()]
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")

@app.post("/projects/active-summary")
async def get_active_projects_summary(api: wandb.Api = Depends(get_wandb_api)):
    try:
        # Fetch jobs with "running" status from Firestore
        jobs_ref = db.collection("a9jobs").where("status", "==", "running")
        active_jobs = {job.to_dict()["job_id"] for job in jobs_ref.stream()}

        active_projects = []
        total_runs = 0
        completed_runs = 0

        for project in api.projects():
            project_data = {"name": project.name, "active_runs": [], "completed_runs": 0}
            for run in api.runs(project.name):
                if run.state == "running":
                    project_data["active_runs"].append(run.name)
                elif run.state == "finished":
                    project_data["completed_runs"] += 1
            
            if project_data["active_runs"]:
                # Check if the project is associated with a running job in Firestore
                if project.name in active_jobs:
                    active_projects.append(project_data)
                    total_runs += len(project_data["active_runs"]) + project_data["completed_runs"]
                    completed_runs += project_data["completed_runs"]

        # Calculate completion rate
        completion_rate = (completed_runs / total_runs) * 100 if total_runs > 0 else 0

        response = {
            "active_project_count": len(active_projects),
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "completion_rate": completion_rate,
        }

        return JSONResponse(content=response)

    except Exception as e:
        error_message = f"Error fetching active projects summary: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/projects/active")
async def get_active_projects(api: wandb.Api = Depends(get_wandb_api)):
    try:
        active_projects = []
        for project in api.projects():
            project_data = {"name": project.name, "active_runs": []}
            for run in api.runs(project.name):
                if run.state in ["running", "killed"]:
                    project_data["active_runs"].append({
                        "run_name": run.name,
                        "state": run.state,
                    })
            if project_data["active_runs"]:
                active_projects.append(project_data)
        return {"active_projects": active_projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching active projects: {str(e)}")

@app.post("/projects/plots")
async def get_project_plots(
    project_name: str = Form(...),
    api: wandb.Api = Depends(get_wandb_api)
):
    """Fetch plot data, leaderboard, and additional project details from the database."""
    try:
        # Fetch all runs for the given project
        runs = [run for run in api.runs(project_name) if run.state in ["running", "killed", "crashed"]]
        
        plot_metrics = {
            'train/grad_norm': 'grad_norm',
            'train/loss': 'loss',
            'train/global_step': 'global_step',
            'train/epoch': 'epoch',
            'train/learning_rate': 'learning_rate'
        }
        
        grouped_plot_data = {metric: {} for metric in plot_metrics.values()}
        leaderboard_data = []

        # Initialize additional variables
        active_miners = 0  # Count active runs (miners)
        model_status = None  # Model status from the database
        dataset_size = None
        minimum_specs = None
        recommended_specs = None
        model_name = None
        days_running = None
        first_place = None
        second_place = None
        third_place = None
        training_period = None
        model_description=None

        # Fetch job data from Firestore
        try:
            job_query = db.collection("a9jobs").where("job_id", "==", project_name).limit(1)
            job_doc = next(job_query.stream(), None)  # Get the first matching document

            if job_doc:
                job_data = job_doc.to_dict()
                model_status = job_data.get("status")
                dataset_size = job_data.get("dataset_size_mb")
                minimum_specs = job_data.get("minimum_specs")
                recommended_specs = job_data.get("recommended_specs")
                model_name = job_data.get("title")
                model_description = job_data.get("description")

                date_created = job_data.get("date_created")
                if date_created:
                    date_created_dt = datetime.fromisoformat(date_created)
                    days_running = (datetime.utcnow() - date_created_dt).days

                    # Check W&B states to determine the training period
                    end_date = None
                    for run in runs:
                        if run.state in ["finished", "killed", "crashed"]:
                            end_date = datetime.fromtimestamp(run.summary.get("_timestamp", datetime.utcnow().timestamp()))
                            break

                    if end_date:
                        training_period = f"{date_created} to {end_date.isoformat()}"
                    elif any(run.state == "running" for run in runs):
                        training_period = f"{date_created} to {datetime.utcnow().isoformat()} (ongoing)"
                    else:
                        training_period = "No active or completed runs found."
        except Exception as e:
            print(f"Error fetching job details from Firestore: {e}")

        try:
            prize_query = db.collection("prizes").where("job_id", "==", project_name).limit(1)
            prize_doc = next(prize_query.stream(), None)

            if prize_doc:
                prize_data = prize_doc.to_dict()
                first_place = prize_data.get("first_place")
                second_place = prize_data.get("second_place")
                third_place = prize_data.get("third_place")
        except Exception as e:
            print(f"Error fetching prize details from Firestore: {e}")
        # Process runs to generate plot data and leaderboard
        for run in runs:
            try:
                history = run.history()
                if history.empty:
                    continue

                for metric, metric_key in plot_metrics.items():
                    if run.name not in grouped_plot_data[metric_key]:
                        grouped_plot_data[metric_key][run.name] = {"x": [], "y": []}

                    for _, row in history.iterrows():
                        step = sanitize_value(row.get('_step', 0))
                        value = sanitize_value(row.get(metric, 0))

                        grouped_plot_data[metric_key][run.name]["x"].append(step or 0)
                        grouped_plot_data[metric_key][run.name]["y"].append(value or 0)

                # Calculate leaderboard data
                summary = run.summary
                leaderboard_entry = {
                    "name": run.name,
                    "final_loss": sanitize_value(summary.get("final_loss")),
                    "train_loss": sanitize_value(summary.get("train_loss")),
                    "learning_rate": sanitize_value(summary.get("train/learning_rate")),
                    "steps_completed": sanitize_value(history["_step"].max() if "_step" in history else 0),
                    "epochs_completed": sanitize_value(history["train/epoch"].max() if "train/epoch" in history else 0),
                    "runtime_hours": sanitize_value(row.get("_runtime", 0) / 3600)
                }
                leaderboard_data.append(leaderboard_entry)

                # Count active miners
                if run.state == "running":
                    active_miners += 1
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")

        # Sort leaderboard by final loss
        leaderboard_data.sort(key=lambda x: x["final_loss"] or float("inf"))
        leaderboard = [{"rank": i + 1, **entry} for i, entry in enumerate(leaderboard_data)]

        # Prepare response data
        response_data = {
            "grouped_plot_data": grouped_plot_data,
            "leaderboard": leaderboard,
            "metrics_available": list(plot_metrics.values()),
            "database_info": {
                "dataset_size_mb": dataset_size,
                "minimum_specs": minimum_specs,
                "recommended_specs": recommended_specs,
                "model_name": model_name,
                "model_description": model_description,
                "days_running": days_running,
                "active_miners": active_miners,
                "model_status": model_status,
                "training_period": training_period,
            },
            "prize_info": {
                "first_place": first_place,
                "second_place": second_place,
                "third_place": third_place,
            },
        }

        # Log debug data
        log_debug_data(response_data)

        return sanitize_response(response_data)

    except Exception as e:
        error_message = f"Error fetching project plots: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/jobs")
async def get_open_jobs(status: str = Form(...)):
    try:
        jobs_ref = db.collection("a9jobs").where("status", "==", f"{status}")
        open_jobs = [job.to_dict() for job in jobs_ref.stream()]
        return {"open_jobs": open_jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching open jobs: {str(e)}")

@app.post("/jobs/prizes/add")
async def add_prizes_for_job(
    job_id: str = Form(...),
    first_place: str = Form(...),
    second_place: str = Form(...),
    third_place: str = Form(...),
):
    """
    Add prizes for a specific job in the Firestore 'prizes' collection.
    """
    try:
        # Check if the job exists in the 'a9jobs' collection
        job_query = db.collection("a9jobs").where("job_id", "==", job_id).limit(1)
        job_doc = next(job_query.stream(), None)

        if not job_doc:
            raise HTTPException(status_code=404, detail=f"Job with job_id '{job_id}' not found.")

        # Prepare prize data
        prize_data = {
            "job_id": job_id,
            "first_place": first_place,
            "second_place": second_place,
            "third_place": third_place,
            "date_added": datetime.utcnow().isoformat()
        }

        # Add or update the prizes in the 'prizes' collection
        db.collection("prizes").document(job_id).set(prize_data)

        return {
            "prizes": prize_data,
        }

    except Exception as e:
        error_message = f"Error adding prizes for job_id '{job_id}': {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.delete("/jobs/prizes/delete")
async def delete_prize(job_id: str = Form(...)):
    """
    Delete a prize entry from the Firestore 'prizes' collection based on the job_id.
    """
    try:
        # Fetch the prize document using the job_id
        prize_doc = db.collection("prizes").document(job_id).get()

        if not prize_doc.exists:
            raise HTTPException(status_code=404, detail=f"Prize for job_id '{job_id}' not found.")

        # Delete the document from Firestore
        db.collection("prizes").document(job_id).delete()

        return {
            "message": f"Prize for job_id '{job_id}' deleted successfully."
        }

    except Exception as e:
        error_message = f"Error deleting prize for job_id '{job_id}': {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/jobs/create")
async def create_job(
    user_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        dataset_size = os.path.getsize(file_path) / (1024 ** 2) 
        job_id = generate_job_id()
        dataset_id = f"A9-Labs/{job_id}"
        transform_csv_to_hf_dataset(file_path, repo_name=dataset_id)
        specs = calculate_system_requirements(dataset_size)

        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "title": title,
            "description": description,
            "date_created": datetime.utcnow().isoformat(),
            "status": "open",
            "dataset_id": dataset_id,
            "dataset_size_mb": dataset_size,
            "minimum_specs": specs["minimum"],
            "recommended_specs": specs["recommended"],
        }
        store_job_in_db(job_data)
        os.remove(file_path)
        return {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "dataset_size_mb": dataset_size,
            "minimum_specs": specs["minimum"],
            "recommended_specs": specs["recommended"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")
# tcp://0.tcp.us-cal-1.ngrok.io:18909
@app.post("/jobs/update_payment_mode")
async def update_job(job_id: str = Form(...), status: str = Form(...)):
    try:
        jobs_collection = db.collection("a9jobs")
        query = jobs_collection.where("job_id", "==", job_id).limit(1)
        docs = query.stream()
        job_doc = next(docs, None)

        if not job_doc:
            raise HTTPException(status_code=404, detail=f"Job with job_id {job_id} not found.")
        
        job_doc.reference.update({"payment_mode": status})
        return {"message": f"Job {job_id} status updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating job: {str(e)}")
@app.post("/jobs/update-status")
async def update_job_status_based_on_runs(api: wandb.Api = Depends(get_wandb_api)):
    """
    Check W&B projects and update job statuses in the Firestore database based on the runs' states.
    """
    try:
        projects = api.projects()
        updated_jobs = []

        for project in projects:
            job_id = project.name  # Assuming the W&B project name corresponds to the Firestore job_id
            runs = api.runs(project.name)

            states = [run.state for run in runs]
            running_runs = any(state == "running" for state in states)
            finished_runs = any(state in ["finished", "completed"] for state in states)
            problematic_runs = all(state in ["crashed", "killed"] for state in states)

            if running_runs:
                # Skip projects with running runs
                print(f"Skipping job_id '{job_id}' as it has active running runs.")
                continue

            job_query = db.collection("a9jobs").where("job_id", "==", job_id).limit(1)
            job_doc = next(job_query.stream(), None)

            if not job_doc:
                print(f"Job with job_id '{job_id}' not found in Firestore.")
                continue

            # Update job status based on the states
            job_data = job_doc.to_dict()
            job_update_data = {}

            if problematic_runs:
                # If all runs are crashed or killed, update status to "open"
                job_update_data["status"] = "open"
                print(f"Updated job_id '{job_id}' to status 'open'.")
            elif finished_runs and not running_runs:
                # If at least one run is finished and no runs are running, update status to "closed"
                job_update_data["status"] = "closed"
                job_update_data["end_date"] = datetime.utcnow().isoformat()
                job_update_data["payment_mode"] = "pending_payment"
                print(f"Updated job_id '{job_id}' to status 'closed' with pending payment.")

            if job_update_data:
                # Apply updates to Firestore
                job_doc.reference.update(job_update_data)
                updated_jobs.append({job_id: job_update_data})

        return {"message": "Job statuses updated successfully.", "updated_jobs": updated_jobs}

    except Exception as e:
        error_message = f"Error updating job statuses: {e}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/jobs/delete")
async def delete_job(job_id: str = Form(...)):
    try:
        job_ref = db.collection("a9jobs").document(job_id)
        job = job_ref.get()
        if not job.exists:
            raise HTTPException(status_code=404, detail="Job not found.")
        job_ref.delete()
        return {"message": f"Job {job_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")


## miner Operations
@app.post("/jobs/submit")
async def submit_job(
    job_id: str = Form(...),
    miner_id: str = Form(...),
    platform: str = Form(...),
    huggingFacerepoId: str = Form(...),
    loss: float = Form(...),
    totalPipelineTime: float = Form(...),
    accuracy: float = Form(...)
):
    """
    Submit job details and store them in Firestore.
    """
    try:
        # Validate that the job_id exists in Firestore
        job_query = db.collection("a9jobs").where("job_id", "==", job_id).limit(1)
        job_doc = next(job_query.stream(), None)

        if not job_doc:
            raise HTTPException(status_code=404, detail=f"Job with job_id '{job_id}' not found.")

        submission_id = f"{job_id}_{miner_id}"
        # Prepare data to be stored
        submission_data = {
            "job_id": job_id,
            "miner_id": miner_id,
            "platform": platform,
            "huggingFacerepoId": huggingFacerepoId,
            "loss": loss,
            "status":"pending_payment",
            "totalPipelineTime": totalPipelineTime,
            "accuracy": accuracy,
            "completedAt": datetime.utcnow().isoformat(),
            "submission_id":submission_id
        }

        # Save to Firestore under a new document in the 'submissions' collection
        
        db.collection("completed_jobs").document(submission_id).set(submission_data)

        return {
            "submission_data": submission_data
        }

    except Exception as e:
        error_message = f"Error submitting job: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/jobs/reward")
async def reward_miner(
    submission_id: str = Form(...),
    reward: float = Form(...)
):
    """
    Reward a miner by updating the status to 'paid' and adding the reward.
    """
    try:
        # Query Firestore to find the submission by submission_id
        submission_doc = db.collection("completed_jobs").document(submission_id).get()

        if not submission_doc.exists:
            raise HTTPException(status_code=404, detail=f"Submission with submission_id '{submission_id}' not found.")

        # Update the status to 'paid' and add the reward
        update_data = {
            "status": "paid",
            "reward": reward,
            "rewardedAt": datetime.utcnow().isoformat()
        }
        db.collection("completed_jobs").document(submission_id).update(update_data)

        return {
            "message": f"Miner rewarded successfully for submission_id '{submission_id}'.",
            "updated_data": update_data
        }

    except Exception as e:
        error_message = f"Error rewarding miner for submission_id '{submission_id}': {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

### Inference distributor 
@app.post("/nodes/online")
async def get_online_nodes():
    """
    Fetch and return all nodes with status 'online' from the Firestore 'compute_nodes' collection.
    """
    try:
        # Query Firestore for nodes with status 'online'
        online_nodes_query = db.collection("compute_nodes").where("status", "==", "online")
        online_nodes = [node.to_dict() for node in online_nodes_query.stream()]

        if not online_nodes:
            return {
                "online_nodes": []
            }

        return {
            "online_nodes": online_nodes
        }

    except Exception as e:
        error_message = f"Error fetching online nodes: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


# models from huggingface
@app.post("/models/add-models")
async def add_models(
    name: str = Form(...),
    provider: str = Form(...),
    modelCategory: str = Form(...),
    ram: str = Form(...),
    cpu: str = Form(...),
    gpu: str = Form(...),
    storage: str = Form(...),
    parameters: str = Form(...),
    contextsize: str = Form(...),
    precision: str = Form(...),
):
    """
    Add a HuggingFace model to the Firestore database.

    Args:
        name (str): HuggingFace model repository ID.
        provider (str): The model provider (e.g., HuggingFace).

    Returns:
        dict: The added model details.
    """
    capabilities=[]

    try:
        model_id = generate_job_id()
        model_spec={
            "precision":precision,
            "contextsize":contextsize,
            "parameters":parameters
        }
        model_requirements={
            "cpu":cpu,
            "ram":ram,
            "gpu":gpu,
            "storage":storage
        }
        get_model_capabilities={"type":modelCategory}
        capabilities.append(get_model_capabilities)
        # Prepare model data for Firestore
        model_data = {
            "name": name,
            "provider": provider,
            "capabilities":capabilities,
            "specifications":model_spec,
            "requirements":model_requirements,
        }

        # Store model data in Firestore
        db.collection("model_catelog").document(model_id).set(model_data)

        return {
            name: model_data
        }

    except Exception as e:
        error_message = f"Error adding model: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/models/catalogue")
async def get_models():
    """
    Retrieve all models from the model catalog collection.
    
    Returns:
        list: A list of models with their details
    """
    try:
        # Fetch all documents from the model_catelog collection
        models_ref = db.collection("model_catelog")
        models_snapshot = models_ref.get()
        
        # Convert snapshot to list of models
        models_list = []
        for doc in models_snapshot:
            model_data = doc.to_dict()
            model_data['id'] = doc.id  # Include document ID
            models_list.append(model_data)
        
        return {
            "total_models": len(models_list),
            "models": models_list
        }
    
    except Exception as e:
        error_message = f"Error retrieving models: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/deploy_model/")
async def deploy_model(
    deployment_name: str = Form(), 
    model: str = Form(),
    compute_id: str = Form(),
    user_id: str = Form()
):
    ssh_username = ""
    ssh_password = ""
    ngrok_ssh_url = ""
    ngrok_token = ""
    initial_port = 8094
    
    try:
        # Fetch Miner and Compute Resource Credentials
        miners_ref = db.collection('miners')
        miner_doc_ref = miners_ref.where('id', '==', compute_id).get()
        
        if not miner_doc_ref:
            raise HTTPException(status_code=404, detail="Miner not found")

        miner_details = miner_doc_ref[0].to_dict()
        compute_resource_id = miner_details.get("compute_resources", [None])[0]
        
        fetch_resource_ref = db.collection('compute_resources').document(compute_resource_id).get()
        if not fetch_resource_ref.exists:
            raise HTTPException(status_code=404, detail="Compute resource not found")

        creds = fetch_resource_ref.to_dict().get('network', {})
        ssh_username = creds.get("username", "")
        ssh_password = creds.get("password", "")
        ngrok_token = creds.get("ngrok_token", "")
        ngrok_ssh_url = creds.get("ssh", "")

        # Validate Deployment Name
        running_nodes_ref = db.collection('running_container')
        existing_deployments = running_nodes_ref.where('endpoint', '==', deployment_name).get()
        if existing_deployments:
            raise HTTPException(status_code=400, detail="Deployment name already in use")

        # Establish SSH Connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        ssh_host = urlparse(ngrok_ssh_url).hostname
        ssh_port = urlparse(ngrok_ssh_url).port or 22

        ssh.connect(
            hostname=ssh_host, 
            port=ssh_port, 
            username=ssh_username, 
            password=ssh_password
        )

        # Find Free Port
        def is_port_free(port):
            _, stdout, _ = ssh.exec_command(f"netstat -tuln | grep :{port}")
            return len(stdout.read().decode().strip()) == 0

        port = initial_port
        while not is_port_free(port):
            port += 1

        # Pull Docker Image
        pull_command = "docker pull tobiusbates/fastapi-agent:latest"
        _, stdout, stderr = ssh.exec_command(pull_command)
        pull_output = stdout.read().decode() + stderr.read().decode()
        
        if "error" in pull_output.lower():
            raise HTTPException(status_code=500, detail=f"Image pull failed: {pull_output}")

        # Run Docker Container
        docker_command = (
            f"docker run -d -t "
            f"-e PORT={port} "
            f"-e ENDPOINT={deployment_name} "
            f"-e MODEL_={model} "
            f"-e NGROK_TOKEN={ngrok_token} "
            f"-p {port}:{port} "
            "tobiusbates/fastapi-agent:latest"
        )

        _, stdout, stderr = ssh.exec_command(docker_command)
        container_output = stderr.read().decode()
        
        if "Error" in container_output:
            raise HTTPException(status_code=500, detail=f"Docker deployment failed: {container_output}")

        container_id = stdout.read().decode().strip()

        # Wait for Port to be in Use
        while True:
            _, stdout, _ = ssh.exec_command(f"netstat -tuln | grep :{port}")
            if len(stdout.read().decode().strip()) > 0:
                break
            time.sleep(5)

        # Wait for Firestore Update
        retry_count, max_retries = 0, 60
        serveo_url = None

        while retry_count < max_retries:
            existing_docs = running_nodes_ref.where('endpoint', '==', deployment_name).get()
            if existing_docs:
                doc = existing_docs[0]
                doc_id = doc.id
                serveo_url = doc.to_dict().get("ngrok_url")

                if serveo_url:
                    running_nodes_ref.document(doc_id).update({
                        'container_id': container_id,
                        'host': ssh_username,
                        'user_id': user_id,
                        'status': "active"
                    })
                    break

            retry_count += 1
            time.sleep(5)

        if not serveo_url:
            raise HTTPException(status_code=500, detail="Timeout waiting for Firestore update")

        return {
            "message": "Deployment successful",
            "port": port,
            "container_id": container_id,
            "service_url": serveo_url,
        }

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if 'ssh' in locals():
                ssh.close()
        except Exception as e:
            logger.error(f"Error closing SSH connection: {e}")

@app.post("/user_containers/")
async def get_user_containers(user_id: str = Form(...)):
    try:
        # Query Firestore for containers associated with the user_id
        running_nodes_ref = db.collection('running_container')
        user_containers = running_nodes_ref.where('user_id', '==', user_id).get()
        
        # Transform documents into list of container details
        containers_list = [
            {
                'endpoint': doc.to_dict().get('endpoint', ''),
                'container_id': doc.to_dict().get('container_id', ''),
                'status': doc.to_dict().get('status', ''),
                'ngrok_url': doc.to_dict().get('ngrok_url', ''),
                'model': doc.to_dict().get('model', '')
            } for doc in user_containers
        ]
        
        return {
            "user_id": user_id,
            "total_containers": len(containers_list),
            "containers": containers_list
        }
    
    except Exception as e:
        logger.error(f"Error retrieving user containers: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user containers")

@app.post("/update_container_status/")
async def update_container_status(
    endpoint: str = Form(...),
    new_status: str = Form(...)
):
    try:
        # Find the document with the specified endpoint
        running_nodes_ref = db.collection('running_container')
        container_docs = running_nodes_ref.where('endpoint', '==', endpoint).get()
        
        # Check if container exists
        if not container_docs:
            raise HTTPException(status_code=404, detail="Container not found")
        
        # Get the first (should be only) document
        container_doc = container_docs[0]
        
        # Update the status
        running_nodes_ref.document(container_doc.id).update({
            'status': new_status
        })
        
        return {
            "endpoint": endpoint,
            "new_status": new_status,
            "message": "Container status updated successfully"
        }
    
    except Exception as e:
        logger.error(f"Error updating container status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update container status")

# async def start_ngrok(port):
#     listener = await ngrok.connect(port, 
#         authtoken=ngrok_token,
#         proto="http"
#     )
#     ngrok_url = listener.url()
#     return ngrok_url

# async def main():
#     port = int(os.getenv('PORT', 8093))
    
#     try:
#         ngrok_url = await start_ngrok(port)
#         logger.info(f"Ngrok tunnel created: {ngrok_url}")
        
#         config = uvicorn.Config(app, host="0.0.0.0", port=port)
#         server = uvicorn.Server(config)
#         await server.serve()
    
#     except Exception as e:
#         logger.error(f"Error setting up Ngrok tunnel: {e}")
#         raise

# if __name__ == "__main__":
#     asyncio.run(main())