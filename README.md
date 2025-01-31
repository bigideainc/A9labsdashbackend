# A9 Labs API

## Overview

A9 Labs API is a FastAPI-based service designed for managing and deploying machine learning models, tracking jobs, and integrating with Weights & Biases (W&B), Firebase Firestore, and Hugging Face.

We have also created a Docker image for streamlined deployment:  
`docker pull tobiusbates/a9labs:latest`

## Features

- **Job Management:** Create, update, and track AI/ML jobs.
- **Model Registry:** Store and manage models in Firestore.
- **Compute Resource Allocation:** Assign compute nodes to ML tasks.
- **Deployment:** Deploy models to remote compute nodes using SSH and Docker.
- **Monitoring & Logging:** Integrated with W&B for experiment tracking.

## Technologies Used

- **FastAPI** – High-performance Python web framework.
- **Firebase Firestore** – NoSQL cloud database for storing jobs and models.
- **Weights & Biases (W&B)** – Model tracking and logging.
- **Hugging Face Transformers** – Model hosting and inference.
- **Ngrok** – Remote access tunneling for secure connections.
- **Paramiko** – SSH management for compute resources.
- **Docker** – Containerized deployment.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Firebase Firestore credentials (`creds.json`)
- Weights & Biases API Key
- Hugging Face API Token

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-org/a9labs-api.git
   cd a9labs-api
   ```

2. Create a virtual environment and install dependencies:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Set environment variables:

   ```sh
   export WANDB_API_KEY="your_wandb_api_key"
   export HF_TOKEN="your_huggingface_api_token"
   ```

4. Start the FastAPI server:

   ```sh
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Deploying with Docker

We have built a Docker image for easy deployment:

```sh
docker pull tobiusbates/a9labs:latest
docker run --gpus all -d -p 8000:8000 tobiusbates/a9labs:latest
```

## API Endpoints

### Job Management

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/jobs/create` | Create a new ML job |
| `POST` | `/jobs/update-status` | Update job status based on W&B runs |
| `POST` | `/jobs/delete` | Delete a job |
| `POST` | `/jobs/submit` | Submit a completed job for review |
| `POST` | `/jobs/reward` | Reward a miner for job completion |

### Model Registry

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/models/add-models` | Add a new ML model |
| `POST` | `/models/catalogue` | Retrieve all registered models |

### Deployment

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/deploy_model/` | Deploy a model to a compute node |
| `POST` | `/update_container_status/` | Update the status of a deployed container |
| `POST` | `/user_containers/` | Get a user's active containers |

## Development & Contribution

1. Fork the repository and create a feature branch.
2. Follow PEP8 coding style.
3. Submit a pull request with a detailed description.

## License

This project is licensed under the MIT License.

---

For inquiries, contact `support@a9labs.com`.