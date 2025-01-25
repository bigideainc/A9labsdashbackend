from transformers import AutoConfig
from huggingface_hub import HfApi,list_repo_files
import json
from typing import Dict, Union, Optional
import requests
import paramiko
import time
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException

def get_model_specs(repo_id: str, hf_token: str = None):
    """
    Retrieve HuggingFace model specifications dynamically.

    Args:
        repo_id (str): HuggingFace model repository ID.
        hf_token (str): HuggingFace API token for private repos (optional).

    Returns:
        dict: Model specifications including parameters, context size, and precision.
    """
    try:
        # Load model configuration
        config = AutoConfig.from_pretrained(repo_id, token=hf_token)

        # Extract specifications
        specs = {
            "model_type": config.model_type,
            "hidden_size": getattr(config, "hidden_size", "Unknown"),
            "num_attention_heads": getattr(config, "num_attention_heads", "Unknown"),
            "num_hidden_layers": getattr(config, "num_hidden_layers", "Unknown"),
            "vocab_size": getattr(config, "vocab_size", "Unknown"),
            "max_position_embeddings": getattr(config, "max_position_embeddings", "Unknown"),  # Context size
            "precision": "float32" if not hasattr(config, "torch_dtype") else str(config.torch_dtype),
            # "total_parameters": sum(v.numel() for v in AutoConfig.from_pretrained(repo_id)._parameters.values()),
        }
        return specs
    except Exception as e:
        print(f"Error retrieving model specs: {e}")
        return None

def get_model_deployment_requirements(repo_id: str, hf_token: str = None) -> Dict[str, Union[float, str]]:
    """
    Estimate compute and storage requirements for a HuggingFace model with improved parameter detection.
    
    Args:
        repo_id (str): HuggingFace model repository ID (e.g., 'microsoft/DialoGPT-medium')
        hf_token (str, optional): HuggingFace API token for private repos
        
    Returns:
        dict: Deployment requirements including GPU, RAM, and storage estimates
    """
    try:
        api = HfApi()
        model_info = api.model_info(repo_id, token=hf_token)
        
        # Initialize variables
        model_size = 0.0
        parameter_count = None
        
        # Calculate total model size from all model files
        for sibling in model_info.siblings:
            if sibling.rfilename.endswith(('.bin', '.safetensors', '.pt', '.ckpt')):
                model_size += sibling.size if sibling.size is not None else 0
        
        model_size = model_size / (1024 ** 3)  # Convert to GB
        
        # Try multiple methods to get parameter count
        # Method 1: Direct from card data
        if model_info.cardData and 'parameters' in model_info.cardData:
            parameter_count = float(model_info.cardData['parameters'])
        elif model_info.cardData and 'parameter_count' in model_info.cardData:
            parameter_count = float(model_info.cardData['parameter_count'])
            
        # Method 2: Parse from model configuration
        if parameter_count is None:
            try:
                config_files = [f for f in model_info.siblings if f.rfilename == 'config.json']
                if config_files:
                    config_info = api.hf_hub_download(repo_id, 'config.json', token=hf_token)
                    with open(config_info, 'r') as f:
                        config = json.load(f)
                        # Look for common parameter indicators
                        for key in ['n_parameters', 'num_parameters', 'total_params']:
                            if key in config:
                                parameter_count = float(config[key])
                                break
            except Exception:
                pass
                
        # Method 3: Estimate from model size if still unknown
        if parameter_count is None and model_size > 0:
            # Assume mixed precision (FP16/FP32) average of 3 bytes per parameter
            parameter_count = (model_size * 1024 ** 3) / 3
        
        # Calculate requirements
        if parameter_count is not None:
            vram_required = parameter_count * 2 / (1024 ** 3)  # Assume FP16
            ram_required = max(model_size * 2, vram_required * 1.5)  # Account for runtime overhead
            storage_required = model_size
            
            # GPU recommendations based on parameter count
            if parameter_count >= 10e9:  # 10B+ parameters
                recommended_gpu = ">=32GB VRAM (A5000, A6000, A100)"
            elif parameter_count >= 1e9:  # 1B-10B parameters
                recommended_gpu = ">=16GB VRAM (RTX 3090, 4090, A4000)"
            elif parameter_count >= 100e6:  # 100M-1B parameters
                recommended_gpu = ">=8GB VRAM (RTX 3070, 4070)"
            else:
                recommended_gpu = ">=4GB VRAM (RTX 3060, 4060)"
        else:
            vram_required = model_size * 1.5  # Rough estimate if parameters unknown
            ram_required = model_size * 2
            storage_required = model_size
            recommended_gpu = "Unable to determine precisely - estimate based on model size"
        
        requirements = {
            "model_size_gb": round(model_size, 2),
            "estimated_parameters": f"{parameter_count/1e6:.1f}M" if parameter_count is not None else "Unknown",
            "vram_required_gb": round(vram_required, 2),
            "ram_required_gb": round(ram_required, 2),
            "storage_required_gb": round(storage_required, 2),
            "recommended_gpu": recommended_gpu,
            "notes": "Estimates assume FP16 precision. Actual requirements may vary based on usage."
        }
        
        return requirements
        
    except Exception as e:
        print(f"Error retrieving deployment requirements for {repo_id}: {str(e)}")
        return {
            "model_size_gb": "Unknown",
            "estimated_parameters": "Unknown",
            "vram_required_gb": "Unknown",
            "ram_required_gb": "Unknown",
            "storage_required_gb": "Unknown",
            "recommended_gpu": "Unknown",
            "notes": f"Error occurred: {str(e)}"
        }

def get_model_capabilities(repo_id: str) -> str:
    """
    Determine the capabilities of a HuggingFace model.

    Args:
        repo_id (str): HuggingFace model repository ID.

    Returns:
        str: Capabilities of the model (e.g., Text Generation, ASR, TTS, Image Generation).
    """
    try:
        config = AutoConfig.from_pretrained(repo_id)
        model_type = config.model_type.lower()

        # Map model types to capabilities
        if model_type in {"gpt", "gpt2", "bloom", "llama","openelm","gemma"}:
            return "Text Generation"
        elif model_type in {"wav2vec2", "whisper"}:
            return "Automatic Speech Recognition (ASR)"
        elif model_type in {"t5", "bart"}:
            return "Text-to-Text Generation"
        elif model_type in {"stable-diffusion", "unet"}:
            return "Image Generation"
        elif model_type in {"tts", "fastspeech2"}:
            return "Text-to-Speech (TTS)"
        else:
            return "Unknown Capability"
    except Exception as e:
        print(f"Error determining model capabilities: {e}")
        return "Unknown Capability"
    

def analyze_model_requirements(repo_id: str, token: str = None) -> Dict[str, Union[float, str]]:
    """
    Analyze model requirements by directly inspecting repository files and metadata.
    
    Args:
        repo_id (str): HuggingFace model repository ID
        token (str, optional): HuggingFace API token
    """
    try:
        # Get list of all files in the repository
        files = list_repo_files(repo_id, token=token)
        
        # Calculate total size of model files
        total_size = 0
        model_files = []
        
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        api_url = f"https://huggingface.co/api/models/{repo_id}"
        
        # Get model metadata from API
        response = requests.get(api_url, headers=headers)
        model_info = response.json() if response.status_code == 200 else {}
        
        # Get direct file sizes
        for file in files:
            if any(file.endswith(ext) for ext in ['.bin', '.safetensors', '.pt', '.ckpt', '.model']):
                model_files.append(file)
                try:
                    # Get file size from API
                    file_url = f"https://huggingface.co/api/models/{repo_id}/tree/main/{file}"
                    file_response = requests.get(file_url, headers=headers)
                    if file_response.status_code == 200:
                        file_info = file_response.json()
                        total_size += file_info.get('size', 0)
                except Exception as e:
                    print(f"Error getting size for {file}: {e}")
        
        # Convert to GB
        total_size_gb = total_size / (1024 ** 3)
        
        # Try to get model details from README
        try:
            readme_url = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            readme = requests.get(readme_url, headers=headers).text
        except:
            readme = ""
        
        return {
            "repository_id": repo_id,
            "model_files_found": model_files,
            "total_size_gb": round(total_size_gb, 2),
            "model_files_count": len(model_files),
            "raw_metadata": model_info,
            "files_analyzed": files,
            "estimated_requirements": {
                "minimum_ram_gb": round(total_size_gb * 1.5, 2),
                "recommended_ram_gb": round(total_size_gb * 2, 2),
                "storage_required_gb": round(total_size_gb, 2),
                "estimated_vram_gb": round(total_size_gb * 1.2, 2)
            }
        }
        
    except Exception as e:
        return {
            "error": f"Error analyzing repository: {str(e)}",
            "repository_id": repo_id
        }
    
def get_ngrok_url(ssh_client: paramiko.SSHClient, port: int, retries: int = 3) -> str:
    """Get ngrok URL after starting tunnel."""
    ngrok_command = f"ngrok http {port} --log=stdout"
    
    # Start ngrok in background
    _, stdout, _ = ssh_client.exec_command(f"nohup {ngrok_command} > ngrok.log 2>&1 &")
    time.sleep(5)  # Allow ngrok to start
    
    # Get URL from ngrok API
    for _ in range(retries):
        _, stdout, _ = ssh_client.exec_command("curl http://localhost:4040/api/tunnels")
        response = stdout.read().decode()
        
        try:
            tunnels = json.loads(response)["tunnels"]
            if tunnels:
                return tunnels[0]["public_url"]
        except (json.JSONDecodeError, KeyError, IndexError):
            time.sleep(2)
            continue
            
    raise HTTPException(status_code=500, detail="Failed to get ngrok URL")