variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "ap-south-1" # Mumbai
}

variable "project_name" {
  description = "Short name used to tag and name resources."
  type        = string
  default     = "call-chat-summarizer"
}

variable "instance_type" {
  description = "EC2 instance type. Whisper-large-v3 + Gemma-4-E2B + SpeechBrain (fp16, ~13GB weights). PREFERRED is a 24GB GPU (g6.xlarge L4 / g5.xlarge A10G), but both hit InsufficientInstanceCapacity in every Mumbai AZ on 2026-07-01. Using g4dn.xlarge (NVIDIA T4 16GB, 4 vCPU / 16 GB) which has abundant ap-south-1c capacity — VRAM headroom over the ~13GB stack is tight (~2-3GB), so watch for OOM under concurrent call load; move back to g6/g5.xlarge when capacity returns."
  type        = string
  default     = "g4dn.xlarge"
}

variable "availability_zone" {
  description = "AZ to launch the GPU instance in. Pinned to steer around per-AZ GPU capacity. g4dn.xlarge has abundant capacity in ap-south-1c (default). If moving back to g5/g6.xlarge, use 1a or 1b (those types are not offered in 1c). Must be an AZ that offers var.instance_type."
  type        = string
  default     = "ap-south-1c"
}

variable "root_volume_gb" {
  description = "Root EBS volume size in GB. DLAMI base snapshot is ~75GB; leave headroom for the CUDA torch stack + the ~13GB Whisper/Gemma/SpeechBrain weights in the HF cache + logs."
  type        = number
  default     = 140
}

variable "domain" {
  description = "Public hostname Caddy serves with auto-HTTPS."
  type        = string
  default     = "summary.chatbucket.chat"
}

variable "acme_email" {
  description = "Email used by Caddy/Let's Encrypt for ACME registration."
  type        = string
  default     = "udathak@gmail.com"
}

variable "repo_url" {
  description = "Git repository cloned onto the instance."
  type        = string
  default     = "https://github.com/jagannathPridh18/summary_repo.git"
}

variable "git_ref" {
  description = "Git branch/tag/commit to check out."
  type        = string
  default     = "master"
}

variable "app_port" {
  description = "Port the FastAPI/uvicorn app listens on (see run.py / summarizer.config PORT, default 8077)."
  type        = number
  default     = 8077
}

variable "torch_cuda_index" {
  description = "PyTorch wheel index for the pinned torch/torchvision/torchaudio==2.11.0 stack. cu128 matches the DLAMI CUDA 12.x driver and the L4 (sm_89). Empty = PyPI default."
  type        = string
  default     = "https://download.pytorch.org/whl/cu128"
}

variable "hf_token" {
  description = "Optional Hugging Face token. google/gemma-4-E2B-it and openai/whisper-large-v3 are UNGATED so this is not required; leave empty. Provide only to avoid anonymous download rate limits."
  type        = string
  default     = ""
  sensitive   = true
}

variable "ssh_allowed_cidr" {
  description = "CIDR allowed to reach SSH (port 22). Lock this down to your IP for production."
  type        = string
  default     = "0.0.0.0/0"
}

variable "log_retention_days" {
  description = "Retention (in days) for the CloudWatch log groups the CW agent ships to. Keeps storage cost bounded."
  type        = number
  default     = 30
}
