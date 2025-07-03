# 🧠 MLOps CI/CD Pipeline with GitHub Actions, DVC, MLflow, and Docker

A fully automated **MLOps pipeline** built with:
- 🐍 Python
- ⚙️ GitHub Actions for CI/CD
- 📦 MLflow for experiment tracking
- 💾 DVC for data and model versioning
- 🐳 Docker for model packaging
- 🔍 Logging & monitoring hooks for real-world observability

---

## 🎯 Project Objectives

✅ Reproducible model training  
✅ Versioned datasets & models using DVC  
✅ CI/CD pipeline via GitHub Actions  
✅ Model artifact tracking with MLflow  
✅ Docker image build for deployment  
✅ Simulated deployment with hooks for real-world systems (SageMaker, K8s, etc.)  
✅ Logging/monitoring integration (placeholder)

---

## 🧱 Project Structure

.
├── app.py # Optional: inference script
├── train_model.py # Model training with MLflow logging
├── generate_data.py # Generates and saves synthetic data
├── Dockerfile # Packages the model as a service
├── dvc.yaml # DVC pipeline config (optional)
├── app_requirements.txt # App dependencies
├── mlruns/ # Local MLflow tracking artifacts
├── data/
│ └── raw_data.csv # Synthetic dataset (DVC tracked)
├── models/ # Saved models (DVC tracked)
└── .github/
└── workflows/
└── mlops_pipeline.yml # GitHub Actions CI/CD workflow

less
Copy
Edit

---

## 🔄 CI/CD Pipeline Overview

The pipeline is triggered on **push to `main`** and follows this flow:

```mermaid
graph TD
A[Code Push] --> B[CI Setup & Tests]
B --> C[Data Generation]
C --> D[Model Training + Logging]
D --> E[DVC Track + Push]
E --> F[Model Evaluation]
F --> G[Docker Build]
G --> H[Simulated Deployment]
🔧 Setup Locally
bash
Copy
Edit
# Clone the repo and enter the project
git clone https://github.com/yourname/mlops-github-actions-pipeline.git
cd mlops-github-actions-pipeline

# Set up Python environment
python -m venv venv && source venv/bin/activate
pip install -r app_requirements.txt

# Run DVC (if installed)
dvc init
dvc repro                # If using dvc.yaml pipeline
📊 MLflow Tracking
Local MLflow URI: file:./mlruns

Tracks:

Params (solver, C, etc.)

Metrics (accuracy)

Artifacts (model.pkl, input schema)

python
Copy
Edit
mlflow.log_param('solver', solver)
mlflow.log_metric('accuracy', accuracy)
mlflow.sklearn.log_model(sk_model=model, artifact_path="logistic-model")
Use mlflow ui to explore results locally.

💾 DVC Integration
DVC is used to version and track:

data/raw_data.csv

models/model.pkl

bash
Copy
Edit
# Track and push data
dvc add data/raw_data.csv
dvc add models/model.pkl
git add data/raw_data.csv.dvc models/model.pkl.dvc .gitignore
git commit -m "Track data and model with DVC"
dvc remote add -d myremote <remote-url>
dvc push
You can use GitHub, S3, GDrive, or Azure Blob as DVC remote.

📦 Docker Packaging
dockerfile
Copy
Edit
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r app_requirements.txt
CMD ["python", "app.py"]
Build locally:
bash
Copy
Edit
docker build -t ml-inference-service .
docker run -p 8080:8080 ml-inference-service
🔁 GitHub Actions: mlops_pipeline.yml
yaml
Copy
Edit
on:
  push:
    branches: [main]
jobs:
  setup_and_test_code:
    ...
  data_prep_and_train:
    ...
  evaluate_and_register_model:
    ...
  package_model:
    ...
  deploy_model:
    ...
Each job:

Installs dependencies

Runs training or evaluation

Verifies artifacts (MLflow, DVC)

Builds Docker image

Simulates deployment

Artifacts from MLflow are uploaded for traceability.

📈 Logging & Monitoring (Pluggable)
You can plug in logging frameworks like:

loguru, logging, wandb, or sentry-sdk

Send logs to CloudWatch / Grafana / Prometheus (via agent sidecars or SDKs)

python
Copy
Edit
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Training accuracy: {accuracy:.4f}")
For real monitoring:

Monitor inference time, accuracy drift, input schema violations

Use tools like Evidently, Prometheus + Grafana, or WhyLabs

✅ Run Entire Pipeline
bash
Copy
Edit
# Step 1: Generate data
python generate_data.py

# Step 2: Train model and log with MLflow
python train_model.py

# Step 3: Version artifacts
dvc add data/raw_data.csv
dvc add models/model.pkl
dvc push

# Step 4: (Optional) View MLflow UI
mlflow ui
🧪 Simulated Deployment
bash
Copy
Edit
echo "Deployment of Docker image ml-inference-service successful!"
In production, this would involve:

kubectl apply -f deployment.yaml

aws sagemaker create-model

gcloud ai-platform deploy-model

terraform apply

🛠️ Future Enhancements
✅ Real model evaluation script with thresholds

✅ GitHub Secrets for pushing to DockerHub or AWS ECR

✅ Add support for Streamlit app serving

⏳ Monitoring dashboard with Grafana or Weights & Biases

🧠 Author
Built with ❤️ by [Your Name]
Twitter | GitHub | LinkedIn

🪪 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

Would you like me to:
- Export this as a `README.md` file and include it in your repo?
- Add `badges` (build passing, license, etc.)?
- Add a `dvc.yaml` pipeline as well to automate data → model steps?

Let me know what you'd like enhanced!
