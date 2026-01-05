

Overview

This project implements a complete end-to-end MLOps pipeline for predicting NYC taxi trip duration using modern MLOps practices.

The goal is to simulate how a real ML team works from local experimentation to containerization and cloud deployment using AWS SageMaker Pipelines.

The pipeline is fully reproducible and runs:
	•	locally
	•	inside Docker
	•	on AWS SageMaker (training + batch inference)

⸻

Dataset
	•	Source: New York City Taxi Trip Duration (Kaggle)
	•	Target: trip_duration (regression)
	•	Data characteristics:
	•	missing values
	•	categorical features
	•	time-based features
	•	outliers
	•	potential data leakage traps

⸻

Tech Stack
	•	Python 3.11
	•	uv (dependency management)
	•	src/ layout Python packaging
	•	Docker & docker-compose
	•	AWS SageMaker Pipelines
	•	boto3 & SageMaker SDK
	•	scikit-learn

⸻

Project Structure

src/mlproject/
├── data/
├── preprocess/
├── features/
├── train/
├── inference/
├── pipelines/
│   ├── training_pipeline.py
│   └── batch_inference_pipeline.py
├── utils/
└── __init__.py

scripts/
├── preprocess.py
├── feature_engineering.py
├── train.py
├── batch_inference.py
├── run_training_pipeline.py
└── run_batch_inference_pipeline.py

Dockerfile
docker-compose.yml
pyproject.toml
uv.lock
README.md


⸻

Local Execution

Install dependencies

uv sync

Run training locally

uv run train

Run inference locally

uv run inference


⸻

Docker Execution

Build image

docker build -t ml-project .

Run training

docker-compose run app train

Run inference

docker-compose run app inference

Docker uses the same codebase and CLI commands as local execution.

⸻

AWS SageMaker – Training Pipeline

Set execution role

export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account-id>:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

Run training pipeline

uv run python scripts/run_training_pipeline.py \
  --train-s3 s3://<bucket>/data/train.csv \
  --test-s3 s3://<bucket>/data/test.csv \
  --output-prefix-s3 s3://<bucket>/artifacts \
  --run

Pipeline stages
	1.	Preprocessing
	2.	Feature engineering
	3.	Training (multiple models)
	4.	Model selection and artifact saving to S3

⸻

AWS SageMaker – Batch Inference Pipeline

Run batch inference pipeline

uv run python scripts/run_batch_inference_pipeline.py \
  --input-s3 s3://<bucket>/batch-input/test.csv \
  --model-artifacts-s3 s3://<bucket>/artifacts/<best-model>/ \
  --output-prefix-s3 s3://<bucket>/predictions \
  --run

Output

Predictions are written to:

s3://<bucket>/predictions/


⸻

Modeling Approach

Models trained
	•	Linear Regression (baseline)
	•	Random Forest Regressor

Metric
	•	RMSE (Root Mean Squared Error)

Justification
	•	RMSE penalizes large errors more heavily
	•	Suitable for continuous regression targets
	•	Interpretable and widely used in regression problems

The best model is selected automatically based on validation RMSE.

⸻

Data Leakage Prevention
	•	Temporal features derived only from pickup timestamps
	•	Target (trip_duration) never used in feature engineering
	•	Train/test separation enforced early in the pipeline

⸻

Reproducibility
	•	Deterministic preprocessing and feature steps
	•	Fixed random seeds where applicable
	•	Dependency versions pinned using uv.lock
	•	Same codebase used across local, Docker, and SageMaker runs

⸻

Git Workflow
	•	No direct commits to master
	•	All work done through feature branches
	•	Pull Requests used for merging
	•	Branches preserved after merge
	•	Both team members contributed (commit history available)

⸻

Team Contributions
	•	Karim Daou
	•	Pipeline architecture
	•	SageMaker integration
	•	Dockerization
	•	Feature engineering
	•	Mathieu Ajaka
	•	Model training logic
	•	Evaluation and metrics
	•	Data preprocessing
	•	Documentation support

⸻

Final Notes

This project prioritizes:
	•	clean structure
	•	reproducibility
	•	real-world MLOps practices

The focus is not on model perfection, but on clarity, structure, and end-to-end delivery.

⸻


git add README.md
git commit -m "Finalize README with local, Docker, and SageMaker workflows"
git push

