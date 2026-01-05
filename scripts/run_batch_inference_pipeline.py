import argparse
import os

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

from mlproject.pipelines.batch_inference_pipeline import build_batch_inference_pipeline

def resolve_role_arn() -> str:
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    if role_arn:
        return role_arn

    try:
        return sagemaker.get_execution_role()
    except Exception:
        raise RuntimeError(
            "No execution role found. Set SAGEMAKER_ROLE_ARN environment variable."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", default="mlops2025-batch-inference")
    parser.add_argument("--input-s3", required=True, help="s3://.../raw.csv")
    parser.add_argument("--model-artifacts-s3", required=True, help="s3://.../model/")
    parser.add_argument("--output-prefix-s3", required=True, help="s3://.../predictions/")
    parser.add_argument("--region", default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    region = args.region or boto3.Session().region_name
    if not region:
        raise RuntimeError("AWS region not configured")

    role_arn = resolve_role_arn()
    print(f"Using role: {role_arn}")
    print(f"Region: {region}")

    sm_session = sagemaker.session.Session(
        boto3.Session(region_name=region)
    )

    pipeline = build_batch_inference_pipeline(
        pipeline_name=args.pipeline_name,
        role_arn=role_arn,
        sagemaker_session=sm_session,
        input_s3_uri=args.input_s3,
        model_artifacts_s3_uri=args.model_artifacts_s3,
        output_prefix_s3=args.output_prefix_s3,
        region=region,
    )

    print("Upserting batch inference pipeline...")
    pipeline.upsert(role_arn=role_arn)

    if args.run:
        print("Starting batch inference execution...")
        execution = pipeline.start()
        print(f"Execution ARN: {execution.arn}")
    else:
        print("Pipeline created/updated. Use --run to execute.")


if __name__ == "__main__":
    main()