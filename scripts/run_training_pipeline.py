import argparse
import os

import boto3
import sagemaker

from mlproject.pipelines.training_pipeline import build_training_pipeline


def resolve_role_arn(sm_session: sagemaker.session.Session) -> str:
    """
    Local (VS Code) runs should use SAGEMAKER_ROLE_ARN from env.
    SageMaker-managed runs can fallback to sagemaker.get_execution_role().
    """
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    if role_arn:
        return role_arn

    try:
        role_arn = sagemaker.get_execution_role()
        if role_arn:
            return role_arn
    except Exception:
        pass

    raise RuntimeError(
        "No execution role found.\n"
        "Fix: set SAGEMAKER_ROLE_ARN in your terminal to your IAM Role ARN, e.g.\n"
        '  export SAGEMAKER_ROLE_ARN="arn:aws:iam::<acct>:role/service-role/<role-name>"\n'
        "Then re-run the script."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline-name", default="mlops2025-training-pipeline")
    p.add_argument("--train-s3", required=True, help="s3://.../train.csv")
    p.add_argument("--test-s3", required=True, help="s3://.../test.csv")
    p.add_argument("--output-prefix-s3", required=True, help="s3://.../artifacts")
    p.add_argument(
        "--region",
        default=None,
        help="AWS region (optional). If not set, uses AWS CLI configured region.",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="If set, starts a pipeline execution after upserting.",
    )
    args = p.parse_args()

    region = args.region or boto3.Session().region_name
    if not region:
        raise RuntimeError(
            "AWS region is not set.\n"
            "Fix: run `aws configure` and set a default region, or pass --region eu-west-1."
        )


    sm_session = sagemaker.session.Session(boto3.Session(region_name=region))

    role_arn = resolve_role_arn(sm_session)
    print(f"Using execution role ARN: {role_arn}")
    print(f"Using region: {region}")


    pipe = build_training_pipeline(
        pipeline_name=args.pipeline_name,
        region=region,
        sagemaker_session=sm_session,
        role_arn=role_arn,
        train_s3_uri=args.train_s3,
        test_s3_uri=args.test_s3,
        output_prefix_s3=args.output_prefix_s3,
    )


    print("Upserting pipeline (create/update)...")
    pipe.upsert(role_arn=role_arn)
    print("Pipeline upserted successfully.")


    if args.run:
        print("Starting pipeline execution...")
        execution = pipe.start()
        print(f" Pipeline execution started: {execution.arn}")
    else:
        print(" Not running execution (use --run to start it).")


if __name__ == "__main__":
    main()