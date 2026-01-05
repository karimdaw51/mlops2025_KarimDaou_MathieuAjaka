from __future__ import annotations

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterString


def build_training_pipeline(
    *,
    pipeline_name: str,
    role_arn: str,
    region: str,
    input_train_s3: str,
    input_test_s3: str,
    output_prefix_s3: str,
) -> Pipeline:
    """
    Training pipeline:
      1) preprocess (CSV -> clean parquet)
      2) feature engineering (clean parquet -> features parquet)
    NOTE: We'll add the TrainingStep after we confirm the CLI args of scripts/train.py.
    """

    session = sagemaker.Session(boto_region_name=region)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=session,
    )

    # --- Preprocess step ---  
    preprocess_step = ProcessingStep(
        name="Preprocess",
        processor=processor,
        code="scripts/preprocess.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_train_s3,
                destination="/opt/ml/processing/input/train",
            ),
            sagemaker.processing.ProcessingInput(
                source=input_test_s3,
                destination="/opt/ml/processing/input/test",
            ),
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"{output_prefix_s3}/preprocess",
            )
        ],
        job_arguments=[
            "--train_csv",
            "/opt/ml/processing/input/train/train.csv",
            "--test_csv",
            "/opt/ml/processing/input/test/test.csv",
            "--out_train",
            "/opt/ml/processing/output/clean_train.parquet",
            "--out_test",
            "/opt/ml/processing/output/clean_test.parquet",
        ],
    )

    # --- Feature engineering step ---
    feature_step = ProcessingStep(
        name="FeatureEngineering",
        processor=processor,
        code="scripts/feature_engineering.py",
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=f"{output_prefix_s3}/preprocess",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"{output_prefix_s3}/features",
            )
        ],
        job_arguments=[
            "--in_path",
            "/opt/ml/processing/input/clean_train.parquet",
            "--out_path",
            "/opt/ml/processing/output/features_train.parquet",
        ],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            ParameterString(name="input_train_s3", default_value=input_train_s3),
            ParameterString(name="input_test_s3", default_value=input_test_s3),
            ParameterString(name="output_prefix_s3", default_value=output_prefix_s3),
        ],
        steps=[preprocess_step, feature_step],
        sagemaker_session=session,
    )