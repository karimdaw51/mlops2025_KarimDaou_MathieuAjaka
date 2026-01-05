from __future__ import annotations

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def build_batch_inference_pipeline(
    *,
    pipeline_name: str,
    role_arn: str,
    sagemaker_session: sagemaker.session.Session,
    region: str,
    input_s3_uri: str,
    model_artifacts_s3_uri: str,
    output_prefix_s3: str,
    instance_type: str = "ml.m5.large",
    instance_count: int = 1,
) -> Pipeline:
    """
    Batch inference pipeline:
      1) Preprocess raw CSV
      2) Feature engineering
      3) Batch predict using trained model artifacts
    Writes predictions to S3.
    """

    pipeline_sess = PipelineSession(
        boto_session=sagemaker_session.boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        default_bucket=sagemaker_session.default_bucket(),
    )

    p_input_s3 = ParameterString(name="InputS3Uri", default_value=input_s3_uri)
    p_model_s3 = ParameterString(name="ModelArtifactsS3Uri", default_value=model_artifacts_s3_uri)
    p_output_s3 = ParameterString(name="OutputPrefixS3", default_value=output_prefix_s3)

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=instance_type,
        instance_count=instance_count,
        sagemaker_session=pipeline_sess,
    )

    preprocess_step = ProcessingStep(
        name="PreprocessRaw",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=p_input_s3,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="preprocessed",
                source="/opt/ml/processing/output",
                destination=f"{p_output_s3}/preprocess",
            )
        ],
        code="scripts/preprocess.py",
        job_arguments=[
            "--input-path", "/opt/ml/processing/input",
            "--output-path", "/opt/ml/processing/output",
            "--mode", "inference",
        ],
    )

    features_step = ProcessingStep(
        name="FeatureEngineering",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs[
                    "preprocessed"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="features",
                source="/opt/ml/processing/output",
                destination=f"{p_output_s3}/features",
            )
        ],
        code="scripts/feature_engineering.py",
        job_arguments=[
            "--input-path", "/opt/ml/processing/input",
            "--output-path", "/opt/ml/processing/output",
            "--mode", "inference",
        ],
    )

    predict_step = ProcessingStep(
        name="BatchPredict",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=features_step.properties.ProcessingOutputConfig.Outputs[
                    "features"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/features",
            ),
            ProcessingInput(
                source=p_model_s3,
                destination="/opt/ml/processing/model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predictions",
                source="/opt/ml/processing/output",
                destination=f"{p_output_s3}/predictions",
            )
        ],
        code="scripts/batch_inference.py",
        job_arguments=[
            "--features-path", "/opt/ml/processing/features",
            "--model-path", "/opt/ml/processing/model",
            "--output-path", "/opt/ml/processing/output",
        ],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[p_input_s3, p_model_s3, p_output_s3],
        steps=[preprocess_step, features_step, predict_step],
        sagemaker_session=pipeline_sess,
    )