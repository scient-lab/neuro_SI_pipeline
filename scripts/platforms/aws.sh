#!/usr/bin/env bash
# Platform: aws - bootstrap + exec on AWS (EC2 or SageMaker).
# Sourced by pipeline.sh; must define exec_phase_on_platform.

exec_phase_on_platform() {
    local phase_script="$1"
    local step_filter="$2"

    # TODO: AWS-side bootstrap (S3 mounts, instance metadata, AWS_REGION, etc.)
    bash "$phase_script" "$step_filter"
}
