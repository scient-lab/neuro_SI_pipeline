#!/usr/bin/env bash
# create-dss-runs-reader.sh — create a READ-ONLY IAM user for the run-dashboard,
# scoped to s3://enlibra/dss/runs (list + get under that prefix, nothing else).
#
# Run from AWS CloudShell (already authenticated as an IAM-capable admin):
#   bash create-dss-runs-reader.sh
#
# Idempotent: safe to re-run. Reuses the user if it exists; overwrites the
# inline policy; won't create a 3rd access key.
#
# Override any of these via env, e.g.:
#   USER_NAME=my-reader BUCKET=enlibra PREFIX=dss/runs MAKE_KEYS=0 bash create-dss-runs-reader.sh
#
#   USER_NAME   IAM user to create            (default slm-factory-runs-reader)
#   POLICY_NAME inline policy name            (default DssRunsReadOnly)
#   BUCKET      S3 bucket                     (default enlibra)
#   PREFIX      key prefix to allow           (default dss/runs)
#   MAKE_KEYS   1 = create access keys        (default 1; set 0 for role-based Lambda)

set -euo pipefail

USER_NAME="${USER_NAME:-slm-factory-runs-reader}"
POLICY_NAME="${POLICY_NAME:-DssRunsReadOnly}"
BUCKET="${BUCKET:-enlibra}"
PREFIX="${PREFIX:-dss/runs}"        # no leading/trailing slash
MAKE_KEYS="${MAKE_KEYS:-1}"

echo "IAM user : $USER_NAME"
echo "Policy   : $POLICY_NAME (inline)"
echo "S3 scope : s3://$BUCKET/$PREFIX/*   (list + read only)"
echo

command -v aws >/dev/null || { echo "aws CLI not found (run this in CloudShell)" >&2; exit 1; }

# --- 1. least-privilege policy document ------------------------------------
POLICY_FILE="$(mktemp)"
trap 'rm -f "$POLICY_FILE"' EXIT
cat > "$POLICY_FILE" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListRunsPrefix",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::$BUCKET",
      "Condition": { "StringLike": { "s3:prefix": ["$PREFIX", "$PREFIX/*"] } }
    },
    {
      "Sid": "ReadRunObjects",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::$BUCKET/$PREFIX/*"
    }
  ]
}
EOF

# --- 2. create the user (idempotent) ---------------------------------------
if aws iam get-user --user-name "$USER_NAME" >/dev/null 2>&1; then
  echo "user exists — reusing: $USER_NAME"
else
  aws iam create-user --user-name "$USER_NAME" \
    --tags Key=project,Value=slm-factory Key=purpose,Value=dss-runs-dashboard Key=access,Value=read-only >/dev/null
  echo "created user: $USER_NAME"
fi

# --- 3. attach the scoped read-only policy (overwrites if present) ----------
aws iam put-user-policy --user-name "$USER_NAME" \
  --policy-name "$POLICY_NAME" --policy-document "file://$POLICY_FILE"
echo "attached inline policy: $POLICY_NAME"

# --- 4. access keys (optional) ---------------------------------------------
if [[ "$MAKE_KEYS" == "1" ]]; then
  existing=$(aws iam list-access-keys --user-name "$USER_NAME" \
              --query 'length(AccessKeyMetadata)' --output text)
  if [[ "$existing" -ge 2 ]]; then
    echo "!! user already has $existing access keys (AWS max 2) — not creating another."
    echo "   rotate: aws iam delete-access-key --user-name $USER_NAME --access-key-id <OLD_ID>"
  else
    echo
    echo "=== ACCESS KEY (shown ONCE — store in Secrets Manager / backend env; NOT git, NOT the browser) ==="
    aws iam create-access-key --user-name "$USER_NAME" \
      --query 'AccessKey.{AccessKeyId:AccessKeyId,SecretAccessKey:SecretAccessKey}' --output table
  fi
else
  echo "MAKE_KEYS=0 — skipped access keys (use an IAM role for Lambda instead)."
fi

# --- 5. verification hints --------------------------------------------------
echo
echo "confirm the scope:"
echo "  aws iam get-user-policy --user-name $USER_NAME --policy-name $POLICY_NAME"
echo
echo "live test with the new key (allow a few seconds for IAM to propagate):"
echo "  aws s3 ls s3://$BUCKET/$PREFIX/                 # ALLOWED  (lists runs)"
echo "  aws s3 ls s3://$BUCKET/                         # DENIED   (bucket root)"
echo
echo "NOTE: if objects use SSE-KMS with a customer-managed key, also grant kms:Decrypt on that key."
echo "      check with: aws s3api get-bucket-encryption --bucket $BUCKET"
