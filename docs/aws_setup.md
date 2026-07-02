# AWS one-time setup

Operator runbook for the AWS resources the pipeline expects. Run **once per
AWS account**, then forget about it until you rotate keys or change scopes.

> **TL;DR** — create an IAM user `kg-si-pipeline` with two scoped inline
> policies (S3 + CloudWatch Logs), generate access keys, drop them into
> `.env.runpod`. That's it.

## What gets created

| Resource | Name / value | Why |
|---|---|---|
| IAM user | `kg-si-pipeline` | Long-lived credentials the pod uses (`launch.sh` forwards them) |
| IAM policy (inline) | `KGSIPipelineS3Access` | Read/write inside `s3://enlibra/dss/*` only |
| IAM policy (inline) | `KGSIPipelineCloudWatchLogs` | Write log streams under `/enlibra/dss/runs/*` only |
| Access key pair | `AKIA…` + secret | The pod uses these via env vars |
| S3 bucket | `enlibra` | Already exists |
| S3 prefix structure | `dss/{corpus,runs,shared}/` | Program namespace inside the bucket |
| CloudWatch log group | `/enlibra/dss/runs/pipeline` | Per-step log streams land here |

## Prerequisites

- AWS CLI installed locally (admin profile with `iam:*`, `logs:CreateLogGroup`,
  `s3:CreateBucket` if you don't have `enlibra` yet)
- AWS region picked (we use `us-east-1` — change throughout if different)
- The `enlibra` bucket already created (assumed)

## Step 1 — S3 prefix structure inside `s3://enlibra/dss/`

Creates empty `.keep` placeholder objects so the prefixes are visible in the
console and `aws s3 ls`. Repeat as needed.

```bash
PROFILE=admin                  # your admin profile name; use --profile flag throughout
BASE=s3://enlibra/dss

for k in \
  corpus/neuroscience/source_pdfs/.keep \
  corpus/neuroscience/source_txt/.keep \
  corpus/medical/source_pdfs/.keep \
  corpus/medical/source_txt/.keep \
  corpus/physics/source_pdfs/.keep \
  corpus/physics/source_txt/.keep \
  runs/.keep \
  shared/seed_kgs/.keep \
  shared/models/.keep ; do
  echo "" | aws --profile $PROFILE s3 cp - $BASE/$k
done

# Verify:
aws --profile $PROFILE s3 ls $BASE/ --recursive
```

You should see prefixes like `dss/corpus/neuroscience/source_txt/`,
`dss/runs/`, `dss/shared/...`.

## Step 2 — IAM user

```bash
USER=kg-si-pipeline

aws iam create-user --user-name "$USER"
```

If the user already exists you'll get an error — safe to ignore (idempotent
intent for the policies in steps 3+4).

## Step 3 — S3 access policy (read/write inside `dss/*`)

```bash
USER=kg-si-pipeline
POLICY=KGSIPipelineS3Access
BUCKET=enlibra
PREFIX=dss

cat > /tmp/${POLICY}.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucketDssOnly",
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:ListBucketVersions"],
      "Resource": "arn:aws:s3:::${BUCKET}",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["${PREFIX}", "${PREFIX}/*"]
        }
      }
    },
    {
      "Sid": "ReadWriteDss",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:DeleteObjectVersion",
        "s3:GetObjectAcl",
        "s3:PutObjectAcl"
      ],
      "Resource": "arn:aws:s3:::${BUCKET}/${PREFIX}/*"
    }
  ]
}
EOF

aws iam put-user-policy \
    --user-name "$USER" \
    --policy-name "$POLICY" \
    --policy-document file:///tmp/${POLICY}.json
```

**What this allows:** list (incl. object versions) + read + write + delete
(incl. old versions) inside `s3://enlibra/dss/*`.
**What it forbids:** touching other prefixes in `enlibra` or other buckets.

> **Why the version actions?** `s3:ListBucketVersions` + `s3:DeleteObjectVersion`
> are what `scripts/s3_prune_runs.sh` needs to retire old runs under
> `dss/runs/`. On a **versioned** bucket a plain `aws s3 rm` only writes
> delete-markers — the old versions (and their storage cost) linger; these two
> actions let the prune actually purge them. They're harmless on an unversioned
> bucket. `s3:ListBucketVersions` sits in the conditioned list statement, so it
> stays scoped to `dss` like `s3:ListBucket` and never widens beyond the program
> namespace.
>
> **Upgrading an existing user:** inline policies are overwritten in place, so
> just re-run this Step 3 block (from CloudShell or an admin profile) — no
> delete-first needed. A `kg-si-pipeline` whose `s3 rm` worked but hit
> `AccessDenied … ListBucketVersions` was on the pre-version policy; re-running
> this fixes it.

## Step 4 — CloudWatch Logs policy (write under `/enlibra/dss/runs/*`)

```bash
USER=kg-si-pipeline
POLICY=KGSIPipelineCloudWatchLogs
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

cat > /tmp/${POLICY}.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListGroupsAccountWide",
      "Effect": "Allow",
      "Action": "logs:DescribeLogGroups",
      "Resource": "*"
    },
    {
      "Sid": "WriteToOwnedLogGroup",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": [
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/enlibra/dss/runs/*",
        "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/enlibra/dss/runs/*:log-stream:*"
      ]
    }
  ]
}
EOF

aws iam put-user-policy \
    --user-name "$USER" \
    --policy-name "$POLICY" \
    --policy-document file:///tmp/${POLICY}.json
```

**What this allows:** create + write log streams under `/enlibra/dss/runs/*`,
plus account-wide `DescribeLogGroups` (list group names only — needed for the
verification commands below and to navigate the CloudWatch console).
**What it forbids:** writing to other log groups, reading anyone's log
contents (`logs:GetLogEvents` not granted), deleting groups.

> **AWS gotcha:** `logs:DescribeLogGroups` is a list operation and AWS does
> **not** support resource-level scoping for it — it always requires
> `Resource: "*"`. If you put `DescribeLogGroups` in the scoped statement
> alongside the writes, every `describe-log-groups` call returns
> `AccessDeniedException` with a malformed-looking ARN.

## Step 5 — Pre-create the CloudWatch log group with retention

You *can* skip this — `cw_ship.py` creates the group on first write. But then
it defaults to "Never Expire" and you'll pay for old logs forever. Recommended:

```bash
LOG_GROUP=/enlibra/dss/runs/pipeline

aws logs create-log-group --log-group-name "$LOG_GROUP"
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 30
```

Tune `--retention-in-days` to your needs. Choices: `1, 3, 5, 7, 14, 30, 60,
90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653`.

## Step 6 — Generate access keys

```bash
USER=kg-si-pipeline
aws iam create-access-key --user-name "$USER" --output json
```

**Output is shown once.** Save both `AccessKeyId` and `SecretAccessKey` before
closing the terminal. If lost, you must rotate (Step 9).

## Step 7 — Add the new IAM credentials to a local AWS profile

So you can `aws --profile kg-si <command>` for local testing without flipping
your default profile:

```bash
aws configure --profile kg-si
# paste AccessKeyId, SecretAccessKey, region (us-east-1), output (json)
```

## Step 8 — Wire into the pipeline (`.env.runpod`)

```bash
# .env.runpod (gitignored; lives next to .env.runpod.example)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# S3 — input corpus + per-run output mirror
S3_URI=s3://enlibra/dss
CORPUS_PATH=corpus/neuroscience/source_txt
S3_SYNC_INTERVAL_SEC=300        # background output sync every 5 min (optional)

# CloudWatch Logs — per-step log push
AWS_CLOUDWATCH_LOG_GROUP=/enlibra/dss/runs/pipeline
```

`scripts/runpod/launch.sh` forwards all of these into the pod's environment;
`scripts/runpod/bootstrap.sh` writes them into the pod's `.env`.

## Step 9 — Verify

The smoke tests below use the new `kg-si` profile (Step 7).

### S3

```bash
# Should list the dss/ prefix
aws --profile kg-si s3 ls s3://enlibra/dss/

# Should succeed — write+delete round-trip inside dss/ (validates the prune grant)
echo probe | aws --profile kg-si s3 cp - s3://enlibra/dss/runs/_probe/x.txt
aws --profile kg-si s3 rm s3://enlibra/dss/runs/_probe/ --recursive

# Should be DENIED — proves the policy is correctly scoped
aws --profile kg-si s3 ls s3://enlibra/
aws --profile kg-si s3 cp /tmp/x.txt s3://enlibra/wrong-prefix/  # → AccessDenied
```

### CloudWatch Logs

```bash
# Should succeed
aws --profile kg-si logs create-log-stream \
    --log-group-name /enlibra/dss/runs/pipeline \
    --log-stream-name test-stream

# Verify it's there
aws --profile kg-si logs describe-log-streams \
    --log-group-name /enlibra/dss/runs/pipeline \
    --log-stream-name-prefix test-stream

# Write a probe event
aws --profile kg-si logs put-log-events \
    --log-group-name /enlibra/dss/runs/pipeline \
    --log-stream-name test-stream \
    --log-events "timestamp=$(date +%s%3N),message=hello-cw"

# Should be DENIED — proves the scope holds
aws --profile kg-si logs create-log-group --log-group-name /unrelated/foo  # → AccessDenied
```

## Operational notes

### Key rotation (recommended every ~90 days)

```bash
USER=kg-si-pipeline

# 1. Create a second key (now there are two active)
aws iam create-access-key --user-name "$USER" --output json

# 2. Update .env.runpod on the workstation with the new key
# 3. Test a smoke run end-to-end (S3 sync + CloudWatch ship) with the new key

# 4. Disable the old key (don't delete yet — quick rollback if needed)
aws iam update-access-key --user-name "$USER" \
    --access-key-id AKIA_OLD --status Inactive

# 5. After a day or two of stability, delete the old key
aws iam delete-access-key --user-name "$USER" --access-key-id AKIA_OLD
```

## Step 10 — (optional) Bedrock Custom Model Import role

Only needed to deploy a trained model to **Amazon Bedrock Custom Model Import
(CMI)** — see `docs/inference_deployment.md §4`. This is a **separate service
role** that *Bedrock itself* assumes to read the model from S3; it is **distinct
from the `kg-si-pipeline` user** above (the pod's S3/CloudWatch identity), which
must NOT be reused here. The deployable model is the `rl.merge_rl` output
(`…/checkpoint-N/merged_final_model/` — full safetensors + config + tokenizer).

Run once from **CloudShell** (or any admin profile):

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ROLE=kg-si-bedrock-import-role
MODEL_PREFIX="dss/shared/models"      # where s3_sync/promote land the merged model in s3://enlibra

# 1) Trust policy — only Bedrock CMI in THIS account/region may assume the role
#    (aws:SourceAccount / SourceArn guard against the confused-deputy problem).
cat > /tmp/trust.json <<EOF
{ "Version":"2012-10-17","Statement":[{
  "Effect":"Allow","Principal":{"Service":"bedrock.amazonaws.com"},"Action":"sts:AssumeRole",
  "Condition":{"StringEquals":{"aws:SourceAccount":"${ACCOUNT_ID}"},
               "ArnLike":{"aws:SourceArn":"arn:aws:bedrock:${REGION}:${ACCOUNT_ID}:model-import-job/*"}}}]}
EOF
aws iam create-role --role-name "$ROLE" --assume-role-policy-document file:///tmp/trust.json

# 2) Read-only on the model prefixes (scoped like KGSIPipelineS3Access). Grants
#    read on BOTH dss/shared/models/* (where s3_sync + promote land the merged
#    model — the shared/models/ prefix created in Step 1) AND the legacy
#    dss/shared/imported-models/* prefix. Both s3:GetObject (read the files) and
#    s3:ListBucket (Bedrock enumerates the model dir) are required.
cat > /tmp/s3.json <<'EOF'
{ "Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["s3:GetObject"],
   "Resource":[
     "arn:aws:s3:::enlibra/dss/shared/models/*",
     "arn:aws:s3:::enlibra/dss/shared/imported-models/*"
   ]},
  {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::enlibra",
   "Condition":{"StringLike":{"s3:prefix":[
     "dss/shared/models/*",
     "dss/shared/imported-models/*"
   ]}}}]}
EOF
aws iam put-role-policy --role-name "$ROLE" \
    --policy-name KGSIBedrockImportS3Read --policy-document file:///tmp/s3.json

# 3) The role ARN — pass it to create-model-import-job:
aws iam get-role --role-name "$ROLE" --query Role.Arn --output text
```

Then upload the merged model and create the import job:

```bash
# Upload the rl.merge_rl output (full safetensors + config + tokenizer):
aws s3 sync <…/checkpoint-N/merged_final_model>/ s3://enlibra/${MODEL_PREFIX}/specialized-slm/

aws bedrock create-model-import-job \
    --imported-model-name specialized-slm \
    --role-arn "$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)" \
    --model-data-source '{"s3DataSource":{"s3Uri":"s3://enlibra/'"${MODEL_PREFIX}"'/specialized-slm/"}}'
```

> **Architecture gate:** Bedrock CMI accepts only specific architectures. The
> trained model is **Qwen3** — confirm Qwen3 is on the current supported list
> before importing, else the job fails validation (fall back to a RunPod or
> SageMaker endpoint). See `docs/inference_deployment.md §4`.

### Listing current state

```bash
aws iam list-attached-user-policies --user-name kg-si-pipeline   # managed (none)
aws iam list-user-policies          --user-name kg-si-pipeline   # inline (2 — S3 + CW)
aws iam list-access-keys            --user-name kg-si-pipeline   # active keys

aws s3 ls s3://enlibra/dss/ --profile kg-si
aws logs describe-log-groups --log-group-name-prefix /enlibra/dss/ --profile kg-si
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `AccessDenied` on `s3 sync` | IAM key not loaded on the pod | Check `cat $SI_HOME/.env` on the pod has `AWS_ACCESS_KEY_ID` |
| `s3_prune_runs.sh` lists runs but `AccessDenied` on delete / `ListBucketVersions` | `kg-si-pipeline` on the pre-version S3 policy | Re-run **Step 3** to overwrite `KGSIPipelineS3Access` with the version actions (it now grants `s3:DeleteObjectVersion` + `s3:ListBucketVersions`) |
| `CloudWatch ship failed for X/Y (non-fatal)` in pipeline logs | Log group missing OR policy ARN mismatch | Re-run Step 5; double-check `LOG_GROUP_PREFIX` in Step 4 matches the actual group name |
| `cw_ship: boto3 not installed; skipping CloudWatch push` | uv missing on pod fallback path | Confirm uv installed in bootstrap (Step 2 in `scripts/runpod/bootstrap.sh`) |
| `s3 sync` succeeds locally but fails on pod | Region mismatch | `AWS_DEFAULT_REGION` must be set; bucket region detection sometimes fails on first call |
| Pipeline logs say `Background S3 sync: every Ns` but nothing shows in S3 | sync_outputs.sh erroring silently | Set `S3_SYNC_INTERVAL_SEC=` (unset) temporarily, then run `./scripts/data_prep/sync_outputs.sh` manually to see the error |

### Replacing a stale policy

If you ever change scope (e.g. adopt a new prefix), inline policies are
overwritten by re-running `put-user-policy`. To explicitly clear before
re-applying:

```bash
aws iam delete-user-policy --user-name kg-si-pipeline --policy-name KGSIPipelineS3Access
aws iam delete-user-policy --user-name kg-si-pipeline --policy-name KGSIPipelineCloudWatchLogs
```

Then re-run Steps 3 and 4 with the new prefixes.

## Cost expectations

| Service | Why we use it | Typical monthly cost (one pilot run/week) |
|---|---|---|
| S3 storage (Standard) | Corpus + per-run outputs | ~$0.50-2 (sub-100 GB) |
| S3 requests | `aws s3 sync` GETs/PUTs | <$0.10 |
| CloudWatch Logs ingest | Per-step push (~MB per step) | ~$0.50 ($0.50/GB ingest) |
| CloudWatch Logs storage | 30-day retention | <$0.10 |

Most costs are bounded; the big variable is compute (RunPod GPU hours), not
AWS storage.

## See also

- [`aws_inference.md`](aws_inference.md) — serving the trained model on Bedrock CMI
- [`inference_deployment.md`](inference_deployment.md) — operational concerns post-training
- [`../scripts/README.md`](../scripts/README.md) — the runtime side (`S3_URI`, `AWS_CLOUDWATCH_LOG_GROUP`, etc.)
