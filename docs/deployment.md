# Deployment Guide

This document covers deploying the Marketing Measurement Hub to Google Cloud.

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Docker (for local container testing)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Cloud Run     │────▶│   Vertex AI     │     │  Cloud Storage  │
│  (Streamlit)    │     │   (Gemini)      │     │    (Data)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                               ▲
        └───────────────────────────────────────────────┘
        
┌─────────────────┐     ┌─────────────────┐
│ Cloud Scheduler │────▶│ Cloud Function  │
│   (Daily 6am)   │     │  (Data Purge)   │
└─────────────────┘     └─────────────────┘
```

## Step 1: Set Up GCP Project

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com
```

## Step 2: Create Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_BUCKET_NAME

# Upload data files
gsutil -m cp -r data/* gs://YOUR_BUCKET_NAME/data/
gsutil -m cp -r knowledge_base/* gs://YOUR_BUCKET_NAME/knowledge_base/
gsutil -m cp -r agents/agent_prompts/* gs://YOUR_BUCKET_NAME/agent_prompts/
```

## Step 3: Deploy to Cloud Run

### Option A: Direct Deploy (Recommended)

```bash
gcloud run deploy mktg-measurement-hub \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1,GOOGLE_GENAI_USE_VERTEXAI=true,GCS_BUCKET_NAME=YOUR_BUCKET_NAME"
```

### Option B: Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml
```

## Step 4: Configure Custom Domain (Optional)

```bash
# Map custom domain
gcloud run domain-mappings create \
  --service mktg-measurement-hub \
  --domain app.yourdomain.com \
  --region us-central1
```

Then add the DNS records shown in the output to your domain registrar.

## Step 5: Set Up Data Purge (Optional)

For demo environments, set up automatic data reset:

### Deploy Cloud Function

```bash
cd cloud_functions/purge_test_data

gcloud functions deploy purge-test-data \
  --gen2 \
  --runtime python312 \
  --region us-central1 \
  --source . \
  --entry-point purge_test_data \
  --trigger-http \
  --no-allow-unauthenticated
```

### Create Scheduler Job

```bash
gcloud scheduler jobs create http purge-test-data-daily \
  --location us-central1 \
  --schedule "0 6 * * *" \
  --uri "https://us-central1-YOUR_PROJECT_ID.cloudfunctions.net/purge-test-data" \
  --http-method GET \
  --oidc-service-account-email YOUR_SERVICE_ACCOUNT@developer.gserviceaccount.com
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `my-project-123` |
| `GOOGLE_CLOUD_LOCATION` | GCP region | `us-central1` |
| `GOOGLE_GENAI_USE_VERTEXAI` | Enable Vertex AI auth | `true` |
| `GCS_BUCKET_NAME` | Data bucket name | `my-data-bucket` |

## IAM Permissions

The Cloud Run service account needs:

```bash
# Storage access
gsutil iam ch serviceAccount:SERVICE_ACCOUNT:objectViewer gs://YOUR_BUCKET_NAME
gsutil iam ch serviceAccount:SERVICE_ACCOUNT:objectCreator gs://YOUR_BUCKET_NAME

# Vertex AI access (automatic with default service account)
```

## Monitoring

### View Logs

```bash
gcloud run services logs read mktg-measurement-hub --region us-central1
```

### View in Console

- **Cloud Run**: https://console.cloud.google.com/run
- **Cloud Functions**: https://console.cloud.google.com/functions
- **Cloud Scheduler**: https://console.cloud.google.com/cloudscheduler

## Troubleshooting

### "Missing key inputs argument"

**Cause**: Vertex AI mode not enabled
**Fix**: Ensure `GOOGLE_GENAI_USE_VERTEXAI=true` is set

### "Reauthentication needed"

**Cause**: Local gcloud credentials expired
**Fix**: Run `gcloud auth application-default login`
**Note**: This only affects local dev; Cloud Run uses service account automatically

### Container fails to start

**Cause**: Usually memory or startup timeout
**Fix**: Increase memory (`--memory 4Gi`) or timeout (`--timeout 300`)

### Data not loading

**Cause**: GCS permissions or bucket name mismatch
**Fix**: Verify `GCS_BUCKET_NAME` and service account permissions

## Cost Optimization

- **Cloud Run**: Set `--min-instances 0` for scale-to-zero
- **Memory**: Start with 2Gi, increase only if needed
- **Region**: Use `us-central1` for lowest Vertex AI costs
