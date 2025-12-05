# Python Multimodal Voice Agent

<p>
  <a href="https://cloud.livekit.io/projects/p_/sandbox"><strong>Deploy a sandbox app</strong></a>
  •
  <a href="https://docs.livekit.io/agents/overview/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>

A basic example of a multimodal voice agent using LiveKit and the Python [Agents Framework](https://github.com/livekit/agents).

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set up the environment by copying `.env.example` to `.env` and filling in the required values:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`

### Create the .env file from .env.example file.
```console
 cd agents
```
Run the agent:

```console
 python3 voice_agent.py dev
```

This agent requires a frontend application to communicate with. You can use one of our example frontends in [livekit-examples](https://github.com/livekit-examples/), create your own following one of our [client quickstarts](https://docs.livekit.io/realtime/quickstarts/), or test instantly against one of our hosted [Sandbox](https://cloud.livekit.io/projects/p_/sandbox) frontends.


## Docker

### Local Development

#### Build Docker image

```console
docker build -t call-center-agent:latest .
```

#### Run locally (development mode)

```console
docker run -it -p 8081:8081 --env-file .env -e AGENT_MODE=dev call-center-agent:latest
```

#### Run locally (production mode)

```console
docker run -it -p 8081:8081 --env-file .env call-center-agent:latest
```

**Note**: The Docker image uses a non-privileged user for security and exposes port 8081 for health checks. The production mode (`start`) is used by default.

## Google Cloud Run Deployment

### Prerequisites

1. Install and configure Google Cloud CLI
2. Enable necessary APIs:
   ```console
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com
   ```

3. Create an Artifact Registry repository:
   ```console
   gcloud artifacts repositories create call-center-images \
     --repository-format=docker \
     --location=us-central1 \
     --description="Call center agent images"
   ```

### Build and Deploy

#### 1. Build and tag the image

```console
# Build the image
docker build -t call-center-agent:latest .

# Tag for Google Artifact Registry
docker tag call-center-agent:latest \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/call-center-images/call-center-agent:latest
```

#### 2. Configure Docker authentication

```console
gcloud auth configure-docker us-central1-docker.pkg.dev
```

#### 3. Push the image

```console
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/call-center-images/call-center-agent:latest
```

#### 4. Deploy to Cloud Run

```console
gcloud run deploy call-center-agent \
  --image=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/call-center-images/call-center-agent:latest \
  --region=us-central1 \
  --platform=managed \
  --port=8081 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=1 \
  --max-instances=10 \
  --timeout=3600 \
  --concurrency=1 \
  --no-allow-unauthenticated \
  --cpu-boost \
  --execution-environment=gen2 \
  --set-env-vars="AGENT_MODE=start"
```

**Important Cloud Run Configuration Notes:**
- `--min-instances=1`: Keeps at least one instance warm to avoid cold starts
- `--concurrency=1`: Each instance handles one LiveKit session at a time  
- `--cpu-boost`: Provides extra CPU during startup for faster initialization
- `--execution-environment=gen2`: Uses the newer runtime environment
- Health checks automatically use port 8081 endpoint (`/health`)

#### 5. Set environment variables

Set your environment variables using Google Cloud Console or CLI:

```console
gcloud run services update call-center-agent \
  --region=us-central1 \
  --set-env-vars="LIVEKIT_URL=wss://your-livekit-server.com,LIVEKIT_API_KEY=your-api-key,LIVEKIT_API_SECRET=your-api-secret,OPENAI_API_KEY=your-openai-key"
```

### Resource Recommendations

Based on LiveKit documentation:
- **Memory**: 2-4GB recommended
- **CPU**: 2-4 cores recommended  
- **Concurrency**: 25 sessions per instance (adjust based on your load)
- **Timeout**: Set to high value (3600s) to handle long calls
- **Min instances**: 0 for cost efficiency, or 1+ for faster cold starts

### Monitoring and Health Checks

Cloud Run will automatically use the exposed port 8081 for health checks. Monitor your deployment:

```console
# View logs
gcloud run services logs read call-center-agent --region=us-central1

# Get service details
gcloud run services describe call-center-agent --region=us-central1

# Test health endpoint locally
curl http://your-service-url/health
```

### Troubleshooting Cloud Run Issues

**Issue: "Shutting down user disabled instance"**
- **Solution**: This happens when Cloud Run doesn't receive HTTP traffic. The agent now includes a health check server on port 8081 that prevents automatic shutdown.

**Issue: Cold starts taking too long**
- **Solution**: Use `--min-instances=1` to keep at least one instance warm, or increase `--cpu-boost` for faster startup.

**Issue: Agent not connecting to LiveKit**
- **Solution**: Check environment variables are set correctly:
  ```console
  gcloud run services describe call-center-agent --region=us-central1 --format="export"
  ```

**Issue: High memory usage**
- **Solution**: Each voice session requires significant memory. Monitor usage and adjust `--memory` and `--concurrency` settings accordingly.

**Issue: Timeouts during long calls**
- **Solution**: The timeout is set to 3600s (1 hour). For longer calls, increase this value:
  ```console
  gcloud run services update call-center-agent --region=us-central1 --timeout=7200
  ```