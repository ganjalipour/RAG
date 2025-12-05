# Use the official Python slim image as the base image
FROM python:3.11.6-slim

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Create a non-privileged user
RUN useradd --create-home --shell /bin/bash appuser

# Install build dependencies and audio libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libasound2-dev \
    portaudio19-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-privileged user
USER appuser

# Set the working directory in the container
WORKDIR /home/appuser

# Copy requirements and install dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY --chown=appuser:appuser . .

# Expose health check port for Cloud Run
EXPOSE 8081

# Use environment variable to control execution mode, default to production
ENV AGENT_MODE=start
CMD ["sh", "-c", "python agents/voice_agent.py ${AGENT_MODE}"]