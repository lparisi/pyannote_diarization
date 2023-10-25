# Speaker Diarization API
This is a FastAPI based service that utilizes the pyannote.audio pipeline for speaker diarization.

Features
Upload an audio file and get the speaker diarization results.
Integration with HuggingFace model hub for the diarization pipeline.
Efficient handling of audio files and storage.
Integrated Swagger UI for API documentation and testing.
Utilizes NVIDIA CUDA for GPU acceleration when available.
Quick Start
Prerequisites
Docker
NVIDIA Docker (for CUDA support)


## Build the Docker Image
To build the Docker image:

```bash
docker build -t diarization-api:latest .
```

---
## Run the API

To start the API:

```bash
docker run --gpus all -p 9009:9009 diarization-api:latest
```
---
After starting the API, navigate to http://localhost:9009/docs in your browser to access the Swagger UI and test the endpoints.

API Endpoint
/diarize
Method: POST

Description: Upload an audio file to get speaker diarization results.

Payload:

file: The audio file to be uploaded.
Response:

Returns a JSON object containing the diarization results with speaker labels, start, and stop times for each speaker.

Technical Details
FastAPI for the web framework.
pyannote.audio for the diarization pipeline.
Uses a HuggingFace access token to download the diarization model.
Audio files are temporarily stored and then deleted post processing.
GPU acceleration is leveraged when CUDA is available.
Containerized using Docker with CUDA support.
Known Limitations
Single worker configuration for Gunicorn, can be scaled as needed.
Contributing
To contribute to this project, please submit a pull request with your changes.

