from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline
import torch
from pathlib import Path
import shutil
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

TEMP_DIR = "temp_audio_files"
os.makedirs(TEMP_DIR, exist_ok=True)

def load_pipeline(huggingface_access_token):
    logger.info("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=huggingface_access_token
    )
    if torch.cuda.is_available():
        logger.info("CUDA is available. Moving the pipeline to GPU.")
        pipeline.to(torch.device("cuda"))
    else:
        logger.info("CUDA is not available. Using the pipeline on CPU.")
    return pipeline

@app.post("/diarize")
async def diarize_audio(file: UploadFile, pipeline: Pipeline = Depends(load_pipeline)):
    temp_file = Path(TEMP_DIR) / file.filename
    try:
        logger.info(f"Processing file: {file.filename}")

        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved file to temporary directory: {temp_file}")

        diarization = pipeline(str(temp_file))

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "speaker": f"speaker_{speaker}",
                "start": f"{turn.start:.1f}s",
                "stop": f"{turn.end:.1f}s"       
            })

        logger.info(f"File {file.filename} processed successfully.")
        return JSONResponse(content={"results": results})

    except Exception as e:
        logger.error(f"Failed to process file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_file.exists():
            temp_file.unlink()
            logger.info(f"Temporary file {temp_file} deleted.")
