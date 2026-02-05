import os
import uuid
import shutil
import gdown
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI()
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "id": "Indonesian",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "pa": "Punjabi",
    "ur": "Urdu"
}
print("Loading model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded.")


VALID_API_KEY = "hackathon-secret-key-123"


def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)

def download_from_drive(url: str, output_path: str):
    try:
        
        gdown.download(url, output_path, quiet=True, fuzzy=True)
        if not os.path.exists(output_path):
            raise Exception("Download failed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download file. Ensure link is public. Error: {str(e)}")

@app.post("/detect-language")
async def detect_language(
    background_tasks: BackgroundTasks,
    gdrive_link: str, 
    x_api_key: str = Header(None)
):


    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    
    unique_filename = f"audio_{uuid.uuid4()}.mp3"
    
    print(f"Downloading: {gdrive_link}")
    download_from_drive(gdrive_link, unique_filename)

    try:
    
        segments, info = model.transcribe(unique_filename, beam_size=1)
        
        detected_lang = info.language
        confidence = info.language_probability
        try:
            full_language_name = LANGUAGE_MAP.get(detected_lang, detected_lang)
        except:
            full_language_name = detected_lang
        
        background_tasks.add_task(cleanup_file, unique_filename)

        return {
            "status": "success",
            "language": full_language_name,
            "confidence": round(confidence, 2),
            "original_link": gdrive_link
        }

    except Exception as e:
        background_tasks.add_task(cleanup_file, unique_filename)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

def home():
    return {"message": "Language Detection API is Active"}