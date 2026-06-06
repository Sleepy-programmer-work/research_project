import os
import json
import logging
import subprocess
import whisper
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)
_whisper_model = None

def has_audio_stream(video_path: str) -> bool:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of",
            "default=noprint_wrappers=1:nokey=1", video_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return "audio" in output
    except Exception as e:
        logger.warning(f"ffprobe failed for {video_path}: {e}")
        return False

def get_audio_duration(video_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return float(output)
    except Exception:
        return 0.0

def transcribe_audio(video_path: str, video_id: str) -> tuple[str, bool, float]:
    cache_dir = Path(settings.experiment.get("cache_dir", "./cache")) / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video_id}.json"
    
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["transcript"], data["audio_present"], data["audio_duration"]
            
    audio_present = has_audio_stream(video_path)
    audio_duration = get_audio_duration(video_path) if audio_present else 0.0
    
    if not audio_present:
        logger.debug(f"No audio stream found for {video_id}. Skipping Whisper.")
        data = {"transcript": "", "audio_present": False, "audio_duration": 0.0}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return "", False, 0.0
        
    audio_path = str(cache_dir / f"{video_id}.wav")
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg extraction failed for {video_id}: {e}")
        return "", True, audio_duration

    global _whisper_model
    if _whisper_model is None:
        model_name = settings.models.get("whisper", {}).get("name", "tiny")
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
        
    logger.debug(f"Transcribing audio for {video_id}")
    threshold = settings.models.get("whisper", {}).get("no_speech_threshold", 0.7)
    
    result = _whisper_model.transcribe(
        audio_path,
        condition_on_previous_text=False,
        no_speech_threshold=threshold
    )
    
    transcript = result.get("text", "").strip()
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    data = {"transcript": transcript, "audio_present": True, "audio_duration": audio_duration}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return transcript, True, audio_duration
