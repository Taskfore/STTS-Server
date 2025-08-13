# File: routers/management/files.py
# File upload and management endpoints

import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

import utils
from config import (
    get_reference_audio_path,
    get_predefined_voices_path,
    config_manager
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["File Management"])


@router.post("/reference-audio/upload")
async def upload_reference_audio(files: List[UploadFile] = File(...)):
    """
    Handles uploading of reference audio files (.wav, .mp3) for voice cloning.
    Validates files and saves them to the configured reference audio path.
    """
    logger.info(f"Request to upload reference audio with {len(files)} file(s).")
    ref_path = get_reference_audio_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = ref_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError("Invalid file type. Only .wav and .mp3 are allowed.")

            if destination_path.exists():
                logger.info(
                    f"Reference file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded reference file to: {destination_path}"
            )

            max_duration = config_manager.get_int(
                "audio_output.max_reference_duration_sec", 30
            )
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration
            )
            if not is_valid:
                logger.warning(
                    f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to reference audio completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@router.post("/predefined-voices/upload")
async def upload_predefined_voice(files: List[UploadFile] = File(...)):
    """
    Handles uploading of predefined voice files (.wav, .mp3).
    Validates files and saves them to the configured predefined voices path.
    """
    logger.info(f"Request to upload predefined voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            # Basic validation (can be extended if predefined voices have specific requirements)
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )  # No duration limit for predefined
            if not is_valid:
                logger.warning(
                    f"Uploaded predefined voice '{safe_filename}' failed basic validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = (
        utils.get_predefined_voices()
    )  # Fetches formatted list
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,  # List of raw filenames uploaded
        "all_predefined_voices": all_current_predefined_voices,  # Formatted list for UI
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to predefined voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@router.get("/reference-audio", response_model=List[str])
async def get_reference_files():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request for reference audio files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )


@router.get("/predefined-voices", response_model=List[Dict[str, str]])
async def get_predefined_voices():
    """Returns a list of predefined voices with display names and filenames."""
    logger.debug("Request for predefined voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )


@router.delete("/reference-audio/{filename}")
async def delete_reference_audio(filename: str):
    """Delete a reference audio file."""
    try:
        safe_filename = utils.sanitize_filename(filename)
        ref_path = get_reference_audio_path(ensure_absolute=True)
        file_path = ref_path / safe_filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Reference audio file '{filename}' not found"
            )
        
        file_path.unlink()
        logger.info(f"Deleted reference audio file: {file_path}")
        
        return {
            "message": f"Reference audio file '{filename}' deleted successfully",
            "remaining_files": utils.get_valid_reference_files()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting reference audio file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete reference audio file: {str(e)}"
        )


@router.delete("/predefined-voices/{filename}")
async def delete_predefined_voice(filename: str):
    """Delete a predefined voice file."""
    try:
        safe_filename = utils.sanitize_filename(filename)
        voices_path = get_predefined_voices_path(ensure_absolute=True)
        file_path = voices_path / safe_filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{filename}' not found"
            )
        
        file_path.unlink()
        logger.info(f"Deleted predefined voice file: {file_path}")
        
        return {
            "message": f"Predefined voice file '{filename}' deleted successfully",
            "remaining_voices": utils.get_predefined_voices()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting predefined voice file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete predefined voice file: {str(e)}"
        )


@router.get("/reference-audio/{filename}/info")
async def get_reference_audio_info(filename: str):
    """Get information about a specific reference audio file."""
    try:
        safe_filename = utils.sanitize_filename(filename)
        ref_path = get_reference_audio_path(ensure_absolute=True)
        file_path = ref_path / safe_filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{filename}' not found"
            )
        
        # Get file stats
        file_stats = file_path.stat()
        
        # Validate audio file
        max_duration = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, validation_msg = utils.validate_reference_audio(file_path, max_duration)
        
        return {
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": file_stats.st_size,
            "modified_time": file_stats.st_mtime,
            "validation": {
                "is_valid": is_valid,
                "message": validation_msg
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reference audio info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get reference audio information: {str(e)}"
        )


@router.get("/predefined-voices/{filename}/info")
async def get_predefined_voice_info(filename: str):
    """Get information about a specific predefined voice file."""
    try:
        safe_filename = utils.sanitize_filename(filename)
        voices_path = get_predefined_voices_path(ensure_absolute=True)
        file_path = voices_path / safe_filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{filename}' not found"
            )
        
        # Get file stats
        file_stats = file_path.stat()
        
        # Basic validation
        is_valid, validation_msg = utils.validate_reference_audio(file_path, max_duration_sec=None)
        
        return {
            "filename": safe_filename,
            "path": str(file_path),
            "size_bytes": file_stats.st_size,
            "modified_time": file_stats.st_mtime,
            "validation": {
                "is_valid": is_valid,
                "message": validation_msg
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predefined voice info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get predefined voice information: {str(e)}"
        )


@router.get("/storage/usage")
async def get_storage_usage():
    """Get storage usage information for audio files."""
    try:
        def get_directory_size(path: Path) -> int:
            """Get total size of all files in directory."""
            total_size = 0
            if path.exists() and path.is_dir():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            return total_size
        
        ref_path = get_reference_audio_path(ensure_absolute=True)
        voices_path = get_predefined_voices_path(ensure_absolute=True)
        
        reference_size = get_directory_size(ref_path)
        predefined_size = get_directory_size(voices_path)
        
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()
        
        return {
            "reference_audio": {
                "path": str(ref_path),
                "size_bytes": reference_size,
                "size_mb": round(reference_size / (1024 * 1024), 2),
                "file_count": len(reference_files)
            },
            "predefined_voices": {
                "path": str(voices_path),
                "size_bytes": predefined_size,
                "size_mb": round(predefined_size / (1024 * 1024), 2),
                "file_count": len(predefined_voices)
            },
            "total": {
                "size_bytes": reference_size + predefined_size,
                "size_mb": round((reference_size + predefined_size) / (1024 * 1024), 2),
                "file_count": len(reference_files) + len(predefined_voices)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting storage usage: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to get storage usage information"
        )


@router.post("/cleanup")
async def cleanup_invalid_files():
    """Remove invalid audio files from both reference and predefined directories."""
    try:
        removed_files = []
        errors = []
        
        # Clean reference audio files
        ref_path = get_reference_audio_path(ensure_absolute=True)
        max_duration = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        
        if ref_path.exists():
            for file_path in ref_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [".wav", ".mp3"]:
                    try:
                        is_valid, msg = utils.validate_reference_audio(file_path, max_duration)
                        if not is_valid:
                            file_path.unlink()
                            removed_files.append({
                                "file": str(file_path.name),
                                "type": "reference_audio",
                                "reason": msg
                            })
                            logger.info(f"Removed invalid reference audio: {file_path}")
                    except Exception as e:
                        errors.append({
                            "file": str(file_path.name),
                            "error": str(e)
                        })
        
        # Clean predefined voice files
        voices_path = get_predefined_voices_path(ensure_absolute=True)
        
        if voices_path.exists():
            for file_path in voices_path.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [".wav", ".mp3"]:
                    try:
                        is_valid, msg = utils.validate_reference_audio(file_path, max_duration_sec=None)
                        if not is_valid:
                            file_path.unlink()
                            removed_files.append({
                                "file": str(file_path.name),
                                "type": "predefined_voice",
                                "reason": msg
                            })
                            logger.info(f"Removed invalid predefined voice: {file_path}")
                    except Exception as e:
                        errors.append({
                            "file": str(file_path.name),
                            "error": str(e)
                        })
        
        return {
            "message": f"Cleanup completed. Removed {len(removed_files)} invalid files.",
            "removed_files": removed_files,
            "errors": errors,
            "remaining": {
                "reference_files": utils.get_valid_reference_files(),
                "predefined_voices": utils.get_predefined_voices()
            }
        }
        
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup files: {str(e)}"
        )