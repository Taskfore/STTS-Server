# File: server_clean.py
# Cleaned-up FastAPI application using the new router structure.
# This demonstrates the refactored architecture using extracted routers.

import logging
import logging.handlers
import time
import yaml
import webbrowser
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_full_config_for_template,
)

import engine  # TTS Engine interface
from stt_engine import STTEngine  # STT Engine class
import utils  # Utility functions

# Import new router structure
from routers.core import tts
from routers.management import config as config_router, files
from routers import stt, conversation, websocket_stt, websocket_conversation, websocket_conversation_v2

logger = logging.getLogger(__name__)

# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()


def _delayed_browser_open(host: str, port: int):
    """
    Waits for the startup_complete_event, then opens the web browser
    to the server's main page after a short delay.
    """
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning(
                "Server startup did not signal completion within timeout. Browser will not be opened automatically."
            )
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        # Ensure required directories exist
        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        # Load TTS model
        tts_loaded = engine.load_model()
        if not tts_loaded:
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. TTS functionality will not work."
            )
        else:
            logger.info("TTS Model loaded successfully.")
        
        # Initialize and load STT engine
        app.state.stt_engine = STTEngine()
        stt_loaded = app.state.stt_engine.load_model()
        if not stt_loaded:
            logger.warning(
                "WARNING: STT Model failed to load on startup. STT functionality will not work."
            )
        else:
            logger.info("STT Model loaded successfully.")
        
        if tts_loaded:  # Only open browser if TTS (primary functionality) is working
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True,
            )
            browser_thread.start()

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.3",  # Version bump for clean architecture
    lifespan=lifespan,
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Include New Router Structure ---
# Core business logic routers
app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(conversation.router)

# Management routers  
app.include_router(config_router.router)
app.include_router(files.router)

# WebSocket routers
app.include_router(websocket_stt.router)
app.include_router(websocket_conversation.router)
app.include_router(websocket_conversation_v2.router)

# --- Static Files and Templates ---
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )

@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")

# Mount outputs directory
outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

templates = Jinja2Templates(directory=str(ui_static_path))

# --- Main UI Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )

# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, and presets.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()
        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )

# --- Legacy Endpoint Compatibility ---
# These endpoints maintain backward compatibility while using new routers

@app.post("/save_settings", include_in_schema=False)
async def save_settings_legacy(request: Request):
    """Legacy compatibility endpoint - redirects to new config router."""
    return await config_router.save_settings(request)

@app.post("/reset_settings", include_in_schema=False)  
async def reset_settings_legacy():
    """Legacy compatibility endpoint - redirects to new config router."""
    return await config_router.reset_settings()

@app.post("/restart_server", include_in_schema=False)
async def restart_server_legacy():
    """Legacy compatibility endpoint - redirects to new config router."""
    return await config_router.restart_server()

@app.get("/get_reference_files", include_in_schema=False)
async def get_reference_files_legacy():
    """Legacy compatibility endpoint - redirects to new files router."""
    return await files.get_reference_files()

@app.get("/get_predefined_voices", include_in_schema=False)
async def get_predefined_voices_legacy():
    """Legacy compatibility endpoint - redirects to new files router."""
    return await files.get_predefined_voices()

@app.post("/upload_reference", include_in_schema=False)
async def upload_reference_legacy(files_list=File(...)):
    """Legacy compatibility endpoint - redirects to new files router."""
    return await files.upload_reference_audio(files_list)

@app.post("/upload_predefined_voice", include_in_schema=False)
async def upload_predefined_voice_legacy(files_list=File(...)):
    """Legacy compatibility endpoint - redirects to new files router."""
    return await files.upload_predefined_voice(files_list)

# --- Health Check Endpoints ---
@app.get("/health", tags=["System"])
async def health_check():
    """System health check."""
    try:
        health_status = {
            "status": "healthy",
            "version": "2.0.3",
            "components": {
                "tts_loaded": engine.MODEL_LOADED if hasattr(engine, 'MODEL_LOADED') else False,
                "stt_loaded": app.state.stt_engine.model_loaded if hasattr(app.state, 'stt_engine') else False,
                "config_manager": config_manager is not None,
            },
            "paths": {
                "output": str(get_output_path()),
                "reference_audio": str(get_reference_audio_path()),
                "predefined_voices": str(get_predefined_voices_path()),
            }
        }
        
        # Check if all critical components are working
        critical_components = ["tts_loaded", "stt_loaded", "config_manager"]
        all_healthy = all(health_status["components"][comp] for comp in critical_components)
        
        if not all_healthy:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/info", tags=["System"])
async def get_system_info():
    """Get system information and router mapping."""
    return {
        "server": {
            "title": get_ui_title(),
            "version": "2.0.3",
            "architecture": "clean_router_structure",
        },
        "routers": {
            "core": {
                "tts": "Text-to-Speech synthesis endpoints",
                "stt": "Speech-to-Text transcription endpoints", 
                "conversation": "STT→TTS pipeline endpoints"
            },
            "management": {
                "config": "Configuration management endpoints",
                "files": "File upload and management endpoints"
            },
            "websocket": {
                "stt": "Real-time speech transcription",
                "conversation": "Real-time audio conversation",
                "conversation_v2": "New library-based conversation system"
            }
        },
        "features": {
            "legacy_compatibility": True,
            "new_router_structure": True,
            "middleware_support": "conversation_v2",
            "library_integration": "realtime_conversation"
        }
    }

# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()

    logger.info(f"Starting TTS Server (Clean Architecture) on http://{server_host}:{server_port}")
    logger.info(f"API documentation: http://{server_host}:{server_port}/docs")
    logger.info(f"Web UI: http://{server_host}:{server_port}/")
    logger.info("Router structure: Core (TTS/STT/Conversation) + Management (Config/Files) + WebSocket")

    import uvicorn

    uvicorn.run(
        "server_clean:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )