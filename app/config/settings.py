from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration for the Face Verification API.

    Every value has a safe default so the service is runnable out of the box;
    production deployments override via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    app_name: str = "Face Verification API"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Security ---
    api_key_enabled: bool = False
    api_key: str = "changeme"
    # Empty by default: this is a server-to-server API, not a browser client,
    # so CORS is off entirely unless a caller explicitly needs browser access.
    # Never combine a "*" origin with credentials -- enforced in main.py
    # regardless of what's configured here.
    cors_allow_origins: list[str] = []
    enforce_https: bool = False
    request_timeout_seconds: float = 30.0
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0

    # --- Models ---
    models_dir: str = "models"
    detector_model_path: str = "models/det_500m.onnx"
    recognizer_model_path: str = "models/w600k_mbf.onnx"
    onnx_intra_op_threads: int = 0
    onnx_inter_op_threads: int = 0

    # --- Detection ---
    detector_input_size: int = 640
    detector_confidence_threshold: float = 0.5
    detector_nms_threshold: float = 0.4
    max_faces_to_consider: int = 5
    strict_single_face_on_enroll: bool = True

    # --- Recognition ---
    recognizer_input_size: int = 112
    embedding_dimension: int = 512
    verification_similarity_threshold: float = 0.36
    identification_similarity_threshold: float = 0.40
    identification_top_k: int = 5

    # --- Enrollment ---
    enrollment_duplicate_policy: Literal["reject", "replace"] = "reject"
    # Raw enrollment photos are saved under <enrollment_images_dir>/<external_id>/<embedding_id>.jpg
    # for operator/audit reference. This is a deliberate deviation from the
    # "embeddings only" minimization stance described in SECURITY.md -- set to
    # "" to disable and keep only derived embeddings, matching the old default.
    enrollment_images_dir: str = "images"

    # --- Multi-frame verification ---
    multi_frame_min_frames: int = 3
    multi_frame_max_frames: int = 5
    multi_frame_min_valid_frames: int = 2
    multi_frame_min_agreeing_frames: int = 2
    multi_frame_consensus_ratio: float = 0.6

    # --- Quality gates ---
    quality_min_detection_confidence: float = 0.5
    quality_min_face_width_px: int = 60
    quality_min_face_height_px: int = 60
    quality_min_face_area_ratio: float = 0.02
    quality_min_brightness: float = 40.0
    quality_max_brightness: float = 230.0
    quality_min_sharpness: float = 60.0
    # Compare-only: size/brightness/confidence gates are off. This is the
    # Laplacian-variance floor for "extremely blurry / unusable" — typical
    # live captures sit well above it; only a smear fails.
    compare_min_sharpness: float = 8.0

    # --- Storage ---
    faiss_index_dir: str = "storage/faiss"
    metadata_dir: str = "storage/metadata"
    faiss_index_filename: str = "index.faiss"
    metadata_filename: str = "metadata.json"

    # --- Uploads ---
    max_upload_size_mb: int = 8
    allowed_content_types: list[str] = ["image/jpeg", "image/png", "image/webp"]
    # Timeout for fetching image1_url / image2_url (S3 presigned or public HTTPS).
    remote_image_timeout_seconds: float = 10.0
    remote_image_max_redirects: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
