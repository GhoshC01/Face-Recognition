from __future__ import annotations


class FaceServiceError(Exception):
    """Base class for all domain errors raised by the face verification pipeline."""

    error_code: str = "face_service_error"
    http_status: int = 400

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidImageError(FaceServiceError):
    error_code = "invalid_image"
    http_status = 400


class NoFaceDetectedError(FaceServiceError):
    error_code = "no_face_detected"
    http_status = 422


class MultipleFacesDetectedError(FaceServiceError):
    error_code = "multiple_faces_detected"
    http_status = 422


class LowImageQualityError(FaceServiceError):
    error_code = "low_image_quality"
    http_status = 422


class IdentityNotFoundError(FaceServiceError):
    error_code = "identity_not_found"
    http_status = 404


class IdentityAlreadyExistsError(FaceServiceError):
    error_code = "identity_already_exists"
    http_status = 409


class ModelNotReadyError(FaceServiceError):
    error_code = "model_not_ready"
    http_status = 503


class InconsistentEnrollmentImagesError(FaceServiceError):
    """Raised when two enrollment images do not appear to be the same person
    (embedding similarity below the configured consistency threshold)."""

    error_code = "inconsistent_enrollment_images"
    http_status = 422


class InvalidFrameCountError(FaceServiceError):
    """Raised when a multi-frame verification request submits fewer or more
    frames than the configured allowed range."""

    error_code = "invalid_frame_count"
    http_status = 400


class UnsupportedMediaTypeError(FaceServiceError):
    """Raised when an upload's declared content type is not in the
    configured allow-list -- checked before any image decoding is attempted."""

    error_code = "unsupported_media_type"
    http_status = 415


class PayloadTooLargeError(FaceServiceError):
    """Raised when an upload exceeds the configured size limit."""

    error_code = "payload_too_large"
    http_status = 413


class InvalidEmbeddingError(FaceServiceError):
    """Raised when a vector cannot be treated as a valid face embedding --
    wrong shape, NaN/Inf values, or near-zero magnitude. Distinct from
    ModelNotReadyError: this signals a bad vector produced during inference,
    not a missing/misconfigured model."""

    error_code = "invalid_embedding"
    http_status = 500
