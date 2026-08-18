from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.exceptions import (
    IdentityNotFoundError,
    InvalidFrameCountError,
    LowImageQualityError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.core.quality import QualityResult
from app.core.recognizer import FaceRecognizer
from app.core.vector_store import VectorStore
from app.schemas.common import quality_result_to_schema
from app.schemas.verification import (
    FaceVerificationResponse,
    FrameDiagnostic,
    IdentificationMatch,
    IdentificationResponse,
    MultiFrameVerificationResponse,
    VerificationResponse,
)

# Domain errors that mean "this particular frame is unusable" rather than
# "the whole request is invalid" -- caught per-frame in verify_multi_frame so
# one bad frame never aborts the others.
_PER_FRAME_RECOVERABLE_ERRORS = (NoFaceDetectedError, MultipleFacesDetectedError, LowImageQualityError)


@dataclass
class _FrameOutcome:
    index: int
    valid: bool
    external_id: str | None = None
    similarity: float | None = None
    passed_threshold: bool | None = None
    detection_score: float | None = None
    quality: QualityResult | None = None
    rejection_reason: str | None = None


class VerificationService:
    """Use cases backed by enrolled identities in the vector store:
    1:1 verification against a claimed external_id, and 1:N identification.
    """

    def __init__(
        self,
        recognizer: FaceRecognizer,
        vector_store: VectorStore,
        verification_threshold: float,
        identification_threshold: float,
        identification_top_k: int,
        multi_frame_min_frames: int = 3,
        multi_frame_max_frames: int = 5,
        multi_frame_min_valid_frames: int = 2,
        multi_frame_min_agreeing_frames: int = 2,
        multi_frame_consensus_ratio: float = 0.6,
    ) -> None:
        self.recognizer = recognizer
        self.vector_store = vector_store
        self.verification_threshold = verification_threshold
        self.identification_threshold = identification_threshold
        self.identification_top_k = identification_top_k
        self.multi_frame_min_frames = multi_frame_min_frames
        self.multi_frame_max_frames = multi_frame_max_frames
        self.multi_frame_min_valid_frames = multi_frame_min_valid_frames
        self.multi_frame_min_agreeing_frames = multi_frame_min_agreeing_frames
        self.multi_frame_consensus_ratio = multi_frame_consensus_ratio

    def verify(self, external_id: str, image) -> VerificationResponse:
        enrolled_embeddings = self.vector_store.get_embeddings(external_id)
        if not enrolled_embeddings:
            raise IdentityNotFoundError(f"No enrolled embeddings found for external_id='{external_id}'")

        result = self.recognizer.process(image, strict_single_face=False)

        similarities = [float(np.dot(result.embedding, stored)) for stored in enrolled_embeddings]
        best_score = max(similarities)
        verified = best_score >= self.verification_threshold

        return VerificationResponse(
            external_id=external_id,
            verified=verified,
            result="PASS" if verified else "FAIL",
            similarity_score=best_score,
            threshold=self.verification_threshold,
            detection_score=result.detection_score,
            quality=quality_result_to_schema(result.quality),
        )

    def verify_or_identify(self, image, external_id: str | None = None) -> FaceVerificationResponse:
        """The main HRMS-facing endpoint's logic: POST /api/v1/faces/verify.

        Requires exactly one face (unlike `verify`/`identify` above, which
        pick the largest face when several are present) -- this is the
        stricter "basic API" contract for the primary verification path.

        Mode B: external_id supplied -> 1:1 compare against that identity's
        enrolled embeddings. Mode A: external_id omitted -> 1:N search for
        the single best matching identity across everyone enrolled. In both
        modes, a similarity below threshold is always FAIL -- the closest
        available candidate is never forced to PASS.
        """
        result = self.recognizer.process(image, strict_single_face=True)

        if external_id is not None:
            enrolled_embeddings = self.vector_store.get_embeddings(external_id)
            if not enrolled_embeddings:
                raise IdentityNotFoundError(f"No enrolled embeddings found for external_id='{external_id}'")

            similarity = max(float(np.dot(result.embedding, stored)) for stored in enrolled_embeddings)
            verified = similarity >= self.verification_threshold

            return FaceVerificationResponse(
                verified=verified,
                status="PASS" if verified else "FAIL",
                external_id=external_id,
                similarity=similarity,
                threshold=self.verification_threshold,
                mode="verification",
                detection_score=result.detection_score,
                quality=quality_result_to_schema(result.quality),
            )

        candidates = self.vector_store.search(result.embedding, top_k=1)
        best = candidates[0] if candidates else None
        similarity = best.similarity_score if best is not None else 0.0
        verified = best is not None and similarity >= self.identification_threshold

        return FaceVerificationResponse(
            verified=verified,
            status="PASS" if verified else "FAIL",
            external_id=best.external_id if verified else None,
            similarity=similarity,
            threshold=self.identification_threshold,
            mode="identification",
            detection_score=result.detection_score,
            quality=quality_result_to_schema(result.quality),
        )

    def verify_multi_frame(
        self,
        images: list,
        external_id: str | None = None,
        debug: bool = False,
    ) -> MultiFrameVerificationResponse:
        """Multi-frame verification for webcam-based capture: run the same
        single-face pipeline independently over several frames and only PASS
        when enough of them agree -- both on *who* matched (identity
        consistency) and that the match *individually* cleared the
        similarity threshold (threshold agreement). This is deliberately two
        conditions combined, not one: a single strong frame can never carry a
        PASS by itself, guarded by both `multi_frame_min_agreeing_frames` (an
        absolute floor) and `multi_frame_consensus_ratio` (a proportion of
        the valid frames).

        A `None` entry in `images` represents a frame that failed to decode
        upstream -- it is treated the same as any other per-frame rejection,
        so one corrupt frame among several good ones never fails the whole
        request.
        """
        frame_count = len(images)
        if not (self.multi_frame_min_frames <= frame_count <= self.multi_frame_max_frames):
            raise InvalidFrameCountError(
                f"Expected between {self.multi_frame_min_frames} and "
                f"{self.multi_frame_max_frames} frames, got {frame_count}",
                min_frames=self.multi_frame_min_frames,
                max_frames=self.multi_frame_max_frames,
                frames_submitted=frame_count,
            )

        mode = "verification" if external_id is not None else "identification"
        threshold = self.verification_threshold if external_id is not None else self.identification_threshold

        enrolled_embeddings = None
        if external_id is not None:
            enrolled_embeddings = self.vector_store.get_embeddings(external_id)
            if not enrolled_embeddings:
                raise IdentityNotFoundError(f"No enrolled embeddings found for external_id='{external_id}'")

        outcomes = [
            self._process_frame(index, image, external_id, enrolled_embeddings, threshold)
            for index, image in enumerate(images)
        ]
        valid_outcomes = [o for o in outcomes if o.valid]

        if len(valid_outcomes) < self.multi_frame_min_valid_frames:
            return self._build_multi_frame_response(
                verified=False,
                external_id=None,
                similarity=None,
                threshold=threshold,
                mode=mode,
                frames_submitted=frame_count,
                frames_valid=len(valid_outcomes),
                frames_agreeing=0,
                consensus_ratio=0.0,
                reasons=["insufficient_valid_frames"],
                outcomes=outcomes,
                debug=debug,
            )

        # Group valid frames by which identity they matched, regardless of
        # whether each individual frame cleared the threshold -- the
        # "identity consistency" half of the decision.
        groups: dict[str, list[_FrameOutcome]] = {}
        for outcome in valid_outcomes:
            if outcome.external_id is not None:
                groups.setdefault(outcome.external_id, []).append(outcome)

        leading_id, leading_frames = (None, [])
        if groups:
            leading_id, leading_frames = max(groups.items(), key=lambda pair: len(pair[1]))

        agreeing_count = sum(1 for o in leading_frames if o.passed_threshold)
        consensus_ratio = agreeing_count / len(valid_outcomes)

        verified = (
            leading_id is not None
            and agreeing_count >= self.multi_frame_min_agreeing_frames
            and consensus_ratio >= self.multi_frame_consensus_ratio
        )

        reasons: list[str] = []
        if not verified:
            if leading_id is None:
                reasons.append("no_matching_identity")
            elif agreeing_count < self.multi_frame_min_agreeing_frames:
                reasons.append("insufficient_agreeing_frames")
            else:
                reasons.append("consensus_ratio_not_met")

        representative_similarity = (
            float(np.mean([o.similarity for o in leading_frames])) if leading_frames else None
        )
        final_external_id = external_id if external_id is not None else (leading_id if verified else None)

        return self._build_multi_frame_response(
            verified=verified,
            external_id=final_external_id,
            similarity=representative_similarity,
            threshold=threshold,
            mode=mode,
            frames_submitted=frame_count,
            frames_valid=len(valid_outcomes),
            frames_agreeing=agreeing_count,
            consensus_ratio=consensus_ratio,
            reasons=reasons,
            outcomes=outcomes,
            debug=debug,
        )

    def _process_frame(self, index, image, external_id, enrolled_embeddings, threshold) -> _FrameOutcome:
        if image is None:
            return _FrameOutcome(index=index, valid=False, rejection_reason="invalid_image")

        try:
            result = self.recognizer.process(image, strict_single_face=True)
        except _PER_FRAME_RECOVERABLE_ERRORS as exc:
            return _FrameOutcome(index=index, valid=False, rejection_reason=exc.error_code)

        if external_id is not None:
            candidate_id = external_id
            similarity = max(float(np.dot(result.embedding, stored)) for stored in enrolled_embeddings)
        else:
            candidates = self.vector_store.search(result.embedding, top_k=1)
            candidate_id = candidates[0].external_id if candidates else None
            similarity = candidates[0].similarity_score if candidates else 0.0

        return _FrameOutcome(
            index=index,
            valid=True,
            external_id=candidate_id,
            similarity=similarity,
            passed_threshold=candidate_id is not None and similarity >= threshold,
            detection_score=result.detection_score,
            quality=result.quality,
        )

    def _build_multi_frame_response(
        self,
        *,
        verified: bool,
        external_id: str | None,
        similarity: float | None,
        threshold: float,
        mode: str,
        frames_submitted: int,
        frames_valid: int,
        frames_agreeing: int,
        consensus_ratio: float,
        reasons: list[str],
        outcomes: list[_FrameOutcome],
        debug: bool,
    ) -> MultiFrameVerificationResponse:
        frame_diagnostics = None
        if debug:
            frame_diagnostics = [
                FrameDiagnostic(
                    frame_index=o.index,
                    valid=o.valid,
                    external_id=o.external_id,
                    similarity=o.similarity,
                    passed_threshold=o.passed_threshold,
                    detection_score=o.detection_score,
                    quality=quality_result_to_schema(o.quality) if o.quality is not None else None,
                    rejection_reason=o.rejection_reason,
                )
                for o in outcomes
            ]

        return MultiFrameVerificationResponse(
            verified=verified,
            status="PASS" if verified else "FAIL",
            external_id=external_id,
            similarity=similarity,
            threshold=threshold,
            mode=mode,
            frames_submitted=frames_submitted,
            frames_valid=frames_valid,
            frames_agreeing=frames_agreeing,
            consensus_ratio=consensus_ratio,
            required_consensus_ratio=self.multi_frame_consensus_ratio,
            reasons=reasons,
            frames=frame_diagnostics,
        )

    def identify(self, image, top_k: int | None = None) -> IdentificationResponse:
        result = self.recognizer.process(image, strict_single_face=False)
        candidates = self.vector_store.search(result.embedding, top_k=top_k or self.identification_top_k)

        matches = [
            IdentificationMatch(external_id=c.external_id, similarity_score=c.similarity_score)
            for c in candidates
            if c.similarity_score >= self.identification_threshold
        ]

        return IdentificationResponse(
            matches=matches,
            threshold=self.identification_threshold,
            detection_score=result.detection_score,
        )
