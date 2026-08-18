# Models

Place the following ONNX models in this directory (not committed to version control — see `.gitignore`):

- `det_500m.onnx` — SCRFD-500MF face detector (outputs box + 5-point landmarks per stride: 8/16/32).
- `w600k_mbf.onnx` — MobileFaceNet ArcFace embedding model (112x112 input, 512-d L2-normalized output).

Both are part of InsightFace's `buffalo_sc` / antelope model packs. `app/core/detector.py` and
`app/core/embedding.py` assume these exact export conventions (output tensor order, mean/std
normalization, 5-point landmark layout). If you substitute different model files, verify their
input/output signature matches (`onnx` package, `onnx.load(path).graph.output`) before relying on
detection/embedding output.

Override the paths via `DETECTOR_MODEL_PATH` / `RECOGNIZER_MODEL_PATH` env vars if you keep models
elsewhere.
