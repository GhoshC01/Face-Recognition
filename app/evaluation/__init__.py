"""Offline accuracy benchmarking for the FaceVerification pipeline.

Deliberately separate from app/services: this module is not part of the
request-serving API. It is meant to be run against an independent, labeled
dataset that is never used as production enrollment data, and by default it
never touches the production FAISS index at all.
"""
