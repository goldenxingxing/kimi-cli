"""Offline analytics derived from session wire logs.

These modules are read-only: they never instrument the runtime, never write to
session data, and never require a database. Everything is reconstructed by
scanning the append-only ``wire.jsonl`` files that sessions already produce.
"""
