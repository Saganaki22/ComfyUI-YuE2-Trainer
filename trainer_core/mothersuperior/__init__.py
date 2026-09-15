"""Mothersuperior artist-training path.

Importing this package silences the HF hub HTTP-request spam (the `httpx`
logger logs every HEAD/GET at INFO, drowning out the training log).
"""
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
