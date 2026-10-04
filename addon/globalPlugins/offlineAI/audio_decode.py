# -*- coding: utf-8 -*-
# Native audio decoding for Whisper, without requiring ffmpeg on PATH.
# Uses the bundled miniaudio (MP3/FLAC/OGG/WAV) to produce a 16 kHz mono float32
# numpy array that Whisper accepts directly. Falls back to ffmpeg only for
# formats miniaudio can't handle (M4A/AAC/WMA).

import os
import sys

import addonHandler
from logHandler import log

addonHandler.initTranslation()

ADDON_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(ADDON_DIR, "lib")

WHISPER_RATE = 16000

# Extensions miniaudio can decode natively.
NATIVE_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".oga")
# Extensions that still need ffmpeg (miniaudio can't decode these).
FFMPEG_EXTS = (".m4a", ".aac", ".wma", ".mp4", ".mkv", ".webm", ".opus")


def _ensure_lib():
	if LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)


def can_decode_natively(path):
	return path.lower().endswith(NATIVE_EXTS)


def decode_to_array(path):
	"""Decode an audio file to a 16 kHz mono float32 numpy array using miniaudio.
	Returns (array, err). Only handles NATIVE_EXTS; other formats return an error
	so the caller can fall back to ffmpeg."""
	if not can_decode_natively(path):
		return None, "not a natively supported format"
	_ensure_lib()
	try:
		import numpy as np
		import miniaudio
	except Exception as e:
		log.error("offlineAI: miniaudio import failed: %s" % e)
		return None, "native decoder unavailable"
	try:
		# Decode and resample to 16 kHz mono signed-16 in one step.
		decoded = miniaudio.decode_file(
			path, output_format=miniaudio.SampleFormat.SIGNED16,
			nchannels=1, sample_rate=WHISPER_RATE)
		# decoded.samples is an array of int16; convert to float32 [-1, 1].
		arr = np.asarray(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
		return arr, None
	except Exception as e:
		log.error("offlineAI: miniaudio decode failed for %s: %s" % (path, e))
		return None, str(e)


def load_for_whisper(path):
	"""Return (media_for_whisper, err): a 16 kHz mono numpy array when the file
	could be decoded natively (WAV of any sample rate, MP3, FLAC, OGG), or the
	original path so Whisper can hand it to FFmpeg."""
	arr, err = decode_to_array(path)
	if arr is not None:
		return arr, None
	if can_decode_natively(path):
		log.warning("offlineAI: native decode failed, will try ffmpeg: %s" % err)
	return path, None


def native_note(path):
	"""Return a short user-facing note about how a file will be handled, or None
	for WAV/native files that need no external tools."""
	ext = os.path.splitext(path)[1].lower()
	if ext == ".wav" or ext in NATIVE_EXTS:
		return None
	import shutil
	if shutil.which("ffmpeg") is None:
		return _(
			"This format ({e}) needs FFmpeg, which was not found. MP3, FLAC, "
			"OGG and WAV work without it.").format(e=ext)
	return None
