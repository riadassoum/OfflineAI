# -*- coding: utf-8 -*-
# Shared speech-to-text helpers (used by the Transcribe window and dictation).

import os
import sys

from logHandler import log

ADDON_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(ADDON_DIR, "lib")
WHISPER_RATE = 16000


def ensureLib():
	if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)


def importWhisper():
	"""Return (Model, error_text)."""
	ensureLib()
	try:
		from pywhispercpp.model import Model
		return Model, None
	except Exception as e:
		log.error("offlineAI whisper import failed: %s" % e)
		return None, str(e)


def newModel(Model, path, n_threads):
	"""Create a Whisper model that uses the user's thread count (pywhispercpp
	defaults to only 4 threads) and never prints to a console NVDA doesn't have."""
	return Model(path, n_threads=int(n_threads), print_progress=False,
	             print_realtime=False, print_timestamps=False)


def decodeParams(language, translate=False):
	"""Explicit decode parameters. pywhispercpp keeps whatever was set on an
	earlier call, so 'auto' and translate=False must be passed explicitly or a
	previous choice silently sticks."""
	return {
		"language": language if (language and language != "auto") else "auto",
		"translate": bool(translate),
	}


def loadAudio(path, Model=None):
	"""Decode any supported file to a 16 kHz mono float32 numpy array.
	Returns (array, None) or (None, error_text). MP3/FLAC/OGG/WAV are decoded by
	the bundled miniaudio; everything else goes through FFmpeg if installed."""
	from . import audio_decode
	arr, err = audio_decode.decode_to_array(path)
	if arr is not None:
		return arr, None
	if Model is None:
		Model, ierr = importWhisper()
		if Model is None:
			return None, ierr
	try:
		return Model._load_audio(path), None
	except Exception as e:
		return None, str(e)


# --- exporters (always UTF-8; the library's own writers use the Windows ANSI
# code page and fail or corrupt any non-Latin transcript) --------------------

def _ts(centis, sep):
	ms = max(0, int(centis)) * 10
	h, ms = divmod(ms, 3600000)
	m, ms = divmod(ms, 60000)
	s, ms = divmod(ms, 1000)
	return "%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)


def _clean(segments):
	out = []
	for s in segments or []:
		t = (getattr(s, "text", "") or "").strip()
		if t:
			out.append((getattr(s, "t0", 0), getattr(s, "t1", 0), t))
	return out


def plainText(segments):
	return " ".join(t for _a, _b, t in _clean(segments))


def writeTxt(segments, path):
	with open(path, "w", encoding="utf-8") as f:
		f.write(plainText(segments) + "\n")


def writeSrt(segments, path):
	with open(path, "w", encoding="utf-8") as f:
		for i, (t0, t1, text) in enumerate(_clean(segments), 1):
			f.write("%d\n%s --> %s\n%s\n\n" % (
				i, _ts(t0, ","), _ts(t1, ","), text))


def writeVtt(segments, path):
	with open(path, "w", encoding="utf-8") as f:
		f.write("WEBVTT\n\n")
		for t0, t1, text in _clean(segments):
			f.write("%s --> %s\n%s\n\n" % (_ts(t0, "."), _ts(t1, "."), text))


WRITERS = {"txt": writeTxt, "srt": writeSrt, "vtt": writeVtt}
