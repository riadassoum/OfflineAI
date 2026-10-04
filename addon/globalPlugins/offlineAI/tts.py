# -*- coding: utf-8 -*-
# Text-to-speech with voice cloning for the Offline AI add-on.
# Drives the bundled llama-tts.exe (mainline llama.cpp) against a Qwen3-TTS GGUF
# pair (backbone + mmproj audio decoder). Multilingual, CPU, with optional voice
# cloning from a reference audio file. Output WAV is played through sounddevice.
#
# This is a subprocess architecture: llama-tts.exe writes a WAV, we read and play
# it. No native Python binding is needed for the TTS engine itself.

import os
import re
import sys
import time
import wave
import tempfile
import threading
import subprocess

import addonHandler
from logHandler import log

addonHandler.initTranslation()

ADDON_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(ADDON_DIR, "lib")
TTS_DIR = os.path.join(LIB_DIR, "tts")
TTS_EXE = os.path.join(TTS_DIR, "llama-tts.exe")

# Where short-lived output WAVs go. Set by the UI to a folder under the user's
# models store so nothing unexpected lands on the system drive.
_TEMP_DIR = None


def set_temp_dir(d):
	global _TEMP_DIR
	_TEMP_DIR = d


def _mktemp(suffix, prefix):
	kw = {"suffix": suffix, "prefix": prefix}
	if _TEMP_DIR and os.path.isdir(_TEMP_DIR):
		kw["dir"] = _TEMP_DIR
	return tempfile.mkstemp(**kw)

# Languages the Qwen3-TTS voice model can speak. This list is a limit of the
# model itself, not of the add-on; names are shown in the user's own language.
TTS_LANGUAGE_CODES = ["en", "zh", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"]


def default_language(settings=None):
	"""The speech language to use when the user has not chosen one: the saved
	choice, else NVDA's own language if the voice supports it, else English."""
	from . import langs
	saved = langs.normalize((settings or {}).get("tts_lang") or "")
	if saved in TTS_LANGUAGE_CODES:
		return saved
	nv = langs.nvdaLanguage()
	return nv if nv in TTS_LANGUAGE_CODES else "en"

# The bundled/downloadable Qwen3-TTS model pair.
TTS_MODEL = {
	"name": "Qwen3-TTS 1.7B (multilingual, voice cloning)",
	"backbone": "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
	"backbone_url": "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/"
	                "resolve/main/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
	"mmproj": "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
	"mmproj_url": "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/"
	              "resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
	"backbone_size": "~1.0 GB",
	"mmproj_size": "~450 MB",
}


def engine_present():
	return os.path.isfile(TTS_EXE)


# Track live subprocesses so windows can terminate them on close.
_active_procs = set()
_cancelled = set()
_active_lock = threading.Lock()


def _register(proc):
	with _active_lock:
		_active_procs.add(proc)


def _unregister(proc):
	with _active_lock:
		_active_procs.discard(proc)


def cancel_all():
	"""Terminate any running TTS subprocesses so the model does not keep running
	in the background after a window closes."""
	with _active_lock:
		procs = list(_active_procs)
	stop_playback()
	for p in procs:
		try:
			if p.poll() is None:
				_cancelled.add(p)
				p.terminate()
				try:
					p.wait(timeout=3)
				except Exception:
					p.kill()
		except Exception as e:
			log.warning("offlineAI TTS: could not terminate process: %s" % e)
	with _active_lock:
		_active_procs.clear()


def stop_playback():
	"""Silence whatever the add-on is currently playing."""
	# Only if the audio library is already loaded: there is nothing to stop
	# otherwise, and loading it just to stop it would be wasteful.
	sd = sys.modules.get("sounddevice")
	if sd is not None:
		try:
			sd.stop()
		except Exception:
			pass


def safe_file_name(name):
	"""Turn a user-typed voice name into a valid Windows file name."""
	cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name or "").strip(" .")
	return cleaned or "voice"


def models_present(store_dir):
	return (os.path.isfile(os.path.join(store_dir, TTS_MODEL["backbone"])) and
	        os.path.isfile(os.path.join(store_dir, TTS_MODEL["mmproj"])))


def _startupinfo():
	# Hide the console window on Windows.
	si = None
	try:
		si = subprocess.STARTUPINFO()
		si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
		si.wShowWindow = 0  # SW_HIDE
	except Exception:
		si = None
	return si


def synthesize(text, store_dir, lang="en", speaker_file=None,
               n_threads=None, out_path=None, use_mmap=False):
	"""Run llama-tts to synthesize `text` to a WAV file. Returns (wav_path, err).
	If speaker_file is given, the output clones that voice. use_mmap=False (the
	default) loads the model into RAM instead of memory-mapping it, which is
	lighter on disk churn and page-outs on most systems."""
	if not engine_present():
		return None, _("The text-to-speech engine is not installed.")
	backbone = os.path.join(store_dir, TTS_MODEL["backbone"])
	mmproj = os.path.join(store_dir, TTS_MODEL["mmproj"])
	if not (os.path.isfile(backbone) and os.path.isfile(mmproj)):
		return None, _("The text-to-speech model has not been downloaded yet.")
	if not (text or "").strip():
		return None, _("No text to speak.")

	created = False
	if out_path is None:
		fd, out_path = _mktemp(".wav", "offlineAI_tts_")
		os.close(fd)
		created = True

	cmd = [
		TTS_EXE,
		"-m", backbone,
		"-mm", mmproj,
		"-p", text,
		"-o", out_path,
		"--tts-lang", lang or "en",
	]
	if not use_mmap:
		cmd.append("--no-mmap")
	if n_threads and n_threads > 0:
		cmd += ["-t", str(int(n_threads))]
	if speaker_file and os.path.isfile(speaker_file):
		cmd += ["--tts-speaker-file", speaker_file]

	proc = None
	try:
		# Ensure the bundled DLLs next to the exe are found.
		env = dict(os.environ)
		env["PATH"] = TTS_DIR + os.pathsep + env.get("PATH", "")
		proc = subprocess.Popen(
			cmd, cwd=TTS_DIR, env=env,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			startupinfo=_startupinfo())
		_register(proc)
		try:
			_out, err_b = proc.communicate(timeout=600)
		except subprocess.TimeoutExpired:
			proc.kill()
			proc.communicate()
			return None, _("Speech generation timed out.")
	except Exception as e:
		log.error("offlineAI TTS: subprocess failed: %s" % e)
		return None, str(e)
	finally:
		if proc is not None:
			_unregister(proc)

	if proc in _cancelled or (proc.returncode is not None and proc.returncode < 0):
		_cancelled.discard(proc)
		try:
			if created and os.path.isfile(out_path):
				os.remove(out_path)
		except OSError:
			pass
		return None, _("Speech was cancelled.")
	if not os.path.isfile(out_path) or os.path.getsize(out_path) < 128:
		msg = (err_b or b"").decode("utf-8", "replace")[-300:]
		log.warning("offlineAI TTS: no audio produced: %s" % msg.strip())
		try:
			if created and os.path.isfile(out_path):
				os.remove(out_path)
		except OSError:
			pass
		return None, _("No audio was produced. See the NVDA log for details.")
	return out_path, None


def _time_stretch_wsola(samples, rate, speed, quality="smooth"):
	"""Change playback speed while preserving pitch, using WSOLA (Waveform
	Similarity Overlap-Add). samples is a 1-D float32 numpy array (mono).
	speed >1 shortens (faster), speed <1 lengthens (slower). quality selects a
	parameter tuning: 'smooth' (natural vocal flow) or 'crisp' (smaller frames to
	keep sharp consonants from smearing at high speeds). Returns float32.

	WSOLA copies overlapping windows of the input to the output at a stretched
	rate, nudging each source window to best align with the previous output for
	waveform continuity, which avoids the pitch shift plain resampling causes."""
	import numpy as np
	if speed == 1.0 or samples.size == 0:
		return samples
	speed = float(max(0.5, min(3.0, speed)))

	# Frame parameters by quality preset.
	if quality == "crisp":
		frame_ms, seek_ms = 0.018, 0.003   # smaller frame, tighter search
	else:  # smooth (default)
		frame_ms, seek_ms = 0.030, 0.004
	frame = max(128, int(rate * frame_ms))
	overlap = frame // 2
	seek = max(1, int(rate * seek_ms))
	hop_syn = frame - overlap
	hop_ana = int(round(hop_syn * speed))

	win = np.hanning(frame).astype(np.float32)
	n = samples.size
	out_len = int(n / speed) + frame
	out = np.zeros(out_len, dtype=np.float32)
	win_sum = np.zeros(out_len, dtype=np.float32)

	a = 0
	s = 0
	prev_tail = None

	while a + frame < n and s + frame < out_len:
		if prev_tail is None:
			best = a
		else:
			lo = max(0, a - seek)
			hi = min(n - frame, a + seek)
			best = a
			best_score = -1e30
			seg_ref = prev_tail
			for cand in range(lo, hi + 1):
				seg = samples[cand:cand + overlap]
				if seg.size < seg_ref.size:
					continue
				score = float(np.dot(seg[:seg_ref.size], seg_ref))
				if score > best_score:
					best_score = score
					best = cand
		frame_data = samples[best:best + frame] * win
		out[s:s + frame] += frame_data
		win_sum[s:s + frame] += win
		prev_tail = samples[best + hop_syn: best + hop_syn + overlap].copy()
		a = best + hop_ana
		s += hop_syn

	nz = win_sum > 1e-6
	out[nz] /= win_sum[nz]
	return out[:s + overlap] if s + overlap <= out_len else out


def play_wav(path, speed=1.0, stretch_mode="smooth"):
	"""Play a WAV file through the bundled sounddevice/PortAudio stack.
	speed >1 plays faster. stretch_mode: 'smooth'/'crisp' use pitch-preserving
	WSOLA; 'legacy' uses raw sample-rate scaling (pitch rises). Returns (ok, err).
	"""
	if LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)
	try:
		import numpy as np
		import sounddevice as sd
	except Exception as e:
		log.error("offlineAI TTS: audio import failed: %s" % e)
		return False, str(e)
	try:
		with wave.open(path, "rb") as wf:
			rate = wf.getframerate()
			nch = wf.getnchannels()
			width = wf.getsampwidth()
			frames = wf.readframes(wf.getnframes())
		if width == 2:
			data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
		elif width == 4:
			data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
		elif width == 1:
			data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
		else:
			return False, _("Unsupported WAV sample width.")
		if nch > 1:
			data = data.reshape(-1, nch)
		try:
			spd = float(speed)
		except (TypeError, ValueError):
			spd = 1.0
		play_rate = rate
		if 0.5 <= spd <= 3.0 and spd != 1.0:
			if stretch_mode == "legacy":
				# Raw sample-rate scaling: fast, but pitch rises.
				play_rate = int(rate * spd)
			else:
				# Pitch-preserving WSOLA (smooth or crisp).
				try:
					if nch > 1:
						chans = [_time_stretch_wsola(
							data[:, c].copy(), rate, spd, stretch_mode)
							for c in range(nch)]
						m = min(len(c) for c in chans)
						data = np.stack([c[:m] for c in chans], axis=1)
					else:
						data = _time_stretch_wsola(data, rate, spd, stretch_mode)
				except Exception as e:
					log.warning("offlineAI TTS: time-stretch failed, "
					            "playing at normal speed: %s" % e)
		sd.play(data, play_rate)
		sd.wait()
		return True, None
	except Exception as e:
		log.error("offlineAI TTS: playback failed: %s" % e)
		return False, str(e)


def speak(text, store_dir, lang="en", speaker_file=None, n_threads=None,
          on_status=None, keep_wav=False, speed=1.0, stretch_mode="smooth",
          use_mmap=False):
	"""Synthesize and play in one call (runs synchronously; call from a worker
	thread). on_status(msg) reports progress. Returns (wav_path_or_None, err)."""
	def status(m):
		if on_status:
			on_status(m)
	status(_("Generating speech."))
	wav, err = synthesize(text, store_dir, lang=lang, speaker_file=speaker_file,
	                      n_threads=n_threads, use_mmap=use_mmap)
	if err:
		status(_("Speech generation failed: {e}").format(e=err))
		return None, err
	status(_("Playing."))
	ok, perr = play_wav(wav, speed=speed, stretch_mode=stretch_mode)
	if not keep_wav:
		try:
			os.remove(wav)
		except OSError:
			pass
		wav = None
	if not ok:
		return None, perr
	return wav, None


# --- sentence splitting + streaming document reader ------------------------

_re = re

# Split on sentence-ending punctuation (Latin + Arabic + CJK), keeping it simple
# and robust. Falls back to chunking very long runs without terminators.
_SENT_END = _re.compile(r"[^.!?\u061f\u06d4\u3002\uff01\uff1f\n]+"
                        r"[.!?\u061f\u06d4\u3002\uff01\uff1f\n]*", _re.UNICODE)


def split_sentences(text, max_chars=300):
	"""Split text into speakable chunks (sentences), merging tiny fragments and
	hard-splitting overly long runs so each chunk renders quickly."""
	text = (text or "").strip()
	if not text:
		return []
	raw = [m.group(0).strip() for m in _SENT_END.finditer(text)]
	raw = [s for s in raw if s]
	chunks = []
	buf = ""
	for s in raw:
		# Hard-split a single very long sentence on spaces.
		while len(s) > max_chars:
			cut = s.rfind(" ", 0, max_chars)
			if cut <= 0:
				cut = max_chars
			chunks.append(s[:cut].strip())
			s = s[cut:].strip()
		# Merge short fragments up to a comfortable size.
		if len(buf) + len(s) + 1 <= max_chars:
			buf = (buf + " " + s).strip()
		else:
			if buf:
				chunks.append(buf)
			buf = s
	if buf:
		chunks.append(buf)
	return chunks


class DocumentReader(object):
	"""Reads a long text aloud sentence by sentence with look-ahead: while one
	chunk plays, the next is already being synthesized. Supports pause, resume,
	and stop. All heavy work happens on internal threads."""

	def __init__(self, store_dir, lang="en", speaker_file=None, n_threads=None,
	             prerender=True, on_status=None, speed=1.0, stretch_mode="smooth",
	             use_mmap=False):
		self.store_dir = store_dir
		self.lang = lang
		self.speaker_file = speaker_file
		self.n_threads = n_threads
		self.prerender = prerender
		self.on_status = on_status
		self.speed = speed
		self.stretch_mode = stretch_mode
		self.use_mmap = use_mmap
		self._chunks = []
		self._stop = threading.Event()
		self._pause = threading.Event()
		self._genThread = None
		self._playThread = None
		self._queue = None
		self._running = False

	def _status(self, m):
		if self.on_status:
			self.on_status(m)

	def start(self, text):
		import queue
		self._chunks = split_sentences(text)
		if not self._chunks:
			self._status(_("Nothing to read."))
			return False
		self._stop.clear()
		self._pause.clear()
		self._running = True
		# Bounded queue: hold at most 2 rendered clips ahead (or 1 if prerender
		# off, which effectively serializes generate/play with a breath between).
		self._queue = queue.Queue(maxsize=2 if self.prerender else 1)
		self._genThread = threading.Thread(target=self._generator, daemon=True)
		self._playThread = threading.Thread(target=self._player, daemon=True)
		self._genThread.start()
		self._playThread.start()
		return True

	def _generator(self):
		for idx, chunk in enumerate(self._chunks):
			if self._stop.is_set():
				break
			wav, err = synthesize(chunk, self.store_dir, lang=self.lang,
			                      speaker_file=self.speaker_file,
			                      n_threads=self.n_threads, use_mmap=self.use_mmap)
			if self._stop.is_set():
				if wav:
					try:
						os.remove(wav)
					except OSError:
						pass
				break
			if err:
				# Skip a failed chunk but keep going.
				log.warning("offlineAI reader: chunk %d failed: %s" % (idx, err))
				continue
			if not self._put((idx, wav)):
				try:
					os.remove(wav)
				except OSError:
					pass
				break
		self._put(None)  # sentinel: no more clips

	def _put(self, item):
		"""Queue an item, giving up if reading is stopped while the queue is full
		(the player is gone then and nothing would ever take it)."""
		import queue
		while True:
			try:
				self._queue.put(item, timeout=0.2)
				return True
			except queue.Full:
				if self._stop.is_set():
					return False

	def _player(self):
		total = len(self._chunks)
		import queue
		while not self._stop.is_set():
			try:
				item = self._queue.get(timeout=0.2)
			except queue.Empty:
				continue
			if item is None:
				break
			idx, wav = item
			# Honor pause before starting the next clip.
			while self._pause.is_set() and not self._stop.is_set():
				time.sleep(0.1)
			if self._stop.is_set():
				try:
					os.remove(wav)
				except OSError:
					pass
				break
			self._status(_("Sentence {n} of {t}").format(n=idx + 1, t=total))
			play_wav(wav, speed=self.speed, stretch_mode=self.stretch_mode)
			try:
				os.remove(wav)
			except OSError:
				pass
		self._running = False
		if not self._stop.is_set():
			self._status(_("Finished reading."))

	def pause(self):
		self._pause.set()
		self._status(_("Paused."))

	def resume(self):
		self._pause.clear()
		self._status(_("Resumed."))

	def stop(self):
		self._stop.set()
		self._pause.clear()
		# Cut the sentence being spoken and the one being generated right away,
		# instead of letting both run to their end.
		cancel_all()
		# Drain queue to unblock the generator.
		try:
			while self._queue and not self._queue.empty():
				item = self._queue.get_nowait()
				if item and item[1]:
					try:
						os.remove(item[1])
					except OSError:
						pass
		except Exception:
			pass
		self._running = False

	def is_running(self):
		return self._running
