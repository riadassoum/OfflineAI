# -*- coding: utf-8 -*-
# Live dictation for the Offline AI add-on.
#
# Press the gesture to start recording from the default microphone, press it
# again to stop. The audio is transcribed locally with Whisper and the text is
# inserted where the user wants it:
#
#   current   - into whatever has the focus when the text is ready
#   origin    - into the window that had the focus when recording stopped, even
#               if the user has moved elsewhere meanwhile (focus then returns)
#   clipboard - only copied
#
# Nothing is ever written to disk: audio stays in memory as 16 kHz float32 and
# is discarded as soon as it has been transcribed.

import os
import ctypes
import threading
import queue

import addonHandler
from logHandler import log

addonHandler.initTranslation()

WHISPER_SAMPLE_RATE = 16000
# whisper.cpp returns nothing at all for clips shorter than one second.
MIN_SAMPLES = int(WHISPER_SAMPLE_RATE * 1.25)
# Hard cap so a forgotten recording cannot eat RAM for ever (30 minutes).
MAX_SECONDS = 30 * 60


def _callMain(func, *args):
	"""Run func on NVDA's main thread (speech and UI are not thread-safe)."""
	try:
		import wx
		wx.CallAfter(func, *args)
	except Exception:
		func(*args)


class DictationController(object):
	"""Singleton controller toggled by a gesture."""
	_instance = None

	def __init__(self):
		self.recording = False
		self._frames = []
		self._nframes = 0
		self._stream = None
		self._model = None
		self._loaded_key = None
		self._modelLock = threading.Lock()   # one transcription at a time
		self._jobs = queue.Queue()
		self._worker = None
		self._pending = 0
		self._cueVolume = 60
		self._cueMute = False
		self._on_status = None

	@classmethod
	def get(cls):
		if cls._instance is None:
			cls._instance = cls()
		return cls._instance

	# -- cues --
	def _beep(self, sequence):
		"""Play a short tone sequence: list of (freq, ms)."""
		if self._cueMute:
			return
		try:
			import tones
			for freq, ms in sequence:
				tones.beep(freq, ms, left=self._cueVolume, right=self._cueVolume)
		except Exception:
			pass

	def set_cue_volume(self, volume, mute=False):
		self._cueVolume = int(volume)
		self._cueMute = bool(mute)

	def _say(self, msg):
		cb = self._on_status
		if cb:
			_callMain(cb, msg)

	# -- recording --
	def abort(self, on_status):
		"""Cancel an in-progress recording without transcribing/pasting."""
		self._on_status = on_status
		self._closeStream()
		self.recording = False
		self._frames = []
		self._nframes = 0
		self._beep([(880, 40), (440, 40)])
		self._say(_("Dictation cancelled."))

	def shutdown(self):
		"""Called when the add-on is unloaded."""
		self._closeStream()
		self.recording = False
		self._frames = []
		with self._modelLock:
			self._model = None
			self._loaded_key = None

	def _closeStream(self):
		st, self._stream = self._stream, None
		if st is not None:
			try:
				st.stop()
				st.close()
			except Exception:
				pass

	def toggle(self, model_path, on_status, options=None):
		"""Start if idle, stop+transcribe if recording. options is a dict:
		language, n_threads, fast, target, paste_delay, restore_clipboard."""
		self._on_status = on_status
		options = dict(options or {})
		if self.recording:
			self._stop_and_transcribe(model_path, options)
		else:
			self._start(model_path, options)

	def _start(self, model_path, options):
		from . import stt
		stt.ensureLib()
		try:
			import sounddevice as sd
		except Exception as e:
			log.error("offlineAI dictation: audio import failed: %s" % e)
			self._say(_("Microphone support could not be loaded: {e}").format(e=str(e)))
			return
		self._frames = []
		self._nframes = 0
		limit = WHISPER_SAMPLE_RATE * MAX_SECONDS

		def callback(indata, frames, time_info, status):
			if self._nframes < limit:
				self._frames.append(indata.copy())
				self._nframes += frames

		try:
			self._stream = sd.InputStream(
				samplerate=WHISPER_SAMPLE_RATE, channels=1, dtype="float32",
				callback=callback)
			self._stream.start()
			self.recording = True
			self._beep([(440, 40), (880, 40)])
			self._say(_("Recording. Press the key again to stop and transcribe."))
		except Exception as e:
			log.error("offlineAI dictation: cannot start mic: %s" % e)
			self._say(_("Could not start the microphone: {e}").format(e=str(e)))
			self._stream = None
			return
		# Load the model while the user is still speaking, so the time it takes
		# is hidden instead of being added after the recording.
		threading.Thread(target=self._preload, args=(model_path, options),
		                 daemon=True).start()

	def _ensureModel(self, model_path, options):
		"""Must be called with _modelLock held. Returns the model or raises."""
		from . import stt
		Model, err = stt.importWhisper()
		if Model is None:
			raise RuntimeError(err or "whisper unavailable")
		threads = int(options.get("n_threads") or 4)
		key = (model_path, threads)
		if self._model is None or self._loaded_key != key:
			self._model = None
			self._model = stt.newModel(Model, model_path, threads)
			self._loaded_key = key
		return self._model

	def _preload(self, model_path, options):
		try:
			with self._modelLock:
				self._ensureModel(model_path, options)
		except Exception as e:
			log.warning("offlineAI dictation: preload failed: %s" % e)

	def _stop_and_transcribe(self, model_path, options):
		from . import stt
		stt.ensureLib()
		import numpy as np
		self._closeStream()
		self.recording = False
		frames, self._frames = self._frames, []
		self._nframes = 0
		if not frames:
			self._say(_("No audio was captured."))
			return
		audio = np.concatenate(frames, axis=0).flatten().astype("float32")
		# Remember where the user was when they finished speaking.
		try:
			import winUser
			options["origin_hwnd"] = winUser.getForegroundWindow()
		except Exception:
			options["origin_hwnd"] = None
		self._beep([(880, 40), (440, 40)])
		self._pending += 1
		if self._pending > 1:
			self._say(_("Transcribing dictation. {n} notes waiting.").format(
				n=self._pending))
		else:
			self._say(_("Transcribing dictation."))
		self._jobs.put((audio, model_path, options))
		if self._worker is None or not self._worker.is_alive():
			self._worker = threading.Thread(target=self._drain, daemon=True,
			                                name="offlineAI-dictation")
			self._worker.start()

	def _drain(self):
		"""Transcribe queued recordings one after another, in order."""
		while True:
			try:
				audio, model_path, options = self._jobs.get(timeout=0.5)
			except queue.Empty:
				return
			try:
				self._transcribe(audio, model_path, options)
			except Exception as e:
				log.error("offlineAI dictation: job failed: %s" % e)
			finally:
				self._pending = max(0, self._pending - 1)

	def _transcribe(self, audio, model_path, options):
		import numpy as np
		from . import stt
		seconds = audio.size / float(WHISPER_SAMPLE_RATE)
		if audio.size < MIN_SAMPLES:
			audio = np.concatenate(
				[audio, np.zeros(MIN_SAMPLES - audio.size, dtype="float32")])
		params = stt.decodeParams(options.get("language") or "auto", False)
		params["single_segment"] = False
		# Whisper always analyses a full 30-second window, which is why a
		# two-second note used to take as long as a thirty-second one. Shrinking
		# the window to the real length (plus a margin) makes short notes several
		# times faster.
		audio_ctx = 0
		if options.get("fast", True) and seconds < 25:
			audio_ctx = min(1500, max(384, int(seconds * 50) + 128))
			if audio_ctx >= 1500:
				audio_ctx = 0
		params["audio_ctx"] = audio_ctx
		try:
			with self._modelLock:
				model = self._ensureModel(model_path, options)
				try:
					segments = model.transcribe(audio, **params)
				except Exception as e:
					if not audio_ctx:
						raise
					log.warning("offlineAI dictation: fast mode failed (%s); "
					            "retrying normally" % e)
					params["audio_ctx"] = 0
					segments = model.transcribe(audio, **params)
			text = " ".join(s.text.strip() for s in segments).strip()
		except Exception as e:
			log.error("offlineAI dictation: transcription failed: %s" % e)
			self._say(_("Transcription failed: {e}").format(e=str(e)))
			return
		if not text or text.strip("[]() .").upper() in ("BLANK_AUDIO", "BLANK AUDIO"):
			self._say(_("No speech was recognized."))
			return
		_callMain(self._deliver, text, options)

	# -- delivery (main thread) --
	def _deliver(self, text, options):
		import wx
		target = options.get("target") or "current"
		if target == "clipboard":
			self._copyOnly(text)
			return
		try:
			delay = max(0.0, min(10.0, float(options.get("paste_delay") or 0)))
		except (TypeError, ValueError):
			delay = 0.0
		if delay > 0:
			# Warning cue: the text is about to be typed into the focused window.
			self._beep([(1320, 60), (1320, 60)])
			self._say(_("Dictation ready. Inserting in {n} seconds.").format(
				n=int(delay) if delay == int(delay) else delay))
			wx.CallLater(int(delay * 1000), self._insert, text, options)
		else:
			self._insert(text, options)

	def _copyOnly(self, text):
		try:
			import api
			api.copyToClip(text)
			self._beep([(1760, 50)])
			self._say(_("Dictation copied to clipboard: {t}").format(t=text))
		except Exception:
			self._say(text)

	def _insert(self, text, options):
		import wx
		origin = options.get("origin_hwnd")
		back = None
		if (options.get("target") == "origin") and origin:
			try:
				import winUser
				cur = winUser.getForegroundWindow()
				if cur != origin:
					if not ctypes.windll.user32.IsWindow(origin):
						self._copyOnly(text)
						return
					back = cur
					winUser.setForegroundWindow(origin)
					# Give the window time to take the focus before pasting.
					wx.CallLater(350, self._paste, text, options, origin, back)
					return
			except Exception as e:
				log.warning("offlineAI dictation: cannot switch window: %s" % e)
		self._paste(text, options, None, None)

	def _paste(self, text, options, expect_hwnd, back_hwnd):
		"""Paste via the clipboard (the most reliable way to enter text of any
		script, including right-to-left and CJK, into any control)."""
		import wx
		try:
			import api
			if expect_hwnd:
				import winUser
				if winUser.getForegroundWindow() != expect_hwnd:
					# Windows refused the switch; do not paste into the wrong place.
					self._copyOnly(text)
					return
			previous = None
			if options.get("restore_clipboard", True):
				try:
					previous = api.getClipData()
				except Exception:
					previous = None
			from keyboardHandler import KeyboardInputGesture
			api.copyToClip(text)
			KeyboardInputGesture.fromName("control+v").send()
			self._beep([(1760, 50)])
			self._say(_("Dictated text inserted."))
			if previous is not None and previous != text:
				wx.CallLater(700, self._restoreClip, previous)
			if back_hwnd:
				wx.CallLater(450, self._goBack, back_hwnd)
		except Exception as e:
			log.error("offlineAI dictation: inject failed: %s" % e)
			self._copyOnly(text)

	def _restoreClip(self, previous):
		try:
			import api
			api.copyToClip(previous)
		except Exception:
			pass

	def _goBack(self, hwnd):
		try:
			import winUser
			if ctypes.windll.user32.IsWindow(hwnd):
				winUser.setForegroundWindow(hwnd)
		except Exception:
			pass


def pickModel(store_dir, models, preferred=""):
	"""Choose the Whisper model file for dictation: the user's preferred one if
	it is downloaded, otherwise the best downloaded model that is still quick
	enough for live use. Returns a path or None."""
	have = [m for m in models
	        if os.path.isfile(os.path.join(store_dir, m["filename"]))]
	if not have:
		return None
	if preferred:
		for m in have:
			if m["filename"] == preferred:
				return os.path.join(store_dir, m["filename"])
	order = ["ggml-large-v3-turbo-q5_0.bin", "ggml-small.bin",
	         "ggml-small-q5_1.bin", "ggml-base.bin", "ggml-large-v3-turbo.bin",
	         "ggml-tiny.bin", "ggml-medium.bin", "ggml-large-v3.bin"]
	by = {m["filename"]: m for m in have}
	for fn in order:
		if fn in by:
			return os.path.join(store_dir, fn)
	return os.path.join(store_dir, have[0]["filename"])
