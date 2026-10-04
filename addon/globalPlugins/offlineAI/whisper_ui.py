# -*- coding: utf-8 -*-
# Whisper speech-to-text window for the Offline AI add-on.
# Open one or many audio/video files, transcribe them in any of Whisper's ~100
# languages, play back the audio behind any line, and export TXT / SRT / VTT.

import os
import json
import threading

import wx
import gui
import ui
import addonHandler
from logHandler import log

from . import stt, langs, store

addonHandler.initTranslation()

ADDON_DIR = os.path.dirname(__file__)
WHISPER_MODELS_JSON = os.path.join(ADDON_DIR, "whisper_models.json")


def modelHint(key):
	hints = {
		# Translators: short descriptions of Whisper speech recognition models.
		"fastest": _("fastest, lower accuracy"),
		"fast": _("fast"),
		"balanced_small": _("balanced, small download"),
		"balanced": _("balanced"),
		"recommended": _("recommended: accurate and quick"),
		"accurate": _("accurate"),
		"accurate_fast": _("accurate, faster than Large"),
		"most_accurate": _("most accurate, slowest"),
	}
	return hints.get(key, "")


def audioWildcard():
	# Translators: file type filter in the "open audio" dialog.
	return (_("Media files") +
	        "|*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg;*.oga;*.opus;*.wma;*.mp4;"
	        "*.mkv;*.mov;*.webm;*.avi|" +
	        # Translators: file type filter.
	        _("All files") + "|*.*")


def loadWhisperModels():
	try:
		with open(WHISPER_MODELS_JSON, "r", encoding="utf-8") as f:
			return json.load(f).get("models", [])
	except Exception as e:
		log.error("offlineAI: cannot load whisper_models.json: %s" % e)
		return []


class WhisperDialog(wx.Dialog):
	_instance = None

	def __init__(self, parent, storeDirProvider):
		# Translators: title of the speech-to-text window.
		super(WhisperDialog, self).__init__(parent, title=_("Offline AI - Transcribe speech"))
		self.getStoreDir = storeDirProvider
		self.models = loadWhisperModels()
		self.model = None
		self.loadedKey = None
		self.segments = []
		self._segmentMap = []      # (char_start, char_end, t0, t1) per segment
		self._decodedAudio = None  # (samples, rate) of the current file
		self.mediaPath = None
		self.batchPaths = []
		self._abort = threading.Event()
		self._alive = True
		self._working = False

		helper = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)

		self.modelCombo = helper.addLabeledControl(
			_("&Whisper model:"), wx.Choice,
			choices=[self._label(m) for m in self.models])
		if self.models:
			self.modelCombo.SetSelection(self._initialModelIndex())
		self.modelCombo.Bind(wx.EVT_CHOICE, lambda e: self._refreshButtons())

		self._langCodes, labels = langs.choiceList(includeAuto=True)
		self.langCombo = helper.addLabeledControl(
			# Translators: the language spoken in the recording.
			_("Spoken &language:"), wx.Choice, choices=labels)
		last = store.loadSettings().get("whisper_lang", "auto")
		self.langCombo.SetSelection(
			self._langCodes.index(last) if last in self._langCodes else 0)

		self.translateCheck = wx.CheckBox(
			self, label=_("Translate to &English while transcribing"))
		helper.addItem(self.translateCheck)

		row = wx.BoxSizer(wx.HORIZONTAL)
		self.downloadButton = wx.Button(self, label=_("&Download model"))
		self.downloadButton.Bind(wx.EVT_BUTTON, self.onDownload)
		row.Add(self.downloadButton, 0, wx.ALL, 3)
		self.deleteButton = wx.Button(self, label=_("Delete &model file"))
		self.deleteButton.Bind(wx.EVT_BUTTON, self.onDeleteModel)
		row.Add(self.deleteButton, 0, wx.ALL, 3)
		self.openButton = wx.Button(self, label=_("&Open audio/video file(s)..."))
		self.openButton.Bind(wx.EVT_BUTTON, self.onOpen)
		row.Add(self.openButton, 0, wx.ALL, 3)
		self.batchButton = wx.Button(self, label=_("&Batch transcribe to files..."))
		self.batchButton.Bind(wx.EVT_BUTTON, self.onBatch)
		row.Add(self.batchButton, 0, wx.ALL, 3)
		helper.addItem(row)

		self.fileLabel = helper.addItem(
			wx.StaticText(self, label=_("No file chosen.")))

		row2 = wx.BoxSizer(wx.HORIZONTAL)
		self.transcribeButton = wx.Button(self, label=_("&Transcribe"))
		self.transcribeButton.Bind(wx.EVT_BUTTON, self.onTranscribe)
		self.transcribeButton.Enable(False)
		row2.Add(self.transcribeButton, 0, wx.ALL, 3)
		self.stopButton = wx.Button(self, label=_("&Stop"))
		self.stopButton.Bind(wx.EVT_BUTTON, self.onStop)
		self.stopButton.Enable(False)
		row2.Add(self.stopButton, 0, wx.ALL, 3)
		helper.addItem(row2)

		self.transcript = helper.addLabeledControl(
			_("Transcr&ipt:"), wx.TextCtrl,
			style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2, size=(560, 240))
		self.transcript.Bind(wx.EVT_KEY_DOWN, self._onTranscriptKey)

		row3 = wx.BoxSizer(wx.HORIZONTAL)
		self.saveTxt = wx.Button(self, label=_("Save as T&XT..."))
		self.saveTxt.Bind(wx.EVT_BUTTON, lambda e: self.onSave("txt"))
		self.saveSrt = wx.Button(self, label=_("Save as S&RT..."))
		self.saveSrt.Bind(wx.EVT_BUTTON, lambda e: self.onSave("srt"))
		self.saveVtt = wx.Button(self, label=_("Save as &VTT..."))
		self.saveVtt.Bind(wx.EVT_BUTTON, lambda e: self.onSave("vtt"))
		for b in (self.saveTxt, self.saveSrt, self.saveVtt):
			b.Enable(False)
			row3.Add(b, 0, wx.ALL, 3)
		helper.addItem(row3)

		closeButton = wx.Button(self, id=wx.ID_CLOSE, label=_("&Close"))
		closeButton.Bind(wx.EVT_BUTTON, lambda e: self.Close())
		helper.addItem(closeButton)
		self.EscapeId = wx.ID_CLOSE

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(helper.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.SetSizerAndFit(main)
		self._refreshButtons()

	# -- helpers --
	def _initialModelIndex(self):
		"""Select the first model that is already downloaded, if any."""
		for i, m in enumerate(self.models):
			if os.path.isfile(self._path(m)):
				return i
		return 0

	def _label(self, m):
		mark = _(" [downloaded]") if os.path.isfile(self._path(m)) else ""
		hint = modelHint(m.get("hint", ""))
		name = m["name"] + ((" - " + hint) if hint else "")
		return u"{n} - {s}{mk}".format(n=name, s=m.get("size", "?"), mk=mark)

	def _selected(self):
		if not self.models:
			return None
		i = self.modelCombo.GetSelection()
		return self.models[i] if i != wx.NOT_FOUND else None

	def _path(self, m):
		return os.path.join(self.getStoreDir(), m["filename"])

	def _relabelModels(self):
		sel = self.modelCombo.GetSelection()
		for i, mm in enumerate(self.models):
			self.modelCombo.SetString(i, self._label(mm))
		self.modelCombo.SetSelection(sel)

	def _refreshButtons(self):
		m = self._selected()
		have = bool(m) and os.path.isfile(self._path(m))
		self.downloadButton.Show(bool(m) and not have)
		self.deleteButton.Show(have)
		busy = self._working
		self.downloadButton.Enable(not busy)
		self.deleteButton.Enable(not busy)
		self.openButton.Enable(not busy)
		self.batchButton.Enable(not busy)
		self.transcribeButton.Enable(
			not busy and bool(self.mediaPath or self.batchPaths))
		self.stopButton.Enable(busy)
		self.Layout()

	def _safe(self, func, *args):
		"""wx.CallAfter that is ignored once the window has been closed."""
		def run():
			if self._alive:
				func(*args)
		wx.CallAfter(run)

	def _modelReady(self):
		m = self._selected()
		if not m or not os.path.isfile(self._path(m)):
			ui.message(_("Download the selected Whisper model first."))
			return None
		return m

	def _options(self):
		"""Read every control on the main thread, before a worker starts."""
		lang = self._langCodes[max(0, self.langCombo.GetSelection())]
		store.updateSettings(whisper_lang=lang)
		return {
			"language": lang,
			"translate": bool(self.translateCheck.GetValue()),
			"threads": store.autoThreads(),
		}

	# -- download / delete --
	def onDownload(self, evt):
		m = self._selected()
		if not m:
			return
		dest = self._path(m)
		if os.path.isfile(dest):
			ui.message(_("Model already downloaded."))
			return
		self.downloadButton.Enable(False)
		from . import downloader
		downloader.download(self, m["url"], dest, m["name"], self._dlDone)

	def _dlDone(self, ok, err):
		if not self._alive:
			return
		ui.message(_("Download complete.") if ok
		           else _("Download failed: {e}").format(e=err))
		self._relabelModels()
		self._refreshButtons()

	def onDeleteModel(self, evt):
		m = self._selected()
		if not m or not os.path.isfile(self._path(m)):
			return
		if gui.messageBox(
			_("Delete the downloaded file for {n} ({s})? You can download it "
			  "again later.").format(n=m["name"], s=m.get("size", "?")),
			_("Delete model"), wx.YES_NO | wx.ICON_QUESTION, self) != wx.YES:
			return
		if self.loadedKey and self.loadedKey[0] == m["filename"]:
			self.model = None
			self.loadedKey = None
		try:
			os.remove(self._path(m))
			ui.message(_("Model file deleted."))
		except OSError as e:
			ui.message(_("Could not delete the file: {e}").format(e=str(e)))
		self._relabelModels()
		self._refreshButtons()

	# -- choosing files --
	def _pickFiles(self, title):
		dlg = wx.FileDialog(self, title, wildcard=audioWildcard(),
		                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)
		paths = dlg.GetPaths() if dlg.ShowModal() == wx.ID_OK else []
		dlg.Destroy()
		return list(paths)

	def _warnFfmpeg(self, paths):
		from . import audio_decode
		for p in paths:
			note = audio_decode.native_note(p)
			if note:
				gui.messageBox(note, _("Audio format"), wx.OK | wx.ICON_INFORMATION, self)
				return

	def onOpen(self, evt):
		paths = self._pickFiles(_("Choose one or more audio or video files"))
		if not paths:
			return
		self._decodedAudio = None
		if len(paths) == 1:
			self.mediaPath = paths[0]
			self.batchPaths = []
			name = os.path.basename(self.mediaPath)
			self.fileLabel.SetLabel(_("File: {n}").format(n=name))
			ui.message(_("Chosen {n}. Press Transcribe.").format(n=name))
		else:
			self.mediaPath = None
			self.batchPaths = paths
			msg = _("{n} files chosen. Press Transcribe to write a transcript "
			        "file next to each one.").format(n=len(paths))
			self.fileLabel.SetLabel(msg)
			ui.message(msg)
		self._refreshButtons()
		self._warnFfmpeg(paths)

	def onBatch(self, evt):
		if not self._modelReady():
			return
		paths = self._pickFiles(_("Choose the audio or video files to transcribe"))
		if not paths:
			return
		self.mediaPath = None
		self.batchPaths = paths
		self.fileLabel.SetLabel(_("{n} files chosen for batch transcription.").format(
			n=len(paths)))
		self._warnFfmpeg(paths)
		self._startBatch()

	# -- transcribe one file --
	def onStop(self, evt):
		self._abort.set()
		self.stopButton.Enable(False)
		ui.message(_("Stopping."))

	def onTranscribe(self, evt):
		if self._working:
			return
		if self.batchPaths:
			self._startBatch()
			return
		if not self.mediaPath:
			ui.message(_("Choose a file first with the Open button."))
			return
		m = self._modelReady()
		if not m:
			return
		self._abort.clear()
		self.segments = []
		self._segmentMap = []
		self.transcript.SetValue("")
		self._working = True
		self._refreshButtons()
		for b in (self.saveTxt, self.saveSrt, self.saveVtt):
			b.Enable(False)
		ui.message(_("Transcribing. This can take a while."))
		threading.Thread(target=self._txWorker,
		                 args=(m, self._path(m), self.mediaPath, self._options()),
		                 daemon=True).start()

	def _getModel(self, m, path, opts):
		"""Worker thread: load (or reuse) the model. Returns (model, error)."""
		Model, err = stt.importWhisper()
		if Model is None:
			return None, _("The speech engine could not be loaded: {e}").format(e=err)
		key = (m["filename"], opts["threads"])
		try:
			if self.model is None or self.loadedKey != key:
				self.model = None
				self.model = stt.newModel(Model, path, opts["threads"])
				self.loadedKey = key
			return self.model, None
		except Exception as e:
			self.model = None
			self.loadedKey = None
			return None, _("Could not load the model: {e}").format(e=str(e))

	@staticmethod
	def _friendlyError(e):
		msg = str(e)
		if "FFMPEG" in msg.upper():
			return _("FFmpeg is required for this file format. Convert it to "
			         "WAV, MP3, FLAC or OGG, or install FFmpeg.")
		return msg

	def _txWorker(self, m, modelPath, mediaPath, opts):
		model, err = self._getModel(m, modelPath, opts)
		if model is None:
			self._safe(self._txDone, err)
			return
		try:
			audio, aerr = stt.loadAudio(mediaPath)
			if audio is None:
				raise RuntimeError(aerr or "cannot decode")
			# Keep the decoded audio so segment playback works for every format.
			self._decodedAudio = (audio, stt.WHISPER_RATE)

			def onSeg(seg):
				self._safe(self._appendSeg, seg.text)

			result = model.transcribe(
				audio, new_segment_callback=onSeg,
				abort_callback=self._abort.is_set,
				**stt.decodeParams(opts["language"], opts["translate"]))
			self.segments = list(result) if result else []
			self._safe(self._txDone, None)
		except Exception as e:
			log.error("offlineAI transcription failed: %s" % e)
			self._safe(self._txDone, _("Transcription error: {e}").format(
				e=self._friendlyError(e)))

	def _appendSeg(self, text):
		self.transcript.AppendText(text)

	def _txDone(self, err):
		self._working = False
		self._refreshButtons()
		if err:
			ui.message(err)
			return
		has = bool(self.segments)
		if has:
			# Rebuild the box from the final segment list and remember where each
			# segment sits, so Space/Enter can play the audio behind it.
			self._segmentMap = []
			parts = []
			pos = 0
			for s in self.segments:
				txt = s.text.strip()
				if not txt:
					continue
				if parts:
					parts.append(" ")
					pos += 1
				start = pos
				parts.append(txt)
				pos += len(txt)
				self._segmentMap.append(
					(start, pos, getattr(s, "t0", 0), getattr(s, "t1", 0)))
			self.transcript.SetValue("".join(parts))
			self.transcript.SetInsertionPoint(0)
		for b in (self.saveTxt, self.saveSrt, self.saveVtt):
			b.Enable(has)
		if has:
			ui.message(_("Transcription complete. Press Space or Enter in the "
			             "transcript to play the audio for the current segment."))
		else:
			ui.message(_("No speech was transcribed."))

	# -- interactive transcript player --
	def _onTranscriptKey(self, evt):
		key = evt.GetKeyCode()
		if key in (wx.WXK_SPACE, wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER) \
				and self._segmentMap:
			sel_start, sel_end = self.transcript.GetSelection()
			if sel_start != sel_end:
				self._playCharRange(sel_start, sel_end)
			else:
				self._playSegmentAt(self.transcript.GetInsertionPoint())
			return
		evt.Skip()

	def _segmentAt(self, pos):
		# The multi-line control counts each line break as two characters;
		# transcripts are single-paragraph, so positions map directly.
		for (cs, ce, t0, t1) in self._segmentMap:
			if cs <= pos <= ce:
				return (t0, t1)
		if self._segmentMap:
			return self._segmentMap[-1][2:]
		return None

	def _playSegmentAt(self, pos):
		seg = self._segmentAt(pos)
		if not seg:
			ui.message(_("No audio segment here."))
			return
		self._playTimeRange(seg[0] / 100.0, seg[1] / 100.0)

	def _playCharRange(self, cstart, cend):
		t0 = t1 = None
		for (cs, ce, s0, s1) in self._segmentMap:
			if ce <= cstart or cs >= cend:
				continue
			if t0 is None:
				t0 = s0
			t1 = s1
		if t0 is None:
			ui.message(_("No audio for the selection."))
			return
		self._playTimeRange(t0 / 100.0, t1 / 100.0)

	def _playTimeRange(self, start_s, end_s):
		if self._decodedAudio is None:
			ui.message(_("Could not load the audio for playback."))
			return
		samples, rate = self._decodedAudio
		a = max(0, int(start_s * rate))
		b = min(len(samples), int(end_s * rate))
		if b <= a:
			b = min(len(samples), a + rate)
		threading.Thread(target=self._playClip, args=(samples[a:b], rate),
		                 daemon=True).start()

	def _playClip(self, clip, rate):
		try:
			stt.ensureLib()
			import sounddevice as sd
			sd.stop()
			sd.play(clip, rate)
			sd.wait()
		except Exception as e:
			log.error("offlineAI: clip playback failed: %s" % e)
			self._safe(ui.message, _("Playback failed."))

	# -- batch transcription --
	def _startBatch(self):
		if self._working:
			return
		m = self._modelReady()
		if not m or not self.batchPaths:
			return
		fmts = ["txt", "srt", "vtt"]
		cdlg = wx.SingleChoiceDialog(
			self,
			_("Each transcript is saved next to its audio file, with the same "
			  "name. Choose the format:"),
			_("Batch transcription"),
			[_("Text (.txt)"), _("SubRip subtitles (.srt)"), _("WebVTT subtitles (.vtt)")])
		ok = cdlg.ShowModal() == wx.ID_OK
		fmt = fmts[cdlg.GetSelection()] if ok else None
		cdlg.Destroy()
		if not ok:
			return
		paths = list(self.batchPaths)
		self._abort.clear()
		self.segments = []
		self._segmentMap = []
		self._decodedAudio = None
		for b in (self.saveTxt, self.saveSrt, self.saveVtt):
			b.Enable(False)
		self.transcript.SetValue("")
		self._working = True
		self._refreshButtons()
		ui.message(_("Batch transcription started for {n} files.").format(n=len(paths)))
		threading.Thread(target=self._batchWorker,
		                 args=(m, self._path(m), paths, fmt, self._options()),
		                 daemon=True).start()

	def _log(self, line, speak=True):
		self.transcript.AppendText(line + "\n")
		if speak:
			ui.message(line)

	def _batchWorker(self, m, modelPath, paths, fmt, opts):
		model, err = self._getModel(m, modelPath, opts)
		if model is None:
			self._safe(self._batchDone, err, 0, len(paths))
			return
		done = 0
		total = len(paths)
		params = stt.decodeParams(opts["language"], opts["translate"])
		for idx, path in enumerate(paths, 1):
			if self._abort.is_set():
				break
			name = os.path.basename(path)
			self._safe(self._log, _("{i} of {t}: transcribing {n}").format(
				i=idx, t=total, n=name))
			try:
				audio, aerr = stt.loadAudio(path)
				if audio is None:
					raise RuntimeError(aerr or "cannot decode")
				segs = model.transcribe(
					audio, abort_callback=self._abort.is_set, **params)
				if self._abort.is_set():
					break
				out = os.path.splitext(path)[0] + "." + fmt
				stt.WRITERS[fmt](segs, out)
				done += 1
				self._safe(self._log, _("Saved {o}").format(o=out), False)
			except Exception as e:
				log.error("offlineAI batch: %s failed: %s" % (path, e))
				self._safe(self._log, _("Skipped {n}: {e}").format(
					n=name, e=self._friendlyError(e)))
		self._safe(self._batchDone, None, done, total)

	def _batchDone(self, err, done, total):
		self._working = False
		self._refreshButtons()
		if err:
			ui.message(err)
			return
		msg = _("Batch complete. {d} of {t} files transcribed. The transcripts "
		        "are saved next to the original files; details are in the "
		        "transcript box.").format(d=done, t=total)
		self._log(msg)

	# -- export --
	def onSave(self, fmt):
		if not self.segments:
			ui.message(_("Nothing to save yet."))
			return
		wildcards = {
			"txt": _("Text file") + "|*.txt",
			"srt": _("SubRip subtitles") + "|*.srt",
			"vtt": _("WebVTT subtitles") + "|*.vtt",
		}
		base = os.path.splitext(os.path.basename(self.mediaPath or "transcript"))[0]
		dlg = wx.FileDialog(self, _("Save transcript"),
		                    defaultDir=os.path.dirname(self.mediaPath or ""),
		                    defaultFile=base + "." + fmt,
		                    wildcard=wildcards[fmt],
		                    style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
			try:
				stt.WRITERS[fmt](self.segments, path)
				ui.message(_("Saved {n}").format(n=os.path.basename(path)))
			except Exception as e:
				log.error("offlineAI export failed: %s" % e)
				ui.message(_("Could not save: {e}").format(e=str(e)))
		dlg.Destroy()

	def onClose(self):
		self._alive = False
		self._abort.set()
		import sys
		sd = sys.modules.get("sounddevice")
		if sd is not None:
			try:
				sd.stop()
			except Exception:
				pass
		self.model = None
		self._decodedAudio = None
		WhisperDialog._instance = None
		self.Destroy()

	@classmethod
	def show(cls, storeDirProvider):
		if cls._instance is not None:
			try:
				cls._instance.Raise()
				return
			except Exception:
				cls._instance = None
		d = cls(gui.mainFrame, storeDirProvider)
		cls._instance = d
		d.Bind(wx.EVT_CLOSE, lambda e: d.onClose())
		gui.mainFrame.prePopup()
		d.Show()
		gui.mainFrame.postPopup()
