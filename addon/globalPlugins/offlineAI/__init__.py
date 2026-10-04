# -*- coding: utf-8 -*-
# Offline AI add-on for NVDA
# Copyright (C) 2026 Riad Assoum
# Released under the GNU General Public License version 2 or later.
# https://github.com/riadassoum/offlineai
#
# Download GGUF language models and chat with them locally, transcribe and
# dictate with Whisper, read images and PDFs with a vision model, and speak
# with a cloned voice - all on the CPU, with no data leaving the computer.
# The engines (llama.cpp, whisper.cpp) are bundled in "lib" for NVDA 2026
# (64-bit, Python 3.13).

import os
import sys
import json
import time
import threading

import globalPluginHandler
import addonHandler
import gui
from gui.settingsDialogs import SettingsPanel
import wx
import ui
import globalVars
from logHandler import log
from scriptHandler import script

from . import store, langs, inference
from .store import (
	DEFAULT_SETTINGS, loadLocalConfig, loadSettings, saveSettings,
	getModelsStoreDir, setModelsStoreDir, getWorkTempDir, clearWorkTemp,
)

addonHandler.initTranslation()

ADDON_DIR = os.path.dirname(__file__)
MODELS_JSON = os.path.join(ADDON_DIR, "models.json")
LIB_DIR = os.path.join(ADDON_DIR, "lib")


def ensureLibOnPath():
	if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)


def browseable(text, title):
	"""Show a long result in NVDA's own browseable window, where it can be read
	with the normal reading keys, copied, and dismissed with Escape. (A plain
	message box cannot be reviewed line by line.)"""
	try:
		ui.browseableMessage(text, title, closeButton=True, copyButton=True)
	except TypeError:
		ui.browseableMessage(text, title)
	except Exception as e:
		log.warning("offlineAI: browseable message failed: %s" % e)
		gui.messageBox(text, title, wx.OK | wx.ICON_INFORMATION)


def readTextFile(path):
	"""Read a text or source-code file in whatever encoding it uses: UTF-8 (with
	or without BOM), UTF-16, or the system's own ANSI code page (Windows-1256
	for Arabic, 1251 for Cyrillic, 932 for Japanese...). Reading everything as
	UTF-8 turned such files into question marks."""
	with open(path, "rb") as f:
		raw = f.read()
	if raw.startswith(b"\xef\xbb\xbf"):
		return raw[3:].decode("utf-8", "replace")
	if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
		return raw.decode("utf-16", "replace")
	if b"\x00" in raw[:4096]:
		# Lots of NULs without a BOM: UTF-16 without BOM, or a binary file.
		even = raw[0:4096:2].count(b"\x00")
		odd = raw[1:4096:2].count(b"\x00")
		if odd > even * 4:
			return raw.decode("utf-16-le", "replace")
		if even > odd * 4:
			return raw.decode("utf-16-be", "replace")
		raise ValueError("binary")
	try:
		return raw.decode("utf-8")
	except UnicodeDecodeError:
		pass
	try:
		return raw.decode("mbcs")
	except Exception:
		return raw.decode("utf-8", "replace")


# File types offered when attaching context. "All files" is always available;
# any text-based file works.
TEXT_FILE_PATTERNS = (
	"*.txt;*.md;*.rst;*.log;*.csv;*.tsv;*.json;*.jsonl;*.xml;*.yaml;*.yml;"
	"*.toml;*.ini;*.cfg;*.conf;*.html;*.htm;*.css;*.srt;*.vtt;*.tex;"
	"*.py;*.pyw;*.js;*.mjs;*.ts;*.jsx;*.tsx;*.c;*.h;*.cpp;*.hpp;*.cc;*.cs;"
	"*.java;*.kt;*.go;*.rs;*.rb;*.php;*.pl;*.lua;*.swift;*.dart;*.scala;*.r;"
	"*.sql;*.sh;*.bash;*.ps1;*.bat;*.cmd;*.vb;*.vbs;*.pas;*.asm;*.bgt;*.nvgt;"
	"*.au3;*.ahk;*.jl;*.m;*.gradle;*.cmake;*.mk;*.diff;*.patch")


# --- model directory + user data --------------------------------------------

def getCustomModels():
	return loadLocalConfig().get("custom_models", [])


def addCustomModel(entry):
	lst = getCustomModels()
	lst = [m for m in lst if m.get("filename") != entry.get("filename")]
	lst.append(entry)
	store.setValue("custom_models", lst)


def removeCustomModel(filename):
	store.setValue("custom_models", [
		m for m in getCustomModels() if m.get("filename") != filename])


def getCatalogUrl():
	return loadLocalConfig().get("catalog_url", "")


def setCatalogUrl(url):
	store.setValue("catalog_url", url or "")


# Custom quick actions: list of {"name","instruction"}.
def getCustomActions():
	return loadLocalConfig().get("custom_actions", [])


def setCustomActions(actions):
	store.setValue("custom_actions", actions)


def builtinPresets():
	"""System-prompt presets. Names are shown in the user's language; the
	prompts themselves stay in English, which every model understands, and ask
	the model to answer in the user's language where that matters."""
	lang = langs.englishName(langs.primaryTranslationLanguage(loadSettings()))
	return [
		# Translators: name of a built-in persona.
		{"name": _("Helpful Assistant"),
		 "system": "You are a helpful assistant.", "temperature": 0.7},
		# Translators: name of a built-in persona.
		{"name": _("Concise (Direct Answers Only)"),
		 "system": "You are a concise assistant. Answer directly and briefly, with "
		           "no preamble or filler.", "temperature": 0.4},
		# Translators: name of a built-in persona.
		{"name": _("Senior Developer"),
		 "system": "You are a senior software engineer. Give precise, idiomatic code "
		           "and explain trade-offs briefly.", "temperature": 0.3},
		# Translators: name of a built-in persona. {language} is the user's
		# translation language, e.g. "Translator (into Spanish)".
		{"name": _("Translator (into {language})").format(
			language=langs.displayName(langs.primaryTranslationLanguage(loadSettings()))),
		 "system": "You are an expert translator. Translate everything the user "
		           "sends into %s, accurately and naturally, preserving meaning "
		           "and tone. Output only the translation." % lang,
		 "temperature": 0.3},
		# Translators: name of a built-in persona that always answers in the
		# user's own language.
		{"name": _("Answer in my language ({language})").format(
			language=langs.displayName(langs.primaryTranslationLanguage(loadSettings()))),
		 "system": "You are a helpful assistant. Always answer in %s, whatever "
		           "language the question is written in." % lang,
		 "temperature": 0.7},
	]


def getUserPresets():
	return loadLocalConfig().get("user_presets", [])


def setUserPresets(presets):
	store.setValue("user_presets", presets)


def allPresets():
	return builtinPresets() + getUserPresets()


# Voice clone profiles: list of {"name","path"}.
def getVoiceProfiles():
	return loadLocalConfig().get("voice_profiles", [])


def setVoiceProfiles(profiles):
	store.setValue("voice_profiles", profiles)


def addVoiceProfile(name, path):
	profiles = getVoiceProfiles()
	profiles = [p for p in profiles if p.get("name") != name]
	profiles.append({"name": name, "path": path})
	setVoiceProfiles(profiles)


def loadModelDirectory(fetch_remote=False):
	"""Load the merged catalog: bundled local + optional remote + user custom.
	Remote fetch is opt-in (only if a catalog URL is set) and best-effort."""
	try:
		from . import catalog
		url = getCatalogUrl() if fetch_remote else None
		return catalog.build_catalog(custom_models=getCustomModels(),
		                             remote_url=url)
	except Exception as e:
		log.error("offlineAI: catalog load failed, using bare local: %s" % e)
		try:
			with open(MODELS_JSON, "r", encoding="utf-8") as f:
				return json.load(f).get("models", [])
		except Exception:
			return []


def modelPath(m, storeDir=None):
	"""Where a catalog entry lives on disk (custom local files stay in place)."""
	lp = m.get("local_path")
	if lp and os.path.isfile(lp):
		return lp
	return os.path.join(storeDir or getModelsStoreDir(), m["filename"])


def categoryLabel(cat):
	labels = {
		# Translators: model categories in the model list.
		"General": _("General"),
		"Multilingual": _("Multilingual"),
		"Translation": _("Translation"),
		"Reasoning": _("Reasoning"),
		"Coding": _("Coding"),
		"Arabic": langs.displayName("ar"),
		"HuggingFace": "HuggingFace",
	}
	return labels.get(cat, cat or "")


def _estimateTokens(text):
	"""Token estimate for user-facing counts (real count when a model is
	loaded, a script-aware guess otherwise)."""
	return max(1, inference.count_tokens(text or ""))


def _ocrMaxLongSide(settings=None):
	"""Map the OCR quality setting to a max image long-side in pixels. Smaller =
	faster (fewer vision tokens), larger = more accurate on small text."""
	s = settings or loadSettings()
	q = s.get("ocr_quality", "balanced")
	return {"fast": 1024, "balanced": 1600, "best": 2200}.get(q, 1600)


def ttsSpeedChoices():
	# Translators: the normal (unchanged) playback speed of the AI voice.
	return [("0.75x", 0.75), (_("Normal"), 1.0), ("1.25x", 1.25),
	        ("1.5x", 1.5), ("1.75x", 1.75), ("2.0x", 2.0)]


def fmtTime(sec):
	if sec is None or sec < 0 or sec != sec:  # None/neg/NaN
		return "?"
	sec = int(sec)
	if sec >= 3600:
		return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)
	return "%d:%02d" % (sec // 60, sec % 60)


def fmtSize(nbytes):
	if nbytes is None or nbytes < 0:
		return "?"
	for unit in ("B", "KB", "MB", "GB"):
		if nbytes < 1024:
			return "%.1f %s" % (nbytes, unit)
		nbytes /= 1024.0
	return "%.1f TB" % nbytes


# --- settings dialog -------------------------------------------------------

class SettingsDialog(wx.Dialog):
	def __init__(self, parent, settings):
		super(SettingsDialog, self).__init__(parent, title=_("Offline AI settings"))
		self.settings = dict(settings)
		self.restored = False   # True after a backup was imported
		self._alive = True
		self.Bind(wx.EVT_WINDOW_DESTROY, self._onDestroy)

		main = wx.BoxSizer(wx.VERTICAL)

		# System prompt (multiline) with its own label above it.
		main.Add(wx.StaticText(self, label=_("&System prompt:")),
		         0, wx.LEFT | wx.TOP, 10)
		self.sysPrompt = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(440, 60))
		self.sysPrompt.SetValue(str(self.settings["system_prompt"]))
		main.Add(self.sysPrompt, 0, wx.ALL | wx.EXPAND, 10)

		# Numeric fields in a two-column grid: label | spinner.
		grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=10)
		grid.AddGrowableCol(1, 1)

		self.temp = self._addDouble(grid,
			_("&Temperature (0-2, higher = more creative):"),
			self.settings["temperature"], 0.0, 2.0, 0.05)
		self.topP = self._addDouble(grid, _("Top-&p (0-1):"),
			self.settings["top_p"], 0.0, 1.0, 0.05)
		self.topK = self._addInt(grid, _("Top-&k (0 = off):"),
			self.settings["top_k"], 0, 200)
		self.minP = self._addDouble(grid, _("&Min-p (0-1):"),
			self.settings["min_p"], 0.0, 1.0, 0.01)
		self.repeat = self._addDouble(grid, _("&Repeat penalty (1 = off):"),
			self.settings["repeat_penalty"], 1.0, 2.0, 0.05)
		self.maxTok = self._addInt(grid, _("Max &answer tokens:"),
			self.settings["max_tokens"], 16, 8192)
		self.nCtx = self._addInt(grid, _("&Context size (n_ctx):"),
			self.settings["n_ctx"], 256, 32768)
		self.nThreads = self._addInt(grid, _("Threads (&0 = auto):"),
			self.settings["n_threads"], 0, 256)
		self.nBatch = self._addInt(grid, _("&Batch size (n_batch):"),
			self.settings["n_batch"], 32, 2048)
		self.seed = self._addInt(grid, _("&Seed (-1 = random):"),
			self.settings["seed"], -1, 2147483647)

		main.Add(grid, 0, wx.ALL | wx.EXPAND, 10)

		# Checkboxes (these are fine on their own; labels are built in).
		self.flash = wx.CheckBox(self, label=_("Flash attention (&faster)"))
		self.flash.SetValue(bool(self.settings["flash_attn"]))
		main.Add(self.flash, 0, wx.LEFT | wx.BOTTOM, 10)
		self.mlock = wx.CheckBox(self, label=_("Lock model in RAM (&mlock)"))
		self.mlock.SetValue(bool(self.settings["use_mlock"]))
		main.Add(self.mlock, 0, wx.LEFT | wx.BOTTOM, 10)
		self.mmap = wx.CheckBox(self, label=_(
			"Memory-map model files (mma&p) - lets a model larger than your "
			"free RAM still run; turn off to load models fully into RAM"))
		self.mmap.SetValue(bool(self.settings.get("use_mmap", True)))
		main.Add(self.mmap, 0, wx.LEFT | wx.BOTTOM, 10)
		self.stream = wx.CheckBox(self, label=_("Stream reply &live (recommended)"))
		self.stream.SetValue(bool(self.settings["stream"]))
		main.Add(self.stream, 0, wx.LEFT | wx.BOTTOM, 10)
		self.heartbeat = wx.CheckBox(self, label=_("Play a soft tone while &computing (heartbeat)"))
		self.heartbeat.SetValue(bool(self.settings.get("heartbeat", True)))
		main.Add(self.heartbeat, 0, wx.LEFT | wx.BOTTOM, 10)

		# Reasoning (<think>) handling for models like DeepSeek-R1.
		self._thinkModes = ["hide", "cue", "show"]
		self._thinkLabels = [
			_("Hide reasoning, show only the answer"),
			_("Announce a short cue, then the answer"),
			_("Show the reasoning too"),
		]
		trow = wx.BoxSizer(wx.HORIZONTAL)
		trow.Add(wx.StaticText(self, label=_("&Reasoning (think) mode:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.thinkMode = wx.Choice(self, choices=self._thinkLabels)
		cur = self.settings.get("think_mode", "hide")
		self.thinkMode.SetSelection(
			self._thinkModes.index(cur) if cur in self._thinkModes else 0)
		try:
			self.thinkMode.SetName(_("Reasoning (think) mode"))
		except Exception:
			pass
		trow.Add(self.thinkMode, 0, wx.EXPAND)
		main.Add(trow, 0, wx.LEFT | wx.BOTTOM, 10)

		# Quick-action output behavior.
		self._quickModes = ["speak", "copy", "both"]
		self._quickLabels = [
			_("Speak the result"),
			_("Copy the result to clipboard"),
			_("Both speak and copy"),
		]
		qrow = wx.BoxSizer(wx.HORIZONTAL)
		qrow.Add(wx.StaticText(self, label=_("&Quick action output:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.quickOutput = wx.Choice(self, choices=self._quickLabels)
		curq = self.settings.get("quick_output", "speak")
		self.quickOutput.SetSelection(
			self._quickModes.index(curq) if curq in self._quickModes else 0)
		try:
			self.quickOutput.SetName(_("Quick action output"))
		except Exception:
			pass
		qrow.Add(self.quickOutput, 0)
		main.Add(qrow, 0, wx.LEFT | wx.BOTTOM, 10)

		# HuggingFace search result count.
		hrow = wx.BoxSizer(wx.HORIZONTAL)
		hrow.Add(wx.StaticText(self, label=_("&HuggingFace search results:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.hfLimit = wx.SpinCtrl(self, min=5, max=500,
		                           initial=int(self.settings.get("hf_result_limit", 60)))
		try:
			self.hfLimit.SetName(_("HuggingFace search results"))
		except Exception:
			pass
		hrow.Add(self.hfLimit, 0)
		main.Add(hrow, 0, wx.LEFT | wx.BOTTOM, 10)

		# AI voice (TTS) default speed and streaming look-ahead.
		self._ttsSpeeds = ttsSpeedChoices()
		srow = wx.BoxSizer(wx.HORIZONTAL)
		srow.Add(wx.StaticText(self, label=_("AI voice &speed:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.ttsSpeed = wx.Choice(self, choices=[l for l, _v in self._ttsSpeeds])
		cs = self.settings.get("tts_speed", 1.0)
		self.ttsSpeed.SetSelection(
			next((i for i, (_l, v) in enumerate(self._ttsSpeeds) if v == cs), 1))
		try:
			self.ttsSpeed.SetName(_("AI voice speed"))
		except Exception:
			pass
		srow.Add(self.ttsSpeed, 0)
		main.Add(srow, 0, wx.LEFT | wx.BOTTOM, 10)

		# Time-stretch quality for faster playback.
		self._stretchModes = [("smooth", _("Smooth speech (default)")),
		                      ("crisp", _("Crisp transients")),
		                      ("legacy", _("Legacy sample-rate scaling"))]
		strow = wx.BoxSizer(wx.HORIZONTAL)
		strow.Add(wx.StaticText(self, label=_("Speed &quality:")),
		          0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.ttsStretch = wx.Choice(
			self, choices=[lbl for _m, lbl in self._stretchModes])
		cm = self.settings.get("tts_stretch", "smooth")
		self.ttsStretch.SetSelection(
			next((i for i, (m, _l) in enumerate(self._stretchModes) if m == cm), 0))
		try:
			self.ttsStretch.SetName(_("Speed quality"))
		except Exception:
			pass
		strow.Add(self.ttsStretch, 0)
		main.Add(strow, 0, wx.LEFT | wx.BOTTOM, 10)

		# OCR speed vs quality (controls image resolution / vision tokens).
		self._ocrQ = [("fast", _("Fast (lower resolution)")),
		              ("balanced", _("Balanced (default)")),
		              ("best", _("Best quality (slower)"))]
		orow = wx.BoxSizer(wx.HORIZONTAL)
		orow.Add(wx.StaticText(self, label=_("&OCR speed vs quality:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.ocrQuality = wx.Choice(self, choices=[l for _m, l in self._ocrQ])
		cq = self.settings.get("ocr_quality", "balanced")
		self.ocrQuality.SetSelection(
			next((i for i, (m, _l) in enumerate(self._ocrQ) if m == cq), 1))
		try:
			self.ocrQuality.SetName(_("OCR speed vs quality"))
		except Exception:
			pass
		orow.Add(self.ocrQuality, 0)
		main.Add(orow, 0, wx.LEFT | wx.BOTTOM, 10)

		self.ttsPrerender = wx.CheckBox(self, label=_(
			"When reading documents, &render the next sentence while the current "
			"one plays (smoother, but no pause between sentences)"))
		self.ttsPrerender.SetValue(bool(self.settings.get("tts_prerender", True)))
		main.Add(self.ttsPrerender, 0, wx.LEFT | wx.BOTTOM, 10)

		# Audio cue volume + mute.
		vrow = wx.BoxSizer(wx.HORIZONTAL)
		vrow.Add(wx.StaticText(self, label=_("Audio cues &volume:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.cueVolume = wx.Slider(
			self, value=int(self.settings.get("cue_volume", 60)), minValue=0,
			maxValue=100, style=wx.SL_HORIZONTAL | wx.SL_LABELS, size=(220, -1))
		try:
			self.cueVolume.SetName(_("Audio cues volume"))
		except Exception:
			pass
		vrow.Add(self.cueVolume, 0)
		main.Add(vrow, 0, wx.LEFT | wx.BOTTOM, 10)
		self.cueMute = wx.CheckBox(self, label=_("&Mute all audio cues"))
		self.cueMute.SetValue(bool(self.settings.get("cue_mute", False)))
		main.Add(self.cueMute, 0, wx.LEFT | wx.BOTTOM, 10)

		self.formatTables = wx.CheckBox(self, label=_(
			"Format &tables for screen readers (rewrite markdown tables as "
			"readable labeled rows)"))
		self.formatTables.SetValue(bool(self.settings.get("format_tables", True)))
		main.Add(self.formatTables, 0, wx.LEFT | wx.BOTTOM, 10)

		# Backup / restore all data.
		bkRow = wx.BoxSizer(wx.HORIZONTAL)
		self.backupButton = wx.Button(self, label=_("Export all data / &backup..."))
		self.backupButton.Bind(wx.EVT_BUTTON, self.onBackup)
		bkRow.Add(self.backupButton, 0, wx.ALL, 3)
		self.restoreButton = wx.Button(self, label=_("&Import backup..."))
		self.restoreButton.Bind(wx.EVT_BUTTON, self.onRestore)
		bkRow.Add(self.restoreButton, 0, wx.ALL, 3)
		main.Add(bkRow, 0, wx.LEFT | wx.BOTTOM, 10)

		self.benchButton = wx.Button(
			self, label=_("Run hardware && speed &benchmark..."))
		self.benchButton.Bind(wx.EVT_BUTTON, self.onBenchmark)
		main.Add(self.benchButton, 0, wx.LEFT | wx.BOTTOM, 10)

		btns = wx.BoxSizer(wx.HORIZONTAL)
		ok = wx.Button(self, id=wx.ID_OK, label=_("&Save"))
		ok.Bind(wx.EVT_BUTTON, self.onSave)
		reset = wx.Button(self, label=_("Restore &defaults"))
		reset.Bind(wx.EVT_BUTTON, self.onReset)
		actions = wx.Button(self, label=_("Custom &actions..."))
		actions.Bind(wx.EVT_BUTTON, self.onCustomActions)
		cancel = wx.Button(self, id=wx.ID_CANCEL, label=_("Cancel"))
		btns.Add(ok, 0, wx.ALL, 4)
		btns.Add(reset, 0, wx.ALL, 4)
		btns.Add(actions, 0, wx.ALL, 4)
		btns.Add(cancel, 0, wx.ALL, 4)
		main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 6)

		self.SetSizerAndFit(main)
		self.temp.SetFocus()

	def _onDestroy(self, evt):
		if evt.GetEventObject() is self:
			self._alive = False
		evt.Skip()

	def _later(self, func, *args):
		"""wx.CallAfter that is skipped if this dialog was closed meanwhile."""
		def run():
			if self._alive:
				func(*args)
		wx.CallAfter(run)

	def onCustomActions(self, evt):
		dlg = CustomActionsDialog(self)
		dlg.ShowModal()
		dlg.Destroy()

	def _cleanName(self, label):
		# Strip accelerator "&" for the accessible name.
		return label.replace("&", "")

	def _addDouble(self, grid, label, val, lo, hi, inc):
		lbl = wx.StaticText(self, label=label)
		ctrl = wx.SpinCtrlDouble(self, min=lo, max=hi, inc=inc, initial=float(val))
		ctrl.SetDigits(2)
		# Ensure NVDA announces the field's purpose even without visual pairing.
		try:
			ctrl.SetName(self._cleanName(label))
		except Exception:
			pass
		grid.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
		grid.Add(ctrl, 0, wx.EXPAND)
		return ctrl

	def _addInt(self, grid, label, val, lo, hi):
		lbl = wx.StaticText(self, label=label)
		ctrl = wx.SpinCtrl(self, min=lo, max=hi, initial=int(val))
		try:
			ctrl.SetName(self._cleanName(label))
		except Exception:
			pass
		grid.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
		grid.Add(ctrl, 0, wx.EXPAND)
		return ctrl

	def onBenchmark(self, evt):
		# Resolve a downloaded model (last-used or first available).
		store_dir = getModelsStoreDir()
		models = loadModelDirectory(fetch_remote=False)
		last = loadLocalConfig().get("last_model_filename")
		chosen = None
		for m in models:
			path = modelPath(m, store_dir)
			if os.path.isfile(path):
				if last and m["filename"] == last:
					chosen = (m, path)
					break
				if chosen is None:
					chosen = (m, path)
		if not chosen:
			ui.message(_("No downloaded model found. Download one first."))
			return
		self.benchButton.Enable(False)
		ui.message(_("Running benchmark with {n}, please wait.").format(
			n=chosen[0]["name"]))
		threading.Thread(target=self._benchWorker, args=(chosen,),
		                 daemon=True).start()

	def _benchWorker(self, chosen):
		m, path = chosen
		if not inference.try_acquire_busy():
			wx.CallAfter(ui.message, _("Offline AI is busy. Please wait."))
			self._later(self.benchButton.Enable, True)
			return
		try:
			ok, err = inference.load_model(path, self.settings,
			                               context=m.get("context"))
			if not ok:
				wx.CallAfter(ui.message,
				             _("Could not load the model: {e}").format(e=err))
				return
			res, err = inference.benchmark(30)
			if err:
				wx.CallAfter(ui.message, _("Benchmark failed: {e}").format(e=err))
				return
			tps = res["tokens_per_second"]
			if tps >= 20:
				verdict = _("Your CPU is well-suited for models up to about 7B.")
			elif tps >= 10:
				verdict = _("Your CPU is comfortable with 1.5B to 3B models.")
			elif tps >= 4:
				verdict = _("Your CPU can run 1.5B models; larger ones will be slow.")
			else:
				verdict = _("Your CPU is slow for this model; try a smaller one.")
			msg = _("Benchmark of {name} complete: {tps:.1f} tokens per second "
			        "({tok} tokens in {sec:.1f} seconds). {verdict}").format(
				name=m["name"], tps=tps, tok=res["tokens"], sec=res["seconds"],
				verdict=verdict)
			wx.CallAfter(gui.messageBox, msg, _("Benchmark result"),
			             wx.OK | wx.ICON_INFORMATION)
		finally:
			inference.release_busy()
			self._later(self.benchButton.Enable, True)

	def onBackup(self, evt):
		from . import backup, history as histmod
		dlg = wx.FileDialog(
			self, _("Save backup"),
			defaultFile="offlineAI_backup_%s.zip" % time.strftime("%Y%m%d_%H%M%S"),
			wildcard=_("Backup zip|*.zip"),
			style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
		path = None
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
		dlg.Destroy()
		if not path:
			return
		vpd = os.path.join(getModelsStoreDir(), "voice_profiles")
		ok, err = backup.create_backup(
			path, store.configPath(), histmod._historyPath(), vpd)
		if ok:
			ui.message(_("Backup saved to {n}.").format(n=os.path.basename(path)))
		else:
			ui.message(_("Backup failed: {e}").format(e=err))

	def onRestore(self, evt):
		from . import backup, history as histmod
		dlg = wx.FileDialog(
			self, _("Choose a backup to import"),
			wildcard=_("Backup zip|*.zip|All files|*.*"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		path = None
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
		dlg.Destroy()
		if not path:
			return
		manifest, err = backup.inspect_backup(path)
		if err:
			ui.message(_("Not a valid backup: {e}").format(e=err))
			return
		contents = ", ".join(manifest.get("contents", []))
		if gui.messageBox(
			_("This backup was created on {d} and contains: {c}.\n\n"
			  "Existing data will be overwritten. Continue?").format(
				d=manifest.get("created", "?"), c=contents or _("nothing")),
			_("Import backup"), wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
			return
		vpd = os.path.join(getModelsStoreDir(), "voice_profiles")
		restored, err = backup.restore_backup(
			path, store.configPath(), histmod._historyPath(), vpd, overwrite=True)
		if err:
			ui.message(_("Restore failed: {e}").format(e=err))
			return
		gui.messageBox(
			_("Restored: {items}.").format(
				items=", ".join(restored) or _("nothing")),
			_("Import backup"), wx.OK | wx.ICON_INFORMATION, self)
		# Close without saving: pressing Save here would write this dialog's old
		# values over the settings that were just restored.
		self.restored = True
		self.EndModal(wx.ID_CANCEL)

	def onReset(self, evt):
		# Reset what this dialog shows; language and dictation choices made in
		# NVDA's Settings are left alone.
		keep = ("translate_lang_1", "translate_lang_2", "tts_lang", "whisper_lang",
		        "ocr_lang", "dictation_lang", "dictation_model",
		        "dictation_target", "dictation_paste_delay", "dictation_fast",
		        "dictation_restore_clipboard", "unload_after_min")
		kept = {k: self.settings[k] for k in keep if k in self.settings}
		self.settings = dict(DEFAULT_SETTINGS)
		self.settings.update(kept)
		self.sysPrompt.SetValue(self.settings["system_prompt"])
		self.temp.SetValue(self.settings["temperature"])
		self.topP.SetValue(self.settings["top_p"])
		self.topK.SetValue(self.settings["top_k"])
		self.minP.SetValue(self.settings["min_p"])
		self.repeat.SetValue(self.settings["repeat_penalty"])
		self.maxTok.SetValue(self.settings["max_tokens"])
		self.nCtx.SetValue(self.settings["n_ctx"])
		self.nThreads.SetValue(self.settings["n_threads"])
		self.nBatch.SetValue(self.settings["n_batch"])
		self.seed.SetValue(self.settings["seed"])
		self.flash.SetValue(self.settings["flash_attn"])
		self.mlock.SetValue(self.settings["use_mlock"])
		self.mmap.SetValue(self.settings.get("use_mmap", True))
		self.stream.SetValue(self.settings["stream"])
		self.heartbeat.SetValue(self.settings.get("heartbeat", True))
		self.thinkMode.SetSelection(
			self._thinkModes.index(self.settings.get("think_mode", "hide")))
		self.quickOutput.SetSelection(
			self._quickModes.index(self.settings.get("quick_output", "speak")))
		self.hfLimit.SetValue(int(self.settings.get("hf_result_limit", 60)))
		cs=self.settings.get("tts_speed",1.0)
		self.ttsSpeed.SetSelection(next((i for i,(_l,v) in enumerate(self._ttsSpeeds) if v==cs),1))
		cm=self.settings.get("tts_stretch","smooth")
		self.ttsStretch.SetSelection(next((i for i,(m,_l) in enumerate(self._stretchModes) if m==cm),0))
		cq=self.settings.get("ocr_quality","balanced")
		self.ocrQuality.SetSelection(next((i for i,(m,_l) in enumerate(self._ocrQ) if m==cq),1))
		self.ttsPrerender.SetValue(bool(self.settings.get("tts_prerender",True)))
		self.cueVolume.SetValue(int(self.settings.get("cue_volume",60)))
		self.cueMute.SetValue(bool(self.settings.get("cue_mute",False)))
		self.formatTables.SetValue(bool(self.settings.get("format_tables",True)))
		ui.message(_("Defaults restored. Choose Save to keep them."))

	def onSave(self, evt):
		self.settings["system_prompt"] = self.sysPrompt.GetValue()
		self.settings["temperature"] = round(self.temp.GetValue(), 3)
		self.settings["top_p"] = round(self.topP.GetValue(), 3)
		self.settings["top_k"] = self.topK.GetValue()
		self.settings["min_p"] = round(self.minP.GetValue(), 3)
		self.settings["repeat_penalty"] = round(self.repeat.GetValue(), 3)
		self.settings["max_tokens"] = self.maxTok.GetValue()
		self.settings["n_ctx"] = self.nCtx.GetValue()
		self.settings["n_threads"] = self.nThreads.GetValue()
		self.settings["n_batch"] = self.nBatch.GetValue()
		self.settings["seed"] = self.seed.GetValue()
		self.settings["flash_attn"] = self.flash.GetValue()
		self.settings["use_mlock"] = self.mlock.GetValue()
		self.settings["use_mmap"] = self.mmap.GetValue()
		self.settings["stream"] = self.stream.GetValue()
		self.settings["heartbeat"] = self.heartbeat.GetValue()
		self.settings["think_mode"] = self._thinkModes[self.thinkMode.GetSelection()]
		self.settings["quick_output"] = self._quickModes[self.quickOutput.GetSelection()]
		self.settings["hf_result_limit"] = int(self.hfLimit.GetValue())
		self.settings["tts_speed"] = self._ttsSpeeds[self.ttsSpeed.GetSelection()][1]
		self.settings["tts_stretch"] = self._stretchModes[self.ttsStretch.GetSelection()][0]
		self.settings["ocr_quality"] = self._ocrQ[self.ocrQuality.GetSelection()][0]
		self.settings["tts_prerender"] = self.ttsPrerender.GetValue()
		self.settings["cue_volume"] = int(self.cueVolume.GetValue())
		self.settings["cue_mute"] = self.cueMute.GetValue()
		self.settings["format_tables"] = self.formatTables.GetValue()
		self.EndModal(wx.ID_OK)


# --- history picker dialog -------------------------------------------------

class HistoryDialog(wx.Dialog):
	def __init__(self, parent, sessions):
		super(HistoryDialog, self).__init__(parent, title=_("Chat history"))
		self.sessions = sessions
		self.selectedId = None
		self.action = None

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(wx.StaticText(self, label=_("Saved &chats:")), 0, wx.ALL, 8)
		self.listBox = wx.ListBox(self, size=(460, 220),
		                          choices=[self._label(s) for s in sessions])
		if sessions:
			self.listBox.SetSelection(0)
		main.Add(self.listBox, 1, wx.ALL | wx.EXPAND, 8)

		btns = wx.BoxSizer(wx.HORIZONTAL)
		load = wx.Button(self, id=wx.ID_OK, label=_("&Load"))
		load.Bind(wx.EVT_BUTTON, self.onLoad)
		delete = wx.Button(self, label=_("&Delete"))
		delete.Bind(wx.EVT_BUTTON, self.onDelete)
		cancel = wx.Button(self, id=wx.ID_CANCEL, label=_("Cancel"))
		btns.Add(load, 0, wx.ALL, 4)
		btns.Add(delete, 0, wx.ALL, 4)
		btns.Add(cancel, 0, wx.ALL, 4)
		main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 6)
		self.SetSizerAndFit(main)
		self.listBox.SetFocus()

	def _label(self, s):
		import time as _t
		when = s.get("updated", 0)
		ts = _t.strftime("%Y-%m-%d %H:%M", _t.localtime(when)) if when else "?"
		# Translators: one saved chat in the history list.
		return _("{title}  ({count} messages, {date})").format(
			title=s.get("title", "?"), count=s.get("count", 0), date=ts)

	def _current(self):
		i = self.listBox.GetSelection()
		return self.sessions[i] if i != wx.NOT_FOUND else None

	def onLoad(self, evt):
		s = self._current()
		if s:
			self.selectedId = s.get("id")
			self.action = "load"
			self.EndModal(wx.ID_OK)

	def onDelete(self, evt):
		s = self._current()
		if not s:
			return
		if gui.messageBox(
			_("Delete this chat permanently?"), _("Confirm"),
			wx.YES_NO | wx.ICON_QUESTION) == wx.YES:
			self.selectedId = s.get("id")
			self.action = "delete"
			self.EndModal(wx.ID_OK)


# --- custom quick actions manager ------------------------------------------

class CustomActionsDialog(wx.Dialog):
	"""Manage user-defined quick actions (name + instruction prompt). Up to 10,
	matching the fixed gesture slots."""
	MAX = 10

	def __init__(self, parent):
		super(CustomActionsDialog, self).__init__(
			parent, title=_("Custom quick actions"))
		self.actions = list(getCustomActions())

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(wx.StaticText(self, label=_(
			"Define up to 10 custom actions. Assign hotkeys in NVDA Input "
			"Gestures (Offline AI category, \"run custom quick action N\").")),
			0, wx.ALL, 8)

		self.listBox = wx.ListBox(self, size=(460, 180),
		                          choices=self._labels())
		if self.actions:
			self.listBox.SetSelection(0)
		main.Add(self.listBox, 1, wx.ALL | wx.EXPAND, 8)

		btns = wx.BoxSizer(wx.HORIZONTAL)
		add = wx.Button(self, label=_("&Add..."))
		add.Bind(wx.EVT_BUTTON, self.onAdd)
		edit = wx.Button(self, label=_("&Edit..."))
		edit.Bind(wx.EVT_BUTTON, self.onEdit)
		delete = wx.Button(self, label=_("&Delete"))
		delete.Bind(wx.EVT_BUTTON, self.onDelete)
		close = wx.Button(self, id=wx.ID_OK, label=_("&Close"))
		for b in (add, edit, delete):
			btns.Add(b, 0, wx.ALL, 4)
		btns.Add(close, 0, wx.ALL, 4)
		main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 6)
		self.SetSizerAndFit(main)
		self.listBox.SetFocus()

	def _labels(self):
		return [u"{n}. {name}".format(n=i + 1, name=a.get("name", "?"))
		        for i, a in enumerate(self.actions)]

	def _refresh(self):
		sel = self.listBox.GetSelection()
		self.listBox.Set(self._labels())
		if self.actions:
			self.listBox.SetSelection(min(max(sel, 0), len(self.actions) - 1))
		setCustomActions(self.actions)

	def onAdd(self, evt):
		if len(self.actions) >= self.MAX:
			ui.message(_("You already have the maximum of 10 custom actions."))
			return
		dlg = _ActionEditDialog(self)
		if dlg.ShowModal() == wx.ID_OK and dlg.result:
			self.actions.append(dlg.result)
			self._refresh()
		dlg.Destroy()

	def onEdit(self, evt):
		i = self.listBox.GetSelection()
		if i == wx.NOT_FOUND:
			return
		dlg = _ActionEditDialog(self, self.actions[i])
		if dlg.ShowModal() == wx.ID_OK and dlg.result:
			self.actions[i] = dlg.result
			self._refresh()
		dlg.Destroy()

	def onDelete(self, evt):
		i = self.listBox.GetSelection()
		if i == wx.NOT_FOUND:
			return
		del self.actions[i]
		self._refresh()
		ui.message(_("Deleted."))


class _ActionEditDialog(wx.Dialog):
	def __init__(self, parent, action=None):
		super(_ActionEditDialog, self).__init__(
			parent, title=_("Custom action"))
		self.result = None
		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(wx.StaticText(self, label=_("&Name:")), 0, wx.LEFT | wx.TOP, 10)
		self.nameCtrl = wx.TextCtrl(self, size=(420, -1))
		main.Add(self.nameCtrl, 0, wx.ALL | wx.EXPAND, 10)
		main.Add(wx.StaticText(self, label=_(
			"&Instruction (what the model should do with the selected text):")),
			0, wx.LEFT, 10)
		self.instrCtrl = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(420, 100))
		main.Add(self.instrCtrl, 0, wx.ALL | wx.EXPAND, 10)
		if action:
			self.nameCtrl.SetValue(action.get("name", ""))
			self.instrCtrl.SetValue(action.get("instruction", ""))
		btns = wx.BoxSizer(wx.HORIZONTAL)
		ok = wx.Button(self, id=wx.ID_OK, label=_("&Save"))
		ok.Bind(wx.EVT_BUTTON, self.onSave)
		cancel = wx.Button(self, id=wx.ID_CANCEL, label=_("Cancel"))
		btns.Add(ok, 0, wx.ALL, 4)
		btns.Add(cancel, 0, wx.ALL, 4)
		main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 6)
		self.SetSizerAndFit(main)
		self.nameCtrl.SetFocus()

	def onSave(self, evt):
		name = self.nameCtrl.GetValue().strip()
		instr = self.instrCtrl.GetValue().strip()
		if not name or not instr:
			ui.message(_("Please give both a name and an instruction."))
			return
		self.result = {"name": name, "instruction": instr}
		self.EndModal(wx.ID_OK)


# --- HuggingFace search dialog ---------------------------------------------

class HFSearchDialog(wx.Dialog):
	"""Search HuggingFace for GGUF models, pick a repo and a quant file, and add
	it to the catalog with a verified download link."""

	def __init__(self, parent):
		super(HFSearchDialog, self).__init__(
			parent, title=_("Search HuggingFace for models"))
		self.result = None
		self._repos = []
		self._files = []

		main = wx.BoxSizer(wx.VERTICAL)

		srow = wx.BoxSizer(wx.HORIZONTAL)
		srow.Add(wx.StaticText(self, label=_("&Search:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
		self.searchCtrl = wx.TextCtrl(self, size=(300, -1),
		                              style=wx.TE_PROCESS_ENTER)
		self.searchCtrl.Bind(wx.EVT_TEXT_ENTER, self.onSearch)
		srow.Add(self.searchCtrl, 1, wx.RIGHT, 6)
		self.searchButton = wx.Button(self, label=_("&Go"))
		self.searchButton.Bind(wx.EVT_BUTTON, self.onSearch)
		srow.Add(self.searchButton, 0)
		main.Add(srow, 0, wx.ALL | wx.EXPAND, 10)

		main.Add(wx.StaticText(self, label=_("&Repositories:")), 0, wx.LEFT, 10)
		self.repoList = wx.ListBox(self, size=(460, 140))
		self.repoList.Bind(wx.EVT_LISTBOX, self.onRepoSelected)
		self.repoList.Bind(wx.EVT_CONTEXT_MENU, self.onRepoContext)
		main.Add(self.repoList, 0, wx.ALL | wx.EXPAND, 10)

		main.Add(wx.StaticText(self, label=_("&Files (quantizations):")), 0, wx.LEFT, 10)
		self.fileList = wx.ListBox(self, size=(460, 120))
		self.fileList.Bind(wx.EVT_LISTBOX, self.onFileSelected)
		self.fileList.Bind(wx.EVT_CONTEXT_MENU, self.onFileContext)
		main.Add(self.fileList, 0, wx.ALL | wx.EXPAND, 10)

		btns = wx.BoxSizer(wx.HORIZONTAL)
		self.addButton = wx.Button(self, id=wx.ID_OK, label=_("&Add selected"))
		self.addButton.Bind(wx.EVT_BUTTON, self.onAdd)
		self.addButton.Enable(False)
		cancel = wx.Button(self, id=wx.ID_CANCEL, label=_("Cancel"))
		btns.Add(self.addButton, 0, wx.ALL, 4)
		btns.Add(cancel, 0, wx.ALL, 4)
		main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 6)

		self.SetSizerAndFit(main)
		self.searchCtrl.SetFocus()

	def onSearch(self, evt):
		q = self.searchCtrl.GetValue().strip()
		if not q:
			return
		self.searchButton.Enable(False)
		ui.message(_("Searching HuggingFace."))
		threading.Thread(target=self._searchWorker, args=(q,), daemon=True).start()

	def _searchWorker(self, q):
		from . import catalog
		limit = loadSettings().get("hf_result_limit", 60)
		repos = catalog.hf_search_repos(q, limit=limit)
		wx.CallAfter(self._searchDone, repos)

	def _searchDone(self, repos):
		self._repos = repos
		self.repoList.Set(repos if repos else [_("No results.")])
		self.fileList.Clear()
		self.addButton.Enable(False)
		self.searchButton.Enable(True)
		if repos:
			self.repoList.SetSelection(0)
			ui.message(_("{n} repositories found.").format(n=len(repos)))
		else:
			ui.message(_("No repositories found."))

	def onRepoSelected(self, evt):
		i = self.repoList.GetSelection()
		if i == wx.NOT_FOUND or not self._repos:
			return
		repo = self._repos[i]
		self.fileList.Set([_("Loading...")])
		threading.Thread(target=self._filesWorker, args=(repo,), daemon=True).start()

	def _filesWorker(self, repo):
		from . import catalog
		files = catalog.hf_list_gguf_files(repo)
		wx.CallAfter(self._filesDone, files)

	def _filesDone(self, files):
		# files is a list of (filename, size_bytes) tuples.
		self._files = files
		if files:
			from . import catalog
			labels = []
			for fn, size in files:
				sz = catalog._fmt_bytes(size) if size else _("size unknown")
				labels.append(u"{f}  ({s})".format(f=fn, s=sz))
			self.fileList.Set(labels)
			# Prefer a Q4_K_M by default.
			default = 0
			for idx, (fn, _sz) in enumerate(files):
				if "Q4_K_M" in fn or "q4_k_m" in fn:
					default = idx
					break
			self.fileList.SetSelection(default)
			self._announceFileSize(default)
			self.addButton.Enable(True)
		else:
			self.fileList.Set([_("No GGUF files found.")])
			self.addButton.Enable(False)

	def _announceFileSize(self, idx):
		if 0 <= idx < len(self._files):
			from . import catalog
			fn, size = self._files[idx]
			sz = catalog._fmt_bytes(size) if size else _("unknown size")
			ui.message(_("{f}, {s}").format(f=fn, s=sz))

	def onFileSelected(self, evt):
		self._announceFileSize(self.fileList.GetSelection())

	# -- context menus --

	def _repoUrl(self, repo):
		return "https://huggingface.co/" + repo

	def _fileUrl(self, repo, filename):
		return "https://huggingface.co/%s/resolve/main/%s" % (repo, filename)

	def onRepoContext(self, evt):
		i = self.repoList.GetSelection()
		if i == wx.NOT_FOUND or not self._repos:
			return
		repo = self._repos[i]
		menu = wx.Menu()
		mUrl = menu.Append(wx.ID_ANY, _("Copy repository &URL"))
		mId = menu.Append(wx.ID_ANY, _("Copy repository &ID"))
		mOpen = menu.Append(wx.ID_ANY, _("Open in &browser"))
		mFiles = menu.Append(wx.ID_ANY, _("List &files for this model"))
		self.Bind(wx.EVT_MENU, lambda e: self._clip(self._repoUrl(repo),
		          _("Repository URL copied.")), mUrl)
		self.Bind(wx.EVT_MENU, lambda e: self._clip(repo,
		          _("Repository ID copied.")), mId)
		self.Bind(wx.EVT_MENU, lambda e: self._openBrowser(self._repoUrl(repo)), mOpen)
		self.Bind(wx.EVT_MENU, lambda e: self.onRepoSelected(None), mFiles)
		self.repoList.PopupMenu(menu)
		menu.Destroy()

	def onFileContext(self, evt):
		fi = self.fileList.GetSelection()
		ri = self.repoList.GetSelection()
		if fi == wx.NOT_FOUND or ri == wx.NOT_FOUND or not self._files:
			return
		repo = self._repos[ri]
		filename, size = self._files[fi]
		menu = wx.Menu()
		mDl = menu.Append(wx.ID_ANY, _("Copy &download URL"))
		mName = menu.Append(wx.ID_ANY, _("Copy &file name"))
		mSize = menu.Append(wx.ID_ANY, _("&Announce size"))
		mAdd = menu.Append(wx.ID_ANY, _("&Add this model"))
		self.Bind(wx.EVT_MENU, lambda e: self._clip(self._fileUrl(repo, filename),
		          _("Download URL copied.")), mDl)
		self.Bind(wx.EVT_MENU, lambda e: self._clip(filename,
		          _("File name copied.")), mName)
		self.Bind(wx.EVT_MENU, lambda e: self._announceFileSize(fi), mSize)
		self.Bind(wx.EVT_MENU, lambda e: self.onAdd(None), mAdd)
		self.fileList.PopupMenu(menu)
		menu.Destroy()

	def _clip(self, text, msg):
		try:
			import api
			api.copyToClip(text)
			ui.message(msg)
		except Exception:
			ui.message(_("Could not copy."))

	def _openBrowser(self, url):
		try:
			import webbrowser
			webbrowser.open(url)
		except Exception:
			ui.message(_("Could not open the browser."))

	def onAdd(self, evt):
		ri = self.repoList.GetSelection()
		fi = self.fileList.GetSelection()
		if ri == wx.NOT_FOUND or fi == wx.NOT_FOUND or not self._files:
			return
		repo = self._repos[ri]
		filename, size = self._files[fi]
		ui.message(_("Verifying download link."))
		self.addButton.Enable(False)
		threading.Thread(target=self._addWorker, args=(repo, filename, size),
		                 daemon=True).start()

	def _addWorker(self, repo, filename, size):
		from . import catalog
		fmt = catalog.guess_prompt_format(repo)
		entry, err = catalog.hf_build_entry(repo, filename, fmt, size_bytes=size)
		wx.CallAfter(self._addDone, entry, err)

	def _addDone(self, entry, err):
		if entry:
			self.result = entry
			self.EndModal(wx.ID_OK)
		else:
			self.addButton.Enable(True)
			ui.message(_("Could not add this model: {e}").format(e=err))


# --- custom model dialog ---------------------------------------------------

class CustomModelDialog(wx.Dialog):
	"""Collects a custom GGUF model: name, URL or local file, prompt format,
	optional context window."""
	PROMPT_FORMATS = list(inference.PROMPT_FORMATS)

	def __init__(self, parent):
		super(CustomModelDialog, self).__init__(parent, title=_("Add custom GGUF model"))
		self.result = None
		self._localPath = None

		main = wx.BoxSizer(wx.VERTICAL)

		main.Add(wx.StaticText(self, label=_("&Name:")), 0, wx.LEFT | wx.TOP, 10)
		self.nameCtrl = wx.TextCtrl(self, size=(440, -1))
		main.Add(self.nameCtrl, 0, wx.ALL | wx.EXPAND, 10)

		main.Add(wx.StaticText(self, label=_(
			"Direct &URL to a .gguf file (or use Browse for a local file):")),
			0, wx.LEFT, 10)
		urow = wx.BoxSizer(wx.HORIZONTAL)
		self.urlCtrl = wx.TextCtrl(self, size=(340, -1))
		urow.Add(self.urlCtrl, 1, wx.RIGHT, 6)
		self.browseButton = wx.Button(self, label=_("&Browse..."))
		self.browseButton.Bind(wx.EVT_BUTTON, self.onBrowse)
		urow.Add(self.browseButton, 0)
		main.Add(urow, 0, wx.ALL | wx.EXPAND, 10)

		frow = wx.BoxSizer(wx.HORIZONTAL)
		frow.Add(wx.StaticText(self, label=_("Prompt &format:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		# Translators: "auto" prompt format = use the template inside the model file.
		self.fmtChoice = wx.Choice(self, choices=[
			_("auto (use the template stored in the model file)")
			if f == "auto" else f for f in self.PROMPT_FORMATS])
		self.fmtChoice.SetSelection(0)
		try:
			self.fmtChoice.SetName(_("Prompt format"))
		except Exception:
			pass
		frow.Add(self.fmtChoice, 0)
		main.Add(frow, 0, wx.LEFT | wx.BOTTOM, 10)

		crow = wx.BoxSizer(wx.HORIZONTAL)
		crow.Add(wx.StaticText(self, label=_("&Context window:")),
		         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
		self.ctxCtrl = wx.SpinCtrl(self, min=256, max=1000000, initial=4096)
		try:
			self.ctxCtrl.SetName(_("Context window"))
		except Exception:
			pass
		crow.Add(self.ctxCtrl, 0)
		main.Add(crow, 0, wx.LEFT | wx.BOTTOM, 10)

		btns = wx.BoxSizer(wx.HORIZONTAL)
		ok = wx.Button(self, id=wx.ID_OK, label=_("&Add"))
		ok.Bind(wx.EVT_BUTTON, self.onOk)
		cancel = wx.Button(self, id=wx.ID_CANCEL, label=_("Cancel"))
		btns.Add(ok, 0, wx.ALL, 4)
		btns.Add(cancel, 0, wx.ALL, 4)
		main.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 6)

		self.SetSizerAndFit(main)
		self.nameCtrl.SetFocus()

	def onBrowse(self, evt):
		dlg = wx.FileDialog(self, _("Choose a .gguf model file"),
		                    wildcard=_("GGUF models|*.gguf|All files|*.*"),
		                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		if dlg.ShowModal() == wx.ID_OK:
			self._localPath = dlg.GetPath()
			self.urlCtrl.SetValue(self._localPath)
			if not self.nameCtrl.GetValue().strip():
				base = os.path.splitext(os.path.basename(self._localPath))[0]
				self.nameCtrl.SetValue(base)
		dlg.Destroy()

	def onOk(self, evt):
		name = self.nameCtrl.GetValue().strip()
		src = self.urlCtrl.GetValue().strip()
		if not name or not src:
			ui.message(_("Please provide a name and a URL or file."))
			return
		fmt = self.PROMPT_FORMATS[self.fmtChoice.GetSelection()]
		ctx = self.ctxCtrl.GetValue()

		# Local file: reference it directly in place by copying into the store on
		# first use is heavy; instead we register the local path as the URL-like
		# source and let the model loader read it directly.
		if os.path.isfile(src):
			filename = os.path.basename(src)
			self.result = {
				"name": name, "filename": filename,
				"url": "file://" + src,  # marker; loader treats existing path as local
				"prompt_format": fmt, "context": ctx,
				"size": _("local file"), "ram": "?",
				"local_path": src,
			}
		else:
			# Treat as a download URL. Derive filename from the URL tail.
			tail = src.split("?")[0].rstrip("/").split("/")[-1]
			if not tail.lower().endswith(".gguf"):
				tail = (name.replace(" ", "_") + ".gguf")
			self.result = {
				"name": name, "filename": tail, "url": src,
				"prompt_format": fmt, "context": ctx,
				"size": "?", "ram": "?",
			}
		self.EndModal(wx.ID_OK)


# --- main dialog -----------------------------------------------------------

class OfflineAIDialog(wx.Dialog):
	_instance = None

	def __init__(self, parent):
		super(OfflineAIDialog, self).__init__(parent, title=_("Offline AI"))
		self.models = loadModelDirectory()
		self.storeDir = getModelsStoreDir()
		self.settings = loadSettings()
		self._streamBuf = []
		self._pendingQuery = None
		self._alive = True
		self._generating = False
		inference.set_pinned(True)
		# Conversation state for multi-turn history + persistence.
		from . import history as histmod
		self.messages = []
		self.sessionId = histmod.new_session_id()
		# Optional reference context (from OCR, an attached file, etc.).
		self.attachedContext = None
		self.attachedLabel = None

		main = wx.BoxSizer(wx.VERTICAL)
		helper = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)

		self.modelCombo = helper.addLabeledControl(
			_("&Model:"), wx.Choice, choices=[self._label(m) for m in self.models])
		if self.models:
			self.modelCombo.SetSelection(self._initialModelIndex())
		self.modelCombo.Bind(wx.EVT_CHOICE, self.onModelChanged)

		# Persona / preset dropdown.
		self._presets = allPresets() + [{"name": _("Custom..."), "system": None}]
		self.presetCombo = helper.addLabeledControl(
			_("&Persona / preset:"), wx.Choice,
			choices=[p["name"] for p in self._presets])
		self.presetCombo.SetSelection(0)
		self.presetCombo.Bind(wx.EVT_CHOICE, self.onPresetChanged)

		row = wx.BoxSizer(wx.HORIZONTAL)
		self.downloadButton = wx.Button(self, label=_("&Download model"))
		self.downloadButton.Bind(wx.EVT_BUTTON, self.onDownload)
		row.Add(self.downloadButton, 0, wx.ALL, 3)
		self.deleteButton = wx.Button(self, label=_("Delete model f&ile"))
		self.deleteButton.Bind(wx.EVT_BUTTON, self.onDeleteModel)
		row.Add(self.deleteButton, 0, wx.ALL, 3)
		self.settingsButton = wx.Button(self, label=_("Se&ttings..."))
		self.settingsButton.Bind(wx.EVT_BUTTON, self.onSettings)
		row.Add(self.settingsButton, 0, wx.ALL, 3)
		self.folderButton = wx.Button(self, label=_("Models &folder..."))
		self.folderButton.Bind(wx.EVT_BUTTON, self.onChangeFolder)
		row.Add(self.folderButton, 0, wx.ALL, 3)
		helper.addItem(row)

		row1b = wx.BoxSizer(wx.HORIZONTAL)
		self.customButton = wx.Button(self, label=_("Add c&ustom GGUF..."))
		self.customButton.Bind(wx.EVT_BUTTON, self.onAddCustom)
		row1b.Add(self.customButton, 0, wx.ALL, 3)
		self.hfSearchButton = wx.Button(self, label=_("Searc&h HuggingFace..."))
		self.hfSearchButton.Bind(wx.EVT_BUTTON, self.onHFSearch)
		row1b.Add(self.hfSearchButton, 0, wx.ALL, 3)
		self.refreshButton = wx.Button(self, label=_("&Refresh catalog"))
		self.refreshButton.Bind(wx.EVT_BUTTON, self.onRefreshCatalog)
		row1b.Add(self.refreshButton, 0, wx.ALL, 3)
		helper.addItem(row1b)

		self.queryEdit = helper.addLabeledControl(
			_("Your &query:"), wx.TextCtrl, style=wx.TE_MULTILINE, size=(520, 90))

		row2 = wx.BoxSizer(wx.HORIZONTAL)
		self.sendButton = wx.Button(self, label=_("&Send"))
		self.sendButton.Bind(wx.EVT_BUTTON, self.onSend)
		row2.Add(self.sendButton, 0, wx.ALL, 3)
		self.stopButton = wx.Button(self, label=_("St&op"))
		self.stopButton.Bind(wx.EVT_BUTTON, self.onStop)
		self.stopButton.Enable(False)
		row2.Add(self.stopButton, 0, wx.ALL, 3)
		helper.addItem(row2)

		self.answerEdit = helper.addLabeledControl(
			_("&Answer:"), wx.TextCtrl,
			style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2, size=(520, 200))
		# Word-like navigation: Ctrl+Up/Down jump by paragraph. (Ctrl+Left/Right
		# word navigation is already native to the multiline control.)
		self.answerEdit.Bind(wx.EVT_KEY_DOWN, self._onAnswerKey)

		row3 = wx.BoxSizer(wx.HORIZONTAL)
		self.newChatButton = wx.Button(self, label=_("&New chat"))
		self.newChatButton.Bind(wx.EVT_BUTTON, self.onNewChat)
		row3.Add(self.newChatButton, 0, wx.ALL, 3)
		self.historyButton = wx.Button(self, label=_("Chat histor&y..."))
		self.historyButton.Bind(wx.EVT_BUTTON, self.onHistory)
		row3.Add(self.historyButton, 0, wx.ALL, 3)
		self.exportButton = wx.Button(self, label=_("&Export chat..."))
		self.exportButton.Bind(wx.EVT_BUTTON, self.onExport)
		row3.Add(self.exportButton, 0, wx.ALL, 3)
		self.copyCodeButton = wx.Button(self, label=_("Copy code &block..."))
		self.copyCodeButton.Bind(wx.EVT_BUTTON, self.onCopyCode)
		row3.Add(self.copyCodeButton, 0, wx.ALL, 3)
		helper.addItem(row3)

		row4 = wx.BoxSizer(wx.HORIZONTAL)
		self.attachButton = wx.Button(self, label=_("Attach conte&xt file..."))
		self.attachButton.Bind(wx.EVT_BUTTON, self.onAttachFile)
		row4.Add(self.attachButton, 0, wx.ALL, 3)
		self.clearContextButton = wx.Button(self, label=_("C&lear context"))
		self.clearContextButton.Bind(wx.EVT_BUTTON, self.onClearContext)
		row4.Add(self.clearContextButton, 0, wx.ALL, 3)
		helper.addItem(row4)
		self.contextLabel = helper.addItem(
			wx.StaticText(self, label=_("Context: none")))

		closeButton = wx.Button(self, id=wx.ID_CLOSE, label=_("&Close"))
		closeButton.Bind(wx.EVT_BUTTON, lambda e: self.Close())
		helper.addItem(closeButton)
		self.EscapeId = wx.ID_CLOSE

		main.Add(helper.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.SetSizerAndFit(main)
		self._stopFlag = threading.Event()
		self._refreshDownloadButton()

	def _initialModelIndex(self):
		"""Open on the model used last time, else the first downloaded one."""
		last = loadLocalConfig().get("last_model_filename")
		firstHave = None
		for i, m in enumerate(self.models):
			if os.path.isfile(self._path(m)):
				if last and m["filename"] == last:
					return i
				if firstHave is None:
					firstHave = i
		return firstHave or 0

	def _label(self, m):
		mark = _(" [downloaded]") if os.path.isfile(self._path(m)) else ""
		cat = categoryLabel(m.get("category"))
		# Translators: one entry of the model list. {ram} is the memory needed.
		return _("{category}{name} - {size}, needs {ram} of RAM{mark}").format(
			category=(cat + ": ") if cat else "", name=m["name"],
			size=m.get("size", "?"), ram=m.get("ram", "?"), mark=mark)

	def _selected(self):
		if not self.models:
			return None
		i = self.modelCombo.GetSelection()
		return self.models[i] if i != wx.NOT_FOUND else None

	def _path(self, m):
		return modelPath(m, self.storeDir)

	def _safe(self, func, *args):
		"""wx.CallAfter that is ignored once the window has been closed."""
		def run():
			if self._alive:
				func(*args)
		wx.CallAfter(run)

	def _relabelModels(self):
		sel = self.modelCombo.GetSelection()
		for i, mm in enumerate(self.models):
			self.modelCombo.SetString(i, self._label(mm))
		if sel != wx.NOT_FOUND:
			self.modelCombo.SetSelection(sel)

	def _refreshDownloadButton(self):
		m = self._selected()
		if not m:
			self.downloadButton.Enable(False)
			self.deleteButton.Enable(False)
			return
		have = os.path.isfile(self._path(m))
		if have:
			self.downloadButton.Enable(False)
			self.downloadButton.SetLabel(_("Already downloaded"))
		else:
			self.downloadButton.Enable(True)
			self.downloadButton.SetLabel(_("&Download model"))
		# Delete is offered for downloaded files and for user-added entries.
		self.deleteButton.Enable(have or bool(m.get("custom")))

	def onModelChanged(self, evt):
		self._refreshDownloadButton()

	def onDeleteModel(self, evt):
		m = self._selected()
		if not m:
			return
		path = self._path(m)
		inStore = os.path.isfile(path) and not m.get("local_path")
		if self._generating:
			ui.message(_("Wait for the current answer to finish first."))
			return
		if inStore:
			if gui.messageBox(
				_("Delete the downloaded file for {n} ({s}) to free disk space? "
				  "You can download it again later.").format(
					n=m["name"], s=m.get("size", "?")),
				_("Delete model"), wx.YES_NO | wx.ICON_QUESTION, self) == wx.YES:
				if inference.loaded_path() and os.path.normcase(
						inference.loaded_path()) == os.path.normcase(path):
					inference.unload(force=True)
				try:
					os.remove(path)
					ui.message(_("Model file deleted."))
				except OSError as e:
					ui.message(_("Could not delete the file: {e}").format(e=str(e)))
		if m.get("custom"):
			if gui.messageBox(
				_("Also remove {n} from your model list?").format(n=m["name"]),
				_("Delete model"), wx.YES_NO | wx.ICON_QUESTION, self) == wx.YES:
				removeCustomModel(m["filename"])
				self._reloadModels(fetch_remote=False)
				return
		self._relabelModels()
		self._refreshDownloadButton()

	def onPresetChanged(self, evt):
		i = self.presetCombo.GetSelection()
		if i == wx.NOT_FOUND:
			return
		preset = self._presets[i]
		if preset.get("system") is None:
			# "Custom..." — offer to save the current system prompt as a preset.
			dlg = wx.TextEntryDialog(
				self, _("Name this preset (uses your current system prompt and "
				        "temperature):"), _("Save preset"))
			if dlg.ShowModal() == wx.ID_OK:
				name = dlg.GetValue().strip()
				if name:
					presets = getUserPresets()
					presets = [p for p in presets if p.get("name") != name]
					presets.append({
						"name": name,
						"system": self.settings.get("system_prompt", ""),
						"temperature": self.settings.get("temperature", 0.7),
					})
					setUserPresets(presets)
					self._presets = allPresets() + [{"name": _("Custom..."), "system": None}]
					self.presetCombo.Set([p["name"] for p in self._presets])
					# Select the newly saved preset.
					for idx, p in enumerate(self._presets):
						if p["name"] == name:
							self.presetCombo.SetSelection(idx)
							break
					ui.message(_("Preset saved."))
			dlg.Destroy()
			return
		# Apply the preset (on top of the settings as they are on disk now, so
		# changes made in NVDA's Settings dialog are not overwritten).
		self.settings = loadSettings()
		self.settings["system_prompt"] = preset.get("system", "")
		if "temperature" in preset:
			self.settings["temperature"] = preset["temperature"]
		saveSettings(self.settings)
		ui.message(_("Preset applied: {n}").format(n=preset["name"]))

	def onSettings(self, evt):
		self.settings = loadSettings()
		dlg = SettingsDialog(self, self.settings)
		if dlg.ShowModal() == wx.ID_OK:
			# Load-time changes (context, threads...) are picked up automatically
			# by the engine on the next question.
			self.settings = dlg.settings
			saveSettings(self.settings)
			ui.message(_("Settings saved."))
		elif dlg.restored:
			# A backup was imported: show its models, presets and settings.
			self.settings = loadSettings()
			self.storeDir = getModelsStoreDir()
			self._reloadModels(fetch_remote=False)
			self._presets = allPresets() + [{"name": _("Custom..."), "system": None}]
			self.presetCombo.Set([p["name"] for p in self._presets])
			self.presetCombo.SetSelection(0)
		dlg.Destroy()

	def onChangeFolder(self, evt):
		dlg = wx.DirDialog(self, _("Choose a folder to store downloaded models"),
		                   defaultPath=self.storeDir)
		if dlg.ShowModal() == wx.ID_OK:
			self.storeDir = dlg.GetPath()
			setModelsStoreDir(self.storeDir)
			self._relabelModels()
			self._refreshDownloadButton()
			ui.message(_("Models folder set to {p}").format(p=self.storeDir))
		dlg.Destroy()

	def _reloadModels(self, fetch_remote=False):
		sel = self.modelCombo.GetSelection()
		self.models = loadModelDirectory(fetch_remote=fetch_remote)
		self.modelCombo.Set([self._label(m) for m in self.models])
		if self.models:
			self.modelCombo.SetSelection(min(max(sel, 0), len(self.models) - 1))
		self._refreshDownloadButton()

	def onAddCustom(self, evt):
		dlg = CustomModelDialog(self)
		if dlg.ShowModal() == wx.ID_OK and dlg.result:
			from . import catalog
			v = catalog.validate_model(dlg.result)
			if not v:
				ui.message(_("That model entry was not valid."))
			else:
				addCustomModel(v)
				self._reloadModels(fetch_remote=False)
				# Select the newly added model.
				for i, m in enumerate(self.models):
					if m["filename"] == v["filename"]:
						self.modelCombo.SetSelection(i)
						self.onModelChanged(None)
						break
				ui.message(_("Custom model added."))
		dlg.Destroy()

	def onHFSearch(self, evt):
		dlg = HFSearchDialog(self)
		if dlg.ShowModal() == wx.ID_OK and dlg.result:
			addCustomModel(dlg.result)
			self._reloadModels(fetch_remote=False)
			for i, m in enumerate(self.models):
				if m["filename"] == dlg.result["filename"]:
					self.modelCombo.SetSelection(i)
					self.onModelChanged(None)
					break
			ui.message(_("Model added from HuggingFace. You can now download it."))
		dlg.Destroy()

	def onRefreshCatalog(self, evt):
		url = getCatalogUrl()
		if not url:
			# Let the user set a catalog URL now.
			dlg = wx.TextEntryDialog(
				self, _("Enter a catalog URL (direct link to a models JSON), "
				        "or leave blank to use only the built-in list:"),
				_("Catalog URL"), value="")
			if dlg.ShowModal() == wx.ID_OK:
				setCatalogUrl(dlg.GetValue().strip())
			dlg.Destroy()
			url = getCatalogUrl()
		self.refreshButton.Enable(False)
		ui.message(_("Refreshing catalog."))
		threading.Thread(target=self._refreshWorker, daemon=True).start()

	def _refreshWorker(self):
		models = loadModelDirectory(fetch_remote=True)
		wx.CallAfter(self._refreshDone, models)

	def _refreshDone(self, models):
		if not self._alive:
			return
		sel = self.modelCombo.GetSelection()
		self.models = models
		self.modelCombo.Set([self._label(m) for m in self.models])
		if self.models:
			self.modelCombo.SetSelection(min(max(sel, 0), len(self.models) - 1))
		self._refreshDownloadButton()
		self.refreshButton.Enable(True)
		ui.message(_("Catalog refreshed. {n} models available.").format(
			n=len(self.models)))

	# -- download with progress dialog --
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
		self._refreshDownloadButton()

	# -- generation (streaming) --
	def onStop(self, evt):
		self._stopFlag.set()
		self.stopButton.Enable(False)
		ui.message(_("Stopping."))

	def onSend(self, evt):
		m = self._selected()
		if not m:
			ui.message(_("No model selected."))
			return
		path = self._path(m)
		if not os.path.isfile(path):
			ui.message(_("This model is not downloaded yet. Download it first."))
			return
		q = self.queryEdit.GetValue().strip()
		if not q:
			ui.message(_("Please type a query first."))
			return
		if self._generating:
			return
		if not inference.try_acquire_busy():
			ui.message(_("Offline AI is busy with another request. Please wait."))
			return
		self._generating = True
		self.settings = loadSettings()
		self.sendButton.Enable(False)
		self.stopButton.Enable(True)
		self._stopFlag.clear()
		self._streamBuf = []
		self._pendingQuery = q
		self.answerEdit.SetValue("")
		ui.message(_("Thinking."))
		self._startHeartbeat()
		threading.Thread(
			target=self._genWorker,
			args=(m, path, q, dict(self.settings), self.attachedContext,
			      self.attachedLabel, list(self.messages)),
			daemon=True).start()

	def _startHeartbeat(self):
		self._heartbeatStop = stop = threading.Event()
		if not self.settings.get("heartbeat", True) or \
				self.settings.get("cue_mute", False):
			return
		vol = int(self.settings.get("cue_volume", 60))

		def beat():
			# small initial delay so quick replies don't beep at all
			if stop.wait(1.5):
				return
			import tones
			while not stop.is_set():
				try:
					wx.CallAfter(tones.beep, 200, 20, left=vol, right=vol)
				except Exception:
					return
				if stop.wait(2.5):
					return

		threading.Thread(target=beat, daemon=True).start()

	def _stopHeartbeat(self):
		hb = getattr(self, "_heartbeatStop", None)
		if hb is not None:
			hb.set()

	def _genWorker(self, m, path, query, settings, context, contextLabel,
	               messages):
		"""Worker thread. Everything it needs was copied on the main thread; it
		talks back only through _safe(), so closing the window mid-answer is
		harmless."""
		try:
			ok, err = inference.load_model(path, settings, context=m.get("context"))
			if not ok:
				self._safe(self._genDone,
				           _("Could not load the model: {e}").format(e=err), True)
				return
			if loadLocalConfig().get("last_model_filename") != m["filename"]:
				store.setValue("last_model_filename", m["filename"])

			from . import history as histmod
			from . import think_filter
			fmt = m.get("prompt_format", "chatml")
			system = settings.get("system_prompt", "")
			ctx = inference.n_ctx() or int(settings.get("n_ctx", 2048))
			want = int(settings.get("max_tokens", 512))
			# Room kept free for the answer when deciding what else fits.
			reserve = min(want, max(64, ctx // 3))
			notice = None
			if context:
				budget = (ctx - reserve - inference.count_tokens(system) -
				          inference.count_tokens(query) - 96)
				text, cut = inference.truncate_to_tokens(context, budget)
				if cut:
					notice = _(
						"Note: the attached context is longer than this model's "
						"context window, so only the first part was used. Raise "
						"the context size in Settings to use more of it.")
				if text:
					# The wrapper stays in English: it is an instruction to the
					# model, not something shown to the user.
					system = ((system + "\n\n") if system else "") + (
						"Reference context from %s:\n%s\n\nUse this context to "
						"answer the user's questions." % (
							contextLabel or "the attached document", text))
			hist = inference.fit_history(
				system, histmod.messages_to_history(messages), query, reserve)
			tfilt = think_filter.ThinkFilter(
				mode=settings.get("think_mode", "hide"),
				cue_text=_("[Reasoning done] "))
			live = bool(settings.get("stream", True))
			collected = []
			started = time.time()
			ntok = 0
			for piece in inference.stream(fmt, query, system, hist, settings,
			                              should_stop=self._stopFlag.is_set):
				ntok += 1
				shown = tfilt.feed(piece)
				if shown:
					if live:
						self._safe(self._appendStream, shown)
					else:
						collected.append(shown)
			tail = tfilt.flush()
			if tail:
				if live:
					self._safe(self._appendStream, tail)
				else:
					collected.append(tail)
			elapsed = time.time() - started
			tps = (ntok / elapsed) if elapsed > 0 else 0
			if live:
				self._safe(self._genDoneStream, tps, notice)
			else:
				text = "".join(collected).strip()
				self._safe(self._genDone,
				           text or _("(The model returned an empty response.)"),
				           False, tps, notice)
		except inference.ContextOverflow:
			self._safe(self._genDone, _(
				"The question, together with its attached context, is longer "
				"than this model's context window. Shorten it, clear the "
				"context, or raise the context size in Settings."), True)
		except Exception as e:
			log.error("offlineAI inference failed: %s" % e, exc_info=True)
			self._safe(self._genDone,
			           _("Error during generation: {e}").format(e=str(e)), True)
		finally:
			inference.release_busy()
			self._generating = False

	def _appendStream(self, piece):
		# First token has arrived — silence the heartbeat.
		self._stopHeartbeat()
		# Append without moving the caret away from where the user is reading:
		# only follow the end of the text if the caret is already there.
		pos = self.answerEdit.GetInsertionPoint()
		atEnd = pos >= self.answerEdit.GetLastPosition()
		self.answerEdit.AppendText(piece)
		if not atEnd:
			self.answerEdit.SetInsertionPoint(pos)

	def _onAnswerKey(self, evt):
		"""Word-processor-style paragraph navigation with Ctrl+Up/Down."""
		key = evt.GetKeyCode()
		if evt.ControlDown() and key in (wx.WXK_UP, wx.WXK_DOWN):
			text = self.answerEdit.GetValue()
			pos = self.answerEdit.GetInsertionPoint()
			target = (self._prevParagraph(text, pos) if key == wx.WXK_UP
			          else self._nextParagraph(text, pos))
			self.answerEdit.SetInsertionPoint(target)
			# Let NVDA announce the line at the new position.
			try:
				import ui
				line_start = text.rfind("\n", 0, target) + 1
				line_end = text.find("\n", target)
				if line_end == -1:
					line_end = len(text)
				ui.message(text[line_start:line_end] or _("blank line"))
			except Exception:
				pass
			return  # consume the key
		evt.Skip()  # everything else behaves normally

	@staticmethod
	def _prevParagraph(text, pos):
		# A paragraph boundary is a blank line (\n\n). Find the previous one.
		idx = text.rfind("\n\n", 0, max(0, pos - 1))
		return idx + 2 if idx != -1 else 0

	@staticmethod
	def _nextParagraph(text, pos):
		idx = text.find("\n\n", pos)
		return idx + 2 if idx != -1 else len(text)

	def _genDoneStream(self, tps, notice=None):
		self._stopHeartbeat()
		self.sendButton.Enable(True)
		self.stopButton.Enable(False)
		# The filtered text was already streamed into the box; only handle empty.
		answer = self.answerEdit.GetValue().strip()
		if not answer:
			self.answerEdit.SetValue(_("(The model returned an empty response.)"))
		elif self.settings.get("format_tables", True):
			# Rewrite any markdown tables into readable blocks now that the full
			# answer is present. Preserve the caret position.
			from . import tables
			formatted = tables.format_tables(answer)
			if formatted != answer:
				try:
					prev = self.answerEdit.GetInsertionPoint()
				except Exception:
					prev = 0
				self.answerEdit.SetValue(formatted)
				try:
					self.answerEdit.SetInsertionPoint(min(prev, len(formatted)))
				except Exception:
					pass
				answer = formatted
		# Do NOT otherwise move the caret or steal focus.
		self._recordExchange(answer)
		self._announceDone(tps, notice)

	def _announceDone(self, tps, notice):
		if self._stopFlag.is_set():
			msg = _("Stopped.")
		elif tps > 0:
			msg = _("Done. {t:.1f} tokens per second.").format(t=tps)
		else:
			msg = _("Done.")
		if notice:
			msg += " " + notice
		ui.message(msg)

	def _genDone(self, text, isError, tps=0, notice=None):
		self._stopHeartbeat()
		if not isError and self.settings.get("format_tables", True):
			from . import tables
			text = tables.format_tables(text)
		# Preserve the caret position across the full-text replace where possible.
		try:
			prev = self.answerEdit.GetInsertionPoint()
		except Exception:
			prev = 0
		self.answerEdit.SetValue(text)
		self.sendButton.Enable(True)
		self.stopButton.Enable(False)
		try:
			self.answerEdit.SetInsertionPoint(min(prev, len(text)))
		except Exception:
			pass
		if not isError:
			self._recordExchange(text)
			self._announceDone(tps, notice)
		else:
			# Errors must be heard, not just written into the answer box.
			ui.message(text)

	def _recordExchange(self, answer):
		"""Append the completed user+assistant turn and persist the session."""
		q = getattr(self, "_pendingQuery", None)
		if not q or not answer:
			return
		self.messages.append({"role": "user", "text": q})
		self.messages.append({"role": "assistant", "text": answer})
		self._pendingQuery = None
		try:
			from . import history as histmod
			m = self._selected()
			histmod.save_session(self.sessionId, self.messages,
			                     model=(m or {}).get("name"))
		except Exception as e:
			log.error("offlineAI: autosave failed: %s" % e)

	def onNewChat(self, evt):
		from . import history as histmod
		self.messages = []
		self.sessionId = histmod.new_session_id()
		self.attachedContext = None
		self.attachedLabel = None
		self.queryEdit.SetValue("")
		self.answerEdit.SetValue("")
		self._refreshContextLabel()
		self.queryEdit.SetFocus()
		ui.message(_("Started a new chat."))

	def setContext(self, text, label=None):
		"""Attach reference context (from OCR or a file) to the conversation."""
		self.attachedContext = text
		self.attachedLabel = label or _("document")
		tokens = _estimateTokens(text)
		self._refreshContextLabel()
		self.queryEdit.SetFocus()
		ui.message(_("Attached {label}, about {n} tokens. Type your question.").format(
			label=self.attachedLabel, n=tokens))

	def _refreshContextLabel(self):
		if getattr(self, "contextLabel", None) is None:
			return
		if self.attachedContext:
			tokens = _estimateTokens(self.attachedContext)
			self.contextLabel.SetLabel(
				_("Context: {label} (~{n} tokens)").format(
					label=self.attachedLabel, n=tokens))
		else:
			self.contextLabel.SetLabel(_("Context: none"))

	def onAttachFile(self, evt):
		dlg = wx.FileDialog(
			self, _("Choose a text or source code file to attach as context"),
			# Translators: file type filters when attaching a file to the chat.
			wildcard=(_("Text and source code files") + "|" + TEXT_FILE_PATTERNS +
			          "|" + _("All files") + "|*.*"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		path = None
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
		dlg.Destroy()
		if not path:
			return
		try:
			if os.path.getsize(path) > 20 * 1024 * 1024:
				ui.message(_("That file is larger than 20 MB, which is far more "
				             "than any local model can read."))
				return
			text = readTextFile(path)
		except ValueError:
			ui.message(_("That does not look like a text file. Only text-based "
			             "files (documents, code, logs, data) can be attached."))
			return
		except Exception as e:
			ui.message(_("Could not read the file: {e}").format(e=str(e)))
			return
		if not text.strip():
			ui.message(_("That file is empty."))
			return
		# Warn if the file is very large relative to the context window.
		tokens = _estimateTokens(text)
		m = self._selected() or {}
		ctx = int(m.get("context") or loadSettings().get("n_ctx", 2048))
		if tokens > ctx * 0.7:
			if gui.messageBox(
				_("This file is about {n} tokens, but the model's context is "
				  "{c} tokens, so only the first part will be used. You can "
				  "raise the context size in Settings. Attach anyway?").format(
					n=tokens, c=ctx),
				_("Large file"), wx.YES_NO | wx.ICON_QUESTION, self) != wx.YES:
				return
		self.setContext(text, os.path.basename(path))

	def onClearContext(self, evt):
		self.attachedContext = None
		self.attachedLabel = None
		self._refreshContextLabel()
		ui.message(_("Context cleared."))

	def onHistory(self, evt):
		from . import history as histmod
		sessions = histmod.list_sessions()
		if not sessions:
			ui.message(_("No saved chats yet."))
			return
		dlg = HistoryDialog(self, sessions)
		if dlg.ShowModal() == wx.ID_OK and dlg.selectedId:
			if dlg.action == "delete":
				histmod.delete_session(dlg.selectedId)
				ui.message(_("Chat deleted."))
			else:
				s = histmod.load_session(dlg.selectedId)
				if s:
					self.messages = list(s.get("messages", []))
					self.sessionId = s.get("id")
					# Rebuild the answer box to show the conversation.
					self._renderConversation()
					ui.message(_("Chat loaded."))
		dlg.Destroy()

	def _renderConversation(self):
		lines = []
		for msg in self.messages:
			who = _("You: ") if msg.get("role") == "user" else _("AI: ")
			lines.append(who + (msg.get("text") or ""))
		self.answerEdit.SetValue("\n\n".join(lines))
		self.answerEdit.SetInsertionPoint(0)

	def onExport(self, evt):
		from . import history as histmod
		if not self.messages:
			ui.message(_("Nothing to export yet."))
			return
		dlg = wx.FileDialog(
			self, _("Export chat"),
			defaultFile="chat.md",
			wildcard=_("Markdown|*.md|Text file|*.txt"),
			style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
			as_md = path.lower().endswith(".md")
			session = {"title": None, "messages": self.messages}
			try:
				histmod.export_session(session, path, as_markdown=as_md)
				ui.message(_("Saved {n}").format(n=os.path.basename(path)))
			except Exception as e:
				log.error("offlineAI export failed: %s" % e)
				ui.message(_("Could not save: {e}").format(e=str(e)))
		dlg.Destroy()

	def onCopyCode(self, evt):
		from . import codeblocks
		# Look at the latest assistant message (full text), not the possibly
		# think-filtered display.
		latest = ""
		for msg in reversed(self.messages):
			if msg.get("role") == "assistant":
				latest = msg.get("text", "")
				break
		if not latest:
			latest = self.answerEdit.GetValue()
		blocks = codeblocks.extract_code_blocks(latest)
		if not blocks:
			ui.message(_("No code blocks found in the latest answer."))
			return
		if len(blocks) == 1:
			self._copyBlock(blocks[0])
			return
		# Multiple: show a picker.
		choices = []
		for i, b in enumerate(blocks):
			label = codeblocks.language_label(b["lang"])
			first = b["code"].strip().split("\n")[0][:40]
			choices.append(u"{n}. {l}: {f}".format(n=i + 1, l=label, f=first))
		dlg = wx.SingleChoiceDialog(
			self, _("Choose a code block to copy:"), _("Copy code block"), choices)
		if dlg.ShowModal() == wx.ID_OK:
			self._copyBlock(blocks[dlg.GetSelection()])
		dlg.Destroy()

	def _copyBlock(self, block):
		try:
			import api
			api.copyToClip(block["code"])
			from . import codeblocks
			ui.message(_("{l} copied.").format(
				l=codeblocks.language_label(block["lang"])))
		except Exception as e:
			log.error("offlineAI: copy code failed: %s" % e)
			ui.message(_("Could not copy the code."))

	def onClose(self):
		self._alive = False
		self._stopFlag.set()
		self._stopHeartbeat()
		inference.set_pinned(False)
		# Free the model's memory. If an answer is still being generated it
		# stops at the next token and the idle timer frees the model later.
		threading.Thread(target=inference.unload, daemon=True).start()
		OfflineAIDialog._instance = None
		self.Destroy()

	@classmethod
	def show(cls, inject_context=None, context_label=None):
		if cls._instance is not None:
			try:
				cls._instance.Raise()
				if inject_context:
					cls._instance.setContext(inject_context, context_label)
				return
			except Exception:
				cls._instance = None
		d = cls(gui.mainFrame)
		cls._instance = d
		if inject_context:
			d.setContext(inject_context, context_label)
		d.Bind(wx.EVT_CLOSE, lambda e: d.onClose())
		gui.mainFrame.prePopup()
		d.Show()
		gui.mainFrame.postPopup()


# --- text-to-speech (voice cloning) dialog ---------------------------------

class TTSDialog(wx.Dialog):
	_instance = None

	def __init__(self, parent, storeDirProvider):
		super(TTSDialog, self).__init__(
			parent, title=_("Offline AI - Text to speech (voice cloning)"))
		from . import tts
		self.tts = tts
		try:
			tts.set_temp_dir(getWorkTempDir())
		except Exception:
			pass
		self.getStoreDir = storeDirProvider
		self._busy = False
		self._alive = True

		helper = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)

		self.textEdit = helper.addLabeledControl(
			_("&Text to speak:"), wx.TextCtrl,
			style=wx.TE_MULTILINE, size=(520, 120))

		# The voice model speaks a fixed set of languages; show them in the
		# user's language and start on the user's own one when it is supported.
		self._langCodes = langs.sortedCodes(
			tts.TTS_LANGUAGE_CODES, pinned=(langs.nvdaLanguage(),))
		self.langCombo = helper.addLabeledControl(
			_("&Language:"), wx.Choice,
			choices=[langs.displayName(c) for c in self._langCodes])
		cur = tts.default_language(loadSettings())
		self.langCombo.SetSelection(
			self._langCodes.index(cur) if cur in self._langCodes else 0)
		self.langCombo.Bind(
			wx.EVT_CHOICE,
			lambda e: store.updateSettings(tts_lang=self._lang()))

		# Voice profile dropdown (cloning).
		self._reloadProfiles()
		self.voiceCombo = helper.addLabeledControl(
			_("&Voice:"), wx.Choice, choices=self._voiceLabels())
		self.voiceCombo.SetSelection(0)

		row = wx.BoxSizer(wx.HORIZONTAL)
		self.addVoiceButton = wx.Button(self, label=_("&Add voice from file..."))
		self.addVoiceButton.Bind(wx.EVT_BUTTON, self.onAddVoice)
		row.Add(self.addVoiceButton, 0, wx.ALL, 3)
		self.manageVoiceButton = wx.Button(self, label=_("&Manage voices..."))
		self.manageVoiceButton.Bind(wx.EVT_BUTTON, self.onManageVoices)
		row.Add(self.manageVoiceButton, 0, wx.ALL, 3)
		helper.addItem(row)

		# Speed control.
		self._speeds = ttsSpeedChoices()
		self.speedCombo = helper.addLabeledControl(
			_("Playback &speed:"), wx.Choice,
			choices=[lbl for lbl, _v in self._speeds])
		# default from settings
		dspeed = loadSettings().get("tts_speed", 1.0)
		sel = next((i for i, (_l, v) in enumerate(self._speeds) if v == dspeed), 1)
		self.speedCombo.SetSelection(sel)

		# Action row.
		row2 = wx.BoxSizer(wx.HORIZONTAL)
		self.speakButton = wx.Button(self, label=_("&Speak"))
		self.speakButton.Bind(wx.EVT_BUTTON, self.onSpeak)
		row2.Add(self.speakButton, 0, wx.ALL, 3)
		self.saveButton = wx.Button(self, label=_("Sa&ve as WAV file..."))
		self.saveButton.Bind(wx.EVT_BUTTON, self.onSaveWav)
		row2.Add(self.saveButton, 0, wx.ALL, 3)
		helper.addItem(row2)

		# Document reader row (sentence streaming with pause/resume/stop).
		row3 = wx.BoxSizer(wx.HORIZONTAL)
		self.readButton = wx.Button(self, label=_("&Read document (streaming)"))
		self.readButton.Bind(wx.EVT_BUTTON, self.onReadDocument)
		row3.Add(self.readButton, 0, wx.ALL, 3)
		self.pauseButton = wx.Button(self, label=_("&Pause/Resume"))
		self.pauseButton.Bind(wx.EVT_BUTTON, self.onPauseResume)
		self.pauseButton.Enable(False)
		row3.Add(self.pauseButton, 0, wx.ALL, 3)
		self.stopReadButton = wx.Button(self, label=_("St&op"))
		self.stopReadButton.Bind(wx.EVT_BUTTON, self.onStopReading)
		self.stopReadButton.Enable(False)
		row3.Add(self.stopReadButton, 0, wx.ALL, 3)
		helper.addItem(row3)
		self._reader = None
		self._paused = False

		# Model download row (two files).
		self.dlRow = wx.BoxSizer(wx.HORIZONTAL)
		self.dlButton = wx.Button(self, label=_("&Download voice model (~1.5 GB)"))
		self.dlButton.Bind(wx.EVT_BUTTON, self.onDownloadModel)
		self.dlRow.Add(self.dlButton, 0, wx.ALL, 3)
		helper.addItem(self.dlRow)

		closeButton = wx.Button(self, id=wx.ID_CLOSE, label=_("&Close"))
		closeButton.Bind(wx.EVT_BUTTON, lambda e: self.Close())
		helper.addItem(closeButton)
		self.EscapeId = wx.ID_CLOSE

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(helper.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.SetSizerAndFit(main)
		self._refreshState()
		if not self.tts.engine_present():
			ui.message(_("Note: the text-to-speech engine is not installed in "
			             "this copy of the add-on."))

	def _safe(self, func, *args):
		def run():
			if self._alive:
				func(*args)
		wx.CallAfter(run)

	def _refreshState(self):
		have = self.tts.models_present(self.getStoreDir())
		engine = self.tts.engine_present()
		ready = have and engine and not self._busy
		self.speakButton.Enable(ready)
		self.saveButton.Enable(ready)
		self.readButton.Enable(have and engine)
		if have:
			self.dlButton.Hide()
		else:
			self.dlButton.Show()
			self.dlButton.Enable(engine)
		try:
			self.dlButton.GetParent().Layout()
		except Exception:
			pass

	def _reloadProfiles(self):
		# First entry is always the default (no cloning).
		self._profiles = [{"name": _("Default voice"), "path": None}] + \
		                 getVoiceProfiles()

	def _voiceLabels(self):
		return [p["name"] for p in self._profiles]

	def _currentSpeaker(self):
		i = self.voiceCombo.GetSelection()
		if i == wx.NOT_FOUND or i >= len(self._profiles):
			return None
		return self._profiles[i].get("path")

	def _currentSpeed(self):
		i = self.speedCombo.GetSelection()
		return self._speeds[i][1] if i != wx.NOT_FOUND else 1.0

	def onAddVoice(self, evt):
		dlg = wx.FileDialog(
			self, _("Choose a voice sample (a few seconds of clear speech)"),
			wildcard=_("Audio files|*.wav;*.mp3;*.flac;*.m4a|All files|*.*"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		path = None
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
		dlg.Destroy()
		if not path:
			return
		nameDlg = wx.TextEntryDialog(
			self, _("Name this voice (e.g. Narrator, Lecturer):"),
			_("Save voice"),
			value=os.path.splitext(os.path.basename(path))[0])
		if nameDlg.ShowModal() == wx.ID_OK:
			name = nameDlg.GetValue().strip()
			if name:
				# Copy the sample into a voice_profiles folder so it persists.
				try:
					dest_dir = os.path.join(self.getStoreDir(), "voice_profiles")
					if not os.path.isdir(dest_dir):
						os.makedirs(dest_dir)
					import shutil
					# The name may contain characters Windows forbids in file
					# names (":", "/", "?"...), in any script; clean it first.
					dest = os.path.join(
						dest_dir,
						self.tts.safe_file_name(name) + os.path.splitext(path)[1])
					if os.path.normcase(os.path.abspath(path)) != \
							os.path.normcase(os.path.abspath(dest)):
						shutil.copyfile(path, dest)
					path = dest
				except Exception as e:
					log.error("offlineAI: voice copy failed: %s" % e)
				addVoiceProfile(name, path)
				self._reloadProfiles()
				self.voiceCombo.Set(self._voiceLabels())
				for idx, p in enumerate(self._profiles):
					if p["name"] == name:
						self.voiceCombo.SetSelection(idx)
						break
				ui.message(_("Voice saved."))
		nameDlg.Destroy()

	def onManageVoices(self, evt):
		profiles = getVoiceProfiles()
		if not profiles:
			ui.message(_("No saved voices yet. Use Add voice from file."))
			return
		names = [p["name"] for p in profiles]
		dlg = wx.SingleChoiceDialog(
			self, _("Select a voice to delete:"), _("Manage voices"), names)
		if dlg.ShowModal() == wx.ID_OK:
			i = dlg.GetSelection()
			victim = profiles[i]
			profiles = [p for p in profiles if p.get("name") != victim["name"]]
			setVoiceProfiles(profiles)
			# Optionally remove the copied sample file.
			try:
				pth = victim.get("path")
				vdir = os.path.join(self.getStoreDir(), "voice_profiles")
				if pth and os.path.isfile(pth) and os.path.normcase(
						os.path.dirname(os.path.abspath(pth))) == \
						os.path.normcase(os.path.abspath(vdir)):
					os.remove(pth)
			except OSError:
				pass
			self._reloadProfiles()
			self.voiceCombo.Set(self._voiceLabels())
			self.voiceCombo.SetSelection(0)
			ui.message(_("Voice deleted."))
		dlg.Destroy()

	def _lang(self):
		i = self.langCombo.GetSelection()
		return self._langCodes[i] if i != wx.NOT_FOUND else "en"

	def onSpeak(self, evt):
		text = self.textEdit.GetValue().strip()
		if not text:
			ui.message(_("Type some text first."))
			return
		if self._busy:
			ui.message(_("Already generating speech."))
			return
		self._busy = True
		self._refreshState()
		self.stopReadButton.Enable(True)
		threading.Thread(target=self._speakWorker,
		                 args=(text, self._lang(), self._currentSpeaker(), None,
		                       self._currentSpeed()),
		                 daemon=True).start()

	def onSaveWav(self, evt):
		text = self.textEdit.GetValue().strip()
		if not text:
			ui.message(_("Type some text first."))
			return
		dlg = wx.FileDialog(
			self, _("Save speech as WAV"), defaultFile="speech.wav",
			wildcard=_("WAV audio|*.wav"),
			style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
		out = None
		if dlg.ShowModal() == wx.ID_OK:
			out = dlg.GetPath()
		dlg.Destroy()
		if not out:
			return
		if self._busy:
			ui.message(_("Already generating speech."))
			return
		self._busy = True
		self._refreshState()
		self.stopReadButton.Enable(True)
		ui.message(_("Generating speech."))
		threading.Thread(target=self._speakWorker,
		                 args=(text, self._lang(), self._currentSpeaker(), out,
		                       self._currentSpeed()),
		                 daemon=True).start()

	# -- streaming document reader --

	def onReadDocument(self, evt):
		text = self.textEdit.GetValue().strip()
		if not text:
			ui.message(_("Type or paste some text first."))
			return
		if self._reader and self._reader.is_running():
			ui.message(_("Already reading."))
			return
		settings = loadSettings()
		self._reader = self.tts.DocumentReader(
			self.getStoreDir(), lang=self._lang(),
			speaker_file=self._currentSpeaker(),
			n_threads=(settings.get("n_threads", 0) or None),
			prerender=bool(settings.get("tts_prerender", True)),
			speed=self._currentSpeed(),
			stretch_mode=settings.get("tts_stretch", "smooth"),
			use_mmap=settings.get("use_mmap", True),
			on_status=lambda m: self._safe(ui.message, m))
		if self._reader.start(text):
			self.pauseButton.Enable(True)
			self.stopReadButton.Enable(True)
			self._paused = False
			# Poll for completion to re-disable controls.
			self._pollReader()

	def _pollReader(self):
		if not self._alive:
			return
		if self._reader and self._reader.is_running():
			wx.CallLater(500, self._pollReader)
		else:
			self.pauseButton.Enable(False)
			self.stopReadButton.Enable(self._busy)

	def onPauseResume(self, evt):
		if not self._reader or not self._reader.is_running():
			return
		if self._paused:
			self._reader.resume()
			self._paused = False
		else:
			self._reader.pause()
			self._paused = True

	def onStopReading(self, evt):
		"""Stop whatever is speaking or being generated in this window."""
		if self._reader:
			self._reader.stop()
		else:
			self.tts.cancel_all()
		ui.message(_("Stopped."))
		self.pauseButton.Enable(False)
		self.stopReadButton.Enable(False)

	def _speakWorker(self, text, lang, speaker, out_path, speed):
		settings = loadSettings()
		nthreads = settings.get("n_threads", 0) or None
		storeDir = self.getStoreDir()
		if out_path:
			wav, err = self.tts.synthesize(
				text, storeDir, lang=lang, speaker_file=speaker,
				n_threads=nthreads, out_path=out_path,
				use_mmap=settings.get("use_mmap", True))
			if err:
				self._safe(ui.message, _("Failed: {e}").format(e=err))
			else:
				self._safe(ui.message, _("Saved {n}").format(
					n=os.path.basename(out_path)))
		else:
			_wav, err = self.tts.speak(
				text, storeDir, lang=lang, speaker_file=speaker,
				n_threads=nthreads, speed=speed,
				stretch_mode=settings.get("tts_stretch", "smooth"),
				use_mmap=settings.get("use_mmap", True),
				on_status=lambda m: self._safe(ui.message, m))
			if err:
				self._safe(ui.message, _("Failed: {e}").format(e=err))
		self._safe(self._speakDone)

	def _speakDone(self):
		self._busy = False
		self._refreshState()
		if not (self._reader and self._reader.is_running()):
			self.stopReadButton.Enable(False)

	def onDownloadModel(self, evt):
		from . import downloader
		store = self.getStoreDir()
		self.dlButton.Enable(False)
		# Download the two files in sequence.
		backbone_dest = os.path.join(store, self.tts.TTS_MODEL["backbone"])
		mmproj_dest = os.path.join(store, self.tts.TTS_MODEL["mmproj"])

		def afterMmproj(ok, err):
			if not self._alive:
				return
			if ok:
				ui.message(_("Voice model ready."))
			else:
				ui.message(_("Model download failed: {e}").format(e=err))
			self._refreshState()

		def afterBackbone(ok, err):
			if not self._alive:
				return
			if not ok:
				ui.message(_("Model download failed: {e}").format(e=err))
				self._refreshState()
				return
			if os.path.isfile(mmproj_dest):
				afterMmproj(True, None)
				return
			ui.message(_("Downloading the second part."))
			downloader.download(self, self.tts.TTS_MODEL["mmproj_url"],
			                    mmproj_dest, self.tts.TTS_MODEL["mmproj"],
			                    afterMmproj)

		if os.path.isfile(backbone_dest):
			afterBackbone(True, None)
		else:
			ui.message(_("Downloading the voice model."))
			downloader.download(self, self.tts.TTS_MODEL["backbone_url"],
			                    backbone_dest, self.tts.TTS_MODEL["backbone"],
			                    afterBackbone)

	def onClose(self):
		# Stop the document reader and terminate any TTS subprocess so the model
		# does not keep running after the window closes.
		self._alive = False
		try:
			if getattr(self, "_reader", None):
				self._reader.stop()
		except Exception:
			pass
		try:
			self.tts.cancel_all()
		except Exception:
			pass
		try:
			clearWorkTemp()
		except Exception:
			pass
		TTSDialog._instance = None
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


def ocrLanguageChoices():
	"""(codes, labels) for the OCR language hint: 'auto / mixed' first, then
	every language, in the user's own language, with theirs on top."""
	codes, labels = langs.choiceList(includeAuto=False)
	# Translators: OCR language hint meaning "do not assume a language".
	return [""] + codes, [_("Automatic / mixed")] + labels


def ocrLanguageIndex(codes):
	saved = langs.normalize(loadSettings().get("ocr_lang") or "")
	return codes.index(saved) if saved in codes else 0


class PdfOcrDialog(wx.Dialog):
	"""OCR a PDF with page-range selection, single-or-batch processing, and
	per-page navigation. Each processed page's text is cached; navigating to a
	page shows its text, or an empty box with a note if not yet processed."""

	def __init__(self, parent, pdf_path, page_count, storeDirProvider):
		super(PdfOcrDialog, self).__init__(
			parent, title=_("OCR a PDF"))
		from . import ocr
		self.ocr = ocr
		try:
			ocr.set_temp_dir(getWorkTempDir())
		except Exception:
			pass
		self.getStoreDir = storeDirProvider
		self.pdf_path = pdf_path
		self.page_count = page_count
		self.currentPage = 1
		self._busy = False
		self._cancelled = False
		self._alive = True
		settings = loadSettings()
		self.job = ocr.PdfOcrJob(
			pdf_path, storeDirProvider(),
			n_threads=(settings.get("n_threads", 0) or None),
			max_long_side=_ocrMaxLongSide(settings),
			use_mmap=settings.get("use_mmap", True))

		helper = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)
		helper.addItem(wx.StaticText(self, label=_(
			"This PDF has {n} pages.").format(n=page_count)))

		# Language hint.
		self._langCodes, labels = ocrLanguageChoices()
		self.langHint = helper.addLabeledControl(
			_("Text &language (hint):"), wx.Choice, choices=labels)
		self.langHint.SetSelection(ocrLanguageIndex(self._langCodes))

		# Page range.
		rangeRow = wx.BoxSizer(wx.HORIZONTAL)
		rangeRow.Add(wx.StaticText(self, label=_("&From page:")),
		             0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
		self.fromPage = wx.SpinCtrl(self, min=1, max=page_count, initial=1)
		try:
			self.fromPage.SetName(_("From page"))
		except Exception:
			pass
		rangeRow.Add(self.fromPage, 0, wx.RIGHT, 12)
		rangeRow.Add(wx.StaticText(self, label=_("&To page:")),
		             0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
		self.toPage = wx.SpinCtrl(self, min=1, max=page_count, initial=page_count)
		try:
			self.toPage.SetName(_("To page"))
		except Exception:
			pass
		rangeRow.Add(self.toPage, 0)
		helper.addItem(rangeRow)

		# Mode + process buttons.
		procRow = wx.BoxSizer(wx.HORIZONTAL)
		self.onePageButton = wx.Button(self, label=_("OCR &this page"))
		self.onePageButton.Bind(wx.EVT_BUTTON, self.onOcrCurrent)
		procRow.Add(self.onePageButton, 0, wx.ALL, 3)
		self.batchButton = wx.Button(self, label=_("OCR the &range"))
		self.batchButton.Bind(wx.EVT_BUTTON, self.onOcrRange)
		procRow.Add(self.batchButton, 0, wx.ALL, 3)
		self.stopButton = wx.Button(self, label=_("St&op"))
		self.stopButton.Bind(wx.EVT_BUTTON, self.onStop)
		self.stopButton.Enable(False)
		procRow.Add(self.stopButton, 0, wx.ALL, 3)
		helper.addItem(procRow)

		# Page navigation.
		navRow = wx.BoxSizer(wx.HORIZONTAL)
		self.prevButton = wx.Button(self, label=_("&Previous page"))
		self.prevButton.Bind(wx.EVT_BUTTON, self.onPrev)
		navRow.Add(self.prevButton, 0, wx.ALL, 3)
		self.nextButton = wx.Button(self, label=_("&Next page"))
		self.nextButton.Bind(wx.EVT_BUTTON, self.onNext)
		navRow.Add(self.nextButton, 0, wx.ALL, 3)
		helper.addItem(navRow)

		self.pageLabel = helper.addItem(wx.StaticText(self, label=""))

		self.pageText = helper.addLabeledControl(
			_("Page te&xt:"), wx.TextCtrl,
			style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
			size=(540, 240))

		self.askButton = wx.Button(self, label=_("&Ask AI about this page..."))
		self.askButton.Bind(wx.EVT_BUTTON, self.onAskAI)
		helper.addItem(self.askButton)

		saveRow = wx.BoxSizer(wx.HORIZONTAL)
		self.saveAllButton = wx.Button(self, label=_("&Save all OCR'd text..."))
		self.saveAllButton.Bind(wx.EVT_BUTTON, self.onSaveAll)
		saveRow.Add(self.saveAllButton, 0, wx.ALL, 3)
		helper.addItem(saveRow)

		closeButton = wx.Button(self, id=wx.ID_CLOSE, label=_("&Close"))
		closeButton.Bind(wx.EVT_BUTTON, lambda e: self.Close())
		helper.addItem(closeButton)
		self.EscapeId = wx.ID_CLOSE

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(helper.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.SetSizerAndFit(main)
		self._showPage(1)

	def _langHint(self):
		i = self.langHint.GetSelection()
		if i <= 0:
			return None
		return langs.englishName(self._langCodes[i])

	def _safe(self, func, *args):
		def run():
			if self._alive:
				func(*args)
		wx.CallAfter(run)

	def _setBusy(self, busy):
		self._busy = busy
		self.onePageButton.Enable(not busy)
		self.batchButton.Enable(not busy)
		self.stopButton.Enable(busy)

	def _begin(self):
		"""Common start of an OCR run: apply the language hint as it is NOW
		(it used to be read only when turning pages, so changing it had no
		effect) and remember it for next time."""
		self._stopRequested = False
		self.job.lang_hint = self._langHint()
		i = max(0, self.langHint.GetSelection())
		store.updateSettings(ocr_lang=self._langCodes[i])
		self._setBusy(True)

	def onStop(self, evt):
		self._stopRequested = True
		self.stopButton.Enable(False)
		try:
			self.ocr.cancel_all()
		except Exception:
			pass
		ui.message(_("Stopping."))

	def _showPage(self, page):
		page = max(1, min(page, self.page_count))
		self.currentPage = page
		self.prevButton.Enable(page > 1)
		self.nextButton.Enable(page < self.page_count)
		self.pageLabel.SetLabel(
			_("Page {n} of {t}").format(n=page, t=self.page_count))
		if self.job.is_done(page):
			self.pageText.SetValue(self.job.get_text(page) or "")
		else:
			# Empty box with a note when the page has not been processed yet.
			self.pageText.SetValue(
				_("(Page {n} not processed yet. Use 'OCR this page' or "
				  "'OCR the range'.)").format(n=page))
		self.pageText.SetInsertionPoint(0)

	def onPrev(self, evt):
		self._showPage(self.currentPage - 1)

	def onNext(self, evt):
		self._showPage(self.currentPage + 1)

	def onOcrCurrent(self, evt):
		if self._busy:
			ui.message(_("Already working."))
			return
		self._begin()
		page = self.currentPage
		ui.message(_("Reading page {n}.").format(n=page))
		threading.Thread(target=self._pageWorker, args=(page,), daemon=True).start()

	def _pageWorker(self, page):
		text, err = self.job.process_page(page)
		self._safe(self._pageDone, page, text, err)

	def _pageDone(self, page, text, err):
		self._setBusy(False)
		if err:
			ui.message(_("Failed: {e}").format(e=err))
			return
		if page == self.currentPage:
			self._showPage(page)
		ui.message(_("Page {n} done.").format(n=page))

	def onOcrRange(self, evt):
		if self._busy:
			ui.message(_("Already working."))
			return
		first = self.fromPage.GetValue()
		last = self.toPage.GetValue()
		if first > last:
			ui.message(_("The from page must not be after the to page."))
			return
		self._begin()
		ui.message(_("OCR of pages {a} to {b} started.").format(a=first, b=last))
		threading.Thread(target=self._rangeWorker, args=(first, last),
		                 daemon=True).start()

	def _rangeWorker(self, first, last):
		total = last - first + 1
		done = 0
		for page in range(first, last + 1):
			if self._cancelled or self._stopRequested:
				break
			text, err = self.job.process_page(page)
			if self._cancelled or self._stopRequested:
				break
			done += 1
			if err:
				self._safe(ui.message,
				           _("Page {n} failed: {e}").format(n=page, e=err))
			else:
				self._safe(ui.message, _("Processed {d} of {t}.").format(
					d=done, t=total))
				# Update the view if the user is looking at this page.
				self._safe(self._refreshIfCurrent, page)
		self._safe(self._rangeDone, done, total)

	def _refreshIfCurrent(self, page):
		if page == self.currentPage:
			self._showPage(page)

	def _rangeDone(self, done, total):
		self._setBusy(False)
		if self._stopRequested:
			ui.message(_("Stopped. {d} of {t} pages processed.").format(
				d=done, t=total))
		else:
			ui.message(_("Range complete."))

	def onAskAI(self, evt):
		text = self.pageText.GetValue().strip()
		if not text or not self.job.is_done(self.currentPage):
			ui.message(_("OCR this page first."))
			return
		label = _("PDF page {n}").format(n=self.currentPage)
		OfflineAIDialog.show(inject_context=text, context_label=label)

	def onClosePdf(self):
		# Stop any running page/range OCR and terminate the subprocess so the
		# model is not left running after the window closes.
		self._alive = False
		self._cancelled = True
		try:
			from . import ocr
			ocr.cancel_all()
		except Exception:
			pass
		try:
			clearWorkTemp()
		except Exception:
			pass
		self.Destroy()

	def onSaveAll(self, evt):
		done_pages = sorted(p for p in self.job.results
		                    if self.job.results[p] is not None)
		if not done_pages:
			ui.message(_("No pages have been processed yet."))
			return
		dlg = wx.FileDialog(
			self, _("Save OCR text"),
			defaultDir=os.path.dirname(self.pdf_path),
			defaultFile=os.path.splitext(os.path.basename(self.pdf_path))[0] + ".txt",
			wildcard=_("Text file|*.txt"),
			style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
			try:
				with open(path, "w", encoding="utf-8") as f:
					for p in done_pages:
						f.write(_("--- Page {n} ---").format(n=p) + "\n")
						f.write((self.job.results[p] or "") + "\n\n")
				ui.message(_("Saved {n}").format(n=os.path.basename(path)))
			except Exception as e:
				ui.message(_("Could not save: {e}").format(e=str(e)))
		dlg.Destroy()


class OCRDialog(wx.Dialog):
	_instance = None

	def __init__(self, parent, storeDirProvider):
		super(OCRDialog, self).__init__(
			parent, title=_("Offline AI - OCR and image reading"))
		from . import ocr
		self.ocr = ocr
		try:
			ocr.set_temp_dir(getWorkTempDir())
		except Exception:
			pass
		self.getStoreDir = storeDirProvider
		self._alive = True
		self._busy = False

		helper = gui.guiHelper.BoxSizerHelper(self, orientation=wx.VERTICAL)
		helper.addItem(wx.StaticText(self, label=_(
			"Read text from the screen or an image file using a local AI vision "
			"model. Nothing leaves your computer.")))

		self._langCodes, labels = ocrLanguageChoices()
		self.ocrLang = helper.addLabeledControl(
			_("Text &language (hint):"), wx.Choice, choices=labels)
		self.ocrLang.SetSelection(ocrLanguageIndex(self._langCodes))

		row = wx.BoxSizer(wx.HORIZONTAL)
		self.screenButton = wx.Button(self, label=_("OCR the &screen"))
		self.screenButton.Bind(wx.EVT_BUTTON, self.onScreen)
		row.Add(self.screenButton, 0, wx.ALL, 3)
		self.fileButton = wx.Button(self, label=_("OCR an image &file..."))
		self.fileButton.Bind(wx.EVT_BUTTON, self.onFile)
		row.Add(self.fileButton, 0, wx.ALL, 3)
		self.describeButton = wx.Button(self, label=_("&Describe an image..."))
		self.describeButton.Bind(wx.EVT_BUTTON, self.onDescribe)
		row.Add(self.describeButton, 0, wx.ALL, 3)
		self.clipButton = wx.Button(self, label=_("OCR the clip&board image"))
		self.clipButton.Bind(wx.EVT_BUTTON, self.onClipboard)
		row.Add(self.clipButton, 0, wx.ALL, 3)
		self.pdfButton = wx.Button(self, label=_("OCR a &PDF..."))
		self.pdfButton.Bind(wx.EVT_BUTTON, self.onPdf)
		row.Add(self.pdfButton, 0, wx.ALL, 3)
		self.stopButton = wx.Button(self, label=_("St&op"))
		self.stopButton.Bind(wx.EVT_BUTTON, self.onStop)
		self.stopButton.Enable(False)
		row.Add(self.stopButton, 0, wx.ALL, 3)
		helper.addItem(row)

		self.result = helper.addLabeledControl(
			_("&Result:"), wx.TextCtrl,
			style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2, size=(520, 200))

		self.askButton = wx.Button(self, label=_("&Ask AI about this text..."))
		self.askButton.Bind(wx.EVT_BUTTON, self.onAskAI)
		self.askButton.Enable(False)
		helper.addItem(self.askButton)

		self.dlButton = wx.Button(self, label=_("&Download OCR model (~2.7 GB)"))
		self.dlButton.Bind(wx.EVT_BUTTON, self.onDownloadModel)
		helper.addItem(self.dlButton)

		closeButton = wx.Button(self, id=wx.ID_CLOSE, label=_("&Close"))
		closeButton.Bind(wx.EVT_BUTTON, lambda e: self.Close())
		helper.addItem(closeButton)
		self.EscapeId = wx.ID_CLOSE

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(helper.sizer, border=gui.guiHelper.BORDER_FOR_DIALOGS, flag=wx.ALL)
		self.SetSizerAndFit(main)
		self._refreshState()
		if not self.ocr.engine_present():
			ui.message(_("Note: the OCR engine is not installed in this copy."))

	def _safe(self, func, *args):
		def run():
			if self._alive:
				func(*args)
		wx.CallAfter(run)

	def _refreshState(self):
		have = self.ocr.models_present(self.getStoreDir())
		engine = self.ocr.engine_present()
		for b in (self.screenButton, self.fileButton, self.describeButton,
		          self.clipButton):
			b.Enable(have and engine and not self._busy)
		self.pdfButton.Enable(have and engine)
		self.stopButton.Enable(self._busy)
		if have:
			self.dlButton.Hide()
		else:
			self.dlButton.Show()
			self.dlButton.Enable(engine)
		self.Layout()

	def _prompt(self, describe):
		"""Build the prompt on the main thread and remember the language."""
		i = max(0, self.ocrLang.GetSelection())
		code = self._langCodes[i]
		store.updateSettings(ocr_lang=code)
		if describe:
			# Describe in the hinted language, else in the user's own language.
			return self.ocr.describe_prompt_for(langs.englishName(
				code or langs.primaryTranslationLanguage(loadSettings())))
		return self.ocr.ocr_prompt_for(langs.englishName(code) if code else None)

	def _start(self, path, describe, cleanup=False):
		self._busy = True
		self._refreshState()
		threading.Thread(
			target=self._worker,
			args=(path, describe, self._prompt(describe), cleanup),
			daemon=True).start()

	def onStop(self, evt):
		try:
			self.ocr.cancel_all()
		except Exception:
			pass
		ui.message(_("Stopping."))

	def onScreen(self, evt):
		if self._busy:
			return
		# Hide this window first, otherwise the capture is mostly a picture of
		# the OCR window itself instead of what was underneath it.
		self.Hide()
		wx.CallLater(400, self._captureScreen)

	def _captureScreen(self):
		# Encoding a large screenshot takes a moment; keep NVDA responsive.
		def work():
			path, err = self.ocr.capture_screen()
			wx.CallAfter(self._afterCapture, path, err)
		threading.Thread(target=work, daemon=True).start()

	def _afterCapture(self, path, err):
		if not self._alive:
			if path:
				try:
					os.remove(path)
				except OSError:
					pass
			return
		self.Show()
		self.Raise()
		if err:
			ui.message(_("Capture failed: {e}").format(e=err))
			return
		ui.message(_("Screen captured. Reading."))
		self._start(path, False, cleanup=True)

	def onClipboard(self, evt):
		if self._busy:
			return
		path, isTemp, err = self.ocr.grab_clipboard_image()
		if err:
			ui.message(err)
			return
		ui.message(_("Reading clipboard image."))
		self._start(path, False, cleanup=isTemp)

	def onFile(self, evt):
		self._pickAndRun(False)

	def onDescribe(self, evt):
		self._pickAndRun(True)

	def onPdf(self, evt):
		dlg = wx.FileDialog(
			self, _("Choose a PDF"),
			wildcard=_("PDF files|*.pdf|All files|*.*"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		path = None
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
		dlg.Destroy()
		if not path:
			return
		from . import ocr
		count = ocr.pdf_page_count(path)
		if not count:
			ui.message(_("Could not open that PDF."))
			return
		sub = PdfOcrDialog(self, path, count, self.getStoreDir)
		sub.Bind(wx.EVT_CLOSE, lambda e: sub.onClosePdf())
		sub.Show()

	def _pickAndRun(self, describe):
		if self._busy:
			return
		dlg = wx.FileDialog(
			self, _("Choose an image"),
			wildcard=(_("Images") + "|*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;"
			          "*.tif;*.tiff|" + _("All files") + "|*.*"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		path = None
		if dlg.ShowModal() == wx.ID_OK:
			path = dlg.GetPath()
		dlg.Destroy()
		if not path:
			return
		ui.message(_("Reading image."))
		self._start(path, describe)

	def _worker(self, path, describe, prompt, cleanup):
		settings = loadSettings()
		text, err = self.ocr.run_vision(
			path, self.getStoreDir(), prompt,
			n_threads=(settings.get("n_threads", 0) or None),
			n_predict=(self.ocr.DESCRIBE_MAX_TOKENS if describe
			           else self.ocr.OCR_MAX_TOKENS),
			max_long_side=_ocrMaxLongSide(settings),
			use_mmap=settings.get("use_mmap", True))
		if cleanup:
			try:
				os.remove(path)
			except OSError:
				pass
		self._safe(self._workDone, text, err)

	def _workDone(self, text, err):
		self._busy = False
		self._refreshState()
		if err:
			ui.message(_("Failed: {e}").format(e=err))
		else:
			self._showResult(text)

	def _showResult(self, text):
		self.result.SetValue(text)
		self.result.SetInsertionPoint(0)
		self.askButton.Enable(bool(text and text.strip()))
		ui.message(text)

	def onAskAI(self, evt):
		text = self.result.GetValue().strip()
		if not text:
			ui.message(_("There is no recognized text yet."))
			return
		OfflineAIDialog.show(inject_context=text, context_label=_("OCR text"))

	def onDownloadModel(self, evt):
		from . import downloader
		store = self.getStoreDir()
		self.dlButton.Enable(False)
		backbone_dest = os.path.join(store, self.ocr.OCR_MODEL["backbone"])
		mmproj_dest = os.path.join(store, self.ocr.OCR_MODEL["mmproj"])

		def afterMmproj(ok, err):
			if not self._alive:
				return
			ui.message(_("OCR model ready.") if ok
			           else _("Download failed: {e}").format(e=err))
			self._refreshState()

		def afterBackbone(ok, err):
			if not self._alive:
				return
			if not ok:
				ui.message(_("Download failed: {e}").format(e=err))
				self._refreshState()
				return
			if os.path.isfile(mmproj_dest):
				afterMmproj(True, None)
				return
			ui.message(_("Downloading the second part."))
			downloader.download(self, self.ocr.OCR_MODEL["mmproj_url"],
			                    mmproj_dest, self.ocr.OCR_MODEL["mmproj"], afterMmproj)

		if os.path.isfile(backbone_dest):
			afterBackbone(True, None)
		else:
			ui.message(_("Downloading the OCR model."))
			downloader.download(self, self.ocr.OCR_MODEL["backbone_url"],
			                    backbone_dest, self.ocr.OCR_MODEL["backbone"],
			                    afterBackbone)

	def onClose(self):
		# Terminate any running OCR subprocess so the model does not keep
		# running in the background after the window is closed.
		self._alive = False
		try:
			self.ocr.cancel_all()
		except Exception:
			pass
		try:
			clearWorkTemp()
		except Exception:
			pass
		OCRDialog._instance = None
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


class CommandPaletteDialog(wx.Dialog):
	"""A fast, type-to-filter launcher for all Offline AI actions. Arrow to a
	command and press Enter to run it. Escape closes."""

	def __init__(self, parent, actions):
		super(CommandPaletteDialog, self).__init__(
			parent, title=_("Offline AI Quick Launcher"),
			style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
		self._actions = actions  # list of (label, callable)
		self._filtered = list(actions)

		main = wx.BoxSizer(wx.VERTICAL)
		main.Add(wx.StaticText(self, label=_("&Type to filter, then press Enter:")),
		         0, wx.ALL, 8)
		self.search = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
		self.search.Bind(wx.EVT_TEXT, self.onType)
		self.search.Bind(wx.EVT_TEXT_ENTER, self.onEnter)
		self.search.Bind(wx.EVT_KEY_DOWN, self.onKey)
		try:
			self.search.SetName(_("Filter commands"))
		except Exception:
			pass
		main.Add(self.search, 0, wx.EXPAND | wx.ALL, 8)

		self.listBox = wx.ListBox(self, size=(420, 260),
		                          choices=[a[0] for a in self._filtered])
		self.listBox.Bind(wx.EVT_LISTBOX_DCLICK, self.onEnter)
		if self._filtered:
			self.listBox.SetSelection(0)
		main.Add(self.listBox, 1, wx.EXPAND | wx.ALL, 8)

		self.SetSizerAndFit(main)
		self.search.SetFocus()
		self.EscapeId = wx.ID_CANCEL

	def onType(self, evt):
		q = self.search.GetValue().strip().lower()
		if q:
			self._filtered = [a for a in self._actions if q in a[0].lower()]
		else:
			self._filtered = list(self._actions)
		self.listBox.Set([a[0] for a in self._filtered])
		if self._filtered:
			self.listBox.SetSelection(0)
			ui.message(_("{n} commands. {first}").format(
				n=len(self._filtered), first=self._filtered[0][0]))
		else:
			ui.message(_("No matching command."))

	def onKey(self, evt):
		# Up/Down move the list selection while focus stays in the search box.
		# The focus never enters the list, so the newly selected command has to
		# be spoken explicitly - otherwise arrowing is silent.
		key = evt.GetKeyCode()
		if key in (wx.WXK_DOWN, wx.WXK_UP):
			count = self.listBox.GetCount()
			if count:
				i = self.listBox.GetSelection()
				i = (i + 1) if key == wx.WXK_DOWN else (i - 1)
				i = max(0, min(count - 1, i))
				self.listBox.SetSelection(i)
				ui.message(self.listBox.GetString(i))
			return
		evt.Skip()

	def onEnter(self, evt):
		i = self.listBox.GetSelection()
		if i == wx.NOT_FOUND or i >= len(self._filtered):
			return
		action = self._filtered[i][1]
		self.Close()
		# Run after the dialog closes so windows open cleanly.
		wx.CallAfter(action)

	@classmethod
	def show(cls, parent, actions):
		d = cls(parent, actions)
		d.Bind(wx.EVT_CLOSE, lambda e: d.Destroy())
		d.Show()


class OfflineAISettingsPanel(SettingsPanel):
	"""Universal Offline AI settings, shown in NVDA's own Settings dialog so they
	apply across every engine (chat, TTS, OCR, Whisper) and every gesture."""
	# Translators: title of the add-on's category in NVDA's Settings dialog.
	title = _("Offline AI")

	def makeSettings(self, settingsSizer):
		s = self._settings = loadSettings()
		helper = gui.guiHelper.BoxSizerHelper(self, sizer=settingsSizer)

		# --- languages ---
		nv = langs.nvdaLanguage()
		codes, labels = langs.choiceList(includeAuto=False)
		# Translators: translation target meaning "whatever language NVDA uses".
		sameAsNvda = _("Same as NVDA ({language})").format(
			language=langs.displayName(nv) if nv else "?")
		self._lang1Codes = [""] + codes
		self.lang1 = helper.addLabeledControl(
			# Translators: label of the first translation target language.
			_("My &language (for \"translate to my language\" and answers):"),
			wx.Choice, choices=[sameAsNvda] + labels)
		cur = langs.normalize(s.get("translate_lang_1") or "")
		self.lang1.SetSelection(
			self._lang1Codes.index(cur) if cur in self._lang1Codes else 0)
		self._lang2Codes = codes
		self.lang2 = helper.addLabeledControl(
			_("&Second translation language:"), wx.Choice, choices=labels)
		cur2 = langs.normalize(s.get("translate_lang_2") or "en")
		self.lang2.SetSelection(codes.index(cur2) if cur2 in codes else 0)

		# --- dictation ---
		self._dictLangCodes = ["auto", "nvda"] + codes
		self.dictLang = helper.addLabeledControl(
			_("&Dictation language:"), wx.Choice,
			choices=[_("Automatic (detect)"), sameAsNvda] + labels)
		dl = s.get("dictation_lang", "auto")
		self.dictLang.SetSelection(
			self._dictLangCodes.index(dl) if dl in self._dictLangCodes else 0)

		from . import whisper_ui
		self._whisperModels = whisper_ui.loadWhisperModels()
		storeDir = getModelsStoreDir()
		modelLabels = [_("Automatic (best downloaded model)")]
		self._dictModelFiles = [""]
		for m in self._whisperModels:
			have = os.path.isfile(os.path.join(storeDir, m["filename"]))
			modelLabels.append(m["name"] + (
				_(" [downloaded]") if have else _(" [not downloaded]")))
			self._dictModelFiles.append(m["filename"])
		self.dictModel = helper.addLabeledControl(
			_("Dictation &model:"), wx.Choice, choices=modelLabels)
		dm = s.get("dictation_model", "")
		self.dictModel.SetSelection(
			self._dictModelFiles.index(dm) if dm in self._dictModelFiles else 0)

		self._dictTargets = [
			("current", _("Wherever the focus is when the text is ready")),
			("origin", _("The window I dictated in, even if I moved away "
			             "(then return to where I was)")),
			("clipboard", _("Only copy it to the clipboard")),
		]
		self.dictTarget = helper.addLabeledControl(
			_("&Insert dictated text into:"), wx.Choice,
			choices=[lbl for _k, lbl in self._dictTargets])
		dt = s.get("dictation_target", "current")
		self.dictTarget.SetSelection(next(
			(i for i, (k, _l) in enumerate(self._dictTargets) if k == dt), 0))
		self.dictDelay = helper.addLabeledControl(
			_("&Warning before inserting (seconds, 0 = none):"), wx.SpinCtrl,
			min=0, max=10, initial=int(s.get("dictation_paste_delay", 0) or 0))
		self.dictFast = helper.addItem(wx.CheckBox(self, label=_(
			"&Faster transcription of short dictations")))
		self.dictFast.SetValue(bool(s.get("dictation_fast", True)))
		self.dictRestore = helper.addItem(wx.CheckBox(self, label=_(
			"&Restore my clipboard after inserting dictated text")))
		self.dictRestore.SetValue(bool(s.get("dictation_restore_clipboard", True)))

		# --- engines ---
		self.mmap = helper.addItem(wx.CheckBox(self, label=_(
			"Memory-map model files (mma&p) - lets a model larger than your "
			"free RAM still run; turn off to load models fully into RAM")))
		self.mmap.SetValue(bool(s.get("use_mmap", True)))
		self.threads = helper.addLabeledControl(
			_("CPU &threads (0 = automatic):"), wx.SpinCtrl,
			min=0, max=256, initial=int(s.get("n_threads", 0)))
		self.unloadAfter = helper.addLabeledControl(
			_("Free the &quick-action model after this many idle minutes "
			  "(0 = keep it loaded):"), wx.SpinCtrl,
			min=0, max=240, initial=int(s.get("unload_after_min", 10) or 0))
		self._ocrQ = [("fast", _("Fast (lower resolution)")),
		              ("balanced", _("Balanced (default)")),
		              ("best", _("Best quality (slower)"))]
		self.ocrQuality = helper.addLabeledControl(
			_("&OCR speed vs quality:"), wx.Choice,
			choices=[lbl for _m, lbl in self._ocrQ])
		cq = s.get("ocr_quality", "balanced")
		self.ocrQuality.SetSelection(
			next((i for i, (m, _l) in enumerate(self._ocrQ) if m == cq), 1))
		self._ttsSpeeds = ttsSpeedChoices()
		self.ttsSpeed = helper.addLabeledControl(
			_("AI voice sp&eed:"), wx.Choice,
			choices=[lbl for lbl, _v in self._ttsSpeeds])
		cs = s.get("tts_speed", 1.0)
		self.ttsSpeed.SetSelection(
			next((i for i, (_l, v) in enumerate(self._ttsSpeeds) if v == cs), 1))
		self._stretchModes = [("smooth", _("Smooth speech (default)")),
		                      ("crisp", _("Crisp transients")),
		                      ("legacy", _("Legacy sample-rate scaling"))]
		self.ttsStretch = helper.addLabeledControl(
			_("Speed q&uality:"), wx.Choice,
			choices=[lbl for _m, lbl in self._stretchModes])
		cm = s.get("tts_stretch", "smooth")
		self.ttsStretch.SetSelection(
			next((i for i, (m, _l) in enumerate(self._stretchModes) if m == cm), 0))
		self.cueVolume = helper.addLabeledControl(
			_("Audio cues &volume:"), wx.Slider, minValue=0, maxValue=100,
			style=wx.SL_HORIZONTAL)
		self.cueVolume.SetValue(int(s.get("cue_volume", 60)))
		self.cueMute = helper.addItem(
			wx.CheckBox(self, label=_("Mute all audio &cues")))
		self.cueMute.SetValue(bool(s.get("cue_mute", False)))
		self.formatTables = helper.addItem(wx.CheckBox(self, label=_(
			"Format ta&bles for screen readers")))
		self.formatTables.SetValue(bool(s.get("format_tables", True)))

	def onSave(self):
		store.updateSettings(
			translate_lang_1=self._lang1Codes[max(0, self.lang1.GetSelection())],
			translate_lang_2=self._lang2Codes[max(0, self.lang2.GetSelection())],
			dictation_lang=self._dictLangCodes[max(0, self.dictLang.GetSelection())],
			dictation_model=self._dictModelFiles[max(0, self.dictModel.GetSelection())],
			dictation_target=self._dictTargets[max(0, self.dictTarget.GetSelection())][0],
			dictation_paste_delay=int(self.dictDelay.GetValue()),
			dictation_fast=self.dictFast.GetValue(),
			dictation_restore_clipboard=self.dictRestore.GetValue(),
			use_mmap=self.mmap.GetValue(),
			n_threads=int(self.threads.GetValue()),
			unload_after_min=int(self.unloadAfter.GetValue()),
			ocr_quality=self._ocrQ[max(0, self.ocrQuality.GetSelection())][0],
			tts_speed=self._ttsSpeeds[max(0, self.ttsSpeed.GetSelection())][1],
			tts_stretch=self._stretchModes[max(0, self.ttsStretch.GetSelection())][0],
			cue_volume=int(self.cueVolume.GetValue()),
			cue_mute=self.cueMute.GetValue(),
			format_tables=self.formatTables.GetValue(),
		)


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	# Translators: the add-on's category in NVDA's Input Gestures dialog.
	scriptCategory = _("Offline AI")

	def __init__(self, *args, **kwargs):
		super(GlobalPlugin, self).__init__(*args, **kwargs)
		self._menuItems = []
		self._quickStop = threading.Event()
		self._selReader = None
		self._dictModel = None
		self._dictOptions = None
		try:
			gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(
				OfflineAISettingsPanel)
		except Exception as e:
			log.error("offlineAI: could not register settings panel: %s" % e)
		try:
			self.toolsMenu = gui.mainFrame.sysTrayIcon.toolsMenu
			for label, helpText, handler in (
				(_("Offline AI..."), _("Open the Offline AI window"),
				 lambda e: OfflineAIDialog.show()),
				(_("Offline AI - Transcribe speech..."),
				 _("Open the speech-to-text window"),
				 lambda e: self._openWhisper()),
				(_("Offline AI - Text to speech..."),
				 _("Open the text-to-speech and voice cloning window"),
				 lambda e: self._openTTS()),
				(_("Offline AI - OCR and image reading..."),
				 _("Open the OCR and image description window"),
				 lambda e: self._openOCR()),
			):
				item = self.toolsMenu.Append(wx.ID_ANY, label, helpText)
				gui.mainFrame.sysTrayIcon.Bind(wx.EVT_MENU, handler, item)
				self._menuItems.append(item)
		except Exception as e:
			log.error("offlineAI: could not add tools menu item: %s" % e)
		# Leftovers of a previous session that ended abruptly.
		threading.Thread(target=self._startupCleanup, daemon=True).start()

	def _startupCleanup(self):
		try:
			clearWorkTemp()
		except Exception:
			pass

	def _openWindow(self, opener, what):
		try:
			opener()
		except Exception as e:
			log.error("offlineAI: cannot open %s: %s" % (what, e), exc_info=True)
			gui.messageBox(
				_("The window could not be opened: {e}").format(e=str(e)),
				_("Offline AI"), wx.OK | wx.ICON_ERROR)

	def _openOCR(self):
		self._openWindow(lambda: OCRDialog.show(getModelsStoreDir), "OCR")

	def _openTTS(self):
		self._openWindow(lambda: TTSDialog.show(getModelsStoreDir), "TTS")

	def _openWhisper(self):
		def opener():
			from . import whisper_ui
			whisper_ui.WhisperDialog.show(getModelsStoreDir)
		self._openWindow(opener, "whisper")

	@script(description=_("Opens the Offline AI window"), gesture=None)
	def script_openOfflineAI(self, gesture):
		self._openWindow(OfflineAIDialog.show, "chat")

	@script(description=_("Opens the Offline AI speech-to-text window"), gesture=None)
	def script_openOfflineAITranscribe(self, gesture):
		self._openWhisper()

	@script(description=_("Opens the Offline AI text-to-speech window"), gesture=None)
	def script_openOfflineAITTS(self, gesture):
		self._openTTS()

	@script(description=_("Opens the Offline AI OCR and image reading window"),
	        gesture=None)
	def script_openOfflineAIOCR(self, gesture):
		self._openOCR()

	# --- stop everything ---

	@script(description=_("Offline AI: stop the current task (quick action, "
	                      "speech or OCR)"), gesture=None)
	def script_stopAll(self, gesture):
		self._quickStop.set()
		try:
			if self._selReader:
				self._selReader.stop()
			from . import tts, ocr
			tts.cancel_all()
			ocr.cancel_all()
		except Exception as e:
			log.warning("offlineAI: stop failed: %s" % e)
		ui.message(_("Stopped."))

	# --- speak selection with the AI voice ---

	@script(description=_("Offline AI: speak selected text aloud with the voice "
	                      "model (press again to stop)"), gesture=None)
	def script_speakSelection(self, gesture):
		# A second press while it is speaking stops it.
		if self._selReader and self._selReader.is_running():
			self._selReader.stop()
			ui.message(_("Stopped."))
			return
		self._speakText(self._getSelectedText())

	def _speakText(self, text):
		from . import tts
		if not text:
			ui.message(_("No text is selected."))
			return
		storeDir = getModelsStoreDir()
		if not tts.engine_present():
			ui.message(_("The text-to-speech engine is not installed."))
			return
		if not tts.models_present(storeDir):
			ui.message(_("Download the voice model first from the Text to speech window."))
			return
		if self._selReader and self._selReader.is_running():
			self._selReader.stop()
		try:
			tts.set_temp_dir(getWorkTempDir())
		except Exception:
			pass
		settings = loadSettings()
		lang = tts.default_language(settings)
		# Sentence-by-sentence streaming, so long selections start speaking
		# quickly instead of after the whole text has been generated.
		self._selReader = tts.DocumentReader(
			storeDir, lang=lang,
			n_threads=(settings.get("n_threads", 0) or None),
			prerender=bool(settings.get("tts_prerender", True)),
			speed=settings.get("tts_speed", 1.0),
			stretch_mode=settings.get("tts_stretch", "smooth"),
			use_mmap=settings.get("use_mmap", True),
			on_status=None)
		ui.message(_("Speaking selection in {language}.").format(
			language=langs.displayName(lang)))
		self._selReader.start(text)

	# --- OCR / vision ---

	def _ocrReady(self):
		from . import ocr
		if not ocr.engine_present():
			ui.message(_("The OCR engine is not installed."))
			return False
		if not ocr.models_present(getModelsStoreDir()):
			ui.message(_("Download the OCR model first from the OCR window."))
			return False
		try:
			ocr.set_temp_dir(getWorkTempDir())
		except Exception:
			pass
		return True

	def _ocrPrompt(self, describe):
		from . import ocr
		settings = loadSettings()
		code = langs.normalize(settings.get("ocr_lang") or "")
		if describe:
			return ocr.describe_prompt_for(langs.englishName(
				code or langs.primaryTranslationLanguage(settings)))
		return ocr.ocr_prompt_for(langs.englishName(code) if code else None)

	def _visionWorker(self, path, describe, cleanup, title):
		from . import ocr
		settings = loadSettings()
		text, err = ocr.run_vision(
			path, getModelsStoreDir(), self._ocrPrompt(describe),
			n_threads=(settings.get("n_threads", 0) or None),
			n_predict=(ocr.DESCRIBE_MAX_TOKENS if describe else ocr.OCR_MAX_TOKENS),
			max_long_side=_ocrMaxLongSide(settings),
			use_mmap=settings.get("use_mmap", True))
		if cleanup:
			try:
				os.remove(path)
			except OSError:
				pass
		if err:
			wx.CallAfter(ui.message, _("Failed: {e}").format(e=err))
		else:
			wx.CallAfter(browseable, text, _("Offline AI - {t}").format(t=title))

	@script(description=_("Offline AI: OCR the whole screen and read the text"),
	        gesture=None)
	def script_ocrScreen(self, gesture):
		if not self._ocrReady():
			return
		ui.message(_("Capturing screen for OCR."))

		def work():
			from . import ocr
			png, err = ocr.capture_screen()
			if err:
				wx.CallAfter(ui.message, _("Capture failed: {e}").format(e=err))
				return
			self._visionWorker(png, False, True, _("Screen OCR"))
		threading.Thread(target=work, daemon=True).start()

	def _pickImage(self, title):
		dlg = wx.FileDialog(
			gui.mainFrame, title,
			wildcard=(_("Images") + "|*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;"
			          "*.tif;*.tiff|" + _("All files") + "|*.*"),
			style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		gui.mainFrame.prePopup()
		path = dlg.GetPath() if dlg.ShowModal() == wx.ID_OK else None
		gui.mainFrame.postPopup()
		dlg.Destroy()
		return path

	@script(description=_("Offline AI: OCR an image file"), gesture=None)
	def script_ocrFile(self, gesture):
		if not self._ocrReady():
			return
		path = self._pickImage(_("Choose an image to read"))
		if not path:
			return
		ui.message(_("Reading image."))
		threading.Thread(target=self._visionWorker,
		                 args=(path, False, False, _("Image OCR")),
		                 daemon=True).start()

	@script(description=_("Offline AI: describe an image file"), gesture=None)
	def script_describeFile(self, gesture):
		if not self._ocrReady():
			return
		path = self._pickImage(_("Choose an image to describe"))
		if not path:
			return
		ui.message(_("Describing image."))
		threading.Thread(target=self._visionWorker,
		                 args=(path, True, False, _("Image description")),
		                 daemon=True).start()

	def _clipboardImage(self, describe):
		if not self._ocrReady():
			return
		from . import ocr
		path, isTemp, err = ocr.grab_clipboard_image()
		if err:
			ui.message(err)
			return
		ui.message(_("Describing clipboard image.") if describe
		           else _("Reading clipboard image."))
		threading.Thread(
			target=self._visionWorker,
			args=(path, describe, isTemp,
			      _("Clipboard image description") if describe
			      else _("Clipboard OCR")),
			daemon=True).start()

	@script(description=_("Offline AI: OCR an image on the clipboard"),
	        gesture=None)
	def script_ocrClipboard(self, gesture):
		self._clipboardImage(False)

	@script(description=_("Offline AI: describe an image on the clipboard"),
	        gesture=None)
	def script_describeClipboard(self, gesture):
		self._clipboardImage(True)

	@script(description=_("Opens the Offline AI Quick Launcher (command palette)"),
	        gesture=None)
	def script_quickLauncher(self, gesture):
		# The selection must be read now: once the launcher opens, the focus
		# (and with it the selection) belongs to the launcher itself.
		selected = self._getSelectedText()
		settings = loadSettings()
		l1 = langs.primaryTranslationLanguage(settings)
		l2 = langs.secondaryTranslationLanguage(settings)

		def onSel(func):
			def run():
				if not selected:
					ui.message(_("No text was selected when the launcher opened."))
					return
				func(selected)
			return run

		actions = [
			(_("Chat with Offline AI"), lambda: OfflineAIDialog.show()),
			(_("Text to speech"), self._openTTS),
			(_("OCR and image reading"), self._openOCR),
			(_("Transcribe speech (Whisper)"), self._openWhisper),
			(_("Start or stop dictation"),
			 lambda: self._runScriptSafe(self.script_toggleDictation)),
			(_("OCR the screen"),
			 lambda: self._runScriptSafe(self.script_ocrScreen)),
			(_("OCR an image file"),
			 lambda: self._runScriptSafe(self.script_ocrFile)),
			(_("Describe an image file"),
			 lambda: self._runScriptSafe(self.script_describeFile)),
			(_("OCR clipboard image"),
			 lambda: self._runScriptSafe(self.script_ocrClipboard)),
			(_("Describe clipboard image"),
			 lambda: self._runScriptSafe(self.script_describeClipboard)),
			(_("Ask AI about clipboard text"),
			 lambda: self._runScriptSafe(self.script_askClipboardText)),
			(_("Translate selection to {language}").format(
				language=langs.displayName(l1)),
			 onSel(lambda t: self._translate(l1, t))),
			(_("Translate selection to {language}").format(
				language=langs.displayName(l2)),
			 onSel(lambda t: self._translate(l2, t))),
			(_("Translate selection to another language..."),
			 onSel(lambda t: self._translateChoose(t))),
			(_("Summarize selection"), onSel(lambda t: self._summarize(t))),
			(_("Proofread selection"), onSel(lambda t: self._proofread(t))),
			(_("Explain selected code or error"), onSel(lambda t: self._explain(t))),
			(_("Speak selection with the AI voice"), onSel(self._speakText)),
			(_("Stop the current task"),
			 lambda: self._runScriptSafe(self.script_stopAll)),
		]
		gui.mainFrame.prePopup()
		CommandPaletteDialog.show(gui.mainFrame, actions)
		gui.mainFrame.postPopup()

	def _runScriptSafe(self, scriptFunc):
		try:
			scriptFunc(None)
		except Exception as e:
			log.error("offlineAI: launcher action failed: %s" % e, exc_info=True)

	@script(description=_("Offline AI: ask AI about the text on the clipboard"),
	        gesture=None)
	def script_askClipboardText(self, gesture):
		text = None
		try:
			import api
			text = api.getClipData()
		except Exception:
			text = None
		if not text or not text.strip():
			ui.message(_("There is no text on the clipboard."))
			return
		OfflineAIDialog.show(inject_context=text.strip(),
		                     context_label=_("clipboard text"))

	# --- quick actions on selected text ---

	def _getSelectedText(self):
		"""Return the current selection text, or None. Works in documents and
		browse mode (web pages, PDFs) as well as in plain edit fields."""
		try:
			import api
			import textInfos
			obj = api.getFocusObject()
			if obj is None:
				return None
			candidates = []
			ti = getattr(obj, "treeInterceptor", None)
			if ti is not None and getattr(ti, "isReady", False) and \
					not getattr(ti, "passThrough", False):
				candidates.append(ti)
			candidates.append(obj)
			for target in candidates:
				try:
					info = target.makeTextInfo(textInfos.POSITION_SELECTION)
				except (RuntimeError, NotImplementedError, LookupError):
					continue
				if info and not info.isCollapsed:
					text = info.text
					if text and text.strip():
						return text.strip()
			return None
		except Exception as e:
			log.error("offlineAI: cannot get selection: %s" % e)
			return None

	def _pickModel(self):
		"""Choose a downloaded model for quick actions: the last-used one if
		set, else the first downloaded model in the catalog."""
		storeDir = getModelsStoreDir()
		last = loadLocalConfig().get("last_model_filename")
		first = None
		for m in loadModelDirectory(fetch_remote=False):
			path = modelPath(m, storeDir)
			if os.path.isfile(path):
				if last and m["filename"] == last:
					return m, path
				if first is None:
					first = (m, path)
		return first or (None, None)

	def _runQuickAction(self, instruction, actionName, text=None, chunked=False):
		"""Run `instruction` over the selected text (or `text`). chunked=True is
		for jobs whose output is as long as their input (translation,
		proofreading): long texts are processed piece by piece instead of being
		cut off at the answer limit."""
		if text is None:
			text = self._getSelectedText()
		if not text:
			ui.message(_("No text is selected."))
			return
		m, path = self._pickModel()
		if not m:
			ui.message(_("No downloaded model found. Open Offline AI and download one first."))
			return
		self._quickStop.clear()
		ui.message(_("{a}. Working.").format(a=actionName))
		threading.Thread(target=self._quickWorker,
		                 args=(instruction, text, m, path, actionName, chunked),
		                 daemon=True).start()

	def _quickWorker(self, instruction, text, m, path, actionName, chunked):
		if not inference.try_acquire_busy():
			wx.CallAfter(ui.message, _("Offline AI is busy. Please wait."))
			return
		try:
			settings = loadSettings()
			ok, err = inference.load_model(path, settings, context=m.get("context"))
			if not ok:
				wx.CallAfter(ui.message, _("Could not load the model: {e}").format(e=err))
				return
			fmt = m.get("prompt_format", "chatml")
			think = settings.get("think_mode", "hide")
			note = None
			if chunked:
				chunks, per = inference.split_for_context(text, instruction, settings)
				limit = per * 2
			else:
				# One answer about the whole text: make the text fit, keeping
				# room for the answer.
				ctx = inference.n_ctx() or int(settings.get("n_ctx", 2048))
				reserve = min(int(settings.get("max_tokens", 512)), max(64, ctx // 3))
				room = ctx - reserve - inference.count_tokens(instruction) - 64
				fitted, cut = inference.truncate_to_tokens(text, room)
				if cut:
					note = _("Note: the text is longer than the model's context "
					         "window, so only the first part was used.")
				chunks, limit = [fitted], None
			results = []
			total = len(chunks)
			for i, chunk in enumerate(chunks, 1):
				if self._quickStop.is_set():
					break
				if total > 1:
					wx.CallAfter(ui.message, _("Part {i} of {n}.").format(i=i, n=total))
				result, err = inference.generate(
					fmt, chunk, instruction, settings, think_mode=think,
					max_tokens=limit, should_stop=self._quickStop.is_set)
				if err == "context":
					wx.CallAfter(ui.message, _(
						"The text is too long for this model's context window. "
						"Raise the context size in Settings or select less text."))
					return
				if err:
					wx.CallAfter(ui.message, _("Error: {e}").format(e=err))
					return
				if result:
					results.append(result)
			if self._quickStop.is_set() and not results:
				return
			final = "\n\n".join(results).strip()
			if not final:
				wx.CallAfter(ui.message, _("The model returned nothing."))
				return
			wx.CallAfter(self._presentQuickResult, final, actionName, note)
		except Exception as e:
			log.error("offlineAI: quick action failed: %s" % e, exc_info=True)
			wx.CallAfter(ui.message, _("Error: {e}").format(e=str(e)))
		finally:
			inference.release_busy()

	def _presentQuickResult(self, result, actionName, note=None):
		settings = loadSettings()
		if settings.get("format_tables", True):
			from . import tables
			result = tables.format_tables(result)
		mode = settings.get("quick_output", "speak")
		copied = False
		if mode in ("copy", "both"):
			try:
				import api
				copied = bool(api.copyToClip(result))
			except Exception:
				copied = False
		if mode == "copy":
			msg = (_("{a} copied to clipboard.") if copied
			       else _("{a} finished, but could not be copied.")).format(a=actionName)
			ui.message((msg + " " + note) if note else msg)
			return
		if note:
			ui.message(note)
		# The result opens in a window NVDA reads automatically and that can be
		# reviewed with the reading keys, copied, and closed with Escape.
		browseable(result, _("Offline AI - {a}").format(a=actionName))

	# -- the individual quick actions (instructions are English prompts for the
	# model; the language the answer should be in is named explicitly) --

	def _myLanguageName(self):
		return langs.englishName(langs.primaryTranslationLanguage(loadSettings()))

	def _translate(self, code, text=None):
		self._runQuickAction(
			langs.translationInstruction(code),
			_("Translate to {language}").format(language=langs.displayName(code)),
			text=text, chunked=True)

	def _translateChoose(self, text=None):
		if text is None:
			text = self._getSelectedText()
		if not text:
			ui.message(_("No text is selected."))
			return
		codes, labels = langs.choiceList(includeAuto=False)
		dlg = wx.SingleChoiceDialog(
			gui.mainFrame, _("Translate the selected text into:"),
			_("Offline AI - Translate"), labels)
		last = langs.normalize(loadSettings().get("translate_last") or "")
		if last in codes:
			dlg.SetSelection(codes.index(last))
		gui.mainFrame.prePopup()
		ok = dlg.ShowModal() == wx.ID_OK
		gui.mainFrame.postPopup()
		code = codes[dlg.GetSelection()] if ok else None
		dlg.Destroy()
		if not code:
			return
		store.updateSettings(translate_last=code)
		self._translate(code, text)

	def _summarize(self, text=None):
		self._runQuickAction(
			"Summarize the following text concisely, keeping the key points. "
			"Write the summary in %s." % self._myLanguageName(),
			_("Summarize"), text=text)

	def _proofread(self, text=None):
		self._runQuickAction(
			"Proofread and correct the grammar, spelling and punctuation of the "
			"following text. Keep it in its original language and keep its "
			"meaning. Output only the corrected text.",
			_("Proofread"), text=text, chunked=True)

	def _explain(self, text=None):
		self._runQuickAction(
			"Explain the following code or error message clearly and concisely. "
			"Write the explanation in %s." % self._myLanguageName(),
			_("Explain"), text=text)

	@script(description=_("Offline AI: translate selected text to my language "
	                      "(NVDA's language unless changed in settings)"),
	        gesture=None)
	def script_translatePrimary(self, gesture):
		self._translate(langs.primaryTranslationLanguage(loadSettings()))

	@script(description=_("Offline AI: translate selected text to my second "
	                      "language (English unless changed in settings)"),
	        gesture=None)
	def script_translateSecondary(self, gesture):
		self._translate(langs.secondaryTranslationLanguage(loadSettings()))

	@script(description=_("Offline AI: translate selected text to a language "
	                      "chosen from a list"), gesture=None)
	def script_translateChoose(self, gesture):
		# The dialog must not open while the gesture's keys are still down.
		text = self._getSelectedText()
		wx.CallAfter(self._translateChoose, text)

	# Kept so key assignments made in versions before 1.0 keep working. They
	# have no description, so they are not listed in Input Gestures any more;
	# the two generic commands above replace them for every language.
	def script_translateArabic(self, gesture):
		self._translate("ar")

	def script_translateEnglish(self, gesture):
		self._translate("en")

	@script(description=_("Offline AI: summarize selected text"), gesture=None)
	def script_summarize(self, gesture):
		self._summarize()

	@script(description=_("Offline AI: proofread and fix grammar of selected text"), gesture=None)
	def script_proofread(self, gesture):
		self._proofread()

	@script(description=_("Offline AI: explain selected code or error"), gesture=None)
	def script_explainCode(self, gesture):
		self._explain()

	# --- custom user-defined quick actions (fixed slots) ---

	def _runCustomAction(self, slot):
		actions = getCustomActions()
		if slot >= len(actions):
			ui.message(_("Custom action {n} is not defined. Add it in Settings.").format(
				n=slot + 1))
			return
		a = actions[slot]
		self._runQuickAction(a.get("instruction", ""), a.get("name", _("Custom action")))

	# --- live dictation ---

	def _dictationSetup(self):
		"""Return (model_path, options) for dictation, or (None, None)."""
		from . import dictation, whisper_ui
		settings = loadSettings()
		path = dictation.pickModel(
			getModelsStoreDir(), whisper_ui.loadWhisperModels(),
			settings.get("dictation_model", ""))
		if not path:
			return None, None
		lang = settings.get("dictation_lang", "auto") or "auto"
		if lang == "nvda":
			nv = langs.nvdaLanguage()
			lang = nv if nv in langs.LANGUAGES else "auto"
		options = {
			"language": lang,
			"n_threads": store.autoThreads(settings),
			"fast": bool(settings.get("dictation_fast", True)),
			"target": settings.get("dictation_target", "current"),
			"paste_delay": settings.get("dictation_paste_delay", 0),
			"restore_clipboard": bool(settings.get("dictation_restore_clipboard", True)),
		}
		return path, options

	def _dictationController(self):
		from . import dictation
		ctrl = dictation.DictationController.get()
		s = loadSettings()
		ctrl.set_cue_volume(s.get("cue_volume", 60), s.get("cue_mute", False))
		return ctrl

	@script(description=_("Offline AI: toggle live dictation (speak, then insert text)"),
	        gesture=None)
	def script_toggleDictation(self, gesture):
		ctrl = self._dictationController()
		if ctrl.recording:
			# Second press: stop and transcribe with what the recording started with.
			ctrl.toggle(self._dictModel, ui.message, self._dictOptions)
			return
		path, options = self._dictationSetup()
		if not path:
			ui.message(_("No Whisper model found. Open the Transcribe speech "
			             "window and download one first."))
			return
		self._dictModel = path
		self._dictOptions = options
		ctrl.toggle(path, ui.message, options)

	@script(description=_("Offline AI: cancel live dictation without inserting text"),
	        gesture=None)
	def script_abortDictation(self, gesture):
		ctrl = self._dictationController()
		if ctrl.recording:
			ctrl.abort(ui.message)
		else:
			ui.message(_("Dictation is not recording."))

	def terminate(self, *args, **kwargs):
		# Stop everything that could outlive the add-on: recording, child
		# processes, playback, and the loaded model.
		self._quickStop.set()
		try:
			from . import dictation, tts, ocr
			dictation.DictationController.get().shutdown()
			if self._selReader:
				self._selReader.stop()
			tts.cancel_all()
			ocr.cancel_all()
		except Exception as e:
			log.warning("offlineAI: shutdown cleanup failed: %s" % e)
		for cls in (OfflineAIDialog, TTSDialog, OCRDialog):
			try:
				if cls._instance is not None:
					cls._instance.Close()
			except Exception:
				pass
		try:
			from . import whisper_ui
			if whisper_ui.WhisperDialog._instance is not None:
				whisper_ui.WhisperDialog._instance.Close()
		except Exception:
			pass
		try:
			inference.unload(force=True)
		except Exception:
			pass
		try:
			for item in self._menuItems:
				self.toolsMenu.Remove(item)
		except Exception:
			pass
		try:
			gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(
				OfflineAISettingsPanel)
		except Exception:
			pass
		try:
			clearWorkTemp()
		except Exception:
			pass
		super(GlobalPlugin, self).terminate(*args, **kwargs)


# Generate a fixed pool of custom-action slot scripts so they appear in NVDA's
# Input Gestures dialog under the "Offline AI" category. Each reads its name and
# instruction from config at run time.
def _make_custom_slot(slot):
	def _script(self, gesture):
		self._runCustomAction(slot)
	# Translators: input gesture description; {n} is the slot number (1-10).
	_script.__doc__ = _("Offline AI: run custom quick action {n}").format(n=slot + 1)
	_script.category = _("Offline AI")
	return _script


for _i in range(10):
	setattr(GlobalPlugin, "script_customAction%d" % (_i + 1), _make_custom_slot(_i))

# Local AI has no business on the Windows sign-in and UAC screens: there the
# add-on stays completely inert.
if globalVars.appArgs.secure:
	GlobalPlugin = globalPluginHandler.GlobalPlugin  # noqa: F811
