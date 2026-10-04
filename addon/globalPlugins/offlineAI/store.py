# -*- coding: utf-8 -*-
# Persistent configuration for the Offline AI add-on.
#
# Everything the user changes (settings, custom models, presets, custom quick
# actions, voice profiles) lives in ONE json file inside NVDA's user
# configuration folder:
#
#     <NVDA config>/offlineAI/config.json
#
# It used to live inside the add-on folder itself, which NVDA deletes on every
# update, so each new version silently wiped the user's settings. The old file
# is migrated automatically the first time this module is used.

import os
import json
import threading

import globalVars
from logHandler import log

ADDON_DIR = os.path.dirname(__file__)
LEGACY_CONFIG_JSON = os.path.join(ADDON_DIR, "offlineAI_config.json")

_lock = threading.RLock()

DEFAULT_SETTINGS = {
	"temperature": 0.7,
	"top_p": 0.95,
	"top_k": 40,
	"min_p": 0.05,
	"repeat_penalty": 1.1,
	"max_tokens": 512,
	"n_ctx": 2048,
	"n_threads": 0,          # 0 = auto
	"n_batch": 512,
	"flash_attn": True,
	"use_mlock": False,
	"use_mmap": True,        # lets Windows page a model that is bigger than free RAM
	"seed": -1,              # -1 = random
	"system_prompt": "You are a helpful assistant.",
	"stream": True,
	"think_mode": "hide",    # reasoning <think> blocks: show | hide | cue
	"quick_output": "speak", # quick actions: speak | copy | both
	"heartbeat": True,
	"hf_result_limit": 60,
	"tts_speed": 1.0,
	"tts_prerender": True,
	"cue_volume": 60,
	"cue_mute": False,
	"format_tables": True,
	"tts_stretch": "smooth", # smooth | crisp | legacy
	"ocr_quality": "balanced", # fast | balanced | best
	# --- language-related (all "" / "auto" values follow NVDA's language) ---
	"translate_lang_1": "",  # "" = NVDA's interface language
	"translate_lang_2": "en",
	"tts_lang": "",          # "" = NVDA's language when the voice supports it
	"whisper_lang": "auto",  # last language used in the Transcribe window
	"ocr_lang": "",          # "" = auto / mixed
	# --- dictation ---
	"dictation_lang": "auto",        # auto | nvda | <whisper code>
	"dictation_model": "",           # "" = best downloaded model for dictation
	"dictation_target": "current",   # current | origin | clipboard
	"dictation_paste_delay": 0,      # seconds of warning before pasting
	"dictation_fast": True,          # shorten the audio window for short notes
	"dictation_restore_clipboard": True,
	# --- memory ---
	"unload_after_min": 10,  # free the quick-action model after N idle minutes (0 = never)
}

# The development configuration that was shipped by mistake inside 7.x builds.
_SHIPPED_SAMPLE_ACTION = {"name": "Extract TODOs",
                          "instruction": "List all action items."}
_SHIPPED_SAMPLE_PRESET = {"name": "My Style", "system": "Be terse.",
                          "temperature": 0.2}


def configDir():
	d = os.path.join(globalVars.appArgs.configPath, "offlineAI")
	if not os.path.isdir(d):
		try:
			os.makedirs(d)
		except OSError as e:
			log.error("offlineAI: cannot create config folder: %s" % e)
	return d


def configPath():
	return os.path.join(configDir(), "config.json")


def cleanLegacyConfig(cfg):
	"""Remove the developer's sample data that 7.x builds shipped with."""
	if not isinstance(cfg, dict):
		return {}
	acts = cfg.get("custom_actions")
	if isinstance(acts, list):
		cfg["custom_actions"] = [a for a in acts if a != _SHIPPED_SAMPLE_ACTION]
	pres = cfg.get("user_presets")
	if isinstance(pres, list):
		cfg["user_presets"] = [p for p in pres if p != _SHIPPED_SAMPLE_PRESET]
	s = cfg.get("settings")
	if isinstance(s, dict):
		# 7.x shipped n_threads=6 (the developer's CPU) with a 512-token cap and
		# "fast" OCR; if that exact trio is present the user never chose it.
		if (s.get("n_threads") == 6 and s.get("ocr_quality") == "fast"
				and s.get("format_tables") is False and s.get("max_tokens") == 512):
			s["n_threads"] = 0
	return cfg


def _migrateLegacy():
	"""One-time copy of the old in-add-on config into NVDA's config folder."""
	new = configPath()
	if os.path.isfile(new) or not os.path.isfile(LEGACY_CONFIG_JSON):
		return
	try:
		with open(LEGACY_CONFIG_JSON, "r", encoding="utf-8") as f:
			cfg = cleanLegacyConfig(json.load(f))
		_write(new, cfg)
		log.info("offlineAI: migrated settings to %s" % new)
	except Exception as e:
		log.warning("offlineAI: could not migrate old settings: %s" % e)


def _write(path, cfg):
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(cfg, f, ensure_ascii=False, indent=2)
	os.replace(tmp, path)


def loadLocalConfig():
	with _lock:
		_migrateLegacy()
		try:
			with open(configPath(), "r", encoding="utf-8") as f:
				data = json.load(f)
			return data if isinstance(data, dict) else {}
		except FileNotFoundError:
			return {}
		except Exception as e:
			log.warning("offlineAI: config unreadable, using defaults: %s" % e)
			return {}


def saveLocalConfig(cfg):
	with _lock:
		try:
			_write(configPath(), cfg)
		except Exception as e:
			log.error("offlineAI: cannot save config: %s" % e)


def loadSettings():
	cfg = loadLocalConfig()
	s = dict(DEFAULT_SETTINGS)
	saved = cfg.get("settings", {})
	if isinstance(saved, dict):
		s.update(saved)
	return s


def saveSettings(s):
	with _lock:
		cfg = loadLocalConfig()
		cfg["settings"] = s
		saveLocalConfig(cfg)


def updateSettings(**changes):
	"""Change a few settings without clobbering the rest."""
	with _lock:
		s = loadSettings()
		s.update(changes)
		saveSettings(s)
		return s


def getValue(key, default=None):
	return loadLocalConfig().get(key, default)


def setValue(key, value):
	with _lock:
		cfg = loadLocalConfig()
		cfg[key] = value
		saveLocalConfig(cfg)


# --- model store (outside the add-on, survives updates) ---------------------

def defaultModelsStoreDir():
	return os.path.join(os.path.expanduser("~"), "Documents", "offlineAI_models")


def getModelsStoreDir():
	d = loadLocalConfig().get("models_dir") or defaultModelsStoreDir()
	if not os.path.isdir(d):
		try:
			os.makedirs(d)
		except OSError:
			d = os.path.join(globalVars.appArgs.configPath, "offlineAI_models")
			if not os.path.isdir(d):
				try:
					os.makedirs(d)
				except OSError as e:
					log.error("offlineAI: cannot create models dir: %s" % e)
	return d


def setModelsStoreDir(d):
	setValue("models_dir", d)


TEMP_PREFIXES = ("offlineAI_ocr_", "offlineAI_clip_", "offlineAI_tts_",
                 "offlineAI_rs_")


def getWorkTempDir():
	"""Folder for short-lived files (OCR page images, speech WAVs). It sits in
	the models folder so nothing unexpected lands on the system drive."""
	d = os.path.join(getModelsStoreDir(), "temp")
	try:
		if not os.path.isdir(d):
			os.makedirs(d)
	except OSError:
		import tempfile
		return tempfile.gettempdir()
	return d


def clearWorkTemp():
	d = os.path.join(getModelsStoreDir(), "temp")
	if not os.path.isdir(d):
		return
	for name in os.listdir(d):
		if name.startswith(TEMP_PREFIXES):
			try:
				os.remove(os.path.join(d, name))
			except OSError:
				pass


def autoThreads(settings=None):
	"""Number of CPU threads to use: the user's choice, or a sensible automatic
	value (physical-ish cores; hyper-threads rarely help llama/whisper)."""
	s = settings if settings is not None else loadSettings()
	try:
		n = int(s.get("n_threads", 0) or 0)
	except (TypeError, ValueError):
		n = 0
	if n > 0:
		return n
	c = os.cpu_count() or 4
	return max(1, c // 2) if c > 4 else max(1, c)
