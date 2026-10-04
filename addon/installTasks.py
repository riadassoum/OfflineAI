# -*- coding: utf-8 -*-
# Offline AI - install tasks.
#
# Versions before 1.0 kept the user's settings inside the add-on folder, which
# NVDA replaces on every update. When 1.0 (or later) is installed over such a
# version, copy that file into NVDA's configuration folder so nothing is lost.

import os
import json

import globalVars
from logHandler import log

_SAMPLE_ACTION = {"name": "Extract TODOs", "instruction": "List all action items."}
_SAMPLE_PRESET = {"name": "My Style", "system": "Be terse.", "temperature": 0.2}


def onInstall():
	try:
		here = os.path.dirname(os.path.abspath(__file__))        # ...\addons\offlineAI.pendingInstall
		addons = os.path.dirname(here)
		old = os.path.join(addons, "offlineAI", "globalPlugins", "offlineAI",
		                   "offlineAI_config.json")
		destDir = os.path.join(globalVars.appArgs.configPath, "offlineAI")
		dest = os.path.join(destDir, "config.json")
		if os.path.isfile(dest) or not os.path.isfile(old):
			return
		with open(old, "r", encoding="utf-8") as f:
			cfg = json.load(f)
		if not isinstance(cfg, dict):
			return
		# Drop the developer's sample data that 7.x builds shipped with.
		if isinstance(cfg.get("custom_actions"), list):
			cfg["custom_actions"] = [a for a in cfg["custom_actions"] if a != _SAMPLE_ACTION]
		if isinstance(cfg.get("user_presets"), list):
			cfg["user_presets"] = [p for p in cfg["user_presets"] if p != _SAMPLE_PRESET]
		s = cfg.get("settings")
		if isinstance(s, dict) and s.get("n_threads") == 6 and \
				s.get("ocr_quality") == "fast" and s.get("format_tables") is False \
				and s.get("max_tokens") == 512:
			s["n_threads"] = 0   # the developer's CPU, not the user's choice
		if not os.path.isdir(destDir):
			os.makedirs(destDir)
		with open(dest, "w", encoding="utf-8") as f:
			json.dump(cfg, f, ensure_ascii=False, indent=2)
		log.info("offlineAI: settings from the previous version were kept")
	except Exception as e:
		log.warning("offlineAI: could not carry settings over: %s" % e)
