# -*- coding: utf-8 -*-
# Backup and restore of all user data: settings, custom quick actions, persona
# presets, cloned voice profiles, and chat history. Everything is packaged into a
# single timestamped .zip and can be restored on another machine.

import os
import json
import time
import zipfile

import addonHandler
from logHandler import log

addonHandler.initTranslation()

MANIFEST_NAME = "offlineAI_backup.json"
BACKUP_VERSION = 1


def create_backup(dest_zip, config_json_path, history_path, voice_profiles_dir):
	"""Write a backup zip containing config, chat history, and voice samples.
	Returns (ok, err)."""
	try:
		with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as z:
			manifest = {
				"version": BACKUP_VERSION,
				"created": time.strftime("%Y-%m-%d %H:%M:%S"),
				"contents": [],
			}
			# Config JSON (settings, presets, custom actions, voice profile list).
			if os.path.isfile(config_json_path):
				z.write(config_json_path, "config/offlineAI_config.json")
				manifest["contents"].append("config")
			# Chat history JSON.
			if history_path and os.path.isfile(history_path):
				z.write(history_path, "history/chat_history.json")
				manifest["contents"].append("history")
			# Voice profile sample files.
			if voice_profiles_dir and os.path.isdir(voice_profiles_dir):
				for name in os.listdir(voice_profiles_dir):
					full = os.path.join(voice_profiles_dir, name)
					if os.path.isfile(full):
						z.write(full, "voice_profiles/" + name)
				manifest["contents"].append("voice_profiles")
			z.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
		return True, None
	except Exception as e:
		log.error("offlineAI backup: create failed: %s" % e)
		return False, str(e)


def inspect_backup(src_zip):
	"""Return (manifest_dict, err) without extracting anything."""
	try:
		with zipfile.ZipFile(src_zip, "r") as z:
			names = z.namelist()
			if MANIFEST_NAME not in names:
				return None, _("This does not look like an Offline AI backup.")
			manifest = json.loads(z.read(MANIFEST_NAME).decode("utf-8"))
		return manifest, None
	except zipfile.BadZipFile:
		return None, _("The file is not a valid zip archive.")
	except Exception as e:
		return None, str(e)


def _safe_members(z):
	"""Yield only members with safe (non-traversing) paths."""
	for info in z.infolist():
		name = info.filename
		if name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
			log.warning("offlineAI backup: skipping unsafe path %s" % name)
			continue
		yield info


def _relocate(data, config_json_path, voice_profiles_dir):
	"""A backup may come from another computer or user account. Keep this
	machine's models folder, and point voice profiles at the samples that are
	restored here, instead of at paths that only existed on the old machine."""
	try:
		cfg = json.loads(data.decode("utf-8"))
		if not isinstance(cfg, dict):
			return data
		current = {}
		try:
			with open(config_json_path, "r", encoding="utf-8") as f:
				current = json.load(f)
		except Exception:
			current = {}
		old_dir = cfg.get("models_dir")
		if old_dir and not os.path.isdir(old_dir):
			if current.get("models_dir"):
				cfg["models_dir"] = current["models_dir"]
			else:
				cfg.pop("models_dir", None)
		if voice_profiles_dir:
			for p in cfg.get("voice_profiles", []) or []:
				pth = p.get("path")
				if pth and not os.path.isfile(pth):
					p["path"] = os.path.join(
						voice_profiles_dir,
						os.path.basename(pth.replace("\\", "/")))
		# Local model files that do not exist here cannot be used.
		cfg["custom_models"] = [
			m for m in (cfg.get("custom_models") or [])
			if not m.get("local_path") or os.path.isfile(m["local_path"])]
		return json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8")
	except Exception as e:
		log.warning("offlineAI backup: could not adapt paths: %s" % e)
		return data


def restore_backup(src_zip, config_json_path, history_path, voice_profiles_dir,
                   overwrite=True):
	"""Restore a backup. Returns (restored_list, err). Validates the manifest and
	guards against path traversal. overwrite controls whether existing files are
	replaced."""
	manifest, err = inspect_backup(src_zip)
	if err:
		return None, err
	restored = []
	try:
		with zipfile.ZipFile(src_zip, "r") as z:
			members = list(_safe_members(z))
			names = [m.filename for m in members]

			# Config.
			if "config/offlineAI_config.json" in names:
				if overwrite or not os.path.isfile(config_json_path):
					data = z.read("config/offlineAI_config.json")
					data = _relocate(data, config_json_path, voice_profiles_dir)
					tmp = config_json_path + ".tmp"
					with open(tmp, "wb") as f:
						f.write(data)
					os.replace(tmp, config_json_path)
					restored.append("config")

			# History.
			if "history/chat_history.json" in names and history_path:
				if overwrite or not os.path.isfile(history_path):
					d = os.path.dirname(history_path)
					if d and not os.path.isdir(d):
						os.makedirs(d)
					with open(history_path, "wb") as f:
						f.write(z.read("history/chat_history.json"))
					restored.append("history")

			# Voice profiles.
			vp_members = [m for m in members
			              if m.filename.startswith("voice_profiles/")
			              and not m.filename.endswith("/")]
			if vp_members and voice_profiles_dir:
				if not os.path.isdir(voice_profiles_dir):
					os.makedirs(voice_profiles_dir)
				wrote_any = False
				for m in vp_members:
					base = os.path.basename(m.filename)
					if not base:
						continue
					dest = os.path.join(voice_profiles_dir, base)
					if overwrite or not os.path.isfile(dest):
						with open(dest, "wb") as f:
							f.write(z.read(m.filename))
						wrote_any = True
				if wrote_any:
					restored.append("voice_profiles")
		return restored, None
	except Exception as e:
		log.error("offlineAI backup: restore failed: %s" % e)
		return None, str(e)
