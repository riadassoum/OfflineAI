# -*- coding: utf-8 -*-
# Chat history persistence for the Offline AI add-on.
# Pure-Python JSON storage (no sqlite3 — it is one of the stdlib modules NVDA's
# embedded Python may omit). One JSON file holds a list of sessions; each session
# is a list of {"role": "user"|"assistant", "text": ...} messages plus metadata.

import os
import json
import time

import globalVars
import addonHandler
from logHandler import log

addonHandler.initTranslation()


def _historyPath():
	base = globalVars.appArgs.configPath
	d = os.path.join(base, "offlineAI")
	if not os.path.isdir(d):
		try:
			os.makedirs(d)
		except OSError:
			pass
	return os.path.join(d, "chat_history.json")


def _load_all():
	p = _historyPath()
	try:
		with open(p, "r", encoding="utf-8") as f:
			data = json.load(f)
		if isinstance(data, dict) and isinstance(data.get("sessions"), list):
			return data["sessions"]
	except Exception:
		pass
	return []


def _save_all(sessions):
	p = _historyPath()
	try:
		tmp = p + ".tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump({"sessions": sessions}, f, ensure_ascii=False, indent=1)
		os.replace(tmp, p)
		return True
	except Exception as e:
		log.error("offlineAI: cannot save chat history: %s" % e)
		return False


def list_sessions():
	"""Return sessions as a list of dicts, newest first, with id/title/when."""
	sessions = _load_all()
	out = []
	for s in sessions:
		out.append({
			"id": s.get("id"),
			"title": s.get("title") or _default_title(s),
			"updated": s.get("updated", 0),
			"count": len(s.get("messages", [])),
		})
	out.sort(key=lambda x: x["updated"], reverse=True)
	return out


def _default_title(session):
	for msg in session.get("messages", []):
		if msg.get("role") == "user" and msg.get("text"):
			t = msg["text"].strip().replace("\n", " ")
			return (t[:40] + "...") if len(t) > 40 else t
	# Translators: title of a saved chat that has no text yet.
	return _("Untitled chat")


def load_session(session_id):
	for s in _load_all():
		if s.get("id") == session_id:
			return s
	return None


def new_session_id():
	return "chat_%d" % int(time.time() * 1000)


def save_session(session_id, messages, title=None, model=None):
	"""Create or update a session with the given messages list
	(list of {"role","text"})."""
	sessions = _load_all()
	found = None
	for s in sessions:
		if s.get("id") == session_id:
			found = s
			break
	if found is None:
		found = {"id": session_id, "created": int(time.time())}
		sessions.append(found)
	found["messages"] = messages
	found["updated"] = int(time.time())
	if title is not None:
		found["title"] = title
	if model is not None:
		found["model"] = model
	# Cap stored sessions to a sane number (keep newest 200).
	sessions.sort(key=lambda x: x.get("updated", 0), reverse=True)
	if len(sessions) > 200:
		sessions = sessions[:200]
	_save_all(sessions)
	return found


def delete_session(session_id):
	sessions = [s for s in _load_all() if s.get("id") != session_id]
	_save_all(sessions)


def export_session(session, path, as_markdown=False):
	"""Write a session to a .txt or .md file."""
	lines = []
	title = session.get("title") or _default_title(session)
	if as_markdown:
		lines.append("# " + title)
		lines.append("")
		for msg in session.get("messages", []):
			role = _("You:") if msg.get("role") == "user" else _("AI:")
			lines.append("**" + role + "** " + (msg.get("text") or ""))
			lines.append("")
	else:
		lines.append(title)
		lines.append("=" * len(title))
		lines.append("")
		for msg in session.get("messages", []):
			role = _("You:") if msg.get("role") == "user" else _("AI:")
			lines.append(role + " " + (msg.get("text") or ""))
			lines.append("")
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))


def messages_to_history(messages):
	"""Convert a flat message list into (user, assistant) tuples for prompting.
	Pairs each user message with the following assistant message."""
	pairs = []
	pending_user = None
	for msg in messages:
		if msg.get("role") == "user":
			pending_user = msg.get("text", "")
		elif msg.get("role") == "assistant" and pending_user is not None:
			pairs.append((pending_user, msg.get("text", "")))
			pending_user = None
	return pairs
