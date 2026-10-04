# -*- coding: utf-8 -*-
# Model catalog for the Offline AI add-on.
#
# Sources, merged in this order (later ones add to / can't break earlier ones):
#   1. Bundled local models.json (always present, always trusted)
#   2. Optional remote manifest fetched via urllib (validated, best-effort)
#   3. User's custom models (added via the UI, stored in config)
#
# All remote data is validated field-by-field; anything malformed is skipped so
# a bad or hostile manifest can never crash the add-on or inject junk entries.

import os
import json
import urllib.request

import addonHandler
from logHandler import log

addonHandler.initTranslation()

ADDON_DIR = os.path.dirname(__file__)
LOCAL_MODELS_JSON = os.path.join(ADDON_DIR, "models.json")

# "auto" = use the chat template stored inside the GGUF file itself.
VALID_FORMATS = {"auto", "chatml", "llama3", "phi3", "gemma", "mistral"}

# A conservative default remote endpoint. Empty by default so nothing is fetched
# unless the user opts in via settings (set "catalog_url" in config).
DEFAULT_CATALOG_URL = ""


def _clean_str(v, maxlen=200):
	if not isinstance(v, str):
		return None
	v = v.strip()
	if not v or len(v) > maxlen:
		return None
	return v


def _valid_url(u):
	u = _clean_str(u, 2000)
	if not u:
		return None
	# Only allow plain http(s) direct links.
	if not (u.startswith("https://") or u.startswith("http://")):
		return None
	return u


def _valid_filename(f):
	f = _clean_str(f, 255)
	if not f:
		return None
	# No path separators or traversal — filename only, lands in the models dir.
	if "/" in f or "\\" in f or ".." in f or f.startswith("."):
		return None
	return f


def validate_model(entry):
	"""Return a sanitized model dict, or None if the entry is unusable.
	Required: name, filename, and either url (http/https) or a valid local_path.
	Optional: size, ram, prompt_format, family, context, category."""
	if not isinstance(entry, dict):
		return None
	name = _clean_str(entry.get("name"))
	filename = _valid_filename(entry.get("filename"))
	local_path = entry.get("local_path")
	has_local = isinstance(local_path, str) and os.path.isfile(local_path)
	url = _valid_url(entry.get("url"))
	if not (name and filename and (url or has_local)):
		return None
	fmt = entry.get("prompt_format", "chatml")
	if fmt not in VALID_FORMATS:
		fmt = "chatml"
	out = {
		"name": name,
		"filename": filename,
		"url": url or ("file://" + local_path if has_local else ""),
		"prompt_format": fmt,
		"size": _clean_str(entry.get("size", "")) or "?",
		"ram": _clean_str(entry.get("ram", "")) or "?",
	}
	if has_local:
		out["local_path"] = local_path
	# Optional metadata (safe extras).
	fam = _clean_str(entry.get("family", ""))
	if fam:
		out["family"] = fam
	cat = _clean_str(entry.get("category", ""))
	if cat:
		out["category"] = cat
	ctx = entry.get("context")
	if isinstance(ctx, int) and 256 <= ctx <= 1000000:
		out["context"] = ctx
	return out


def load_local_models():
	try:
		with open(LOCAL_MODELS_JSON, "r", encoding="utf-8") as f:
			data = json.load(f)
		out = []
		for m in data.get("models", []):
			v = validate_model(m)
			if v:
				out.append(v)
		return out
	except Exception as e:
		log.error("offlineAI: cannot load local models.json: %s" % e)
		return []


def fetch_remote_models(url, timeout=8):
	"""Best-effort remote fetch. Returns a list (possibly empty). Never raises."""
	url = _valid_url(url)
	if not url:
		return []
	try:
		req = urllib.request.Request(url, headers={"User-Agent": "offlineAI"})
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			raw = resp.read(2 * 1024 * 1024)  # cap at 2 MB
		data = json.loads(raw.decode("utf-8", "replace"))
		models = data.get("models", []) if isinstance(data, dict) else []
		out = []
		for m in models:
			v = validate_model(m)
			if v:
				out.append(v)
		return out
	except Exception as e:
		log.warning("offlineAI: remote catalog fetch failed: %s" % e)
		return []


def merge_models(*lists):
	"""Merge model lists, de-duplicating by filename (first wins)."""
	seen = set()
	out = []
	for lst in lists:
		for m in lst:
			key = m.get("filename")
			if key and key not in seen:
				seen.add(key)
				out.append(m)
	return out


def build_catalog(custom_models=None, remote_url=None):
	"""Assemble the full catalog: local + remote (if any) + custom."""
	local = load_local_models()
	remote = fetch_remote_models(remote_url) if remote_url else []
	custom = []
	for m in (custom_models or []):
		v = validate_model(m)
		if v:
			v["custom"] = True
			custom.append(v)
	# Local first (trusted, ordered), then remote extras, then custom.
	return merge_models(local, remote, custom)


# --- HuggingFace live search (unlimited models on demand) ------------------

HF_API = "https://huggingface.co/api/models"


def hf_search_repos(query, limit=60):
	"""Search HuggingFace for GGUF model repositories matching a query.
	Returns a list of repo ids. Pages through the API to satisfy limits above
	the API's per-request maximum. Never raises."""
	q = _clean_str(query, 200)
	if not q:
		return []
	try:
		import urllib.parse
		want = max(1, int(limit))
		out = []
		fetched = 0
		# The API returns at most ~100 per call; page with skip/limit.
		per = 100 if want > 100 else want
		while fetched < want:
			params = urllib.parse.urlencode({
				"search": q, "filter": "gguf",
				"limit": min(per, want - fetched),
				"skip": fetched,
				"sort": "downloads", "direction": "-1",
			})
			url = HF_API + "?" + params
			req = urllib.request.Request(url, headers={"User-Agent": "offlineAI"})
			with urllib.request.urlopen(req, timeout=15) as resp:
				data = json.loads(resp.read(4 * 1024 * 1024).decode("utf-8", "replace"))
			if not data:
				break
			for m in data:
				rid = m.get("id") or m.get("modelId")
				if isinstance(rid, str) and rid:
					out.append(rid)
			got = len(data)
			fetched += got
			if got < per:
				break  # no more results
		return out
	except Exception as e:
		log.warning("offlineAI: HF search failed: %s" % e)
		return []


def hf_list_gguf_files(repo_id):
	"""List the .gguf files in a HuggingFace repo with sizes. Returns a list of
	(filename, size_bytes) tuples (size may be None). Never raises."""
	rid = _clean_str(repo_id, 200)
	if not rid or ".." in rid:
		return []
	try:
		import urllib.parse
		url = HF_API + "/" + urllib.parse.quote(rid) + "?blobs=true"
		req = urllib.request.Request(url, headers={"User-Agent": "offlineAI"})
		with urllib.request.urlopen(req, timeout=20) as resp:
			data = json.loads(resp.read(8 * 1024 * 1024).decode("utf-8", "replace"))
		files = []
		for s in data.get("siblings", []):
			fn = s.get("rfilename", "")
			# Only top-level single-file GGUFs (skip sharded subfolder parts).
			if fn.endswith(".gguf") and "/" not in fn:
				size = s.get("size")
				files.append((fn, size if isinstance(size, int) else None))
		files.sort(key=lambda x: x[0])
		return files
	except Exception as e:
		log.warning("offlineAI: HF file list failed: %s" % e)
		return []


def _fmt_bytes(n):
	if not isinstance(n, int) or n < 0:
		return "?"
	x = float(n)
	for u in ("B", "KB", "MB", "GB"):
		if x < 1024:
			return "%.1f %s" % (x, u)
		x /= 1024.0
	return "%.1f TB" % x


def hf_build_entry(repo_id, filename, prompt_format="chatml", size_bytes=None):
	"""Build a validated model entry for a specific repo file, verifying the
	download link resolves. Returns (entry, error)."""
	rid = _clean_str(repo_id, 200)
	fn = _valid_filename(filename)
	if not rid or not fn or ".." in rid:
		return None, _("invalid repository or file name")
	url = "https://huggingface.co/%s/resolve/main/%s" % (rid, fn)
	# Verify the link resolves (HEAD).
	try:
		req = urllib.request.Request(url, method="HEAD",
		                             headers={"User-Agent": "offlineAI"})
		with urllib.request.urlopen(req, timeout=15) as resp:
			status = resp.status
	except Exception as e:
		try:
			status = e.code  # HTTPError carries a code
		except Exception:
			status = None
	if status not in (200, 302):
		return None, _("the download link is not available (status {s})").format(
			s=status)
	# Derive a friendly name.
	short = rid.split("/")[-1]
	name = "%s (%s)" % (short, fn.replace(".gguf", ""))
	entry = validate_model({
		"name": name[:120], "filename": fn, "url": url,
		"prompt_format": prompt_format, "family": short.split("-")[0],
		"category": "HuggingFace",
		"size": _fmt_bytes(size_bytes) if size_bytes else "?",
	})
	if not entry:
		return None, _("the model entry is not valid")
	return entry, None


def guess_prompt_format(repo_id):
	"""Best-effort prompt format guess from the repo name."""
	r = (repo_id or "").lower()
	if "llama-3" in r or "llama3" in r or "llama_3" in r:
		return "llama3"
	if "phi-3" in r or "phi3" in r or "phi_3" in r:
		return "phi3"
	if "gemma" in r:
		return "gemma"
	if "mistral" in r or "mixtral" in r or "nemo" in r:
		return "mistral"
	if "qwen" in r or "smollm" in r or "deepseek" in r or "eurollm" in r:
		return "chatml"
	# Anything else: let the engine use the template shipped in the file, which
	# is right for every model instead of a guess that is wrong for many.
	return "auto"
