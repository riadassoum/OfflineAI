# -*- coding: utf-8 -*-
# The single in-process LLM engine of the Offline AI add-on.
#
# One GGUF model is kept loaded at a time and is shared by the chat window and
# the quick-action gestures (previously each kept its own copy, so using both
# held the model in RAM twice). The model is released when the chat window
# closes and, for quick actions, after a configurable idle time.

import os
import sys
import time
import threading

from logHandler import log

ADDON_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(ADDON_DIR, "lib")

_lock = threading.RLock()     # protects load/unload
_busy = threading.Lock()      # held while a generation is running
_llm = None
_signature = None
_loaded_path = None
_last_used = 0.0
_pinned = False               # True while the chat window is open
_watchdog = None

STOP_TOKENS = ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>",
               "<end_of_turn>", "</s>", "<|user|>"]

PROMPT_FORMATS = ("auto", "chatml", "llama3", "phi3", "gemma", "mistral")


class ContextOverflow(Exception):
	"""The prompt alone does not fit in the model's context window."""


def try_acquire_busy():
	return _busy.acquire(blocking=False)


def release_busy():
	try:
		_busy.release()
	except RuntimeError:
		pass


def is_busy():
	if _busy.acquire(blocking=False):
		_busy.release()
		return False
	return True


def _ensureLib():
	if os.path.isdir(LIB_DIR) and LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)


def _autoThreads(settings):
	from . import store
	return store.autoThreads(settings)


def _makeSignature(model_path, settings, ctx):
	return (os.path.normcase(os.path.abspath(model_path)), int(ctx),
	        _autoThreads(settings), int(settings.get("n_batch", 512)),
	        bool(settings.get("use_mlock", False)),
	        bool(settings.get("use_mmap", True)),
	        bool(settings.get("flash_attn", True)))


def load_model(model_path, settings, context=None):
	"""Load (or reuse) a model. Returns (ok, error). A change to any load-time
	setting (context, threads, batch, mlock, mmap, flash attention) reloads."""
	global _llm, _signature, _loaded_path, _last_used
	_ensureLib()
	ctx = int(context or settings.get("n_ctx", 2048))
	sig = _makeSignature(model_path, settings, ctx)
	with _lock:
		if _llm is not None and _signature == sig:
			_last_used = time.time()
			return True, None
		try:
			from llama_cpp import Llama
		except Exception as e:
			log.error("offlineAI: engine import failed: %s" % e)
			return False, str(e)
		_unload_locked()
		try:
			kwargs = dict(
				model_path=model_path,
				n_ctx=ctx,
				n_threads=sig[2],
				n_batch=sig[3],
				use_mlock=sig[4],
				use_mmap=sig[5],
				verbose=False,
			)
			try:
				_llm = Llama(flash_attn=sig[6], **kwargs)
			except Exception:
				# flash attention is refused by some builds/models.
				_llm = Llama(**kwargs)
			_signature = sig
			_loaded_path = model_path
			_last_used = time.time()
			_startWatchdog()
			return True, None
		except Exception as e:
			log.error("offlineAI: model load failed: %s" % e)
			_llm = None
			_signature = None
			_loaded_path = None
			return False, str(e)


def is_loaded():
	return _llm is not None


def loaded_path():
	return _loaded_path


def n_ctx():
	try:
		return int(_llm.n_ctx())
	except Exception:
		return 0


def count_tokens(text):
	"""Real token count from the loaded model's tokenizer. The old '4 characters
	per token' guess is wrong by 2-4x for Arabic, CJK, Cyrillic and most other
	non-English text, which made long chats overflow the context."""
	if not text:
		return 0
	llm = _llm
	if llm is not None:
		try:
			return len(llm.tokenize(text.encode("utf-8"), add_bos=False,
			                        special=False))
		except Exception:
			pass
	return estimate_tokens(text)


def estimate_tokens(text):
	"""Tokenizer-free estimate used when no model is loaded. Non-ASCII text is
	counted at ~1.5 characters per token, ASCII at ~4."""
	if not text:
		return 0
	ascii_n = sum(1 for ch in text if ord(ch) < 128)
	other = len(text) - ascii_n
	return max(1, int(ascii_n / 4.0 + other / 1.5))


def truncate_to_tokens(text, max_tokens):
	"""Cut text so it is at most max_tokens long. Returns (text, was_cut)."""
	if max_tokens <= 0:
		return "", bool(text)
	n = count_tokens(text)
	if n <= max_tokens:
		return text, False
	keep = text
	# Shrink proportionally, then tighten; a handful of rounds always suffices.
	for _round in range(8):
		ratio = max_tokens / float(max(1, n))
		keep = keep[:max(1, int(len(keep) * ratio * 0.97))]
		n = count_tokens(keep)
		if n <= max_tokens:
			break
	return keep, True


# --- prompt building --------------------------------------------------------

def buildPrompt(fmt, userText, system, history=None):
	"""Render a conversation in a model's native prompt format. history is a
	list of (user, assistant) tuples. The BOS token is NOT written here: the
	tokenizer adds it, and writing it as text too produced a double BOS on
	Llama 3 and Mistral models."""
	system = (system or "").strip()
	history = history or []
	if fmt == "llama3":
		out = ("<|start_header_id|>system<|end_header_id|>\n\n" + system +
		       "<|eot_id|>") if system else ""
		for u, a in history:
			out += ("<|start_header_id|>user<|end_header_id|>\n\n" + u +
			        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" +
			        a + "<|eot_id|>")
		out += ("<|start_header_id|>user<|end_header_id|>\n\n" + userText +
		        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
		return out
	if fmt == "phi3":
		out = ("<|system|>\n" + system + "<|end|>\n") if system else ""
		for u, a in history:
			out += "<|user|>\n" + u + "<|end|>\n<|assistant|>\n" + a + "<|end|>\n"
		out += "<|user|>\n" + userText + "<|end|>\n<|assistant|>\n"
		return out
	if fmt == "gemma":
		# Gemma has no system role; prepend system text to the first user turn.
		out = ""
		first = True
		for u, a in history:
			uu = (system + "\n\n" + u) if (system and first) else u
			first = False
			out += ("<start_of_turn>user\n" + uu + "<end_of_turn>\n"
			        "<start_of_turn>model\n" + a + "<end_of_turn>\n")
		uu = (system + "\n\n" + userText) if (system and first) else userText
		out += ("<start_of_turn>user\n" + uu +
		        "<end_of_turn>\n<start_of_turn>model\n")
		return out
	if fmt == "mistral":
		out = ""
		first = True
		for u, a in history:
			uu = (system + "\n\n" + u) if (system and first) else u
			first = False
			out += "[INST] " + uu + " [/INST] " + a + "</s>"
		uu = (system + "\n\n" + userText) if (system and first) else userText
		out += "[INST] " + uu + " [/INST]"
		return out
	# default: chatml (Qwen, SmolLM2, EuroLLM, ...)
	out = ("<|im_start|>system\n" + system + "<|im_end|>\n") if system else ""
	for u, a in history:
		out += ("<|im_start|>user\n" + u + "<|im_end|>\n"
		        "<|im_start|>assistant\n" + a + "<|im_end|>\n")
	out += ("<|im_start|>user\n" + userText + "<|im_end|>\n"
	        "<|im_start|>assistant\n")
	return out


def _messages(userText, system, history):
	msgs = []
	if (system or "").strip():
		msgs.append({"role": "system", "content": system.strip()})
	for u, a in history or []:
		msgs.append({"role": "user", "content": u})
		msgs.append({"role": "assistant", "content": a})
	msgs.append({"role": "user", "content": userText})
	return msgs


def fit_history(system, history, userText, reserve):
	"""Drop the oldest turns until system + history + question + `reserve`
	tokens of answer fit in the context. Uses real token counts."""
	ctx = n_ctx() or 2048
	budget = ctx - int(reserve) - count_tokens(system) - count_tokens(userText) - 48
	kept = []
	used = 0
	for u, a in reversed(history or []):
		cost = count_tokens(u) + count_tokens(a) + 16
		if used + cost > budget:
			break
		kept.append((u, a))
		used += cost
	return list(reversed(kept))


def _sampling(settings):
	params = dict(
		temperature=float(settings.get("temperature", 0.7)),
		top_p=float(settings.get("top_p", 0.95)),
		top_k=int(settings.get("top_k", 40)),
		min_p=float(settings.get("min_p", 0.05)),
		repeat_penalty=float(settings.get("repeat_penalty", 1.1)),
	)
	seed = int(settings.get("seed", -1))
	if seed >= 0:
		params["seed"] = seed
	return params


def stream(fmt, userText, system, history, settings, max_tokens=None,
           should_stop=None):
	"""Generate a reply, yielding text pieces as they are produced.

	fmt "auto" uses the chat template stored inside the GGUF file (correct for
	any model); the named formats use the hand-written templates above. The
	answer length is clamped to what is left of the context window, and a
	prompt that cannot fit raises ContextOverflow instead of a raw engine
	error. Call with the busy lock held."""
	global _last_used
	llm = _llm
	if llm is None:
		raise RuntimeError("no model loaded")
	_last_used = time.time()
	want = int(max_tokens or settings.get("max_tokens", 512))
	params = _sampling(settings)
	ctx = n_ctx() or int(settings.get("n_ctx", 2048))

	use_chat = (fmt == "auto")
	prompt = None
	if use_chat:
		used = sum(count_tokens(m["content"]) + 8
		           for m in _messages(userText, system, history)) + 16
	else:
		prompt = buildPrompt(fmt, userText, system, history)
		try:
			used = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True,
			                        special=True))
		except Exception:
			used = estimate_tokens(prompt)
	room = ctx - used - 8
	if room < 16:
		raise ContextOverflow("%d>%d" % (used, ctx))
	params["max_tokens"] = max(16, min(want, room))

	yielded = False
	try:
		if use_chat:
			try:
				it = llm.create_chat_completion(
					messages=_messages(userText, system, history), stream=True,
					stop=STOP_TOKENS, **params)
				for chunk in it:
					if should_stop and should_stop():
						break
					delta = chunk["choices"][0].get("delta") or {}
					piece = delta.get("content")
					if piece:
						yielded = True
						yield piece
				return
			except (ValueError, KeyError, TypeError) as e:
				if yielded:
					raise
				# No usable template in the file: fall back to ChatML.
				log.warning("offlineAI: built-in chat template failed (%s); "
				            "falling back to ChatML" % e)
				prompt = buildPrompt("chatml", userText, system, history)
		for out in llm(prompt, stream=True, stop=STOP_TOKENS, **params):
			if should_stop and should_stop():
				break
			piece = out["choices"][0]["text"]
			if piece:
				yield piece
	finally:
		_last_used = time.time()


def generate(fmt, userText, system, settings, think_mode="hide",
             max_tokens=None, should_stop=None):
	"""One-shot generation for quick actions. Returns (text, error)."""
	try:
		raw = "".join(stream(fmt, userText, system, None, settings,
		                     max_tokens=max_tokens, should_stop=should_stop))
	except ContextOverflow:
		return None, "context"
	except Exception as e:
		log.error("offlineAI: generate failed: %s" % e)
		return None, str(e)
	try:
		from . import think_filter
		tf = think_filter.ThinkFilter(mode=think_mode)
		raw = tf.feed(raw) + tf.flush()
	except Exception:
		pass
	return raw.strip(), None


def split_for_context(text, instruction, settings):
	"""Split a long text into pieces that each fit the context together with an
	answer up to twice as long (a translation can need more tokens than its
	source). Splits on paragraph, then sentence, then hard boundaries."""
	ctx = n_ctx() or int(settings.get("n_ctx", 2048))
	per = max(64, (ctx - count_tokens(instruction) - 96) // 3)
	if count_tokens(text) <= per:
		return [text], per
	import re
	units = []
	for para in re.split(r"(\n\s*\n)", text):
		if not para:
			continue
		if count_tokens(para) <= per:
			units.append(para)
			continue
		for sent in re.findall(
				r"[^.!?\u061f\u06d4\u3002\uff01\uff1f\u0964\n]+"
				r"[.!?\u061f\u06d4\u3002\uff01\uff1f\u0964\n]*\s*", para) or [para]:
			while count_tokens(sent) > per:
				head, _cut = truncate_to_tokens(sent, per)
				units.append(head)
				sent = sent[len(head):]
			if sent:
				units.append(sent)
	chunks, buf, used = [], "", 0
	for u in units:
		c = count_tokens(u)
		if buf and used + c > per:
			chunks.append(buf)
			buf, used = "", 0
		buf += u
		used += c
	if buf.strip():
		chunks.append(buf)
	return [c for c in chunks if c.strip()], per


def benchmark(n_tokens=30):
	"""Fixed generation to measure speed. Returns (result_dict, err)."""
	global _last_used
	llm = _llm
	if llm is None:
		return None, "no model loaded"
	prompt = ("Write a short paragraph about the importance of accessibility "
	          "in modern software design.")
	try:
		t0 = time.time()
		out = llm(prompt, stream=False, max_tokens=int(n_tokens),
		          temperature=0.7, top_p=0.95, stop=[])
		elapsed = time.time() - t0
		text = out["choices"][0]["text"] or ""
		usage = out.get("usage", {}) if isinstance(out, dict) else {}
		toks = int(usage.get("completion_tokens", 0)) or max(1, len(text) // 4)
		_last_used = time.time()
		return {"tokens": toks, "seconds": elapsed,
		        "tokens_per_second": toks / elapsed if elapsed > 0 else 0.0}, None
	except Exception as e:
		log.error("offlineAI benchmark: failed: %s" % e)
		return None, str(e)


def _unload_locked():
	global _llm, _signature, _loaded_path
	old = _llm
	_llm = None
	_signature = None
	_loaded_path = None
	if old is not None:
		try:
			old.close()
		except Exception:
			pass


def unload(force=False):
	"""Free the model. Unless forced, does nothing while a generation runs."""
	if not force and is_busy():
		return False
	with _lock:
		_unload_locked()
	return True


def set_pinned(flag):
	"""The chat window pins the model so the idle timer leaves it alone."""
	global _pinned, _last_used
	_pinned = bool(flag)
	_last_used = time.time()


def _startWatchdog():
	global _watchdog
	if _watchdog is not None and _watchdog.is_alive():
		return

	def run():
		from . import store
		while True:
			time.sleep(30)
			if _llm is None:
				return  # nothing to watch; restarted on the next load
			if _pinned or is_busy():
				continue
			try:
				minutes = float(store.loadSettings().get("unload_after_min", 10))
			except Exception:
				minutes = 10
			if minutes > 0 and time.time() - _last_used > minutes * 60:
				if unload():
					log.info("offlineAI: idle model unloaded")
					return

	_watchdog = threading.Thread(target=run, daemon=True,
	                             name="offlineAI-idle-unload")
	_watchdog.start()
