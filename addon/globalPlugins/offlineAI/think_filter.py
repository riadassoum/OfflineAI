# -*- coding: utf-8 -*-
# Streaming filter for reasoning models that emit <think>...</think> blocks
# (e.g. DeepSeek-R1 distillations). Handles tags split across token boundaries.
#
# Modes:
#   "show"     - pass everything through unchanged (reasoning included)
#   "hide"     - drop everything inside <think>...</think>; keep only final answer
#   "cue"      - drop the reasoning, but emit a one-time cue when it ends
#
# Usage:
#   f = ThinkFilter(mode="hide", cue_text="[Thinking complete] ")
#   for piece in stream:
#       out = f.feed(piece)   # returns text to display now (may be "")
#       if out: show(out)
#   tail = f.flush()          # any buffered text at end
#   if tail: show(tail)

OPEN = "<think>"
CLOSE = "</think>"


class ThinkFilter(object):
	def __init__(self, mode="show", cue_text="[Thinking complete] "):
		self.mode = mode if mode in ("show", "hide", "cue") else "show"
		self.cue_text = cue_text
		self.in_think = False          # currently inside a think block
		self.buf = ""                  # holds a possible partial tag
		self.cue_pending = False       # a cue is owed when reasoning ends
		self.saw_think = False         # whether any think block was seen

	def _emit_cue_if_needed(self):
		if self.mode == "cue" and self.cue_pending:
			self.cue_pending = False
			return self.cue_text
		return ""

	def feed(self, piece):
		"""Feed one streamed chunk; return text that should be shown now."""
		if self.mode == "show":
			return piece

		self.buf += piece
		out = []

		# Process the buffer, peeling off complete regions. We keep at most
		# len(longest_tag)-1 chars back in the buffer to catch split tags.
		while True:
			if not self.in_think:
				# Look for an opening tag.
				idx = self.buf.find(OPEN)
				if idx != -1:
					# Text before the tag is real answer text -> show it.
					out.append(self.buf[:idx])
					self.buf = self.buf[idx + len(OPEN):]
					self.in_think = True
					self.saw_think = True
					continue
				# No full opening tag. Could a partial one be at the buffer tail?
				keep = self._partial_tail_len(self.buf, OPEN)
				if keep:
					out.append(self.buf[:-keep])
					self.buf = self.buf[-keep:]
				else:
					out.append(self.buf)
					self.buf = ""
				break
			else:
				# Inside a think block: look for the closing tag.
				idx = self.buf.find(CLOSE)
				if idx != -1:
					# Drop everything up to and including the close tag.
					self.buf = self.buf[idx + len(CLOSE):]
					self.in_think = False
					self.cue_pending = True
					# Emit cue immediately (if in cue mode) before continuing.
					out.append(self._emit_cue_if_needed())
					continue
				# No full close tag. Keep a possible partial tail; drop the rest.
				keep = self._partial_tail_len(self.buf, CLOSE)
				self.buf = self.buf[-keep:] if keep else ""
				break

		return "".join(out)

	def flush(self):
		"""Call once the stream ends. Returns any remaining showable text."""
		tail = []
		if not self.in_think:
			# Any buffered non-think text is real; show it.
			tail.append(self.buf)
		# If we ended still inside a think block (model didn't close it),
		# in hide/cue mode we simply drop it. Emit a pending cue if owed.
		tail.append(self._emit_cue_if_needed())
		self.buf = ""
		return "".join(t for t in tail if t)

	@staticmethod
	def _partial_tail_len(s, tag):
		"""Return the length of the longest suffix of s that is a proper prefix
		of tag (so we hold it back in case the tag is split across chunks)."""
		maxlen = min(len(s), len(tag) - 1)
		for n in range(maxlen, 0, -1):
			if s[-n:] == tag[:n]:
				return n
		return 0
