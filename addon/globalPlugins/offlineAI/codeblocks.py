# -*- coding: utf-8 -*-
# Extract fenced code blocks from markdown text (LLM answers).
# Handles ``` fences with optional language tags, and leaves prose alone.

import re

import addonHandler

addonHandler.initTranslation()

# Matches ```lang\n ... \n``` (lang optional). Non-greedy body, DOTALL.
_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


def extract_code_blocks(text):
	"""Return a list of {'lang': str, 'code': str} for each fenced block.
	lang is '' when no language tag is present."""
	if not text:
		return []
	blocks = []
	for m in _FENCE.finditer(text):
		lang = (m.group(1) or "").strip()
		code = m.group(2)
		# Strip a single trailing newline the fence adds, keep internal ones.
		if code.endswith("\n"):
			code = code[:-1]
		if code.strip():
			blocks.append({"lang": lang, "code": code})
	return blocks


def language_label(lang):
	"""Human-friendly label for a language tag, for speech cues."""
	if not lang:
		# Translators: a block of program code with no language given.
		return _("Code block")
	pretty = {
		"py": "Python", "python": "Python", "js": "JavaScript",
		"javascript": "JavaScript", "ts": "TypeScript", "cpp": "C++",
		"c": "C", "cs": "C sharp", "csharp": "C sharp", "java": "Java",
		"go": "Go", "rs": "Rust", "rust": "Rust", "rb": "Ruby",
		"php": "PHP", "sh": "Shell", "bash": "Bash", "shell": "Shell",
		"sql": "SQL", "html": "HTML", "css": "CSS", "json": "JSON",
		"xml": "XML", "yaml": "YAML", "yml": "YAML", "md": "Markdown",
		"kt": "Kotlin", "swift": "Swift", "r": "R", "lua": "Lua",
		"pl": "Perl", "dart": "Dart", "scala": "Scala",
	}
	key = lang.lower()
	# Translators: e.g. "Python code block".
	return _("{language} code block").format(language=pretty.get(key, lang))
