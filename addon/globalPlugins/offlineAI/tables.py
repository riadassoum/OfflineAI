# -*- coding: utf-8 -*-
# Convert markdown tables into screen-reader-friendly text.
# LLMs love ASCII tables (| a | b |), which read terribly character-by-character.
# We turn each table into labeled key-value blocks per row, using the header row
# as the labels, so NVDA speaks "Column = Value" naturally.

import re

import addonHandler

addonHandler.initTranslation()

# A separator row like |---|:--:|---| (dashes, colons, pipes, spaces only).
_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _split_cells(line):
	# Strip one leading/trailing pipe, then split on unescaped pipes.
	s = line.strip()
	if s.startswith("|"):
		s = s[1:]
	if s.endswith("|"):
		s = s[:-1]
	cells = re.split(r"(?<!\\)\|", s)
	return [c.strip().replace("\\|", "|") for c in cells]


def _is_separator(line):
	stripped = line.strip()
	if not stripped or "-" not in stripped:
		return False
	# Only pipes, dashes, colons, spaces — and at least one dash.
	return bool(_SEP.match(stripped))


def format_tables(text):
	"""Find markdown tables in text and replace each with a labeled, readable
	block. Non-table text is left untouched. Returns the transformed text."""
	if not text or "|" not in text:
		return text
	lines = text.split("\n")
	out = []
	i = 0
	n = len(lines)
	while i < n:
		# A table is: a header row, a separator row, then >=1 data rows.
		if (i + 1 < n and "|" in lines[i] and _is_separator(lines[i + 1])
				and not _is_separator(lines[i])):
			header = _split_cells(lines[i])
			j = i + 2
			rows = []
			while j < n and "|" in lines[j] and not _is_separator(lines[j]):
				if lines[j].strip() == "":
					break
				rows.append(_split_cells(lines[j]))
				j += 1
			if rows:  # a real table
				out.append(_render_table(header, rows))
				i = j
				continue
		out.append(lines[i])
		i += 1
	return "\n".join(out)


def _render_table(header, rows):
	"""Render a header + rows as readable labeled blocks."""
	ncol = len(header)
	blocks = []
	for idx, row in enumerate(rows, 1):
		cells = (row + [""] * ncol)[:ncol]
		parts = []
		for c in range(ncol):
			label = (header[c] if c < len(header) and header[c]
			         else _("Column {n}").format(n=c + 1))
			parts.append("%s: %s" % (label, cells[c]))
		blocks.append(_("Row {n} - ").format(n=idx) + "; ".join(parts))
	return "\n".join(blocks)
