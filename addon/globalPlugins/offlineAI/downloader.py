# -*- coding: utf-8 -*-
# Universal model downloader for the Offline AI add-on.
# One implementation used by both the LLM window and the Whisper window:
# same save location logic (caller passes the destination), same accessible
# progress dialog (percent, size, speed, elapsed, remaining, cancel).

import os
import time
import threading
import urllib.request
import urllib.error

import wx
import ui
import addonHandler
from logHandler import log

addonHandler.initTranslation()


def fmtSize(nbytes):
	if nbytes is None or nbytes < 0:
		return "?"
	for unit in ("B", "KB", "MB", "GB"):
		if nbytes < 1024:
			return "%.1f %s" % (nbytes, unit)
		nbytes /= 1024.0
	return "%.1f TB" % nbytes


def fmtTime(sec):
	if sec is None or sec < 0 or sec != sec:
		return "?"
	sec = int(sec)
	if sec >= 3600:
		return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)
	return "%d:%02d" % (sec // 60, sec % 60)


class DownloadProgressDialog(wx.Dialog):
	"""Accessible progress window: percent, amount, speed, elapsed, remaining,
	and a working Cancel button. The status line is a read-only text field NVDA
	re-reads as it updates."""

	def __init__(self, parent, modelName):
		super(DownloadProgressDialog, self).__init__(
			parent, title=_("Downloading {n}").format(n=modelName))
		self.cancelled = False
		s = wx.BoxSizer(wx.VERTICAL)
		self.gauge = wx.Gauge(self, range=1000, size=(420, 24))
		s.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 10)
		self.status = wx.TextCtrl(
			self, style=wx.TE_READONLY | wx.TE_MULTILINE, size=(420, 90))
		s.Add(self.status, 0, wx.ALL | wx.EXPAND, 10)
		self.cancelButton = wx.Button(self, id=wx.ID_CANCEL, label=_("&Cancel"))
		self.cancelButton.Bind(wx.EVT_BUTTON, self.onCancel)
		s.Add(self.cancelButton, 0, wx.ALL | wx.ALIGN_RIGHT, 10)
		self.Bind(wx.EVT_CLOSE, self.onCancel)
		self.SetSizerAndFit(s)

	def onCancel(self, evt):
		self.cancelled = True
		self.cancelButton.Enable(False)
		self.cancelButton.SetLabel(_("Cancelling..."))

	def update(self, done, total, speed, elapsed):
		if total > 0:
			frac = min(1.0, done / total)
			self.gauge.SetValue(int(frac * 1000))
			remain = (total - done) / speed if speed > 0 else None
			txt = _(
				"{pct:.0f} percent - {done} of {total}\n"
				"Speed: {speed}/s\n"
				"Elapsed: {el}   Remaining: {rem}"
			).format(pct=frac * 100, done=fmtSize(done), total=fmtSize(total),
			         speed=fmtSize(speed), el=fmtTime(elapsed),
			         rem=fmtTime(remain))
		else:
			self.gauge.Pulse()
			txt = _("{done} downloaded\nSpeed: {speed}/s\nElapsed: {el}").format(
				done=fmtSize(done), speed=fmtSize(speed), el=fmtTime(elapsed))
		# ChangeValue keeps the caret where the user is reading.
		pos = self.status.GetInsertionPoint()
		self.status.ChangeValue(txt)
		try:
			self.status.SetInsertionPoint(min(pos, len(txt)))
		except Exception:
			pass

	def announce(self, msg):
		ui.message(msg)


def _ui(dlg, func, *args):
	"""Call a dialog method on the main thread, unless it is already gone."""
	def run():
		try:
			if dlg and not getattr(dlg, "_closed", False):
				func(*args)
		except RuntimeError:
			pass  # the window was destroyed in the meantime
	wx.CallAfter(run)


def head_size(url, timeout=15):
	"""Return the download size in bytes from a HEAD request, or None."""
	try:
		req = urllib.request.Request(url, method="HEAD",
		                             headers={"User-Agent": "offlineAI"})
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			cl = resp.headers.get("Content-Length")
			return int(cl) if cl else None
	except Exception:
		return None


def download(parent, url, dest, model_name, on_done):
	"""Start a download with a progress dialog. Calls on_done(ok, err) on the
	main thread when finished. Returns the progress dialog."""
	dlg = DownloadProgressDialog(parent, model_name)
	dlg.Show()
	t = threading.Thread(target=_worker, args=(url, dest, dlg, on_done),
	                     daemon=True)
	t.daemon = True
	t.start()
	return dlg


class _Cancelled(Exception):
	pass


def _worker(url, dest, dlg, on_done):
	"""Download to <dest>.part and rename when complete. An interrupted
	download is resumed from where it stopped (HTTP Range) the next time, which
	matters for multi-gigabyte models on unreliable connections."""
	tmp = dest + ".part"
	start = time.time()
	lastPct = 0
	lastUi = 0.0
	try:
		d = os.path.dirname(dest)
		if d and not os.path.isdir(d):
			os.makedirs(d)
		have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
		headers = {"User-Agent": "offlineAI"}
		if have > 0:
			headers["Range"] = "bytes=%d-" % have
		req = urllib.request.Request(url, headers=headers)
		try:
			resp = urllib.request.urlopen(req, timeout=30)
		except urllib.error.HTTPError as e:
			if e.code == 416 and have > 0:
				# The partial file is stale or already complete: start again.
				os.remove(tmp)
				have = 0
				resp = urllib.request.urlopen(urllib.request.Request(
					url, headers={"User-Agent": "offlineAI"}), timeout=30)
			else:
				raise
		with resp:
			resumed = have > 0 and getattr(resp, "status", 200) == 206
			if not resumed:
				have = 0
			length = int(resp.headers.get("Content-Length", 0) or 0)
			total = (have + length) if length > 0 else 0
			done = have
			session = 0
			chunk = 1024 * 256
			with open(tmp, "ab" if resumed else "wb") as f:
				while True:
					if dlg.cancelled:
						raise _Cancelled()
					buf = resp.read(chunk)
					if not buf:
						break
					f.write(buf)
					done += len(buf)
					session += len(buf)
					now = time.time()
					if now - lastUi >= 0.5:
						lastUi = now
						elapsed = now - start
						speed = session / elapsed if elapsed > 0 else 0
						_ui(dlg, dlg.update, done, total, speed, elapsed)
					if total > 0:
						pct = int(done * 100 / total)
						if pct >= lastPct + 10:
							lastPct = pct - (pct % 10)
							if lastPct < 100:
								_ui(dlg, dlg.announce,
								    _("{p} percent").format(p=lastPct))
		if total > 0 and done < total:
			# The connection dropped. Keep the partial file for resuming.
			raise IOError(_("the connection was interrupted at {p} percent; "
			                "press Download again to resume").format(
				p=int(done * 100 / total)))
		os.replace(tmp, dest)
		wx.CallAfter(_finish, dlg, on_done, True, None)
	except _Cancelled:
		try:
			if os.path.isfile(tmp):
				os.remove(tmp)
		except OSError:
			pass
		wx.CallAfter(_finish, dlg, on_done, False, _("Cancelled."))
	except Exception as e:
		log.warning("offlineAI download failed: %s" % e)
		wx.CallAfter(_finish, dlg, on_done, False, str(e))


def _finish(dlg, on_done, ok, err):
	try:
		dlg._closed = True
		dlg.Destroy()
	except Exception:
		pass
	try:
		on_done(ok, err)
	except RuntimeError:
		pass  # the window that started the download was closed
