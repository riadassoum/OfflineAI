# -*- coding: utf-8 -*-
# Local AI OCR and image description for the Offline AI add-on.
# Drives the bundled llama-mtmd-cli.exe (mainline llama.cpp) against a vision
# GGUF pair (backbone + mmproj). Screen capture uses Windows GDI via ctypes;
# no Tesseract, no PyTorch.

import os
import sys
import tempfile
import subprocess

import addonHandler
from logHandler import log

addonHandler.initTranslation()

ADDON_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(ADDON_DIR, "lib")
OCR_DIR = os.path.join(LIB_DIR, "ocr")
TTS_DIR = os.path.join(LIB_DIR, "tts")  # shared DLLs live here
MTMD_EXE = os.path.join(OCR_DIR, "llama-mtmd-cli.exe")

# Where short-lived page images go. Set by the UI to a folder under the user's
# models store so nothing unexpected lands on the system drive.
_TEMP_DIR = None


def set_temp_dir(d):
	global _TEMP_DIR
	_TEMP_DIR = d


def _mktemp(suffix, prefix):
	kw = {"suffix": suffix, "prefix": prefix}
	if _TEMP_DIR and os.path.isdir(_TEMP_DIR):
		kw["dir"] = _TEMP_DIR
	return tempfile.mkstemp(**kw)

# The vision model pair (downloaded like other models).
OCR_MODEL = {
	"name": "Qwen2.5-VL 3B (OCR and image description)",
	"backbone": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
	"backbone_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/"
	                "resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
	"mmproj": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
	"mmproj_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/"
	              "resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
	"backbone_size": "~1.9 GB",
	"mmproj_size": "~850 MB",
}

OCR_PROMPT = ("Extract all text from this image exactly as it appears, "
              "preserving line breaks. Output only the text, in its original "
              "language, with no translation and no commentary.")
DESCRIBE_PROMPT = ("Describe this image in detail, including any text, layout, "
                   "and important visual elements.")


def ocr_prompt_for(lang_hint=None):
	"""Return an OCR prompt, optionally hinting the expected language so the
	model handles Arabic, French, etc. correctly. The model is multilingual;
	the hint just improves accuracy."""
	if lang_hint:
		return ("Extract all text from this image exactly as it appears, "
		        "preserving line breaks and the original script. The text is "
		        "primarily in %s. Output only the text, with no translation and "
		        "no commentary." % lang_hint)
	return OCR_PROMPT


def describe_prompt_for(lang_name=None):
	"""Image description prompt. With lang_name the description is written in
	that language, so users are not forced to read descriptions in English."""
	if lang_name and lang_name.lower() != "english":
		return DESCRIBE_PROMPT + " Write the description in %s." % lang_name
	return DESCRIBE_PROMPT


# Output budgets (tokens). A dense page of text is 600-1500 tokens; the old
# limit of 256 silently cut OCR results off after a few paragraphs.
OCR_MAX_TOKENS = 2048
DESCRIBE_MAX_TOKENS = 700
# Room for the image (up to ~3000 tokens at "best" quality) plus the answer.
VISION_CONTEXT = 8192


def render_pdf_pages(pdf_path, first_page=1, last_page=None, scale=None,
                     max_long_side=1600):
	"""Render a 1-based inclusive page range of a PDF to temporary PNG files.
	Returns (list_of_(page_number, png_path), err). Uses bundled pypdfium2.

	If scale is None, an adaptive scale is chosen per page so its long side is
	about max_long_side pixels. Smaller images mean far fewer vision tokens and
	much faster OCR, while staying legible for text recognition."""
	if LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)
	try:
		import pypdfium2 as pdfium
	except Exception as e:
		log.error("offlineAI OCR: pypdfium2 import failed: %s" % e)
		return None, _("PDF support is unavailable.")
	try:
		pdf = pdfium.PdfDocument(pdf_path)
		total = len(pdf)
		first = max(1, int(first_page))
		last = total if last_page is None else min(int(last_page), total)
		if first > last:
			return None, _("Invalid page range.")
		out = []
		for pno in range(first, last + 1):
			page = pdf[pno - 1]
			if scale is None:
				# Page size is in points (1/72 inch). Choose a scale so the long
				# side is about max_long_side pixels, clamped to a sane range.
				try:
					w_pt, h_pt = page.get_size()
					long_pt = max(w_pt, h_pt) or 612
					s = max(0.5, min(3.0, float(max_long_side) / long_pt))
				except Exception:
					s = 1.3
			else:
				s = scale
			bitmap = page.render(scale=s)
			w, h, stride = bitmap.width, bitmap.height, bitmap.stride
			buf = bytes(bitmap.buffer)
			png = _write_png(buf, w, h, stride)
			out.append((pno, png))
		return out, None
	except Exception as e:
		log.error("offlineAI OCR: PDF render failed: %s" % e)
		return None, str(e)


def pdf_page_count(pdf_path):
	"""Return the number of pages in a PDF, or None on failure."""
	if LIB_DIR not in sys.path:
		sys.path.insert(0, LIB_DIR)
	try:
		import pypdfium2 as pdfium
		return len(pdfium.PdfDocument(pdf_path))
	except Exception as e:
		log.error("offlineAI OCR: page count failed: %s" % e)
		return None


def engine_present():
	return os.path.isfile(MTMD_EXE)


def models_present(store_dir):
	return (os.path.isfile(os.path.join(store_dir, OCR_MODEL["backbone"])) and
	        os.path.isfile(os.path.join(store_dir, OCR_MODEL["mmproj"])))


def _startupinfo():
	try:
		si = subprocess.STARTUPINFO()
		si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
		si.wShowWindow = 0
		return si
	except Exception:
		return None


def _downscale_png(src_path, max_long_side=1600):
	"""If the image's long side exceeds max_long_side, write a downscaled copy and
	return its path; otherwise return src_path unchanged. Uses pypdfium2's bitmap
	loader is not applicable here, so we use a light numpy nearest-neighbour
	resize on the decoded PNG. Returns (path, is_temp)."""
	# For robustness across formats, use wx (present in NVDA) to load + scale.
	try:
		import wx
		img = wx.Image(src_path)
		if not img.IsOk():
			return src_path, False
		w, h = img.GetWidth(), img.GetHeight()
		long_side = max(w, h)
		native = src_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
		if long_side <= max_long_side and native:
			return src_path, False
		if long_side <= max_long_side:
			# Formats the vision engine cannot open itself (GIF, WebP, TIFF):
			# re-save as PNG without resizing.
			fd, out = _mktemp(".png", "offlineAI_rs_")
			os.close(fd)
			img.SaveFile(out, wx.BITMAP_TYPE_PNG)
			return out, True
		ratio = float(max_long_side) / long_side
		nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
		img = img.Scale(nw, nh, wx.IMAGE_QUALITY_HIGH)
		fd, out = _mktemp(".png", "offlineAI_rs_")
		os.close(fd)
		img.SaveFile(out, wx.BITMAP_TYPE_PNG)
		return out, True
	except Exception as e:
		log.warning("offlineAI OCR: downscale skipped: %s" % e)
		return src_path, False


def capture_screen(region=None):
	"""Capture the whole (virtual) screen or a region to a temp PNG using
	Windows GDI. region is an optional (left, top, width, height).
	Returns (path, err)."""
	if not sys.platform.startswith("win"):
		return None, "screen capture is only available on Windows"
	try:
		import ctypes
		from ctypes import wintypes

		user32 = ctypes.WinDLL("user32", use_last_error=True)
		gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
		# Handles are pointer-sized: without these declarations ctypes truncates
		# them to 32 bits on 64-bit NVDA and the capture fails at random.
		H = ctypes.c_void_p
		user32.GetDC.restype = H
		user32.GetDC.argtypes = [H]
		user32.ReleaseDC.argtypes = [H, H]
		gdi32.CreateCompatibleDC.restype = H
		gdi32.CreateCompatibleDC.argtypes = [H]
		gdi32.CreateCompatibleBitmap.restype = H
		gdi32.CreateCompatibleBitmap.argtypes = [H, ctypes.c_int, ctypes.c_int]
		gdi32.SelectObject.restype = H
		gdi32.SelectObject.argtypes = [H, H]
		gdi32.BitBlt.argtypes = [H, ctypes.c_int, ctypes.c_int, ctypes.c_int,
		                         ctypes.c_int, H, ctypes.c_int, ctypes.c_int,
		                         wintypes.DWORD]
		gdi32.GetDIBits.argtypes = [H, H, wintypes.UINT, wintypes.UINT,
		                            ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
		gdi32.DeleteObject.argtypes = [H]
		gdi32.DeleteDC.argtypes = [H]

		if region:
			left, top, width, height = region
		else:
			left = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
			top = user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN
			width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
			height = user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
		if width <= 0 or height <= 0:
			return None, "empty screen area"

		class BITMAPINFOHEADER(ctypes.Structure):
			_fields_ = [
				("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
				("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
				("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
				("biSizeImage", wintypes.DWORD),
				("biXPelsPerMeter", wintypes.LONG),
				("biYPelsPerMeter", wintypes.LONG),
				("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
			]

		screen_dc = user32.GetDC(None)
		mem_dc = gdi32.CreateCompatibleDC(screen_dc)
		bmp = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
		try:
			old = gdi32.SelectObject(mem_dc, bmp)
			SRCCOPY = 0x00CC0020
			CAPTUREBLT = 0x40000000
			gdi32.BitBlt(mem_dc, 0, 0, width, height, screen_dc, left, top,
			             SRCCOPY | CAPTUREBLT)
			gdi32.SelectObject(mem_dc, old)

			bmi = BITMAPINFOHEADER()
			bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
			bmi.biWidth = width
			bmi.biHeight = -height  # top-down
			bmi.biPlanes = 1
			bmi.biBitCount = 24
			bmi.biCompression = 0  # BI_RGB

			row_stride = ((width * 3 + 3) // 4) * 4
			buffer = ctypes.create_string_buffer(row_stride * height)
			got = gdi32.GetDIBits(mem_dc, bmp, 0, height, buffer,
			                      ctypes.byref(bmi), 0)
		finally:
			gdi32.DeleteObject(bmp)
			gdi32.DeleteDC(mem_dc)
			user32.ReleaseDC(None, screen_dc)
		if not got:
			return None, "could not read the screen"
		return _write_png(buffer.raw, width, height, row_stride), None
	except Exception as e:
		log.error("offlineAI OCR: screen capture failed: %s" % e)
		return None, str(e)


def _write_png(bgr_bytes, width, height, row_stride):
	"""Write 24-bit BGR bytes (top-down, padded rows) to a PNG file."""
	import zlib
	import struct as _struct

	def chunk(tag, data):
		c = _struct.pack(">I", len(data)) + tag + data
		crc = zlib.crc32(tag + data) & 0xffffffff
		return c + _struct.pack(">I", crc)

	mv = memoryview(bgr_bytes)
	line = width * 3
	raw = bytearray((line + 1) * height)  # one filter byte (0) + RGB per row
	for y in range(height):
		row = mv[y * row_stride: y * row_stride + line]
		base = y * (line + 1) + 1
		# BGR -> RGB
		raw[base:base + line:3] = row[2::3]
		raw[base + 1:base + line:3] = row[1::3]
		raw[base + 2:base + line:3] = row[0::3]

	sig = b"\x89PNG\r\n\x1a\n"
	ihdr = _struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
	idat = zlib.compress(bytes(raw), 6)
	fd, path = _mktemp(".png", "offlineAI_ocr_")
	os.close(fd)
	with open(path, "wb") as f:
		f.write(sig)
		f.write(chunk(b"IHDR", ihdr))
		f.write(chunk(b"IDAT", idat))
		f.write(chunk(b"IEND", b""))
	return path


import threading as _threading

# Track live subprocesses so windows can terminate them on close.
_active_procs = set()
_cancelled = set()
_active_lock = _threading.Lock()


def _register(proc):
	with _active_lock:
		_active_procs.add(proc)


def _unregister(proc):
	with _active_lock:
		_active_procs.discard(proc)


def cancel_all():
	"""Terminate any running OCR subprocesses. Called when a window closes so the
	model does not keep running in the background."""
	with _active_lock:
		procs = list(_active_procs)
	for p in procs:
		try:
			if p.poll() is None:
				_cancelled.add(p)
				p.terminate()
				try:
					p.wait(timeout=3)
				except Exception:
					p.kill()
		except Exception as e:
			log.warning("offlineAI OCR: could not terminate process: %s" % e)
	with _active_lock:
		_active_procs.clear()


def run_vision(image_path, store_dir, prompt, n_threads=None, timeout=1200,
               n_predict=OCR_MAX_TOKENS, n_batch=512, max_long_side=1600,
               use_mmap=False):
	"""Run the vision model on an image with a prompt. Returns (text, err).
	The subprocess is registered so it can be terminated if the window closes.
	n_predict caps output length (OCR rarely needs more than a couple hundred
	tokens); n_batch speeds prompt processing; n_threads uses more cores;
	max_long_side downscales oversized images to cut vision tokens (big speedup);
	use_mmap=False loads the model into RAM instead of memory-mapping it, which is
	lighter on disk churn and page-outs on most systems.
	"""
	if not engine_present():
		return None, _("The OCR engine is not installed.")
	backbone = os.path.join(store_dir, OCR_MODEL["backbone"])
	mmproj = os.path.join(store_dir, OCR_MODEL["mmproj"])
	if not (os.path.isfile(backbone) and os.path.isfile(mmproj)):
		return None, _("The OCR model has not been downloaded yet.")
	if not os.path.isfile(image_path):
		return None, _("Image file not found.")

	# Downscale an oversized image first — the single biggest speed lever, since
	# vision tokens scale with pixel count.
	scaled_path, scaled_is_temp = (image_path, False)
	if max_long_side:
		scaled_path, scaled_is_temp = _downscale_png(image_path, max_long_side)

	# Default to using all logical CPUs when the caller didn't specify.
	if not n_threads or n_threads <= 0:
		from . import store
		n_threads = store.autoThreads()

	cmd = [
		MTMD_EXE,
		"-m", backbone,
		"--mmproj", mmproj,
		"--image", scaled_path,
		"-p", prompt,
		"-t", str(int(n_threads)),
		"-n", str(int(n_predict)),
		"-b", str(int(n_batch)),
		"-c", str(VISION_CONTEXT),
	]
	if not use_mmap:
		cmd.append("--no-mmap")
	proc = None
	try:
		# Shared DLLs live in the tts folder; expose both on PATH.
		env = dict(os.environ)
		env["PATH"] = (OCR_DIR + os.pathsep + TTS_DIR + os.pathsep +
		               env.get("PATH", ""))
		proc = subprocess.Popen(
			cmd, cwd=TTS_DIR, env=env,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			startupinfo=_startupinfo())
		_register(proc)
		try:
			out_b, err_b = proc.communicate(timeout=timeout)
		except subprocess.TimeoutExpired:
			proc.kill()
			proc.communicate()
			return None, _("OCR timed out.")
	except Exception as e:
		log.error("offlineAI OCR: subprocess failed: %s" % e)
		return None, str(e)
	finally:
		if proc is not None:
			_unregister(proc)
		# Remove the temporary downscaled image if we created one.
		if scaled_is_temp and scaled_path != image_path:
			try:
				os.remove(scaled_path)
			except OSError:
				pass

	# If the process was terminated (window closed), report cancellation.
	if proc in _cancelled or (proc.returncode is not None and proc.returncode < 0):
		_cancelled.discard(proc)
		return None, _("OCR was cancelled.")

	out = (out_b or b"").decode("utf-8", "replace").strip()
	if not out:
		err = (err_b or b"").decode("utf-8", "replace")[-400:]
		log.warning("offlineAI OCR: no output: %s" % err.strip())
		return None, _("No text was produced. See the NVDA log for details.")
	return out, None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")


def grab_clipboard_image():
	"""Return (path, is_temp, err). Extracts an image from the Windows
	clipboard, whether it is a bitmap (e.g. from Win+Shift+S) or a copied image
	file (from Explorer). is_temp tells the caller whether the file is ours to
	delete; a user's own file is never deleted."""
	try:
		import wx
	except Exception:
		return None, False, "clipboard access unavailable"
	try:
		if not wx.TheClipboard.Open():
			return None, False, _("Could not open the clipboard.")
		try:
			if wx.TheClipboard.IsSupported(wx.DataFormat(wx.DF_BITMAP)):
				bmpData = wx.BitmapDataObject()
				if wx.TheClipboard.GetData(bmpData):
					bmp = bmpData.GetBitmap()
					if bmp and bmp.IsOk():
						img = bmp.ConvertToImage()
						fd, path = _mktemp(".png", "offlineAI_clip_")
						os.close(fd)
						img.SaveFile(path, wx.BITMAP_TYPE_PNG)
						return path, True, None
			if wx.TheClipboard.IsSupported(wx.DataFormat(wx.DF_FILENAME)):
				fileData = wx.FileDataObject()
				if wx.TheClipboard.GetData(fileData):
					for f in fileData.GetFilenames():
						if f.lower().endswith(IMAGE_EXTS):
							return f, False, None
		finally:
			wx.TheClipboard.Close()
	except Exception as e:
		log.warning("offlineAI OCR: clipboard failed: %s" % e)
		return None, False, str(e)
	return None, False, _("No image was found on the clipboard.")


class PdfOcrJob(object):
	"""Manages OCR of a PDF: renders pages on demand, caches recognized text per
	page, and supports processing a single page or a batch. All heavy work is
	driven by the caller from a worker thread; this class is just state + steps.
	"""

	def __init__(self, pdf_path, store_dir, lang_hint=None, n_threads=None,
	             max_long_side=1600, use_mmap=False):
		self.pdf_path = pdf_path
		self.store_dir = store_dir
		self.lang_hint = lang_hint
		self.n_threads = n_threads
		self.max_long_side = max_long_side
		self.use_mmap = use_mmap
		self.page_count = pdf_page_count(pdf_path) or 0
		# page number -> text (or None if not yet processed)
		self.results = {}

	def is_done(self, page):
		return page in self.results and self.results[page] is not None

	def get_text(self, page):
		return self.results.get(page)

	def process_page(self, page):
		"""Render + OCR a single page. Returns (text, err) and caches it."""
		if self.is_done(page):
			return self.results[page], None
		# Render adaptively so the page long side is about max_long_side pixels.
		pages, err = render_pdf_pages(self.pdf_path, page, page, scale=None,
		                              max_long_side=self.max_long_side)
		if err:
			return None, err
		if not pages:
			return None, _("Could not render page.")
		_pno, png = pages[0]
		try:
			# The render already capped resolution, so skip the extra downscale.
			text, err = run_vision(
				png, self.store_dir, ocr_prompt_for(self.lang_hint),
				n_threads=self.n_threads, max_long_side=0,
				use_mmap=self.use_mmap)
		finally:
			try:
				os.remove(png)
			except OSError:
				pass
		if err:
			return None, err
		self.results[page] = text
		return text, None

