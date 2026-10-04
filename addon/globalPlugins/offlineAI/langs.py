# -*- coding: utf-8 -*-
# Language tables for the Offline AI add-on.
#
# The add-on must be usable in any language, so nothing here assumes English or
# Arabic. Codes are ISO 639-1 where one exists (the same codes Whisper uses).
# English names are kept because they are what goes inside prompts sent to the
# models (models follow "Translate into Portuguese" far more reliably than a
# bare code); the names shown to the user come from NVDA/Windows in the user's
# own language whenever available.

import addonHandler
addonHandler.initTranslation()

# code -> English name. This is Whisper's full language list.
LANGUAGES = {
	"af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
	"az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
	"bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
	"ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
	"de": "German", "el": "Greek", "en": "English", "es": "Spanish",
	"et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
	"fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
	"ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
	"hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian",
	"hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian",
	"ja": "Japanese", "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh",
	"km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin",
	"lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian",
	"lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian",
	"ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay",
	"mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
	"nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan",
	"pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
	"ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi",
	"si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona",
	"so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese",
	"sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
	"tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog",
	"tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu",
	"uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba",
	"yue": "Cantonese", "zh": "Chinese",
}

# NVDA/Windows locale codes that differ from the codes above.
_ALIASES = {"nb": "no", "iw": "he", "in": "id", "jv": "jw", "fil": "tl",
            "ckb": "ku", "kmr": "ku"}


def normalize(code):
	"""'pt_BR' / 'pt-BR' / 'PT' -> 'pt'. Returns '' for unusable input."""
	if not code or not isinstance(code, str):
		return ""
	base = code.replace("-", "_").split("_")[0].strip().lower()
	return _ALIASES.get(base, base)


def nvdaLanguage():
	"""The base code of the language NVDA's interface is using ('' if unknown)."""
	try:
		import languageHandler
		return normalize(languageHandler.getLanguage())
	except Exception:
		return ""


def englishName(code):
	"""Name used inside prompts. Unknown codes are returned unchanged so a user
	can type any language name by hand in the config."""
	c = normalize(code)
	return LANGUAGES.get(c) or (code or "")


def displayName(code):
	"""Name shown to the user, in the user's language when NVDA knows it."""
	c = normalize(code)
	name = None
	try:
		import languageHandler
		name = languageHandler.getLanguageDescription(c)
	except Exception:
		name = None
	if not name or name == c:
		name = LANGUAGES.get(c, code)
	return name


def sortedCodes(codes=None, pinned=()):
	"""Codes sorted by their displayed name, with the `pinned` ones first (the
	user's own language, usually), without duplicates."""
	codes = list(codes if codes is not None else LANGUAGES.keys())
	head = []
	for p in pinned:
		p = normalize(p)
		if p and p in codes and p not in head:
			head.append(p)
	rest = sorted((c for c in codes if c not in head),
	              key=lambda c: displayName(c).casefold())
	return head + rest


def choiceList(codes=None, includeAuto=True, pinned=None):
	"""Build (codes, labels) for a wx.Choice. With includeAuto the first entry
	has the code 'auto'."""
	if pinned is None:
		pinned = (nvdaLanguage(), "en")
	ordered = sortedCodes(codes, pinned)
	outCodes, labels = [], []
	if includeAuto:
		outCodes.append("auto")
		# Translators: language choice meaning "detect the language automatically".
		labels.append(_("Automatic (detect)"))
	for c in ordered:
		outCodes.append(c)
		labels.append(displayName(c))
	return outCodes, labels


def primaryTranslationLanguage(settings):
	"""The user's 'my language' translation target: the configured one, else
	NVDA's interface language, else English."""
	c = normalize(settings.get("translate_lang_1") or "") or nvdaLanguage()
	return c or "en"


def secondaryTranslationLanguage(settings):
	c = normalize(settings.get("translate_lang_2") or "") or "en"
	# If both would be the same, fall back to something useful.
	if c == primaryTranslationLanguage(settings):
		c = "en" if c != "en" else (nvdaLanguage() if nvdaLanguage() not in ("", "en") else "fr")
	return c


def translationInstruction(code):
	"""The (English) instruction given to the model for a translation."""
	name = englishName(code)
	return ("Translate the following text into %s. Preserve the meaning, tone "
	        "and formatting. Output only the translation, with no explanation, "
	        "notes or quotation marks." % name)
