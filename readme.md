# Offline AI

* Author: Riad Assoum
* Repository: <http://github.com/riadassoum/offlineai>
* NVDA compatibility: 2026.1 and later (64-bit)
* License: GNU GPL version 2

Offline AI brings local artificial intelligence to NVDA. Language models, speech recognition, image reading and voice synthesis all run on your own processor. There is no separate Python to install, no command line, no account, and not one byte of your text, audio or images leaves the computer. The internet is used only when you choose to download a model.

The add-on is built for screen reader users of every language. It follows the language NVDA is set to, lets you translate into any language, recognises speech in about 100 languages, and its whole interface can be translated.

## What it does

* **Chat with local language models.** Download GGUF models from a built-in catalogue (Qwen, Llama, Gemma, Phi, Mistral, DeepSeek-R1 distillations, multilingual models such as Gemma 3, Aya Expanse and EuroLLM, a dedicated translation model, coding models, Arabic models), search HuggingFace from inside the add-on, or load any GGUF file you already have. Answers stream into a read-only box you can read while they are being written.
* **Quick actions on selected text.** Translate, summarise, proofread or explain whatever is selected in any application, with a keystroke. Up to ten custom actions with your own prompts.
* **Dictation.** Press a key, speak, press it again: the text is typed where you want it.
* **Transcription.** Turn audio and video files into text, SRT or VTT subtitles, one file or many at once, and play back the audio behind any line of the transcript.
* **OCR and image description.** Read the screen, image files, clipboard images and multi-page PDFs with a vision model, and ask the chat model questions about the result.
* **Text to speech with voice cloning.** Speak text or read long documents aloud with an AI voice, optionally imitating a short sample of someone's voice.

## Getting started

1. Open the NVDA menu, Tools, **Offline AI...**.
2. Choose a model in the Model list. Each entry says how big the download is and how much memory it needs. If you are unsure, start with a small one (0.5B to 2B); for languages other than English, the entries in the **Multilingual** category are the best choice, and "Gemma 3 4B" is a good all-rounder.
3. Press **Download model**. A progress window reports percentage, speed and time remaining, and announces every ten percent. If the connection drops, press Download again and it continues from where it stopped.
4. Type a question in **Your query** and press **Send**.

No keys are assigned by default, so nothing can conflict with your other add-ons. Assign the ones you want in NVDA menu, Preferences, Input Gestures, under the **Offline AI** category. The **Quick Launcher** command is a good first one: it opens a searchable list of everything the add-on can do.

## Your language

* **Interface.** The add-on's messages and windows are translatable like any NVDA add-on (see "Translating the add-on" below). Language names in every list are shown in your own language.
* **My language.** In NVDA menu, Preferences, Settings, Offline AI, "My language" decides where "translate to my language" translates to, and which language summaries, explanations and image descriptions are written in. It starts as "Same as NVDA".
* **Translation commands.** "Translate selected text to my language", "Translate selected text to my second language" (English unless you change it) and "Translate selected text to a language chosen from a list" (any of about 100 languages). Long selections are translated piece by piece, so they are not cut off.
* **Speech recognition.** All of Whisper's languages are available, plus automatic detection.
* **OCR.** The language hint accepts any language, or automatic.
* **AI voice.** The voice model itself speaks ten languages: English, Chinese, German, Italian, Portuguese, Spanish, Japanese, Korean, French and Russian. This is a limit of the model, not of the add-on.
* **Which model for which language?** Small models are usually good at only one or two languages. For reliable work in another language, or for translation, use a model from the Multilingual or Translation category, or search HuggingFace for one trained on your language. How well a model handles a language depends on the model, not on the add-on.

## The chat window

* **Model** and **Download model / Delete model file.** Deleting removes the file from the models folder to free disk space; you can download it again at any time. Models you added yourself can also be removed from the list.
* **Persona / preset.** Ready-made system prompts, including a translator into your language and "Answer in my language". "Custom..." saves your current system prompt as a new preset.
* **Settings...** Generation options: system prompt, temperature, top-p, top-k, min-p, repeat penalty, answer length, context size, threads, batch size, seed, flash attention, memory locking and memory-mapping, streaming, the working tone, how reasoning ("think") blocks of models like DeepSeek-R1 are handled, how quick actions deliver their result, custom actions, backup and restore, and a speed benchmark.
* **Models folder...** Where downloads are stored (by default `Documents\offlineAI_models`).
* **Add custom GGUF..., Search HuggingFace..., Refresh catalog.** Ways to add models. For models added this way the prompt format defaults to "auto", which uses the chat template stored in the model file.
* **Attach context file...** Attach any text-based file to talk about: documents, logs, data, and source code in any common programming language. Files in UTF-8, UTF-16 or your system's own encoding are read correctly. If a file is longer than the model's context, the first part is used and you are told.
* **Copy code block...** Copies just the code from the latest answer.
* **Chat history..., Export chat..., New chat.** Conversations are saved automatically.
* In the answer box, Control+Up and Control+Down move by paragraph.

## Quick actions

Select text anywhere, then use one of these commands (assign keys in Input Gestures):

* Translate selected text to my language
* Translate selected text to my second language
* Translate selected text to a language chosen from a list
* Summarize selected text
* Proofread and fix grammar of selected text
* Explain selected code or error
* Run custom quick action 1 to 10
* Speak selected text aloud with the voice model (press again to stop)
* Ask AI about the text on the clipboard
* Stop the current task

The result opens in a window that NVDA reads, that you can review with the normal reading keys, copy from, and close with Escape. In Settings you can choose to have results copied to the clipboard instead, or both. The model used is the one you last chatted with. To save memory it is released after ten idle minutes (changeable in NVDA's settings).

## Dictation

Assign a key to **Toggle live dictation**. Press it, speak, press it again. A rising tone means recording, a falling tone means the recording is being transcribed, a high tone means the text was inserted. **Cancel live dictation** discards a recording.

In NVDA menu, Preferences, Settings, Offline AI you can choose:

* **Dictation language**: automatic, same as NVDA, or a specific language. Choosing your language is faster and more accurate than automatic detection.
* **Dictation model**: by default the best downloaded model that is still quick enough. "Whisper Large v3 Turbo (compressed)" is the recommended one.
* **Insert dictated text into**: wherever the focus is when the text is ready; or the window you dictated in, even if you have moved to another window meanwhile (the text is inserted there and you are returned to where you were); or only the clipboard.
* **Warning before inserting**: a number of seconds, announced with a double beep, before the text is typed.
* **Faster transcription of short dictations**: makes short notes several times quicker.
* **Restore my clipboard after inserting**.

You can dictate several notes in a row without waiting; they are transcribed in order. Dictation also works while a batch transcription is running, each with its own copy of the model.

**Privacy:** recordings are never written to disk. Audio is kept in memory only until it has been transcribed, then discarded.

## Transcribing files

Open NVDA menu, Tools, **Offline AI - Transcribe speech...**.

1. Choose a Whisper model and download it.
2. Choose the spoken language, or leave it on automatic.
3. Press **Open audio/video file(s)...**.
   * One file: press **Transcribe**. The transcript appears in the box. Press Space or Enter on any part of it to hear that part of the recording; select text first to hear exactly that range. Save it as TXT, SRT or VTT.
   * Several files: press **Transcribe**, choose the output format, and a transcript file is written next to each original with the same name. The box lists each file as it is done and reports any that were skipped, with the reason. The **Batch transcribe to files...** button does the same in one step.
4. **Translate to English while transcribing** makes Whisper produce English text directly.

MP3, FLAC, OGG and WAV work out of the box. M4A, AAC, WMA, Opus and video files need FFmpeg installed and on the PATH; you are told when a file needs it.

## OCR and image reading

Open NVDA menu, Tools, **Offline AI - OCR and image reading...**, and download the vision model once (about 2.7 GB). Then you can OCR the screen, an image file, the image on the clipboard, or a PDF; or get a description of an image, written in your language. **Ask AI about this text...** sends the recognised text to the chat window as context.

The PDF reader lets you OCR the current page or a range of pages, stop a range part-way, move between pages, and save all recognised text.

The same functions are available as commands without opening the window: OCR the whole screen, OCR an image file, describe an image file, OCR the clipboard image, describe the clipboard image.

## Text to speech and voice cloning

Open NVDA menu, Tools, **Offline AI - Text to speech...**, and download the voice model once (about 1.5 GB). Type or paste text, choose the language, and press **Speak**, **Save as WAV file...**, or **Read document (streaming)**, which speaks sentence by sentence and can be paused and stopped. **Add voice from file...** makes a named voice from a few seconds of clear speech in a WAV or MP3 file.

## Settings in NVDA's Settings dialog

NVDA menu, Preferences, Settings, **Offline AI** holds the options shared by everything: your languages, dictation options, memory-mapping, CPU threads, how long an unused model stays in memory, OCR speed versus quality, AI voice speed, audio cue volume and mute, and table formatting.

## Where things are stored

* **Models:** `Documents\offlineAI_models` (changeable). They are never touched by updating or removing the add-on. Delete them from the add-on's windows or simply delete the files.
* **Settings, presets, custom actions, voice list:** `offlineAI\config.json` in NVDA's user configuration folder.
* **Chat history:** `offlineAI\chat_history.json` in the same place.
* **Temporary files:** a `temp` folder inside the models folder, emptied automatically.

Settings and history therefore survive add-on updates. The Settings window can export everything to one zip file and import it on another computer.

## Hardware notes

Everything runs on the CPU. A model needs roughly its file size in free memory, plus a little. With memory-mapping on (the default), Windows can run a model that is somewhat larger than your free memory by paging it from disk, at reduced speed. The benchmark in Settings tells you what size of model your processor handles comfortably.

## Translating the add-on

The file `offlineAI.pot` in the repository contains every message. Copy it to `addon/locale/<your language code>/LC_MESSAGES/nvda.po`, translate it with Poedit or any text editor, and send it as a pull request. Prompts sent to the models are deliberately not translatable: they are instructions for the model, and they already name the language the answer must be in.

## Building from source

Install Python 3.13, then `pip install scons markdown` and GNU gettext, and run `scons` in the repository root. This produces `offlineAI-<version>.nvda-addon`. `scons pot` regenerates the translation template.

The bundled engines in `addon/globalPlugins/offlineAI/lib` are prebuilt Windows x64 binaries of llama.cpp (through llama-cpp-python, and the `llama-tts` and `llama-mtmd-cli` programs), whisper.cpp (through pywhispercpp), NumPy, miniaudio, sounddevice/PortAudio and pypdfium2. See `THIRD_PARTY_NOTICES.md`.
