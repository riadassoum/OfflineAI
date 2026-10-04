# Third-party components

Offline AI itself is licensed under the GNU General Public License, version 2 (see `COPYING.txt`).

The add-on bundles the following components, unmodified, under `addon/globalPlugins/offlineAI/lib`. Each remains under its own license; the licenses below are all compatible with distribution alongside GPL software.

| Component | Used for | License |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama.dll`, `ggml*.dll`, `mtmd.dll`, `llama-tts.exe`, `llama-mtmd-cli.exe`) | Language models, vision, speech synthesis | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Python bindings for llama.cpp | MIT |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Speech recognition | MIT |
| [pywhispercpp](https://github.com/absadiki/pywhispercpp) | Python bindings for whisper.cpp | MIT |
| [NumPy](https://numpy.org) | Audio buffers | BSD 3-Clause |
| [miniaudio](https://miniaud.io) and [pyminiaudio](https://github.com/irmen/pyminiaudio) | Decoding MP3, FLAC, OGG, WAV | Public domain / MIT No Attribution, MIT |
| [python-sounddevice](https://github.com/spatialaudio/python-sounddevice) and [PortAudio](https://www.portaudio.com) | Microphone capture and playback | MIT |
| [cffi](https://cffi.readthedocs.io) and [pycparser](https://github.com/eliben/pycparser) | Required by the audio libraries | MIT, BSD 3-Clause |
| [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) and [PDFium](https://pdfium.googlesource.com/pdfium/) | Rendering PDF pages | Apache-2.0 or BSD 3-Clause |
| [Jinja2](https://palletsprojects.com/p/jinja/) and [MarkupSafe](https://palletsprojects.com/p/markupsafe/) | Chat templates stored in model files | BSD 3-Clause |
| [diskcache](https://github.com/grantjenks/python-diskcache) | Required by llama-cpp-python | Apache-2.0 |
| [typing_extensions](https://github.com/python/typing_extensions), `pickletools` | Python standard-library backports | PSF License |
| [platformdirs](https://github.com/tox-dev/platformdirs) | Required by pywhispercpp | MIT |
| Microsoft Visual C++ and OpenMP runtime DLLs (`msvcp140`, `vcomp140`, `libomp`) | Runtime support for the above | Microsoft redistributable terms / Apache-2.0 with LLVM exception |

Models are **not** part of the add-on. They are downloaded by the user from their publishers (mostly through HuggingFace) and each has its own license, shown on its HuggingFace page. Check a model's license before using it commercially.

The build system (`sconstruct`, `site_scons`) comes from the [NVDA add-on template](https://github.com/nvaccess/AddonTemplate), GPL v2.
