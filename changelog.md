# Offline AI 1.0.2

- Added a complete Arabic interface, installation description, user guide and changelog, with right-to-left HTML help.
- Fixed lost leading punctuation and punctuation-only text when splitting long selections for translation or proofreading. Token truncation now respects the budget even for a single multi-token character.
- Prevented idle cleanup, shutdown and model deletion from closing a native model while a request is using it. Shutdown cleanup is deferred until the active request releases the engine.
- Validated backup manifests before import and allowed restoration into a new configuration directory. Backup content labels and quick-action reasoning cues are now translatable.
- Regenerated the translation template, including the previously missing cancelled-download message.
- Merged Dependabot updates for actions/download-artifact v8 (#4) and softprops/action-gh-release v3 (#3).

GPU/NPU acceleration remains unsupported by the bundled CPU engines; enhancement request #8 remains open. Arabic speech recognition and language-model tasks are available, but the bundled speech-synthesis model does not support Arabic speech output.

Validation: 11 platform-independent regression tests passed. Arabic translation placeholders, catalog completeness, compiled translations and packaged documentation were checked. The add-on was built with SCons. Interactive Windows/NVDA and native engine execution were not tested in this Linux environment.

Thanks to @animeshahilya for the earlier download and hardware-documentation contributions, and Dependabot for the build updates.

## 1.0.1

- Cancelled model downloads retain partial files and can resume when the server supports range requests (#7).
- Clarified CPU, GPU and NPU hardware support (#6).
- Updated build automation dependencies (#2, #5).
