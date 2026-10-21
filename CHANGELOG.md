# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## 0.1.0

The first release.

- Analyses a recorded IELTS Speaking practice test on your own computer, on Windows 10 22H2 or later, Windows 11, and Ubuntu 22.04 or 24.04.
- Three modes (solo, interview, role swap) and three recording layouts (one recording of everyone, one track per person, two files).
- Speech recognition with Parakeet (the default) or Whisper; speaker separation with pyannote (with your own Hugging Face token) or sherpa-onnx (no account).
- Finds the questions, the three Parts and who asks and who answers, with a review form to correct them.
- Measures fluency, hesitation, linking words, vocabulary, grammar pointers, pitch and loudness. The MFA aligner is experimental and off by default, because it did not pass its accuracy check. The pronunciation check did not pass its accuracy check either: it is on by default, marked experimental, and its words to check go to the AI only as hints.
- Copy for AI: one text with a grading prompt for any AI chat, and an optional short audio file for chats that can listen.
- A web app in your browser and a `fluentfolio` command for scripts and AI agents, with JSON output, an answer checker and a guide for agents.
- An installer that needs no administrator rights, with update and uninstall.
