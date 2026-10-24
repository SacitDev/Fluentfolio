# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

## 0.1.1

- Updates filelock to 4.2.1, opentelemetry-api to 1.45.1 and setuptools to 84.1.0 to address five published security advisories. The matching OpenTelemetry SDK and semantic conventions are updated with them. The usual two-week wait for other new package releases stays in place.
- Updates filelock and setuptools in the optional MFA aligner's Windows and Linux environments too. An older environment is shown as needing an update and cannot run until it is reinstalled from Setup, or with `fluentfolio components install mfa`. Reinstallation needs an internet connection and downloads the environment's packages again; it does not remove recordings, analyses, settings or the downloaded speech models.
- Install.cmd and Update.cmd now start Windows PowerShell with its own module path, also from a PowerShell 7 window. Before, Windows PowerShell took the modules of PowerShell 7 there. Two commands that the installer needs were then missing, so a new install stopped with an error.
- When the installer uses the other folder it offers (if the usual one cannot be used), it makes that folder readable by you only. That step now also removes an "OWNER RIGHTS" entry, if the folder has one. The entry gives nobody else access, but before, the step counted it as another account and failed. Such an entry was seen only in tests on GitHub's Windows Server, on folders made inside a folder that Python had made private. A new folder at the top of a drive, where the installer's other folder goes, was not affected.

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
