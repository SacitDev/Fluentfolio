# Security

## Reporting a problem

Please report security problems privately, not in a public issue. On this repository's GitHub page, open the **Security** tab and choose **Report a vulnerability**. Describe what you found and how to reproduce it. Never attach recordings, transcripts or tokens.

Fluentfolio is a free, open-source project. It can be maintained and improved by its authors, or by anyone under its MIT licence, but nobody promises an answer, a fix or a time frame. As the [LICENSE](LICENSE) says, it comes without any warranty.

If a fix is made, it comes in a new release.

## What is in scope

Fluentfolio's own code and its installer, in particular:

- **The local web server.** It listens only on `127.0.0.1`. A launch link with a random key, kept in a file in your data folder, sets a session cookie (HttpOnly, SameSite=Strict). Requests with another host name, or a cross-site origin, are refused. Pages carry a strict content policy and are not cached.
- **Downloads.** Only over https. Every model file is checked against the SHA-256 pinned in `src/fluentfolio/data/models.toml`, and Hugging Face models are pinned to a fixed revision. An archive that would write outside its folder, or unpack far larger than expected, is refused. The MFA environment is built from a lock file that names every package with its checksum.
- **The Hugging Face token.** It is saved and removed only through Hugging Face's own library. You type it into a hidden field, in the app or at a hidden prompt, or pass it on standard input. It is never written to a log, an export, a diagnostic report or a child process's environment.
- **Child processes.** Each step that runs a model or an outside tool runs in its own process, started without a shell, with an absolute program path and an environment built for it.
- **Privacy leaks** of recordings, transcripts, names or paths into logs, exports or reports.

## What is out of scope

- Malware or another person already running as your user, or as an administrator, on your computer. Nothing a local app does can stop them.
- Theft of a computer without disk encryption. Encrypting the disk is a choice in your operating system.
- What an AI chat's company does with text or audio you upload to it.
- Crash reports that your operating system sends by itself.
- Vulnerabilities in the third-party parts (Python packages, models, MFA, FFmpeg). Please report those to their own projects. If Fluentfolio uses one in an unsafe way, that is in scope.

## Known risks

These are known, and this release leaves them as they are:

- **Another program on your own computer.** Browsers do not keep cookies apart by port, so a different local web app that you open in the same browser could receive Fluentfolio's session cookie. This is treated like local malware.
- **A data or models folder you move on Windows.** The default folders are inside your own user profile, or in a folder the installer makes readable by you only. If you move one in Settings to another place, such as a second drive, it keeps the permissions of that place. Other accounts on the computer may then be able to read your recordings, or change the model files, which run as code when they load. On Linux, Fluentfolio makes its files readable by your user only, wherever they are.
- **The pyannote model files are Python pickles**, a format that can run code when loaded. They are pinned to a fixed revision, and their SHA-256 is checked after each download. Before each analysis only their size is checked.
- **The MFA environment is not scanned for known vulnerabilities**, because the scanner does not read its lock format. It runs offline, on local files only.
- **Older copies of pillow and certifi in the MFA environment.** On 20 October 2026, nine flaws were published for the image library pillow 12.3.0, and the certificate list certifi 2026.7.22 was found to trust a certificate authority whose keys leaked. The app's own packages use the fixed versions, pillow 13.0.0 and certifi 2026.10.19. The environment of the optional MFA aligner keeps its own pillow 12.3.0 and certifi 2026.7.22, and that environment is not scanned (see above). Fluentfolio opens only audio and text files and never gives pillow an image.
- **The other person's agreement is your statement**, given by the tick in the app or by an option on the command line. Fluentfolio records it but cannot prove it.
- **Prompt injection is contained, not prevented.** A recording can contain words that look like instructions. The prompt tells the AI chat to treat everything except the prompt as data, and the agent guide tells AI agents the same, but no prompt can guarantee it.
- **Antivirus false alarms** can still happen, because the install scripts and the speech tools are not code-signed.
