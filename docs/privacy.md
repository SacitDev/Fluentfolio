# Privacy

Fluentfolio analyses your recordings on your own computer. It has no account, no telemetry and no crash upload, and it sends nothing about you or your recordings anywhere.

## What stays on your computer

Everything an analysis makes: the recording you chose, the transcripts, the measures, the exports and the logs. An analysis makes no network calls at all.

If you add your own transcript, Fluentfolio keeps a copy in which the names found in its speaker labels are replaced. It keeps those names in a list beside that copy, so that the answer check can find them in an AI chat's answer. The list is never exported. Deleting the analysis deletes it; an analysis made again from it in another mode has its own copy, which goes when you delete that one.

Fluentfolio keeps two folders:

- The **data folder** holds your analyses, your settings and the logs. On Windows it is `%LOCALAPPDATA%\Fluentfolio-data`, on Linux `~/.local/share/fluentfolio-data`.
- The **app folder** holds the program and the downloaded models. On Windows it is `%LOCALAPPDATA%\Fluentfolio`, on Linux `~/.local/share/fluentfolio`.

You can move the data folder and the models folder in Settings, on the Advanced tab, under Folders. On Windows, a moved folder keeps the access rights of its new place, such as a second drive, so other accounts on the computer may be able to open it ([SECURITY.md](../SECURITY.md), "Known risks"). The app's pages write the start of a path in your user folder as `%LOCALAPPDATA%` or `%USERPROFILE%` (on Linux `~`), so a screenshot does not show your user name. The command line prints full paths, for scripts and AI agents.

The app opens in your browser, but it is served by your own computer at `127.0.0.1`, which other computers cannot reach. It opens through a link with a secret key, kept in a file that only your user account can read, and your browser is told to keep no copy of its pages.

## When Fluentfolio uses the internet

Only when you ask it to:

- **Downloading parts.** The installer downloads uv, Python 3.12 and the program's packages from github.com, release-assets.githubusercontent.com, releases.astral.sh, pypi.org, files.pythonhosted.org, download.pytorch.org and download-r2.pytorch.org. Before it asks you to continue, it also checks that it can reach three sites for the setup page's downloads: huggingface.co, cas-bridge.xethub.hf.co and conda.anaconda.org. The setup page, later called Components, downloads the speech models and tools from huggingface.co and github.com, and the MFA aligner from conda-forge and github.com. Each model file is checked against a checksum fixed in Fluentfolio when it is downloaded, before it is kept, and the MFA aligner is installed from a lock file that names every package with its checksum.
- **Your Hugging Face token.** When you save or check the token, it is sent to huggingface.co to see whether it works and whether you accepted the pyannote model's terms.
- **Check for updates.** On the Help page, this asks GitHub for the newest version number. Nothing about you is sent.

An analysis never downloads anything. If a part is missing, it says so and runs without it, or stops and tells you which part to install.

## Your Hugging Face token

Fluentfolio saves the token with Hugging Face's own tool, in Hugging Face's standard place in your user profile, not in Fluentfolio's folders. It never writes the token into a log, an export, a diagnostic report or the settings. Removing the token on the token page removes only a token that Fluentfolio saved itself.

## Logs and diagnostic reports

Logs hold counts, durations, step names and error codes. They never hold the words of a transcript, file names or the names of people in a meeting. The diagnostic report replaces your folders with placeholders such as `<DATA>` and removes tokens and your user name. Read a report before you share it.

## What you share with an AI chat

Copy for AI and the export hold your transcript: the words you said and, in interview and role-swap recordings, the other person's words too. When you paste them into an AI chat, or upload your recording, that chat's service receives them under its own terms. Exports use neutral labels such as "the candidate", "the examiner" or "Person 1", never your file names. If you add your own transcript, the names in its speaker labels, such as "Anna Smith:", are replaced by neutral labels where they are written as in the label. Some labels, a name written in lower case and some rarer forms of a name, such as "Anna've", are not seen; see [limitations](limitations.md).

## The other person in a recording

A recording of two people holds the other person's voice and words. Fluentfolio analyses it only after you confirm that the other person agreed to be recorded and analysed on this computer. Before an audio export, and beside the copy buttons, it reminds you that the file holds their voice. Share it only if they agreed to that too.

## Deleting

- **Delete an analysis** (on its page, or `fluentfolio jobs delete JOB`) removes its folder, with the recording copy, the transcripts, the results and the exports in it, and its temporary files.
- **Delete all my data** (at the bottom of History) removes every analysis, both temporary folders, the logs and the openSMILE licence consent. The models, the token and your settings stay; each has its own place to remove it.
- **Keep analyses for (days)**, in Settings on the Advanced tab, deletes older analyses when the next one starts. It is 0 at first, which keeps every analysis.

Deleting skips the Recycle Bin. It is not secure erasure: on an SSD, or in a folder that OneDrive or Dropbox syncs, copies can remain. Fluentfolio warns you if the data folder is in such a synced folder. Files you saved or downloaded elsewhere, such as an export you copied to your Documents, are outside Fluentfolio's reach: delete those yourself.

Uninstalling removes the app folder and keeps your data folder, a models folder you moved out of the app folder and the token unless you say yes to removing them.
