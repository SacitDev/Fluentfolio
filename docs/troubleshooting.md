# Troubleshooting

Every problem Fluentfolio reports has a code, such as `E_INPUT_TOO_SHORT`. In the app the code is at the bottom of the error card. On the command line it is in the output, and `--json` gives it as `error.code`. Find the code on this page.

Codes that start with `E_` are errors: the action did not happen. A few also appear as a note beside an action that did happen. For example, the system check shows `E_TOKEN_MISSING` on a row that does not stop an analysis, and a saved token can come with `E_HF_TERMS_NOT_ACCEPTED`. Codes that start with `W_` are warnings: the action happened, but something limits the result.

## Before anything else

- Run the system check. In the app, open Help and choose "run the system check". On the command line, run `fluentfolio doctor`. It checks the install, the disks, the computer and each part, and says what to fix.
- If you ask for help, share a diagnostic report. In the app, open "Diagnostic report" on the error card and choose "Copy diagnostic report". On the command line, run `fluentfolio doctor --report report.txt`. Read the report before you share it. Your folders appear as placeholders such as `<DATA>`, tokens and your user name are removed, and it holds no recording, transcript or recording name.

## Installing, updating and removing

The installer and the uninstaller print these codes, with a line that says what to do. The installer can also stop with five codes from other sections: `E_PATH_RULE`, `E_DISK_FULL`, `E_NETWORK`, `E_TLS_INTERCEPTED` and `E_HASH_MISMATCH`. Most of those sections are written for the app, so for the installer follow the line it prints, then run it again. When the GPU test in the installer's last step fails, it names that test's code, such as `E_CUDA_RUNTIME`, and uses the processor; with `-Device gpu` (Linux: `--device gpu`) it keeps GPU support and warns instead. Once its first checks have passed, the installer also writes its run to `install.log` in the `logs` folder of the app folder, and a stop message names that file.

### E_NOT_SUPPORTED

**This computer is not supported.** Version 1 needs Windows 10 22H2 or later, or Windows 11, or Ubuntu 22.04 or 24.04, on a 64-bit Intel or AMD processor.

What to do: use a supported computer. Computers with an ARM processor are not supported yet.

### E_ELEVATED

**Started as administrator.** The installer was started as administrator, or with sudo. Fluentfolio installs for your user only and never needs administrator rights.

What to do: start the installer again the normal way: double-click Install.cmd (on Linux, run `./install.sh`), without "Run as administrator" or sudo.

### E_SAC_ENFORCED

**Smart App Control blocks Fluentfolio.** Smart App Control is a Windows 11 security feature. When it is on, it blocks programs that are not signed by a known publisher. Fluentfolio runs Python and speech tools that are not signed that way, so it cannot work while Smart App Control is on. The installer checks this before it downloads anything, and it never changes the setting itself.

What to do: decide whether to switch Smart App Control off. It is in Windows Security, under App and browser control, Smart App Control settings. Switching it off is your own decision: Windows may not let you switch it on again without resetting Windows, so read Microsoft's page about Smart App Control first. If you keep it on, Fluentfolio cannot run on this computer.

### W_SAC_EVALUATION

**Smart App Control is in evaluation mode.** Smart App Control is watching this computer. Windows may switch it on later by itself, and then it blocks Fluentfolio.

What to do: nothing now. If Fluentfolio stops starting one day, read `E_SAC_ENFORCED` above.

### E_VCRUNTIME_MISSING

**The Visual C++ runtime is missing.** Windows lacks the Microsoft Visual C++ runtime, which Python and the speech tools need.

What to do: install "Visual C++ Redistributable x64" from Microsoft (`https://aka.ms/vs/17/release/vc_redist.x64.exe`), then run the installer again.

### E_SOURCE

**No release found.** The installer did not find the Fluentfolio files next to it. The download was not fully extracted.

What to do: extract the whole release download (on Windows, right-click the ZIP and choose "Extract All"), then start the installer inside the extracted folder.

On Linux, the installer also stops with this code when `--source` is a `.zip` and the computer has no unzip program. Then use the release file that ends in `.tar.gz`, or extract the zip first.

### E_DISK_PREFLIGHT

**Not enough disk space to install.** The disk does not have the space the install needs. Nothing was downloaded.

What to do: free the space the message names, then run the installer again.

### E_APP_RUNNING

**Fluentfolio is running.** The installer cannot change the program, and the uninstaller cannot remove it, while Fluentfolio is running.

What to do: close the Fluentfolio window (the one that says "keep this window open"), then run the installer or the uninstaller again. An analysis started in the background (with `--detach`) also counts: wait until it ends, or stop it with `fluentfolio cancel JOB` (`fluentfolio jobs list` shows the JOB name).

### E_FILE_BLOCKED

**A security program blocked a file.** A file of the install was blocked or removed right after it was written, most likely by an antivirus program.

What to do: check your security program's list of blocked files, then run the installer again.

### E_INSTALL_FAILED

**An install step failed.** A step of the install failed for a reason the installer does not know. What was already done is kept.

What to do: run the installer again. If it fails again, share `install.log` when you ask for help. Read it first: it holds folder names but no recordings.

### E_UPDATE_FAILED_OLD_KEPT

**The update failed; the old version is kept.** The new version failed before it replaced the old one. The old version is unchanged and still works.

What to do: start Fluentfolio as before. Try the update again later.

### E_NOT_INSTALLED

**Fluentfolio is not installed.** The uninstaller found no installed Fluentfolio for this user.

What to do: nothing needs to be removed. If you installed into another folder, choose **Uninstall Fluentfolio** in the Start menu, which finds its own folder, or give the folder with `-AppRoot <folder>` (Linux: `--app-root <folder>`). If an app folder is still there after an uninstall that stopped, remove the rest by hand, as `E_UNINSTALL_FAILED` says.

### E_UNINSTALL_FAILED

**The uninstall stopped.** The uninstaller stopped before it finished, for example at a file that another program has open. The text after "Problem:" says how far it got.

- **"Nothing was removed"**: Fluentfolio is still installed as it was. Do what the message said, then run the uninstaller again. If it says that your recordings folder lies inside the app folder, move that folder out first, because the uninstaller deletes the app folder with everything in it. Move it in Fluentfolio under Settings, Advanced tab. If you set `FLUENTFOLIO_HOME` to it, move it by hand and change the variable instead. If it says that a folder you keep lies inside a folder you said yes to deleting, answer no to deleting the outer folder, or first move the inner one out, in Settings as above. If the install record (`state.json` in the app folder) still cannot be read, you can remove the program by hand, as below.
- **"The uninstall stopped before removing the program"**: removing the Hugging Face token did not finish, so the program and your recordings are still there. Run the uninstaller again. To remove the program and keep the token, answer no to the token question.
- **"The uninstall stopped while removing files"**: the uninstaller had begun to delete, so part of the program and its Start-menu entries may be gone already. What was removed stays removed. The message names the file or folder that could not be deleted, often because a program still uses it. Close any program that uses it. A window opened inside the app folder counts too, such as a PowerShell window. Then remove the rest by hand, as below, together with any folder of yours that you said yes to deleting.

On Linux, the uninstaller prints only the "Nothing was removed" messages about a folder that lies inside another one. If anything else fails there, such as reading its `state.json`, removing the token or deleting the files, it stops with the failing command's own message instead. What it removed stays removed. Run it again, or remove the rest by hand.

To remove the program by hand on Windows: in File Explorer, paste the folder that holds the app folder into the address bar (for the usual app folder `%LOCALAPPDATA%`, for `C:\Fluentfolio` the drive `C:\`), then delete **Fluentfolio** there. If the Start menu still lists Fluentfolio, paste `%APPDATA%\Microsoft\Windows\Start Menu\Programs` and delete **Fluentfolio** and **Uninstall Fluentfolio** there.

To remove the program by hand on Linux: delete the app folder (usually `~/.local/share/fluentfolio`) and the menu entry `fluentfolio.desktop` (usually in `~/.local/share/applications`), for example in your file manager.

A Hugging Face token that Fluentfolio saved stays on this computer. To make it useless, delete it on huggingface.co, under Settings, Access Tokens.

## Recordings and files

### E_INPUT_UNSUPPORTED

**This file type cannot be used.** Fluentfolio reads WAV, MP3, MP4, M4A, MOV, WebM, MKV, Ogg, Opus and FLAC recordings. It looks at the file's contents, not its name, so renaming a file does not help.

What to do: save or export the recording again as one of these types, for example as M4A or WAV from your recording app.

### E_INPUT_DAMAGED

**The recording could not be read.** The file starts like a recording, but it is damaged or cut short. This often happens when a recording app is closed before it finishes saving.

What to do: use another copy of the recording, or export it again from the app that made it.

### E_INPUT_UNREADABLE

**The file could not be opened.** Another program may have the file open, or your account may not be allowed to read it.

What to do: close the program that recorded or opened the file, then try again. Or copy the file to your own folder, such as Documents, and use the copy.

### E_INPUT_NO_AUDIO

**The file has no sound.** The file has no audio track, so there is nothing to analyse. A screen recording made with the microphone off looks like this.

What to do: choose the file that holds the recorded sound.

### E_INPUT_TOO_SHORT

**The recording is too short.** It is shorter than the shortest length in Settings, which is 10 seconds unless you changed it.

What to do: record a longer answer. To analyse short clips, lower "Shortest recording (s)" in Settings, on the Advanced tab, under Recording limits.

### E_INPUT_TOO_LONG

**The recording is too long.** It is longer than the longest length in Settings, which is 180 minutes unless you changed it.

What to do: cut the recording into parts, or raise "Longest recording (min)" in Settings, on the Advanced tab.

### E_INPUT_TOO_LARGE

**The file is too large.** It is larger than the largest size in Settings, which is 2 GB unless you changed it. Video files are often this large.

What to do: export the recording as audio only, for example as M4A. Or raise "Largest upload (MB)" in Settings, on the Advanced tab.

### E_INPUT_SILENT

**No speech found.** Fluentfolio found only silence, noise, music or a steady tone in the recording.

What to do: check that the microphone was on when you recorded, and that you chose the right file.

### E_INPUT_TRACKS

**One track per person was not found.** The per-track layout needs a stereo recording or a file with two audio tracks, one for each person. This file has neither.

What to do: choose the mixed layout, which tells the two voices apart by how they sound. Or export the recording with one track per person (see the [recording guide](recording.md)).

### E_INPUT_LAYOUT

**The layout does not fit.** The capture layout does not fit the mode or the files you gave.

What to do: use mixed for one file with both voices, per-track for a file with one track per person, or two-files with one file for each person.

### E_INPUT_TEXT

**This text cannot be used.** A question list or an AI chat's answer must be a .txt or .md file. Your own transcript must be a .txt, .md, .vtt or .srt file. Each can be at most 2 MB, saved as UTF-8 or UTF-16 text. A UTF-16 file needs the byte-order mark that Notepad and Word write at its start. A text you paste into the app counts the same way. Word documents are not read.

What to do: save the text as a plain .txt file in UTF-8, then try again.

### E_UPLOAD_GONE

**Choose the recording again.** The recording you chose is no longer waiting on this computer, for example because Fluentfolio was started again in between.

What to do: choose the file again, then start the analysis.

### E_NO_RECORDING

**Choose a recording first.** No recording was chosen. With the two-files layout, each person needs a file.

What to do: choose the file, then start the analysis.

### E_NO_MODE

**Choose who speaks in the recording.** No mode was chosen, so Fluentfolio does not know who speaks. It never guesses this.

What to do: under "Who speaks", choose Solo, Interview or Role swap, then start the analysis.

### W_INPUT_SHORT

**A short recording.** The recording is under 60 seconds (or the length set in Settings). The analysis ran, but a short recording gives little evidence, so the measures are less reliable.

What to do: for a fuller picture, record a longer answer.

### W_TRACK_LENGTHS

**The two files differ in length.** With the two-files layout, after the offset, the files differ in length by more than 2 seconds. They may not start at the same moment.

What to do: check the offset between the two files (on the command line, `--offset mm:ss`), then run the analysis again.

### W_TRACK_BLEED

**The tracks hear each other.** Each voice is loud on both tracks, so the tracks cannot tell the two people apart well.

What to do: run the analysis again with the mixed layout, which tells voices apart by how they sound.

### W_BACKGROUND_MUSIC

**Background music (a heuristic check).** A simple check heard what may be music under the speech. It can be wrong. Words heard over the music are marked unsure and left out of the words to check, the rare words, and the pitch and voice measures.

What to do: record again without music playing in the background.

## Decisions only you can make

### E_CONFIRM_LONG

**A long recording needs your confirmation.** The recording is longer than 60 minutes (or the length in Settings), so the analysis takes a long time.

What to do: in the app, tick the confirmation and start again. On the command line, add `--confirm-long`. For a recording over 90 minutes (or "Offer a light run over (min)" in Settings), `--light-run` instead gives a faster run with the slowest optional steps off; for a shorter one it is refused.

### E_CONSENT_REQUIRED

**The other person's agreement is needed.** A recording of two people is analysed only when the other person agreed to be recorded and analysed on this computer.

What to do: ask the other person. If they agreed, tick the box in the app, or add `--partner-consent` on the command line.

### E_CONFIRM_NEEDED

**This step asks first.** The step downloads or deletes files, so it asks before it starts. There was no terminal to ask in, so nothing was changed.

What to do: run the command in a terminal and answer the question, or add `--yes`. `--yes` never accepts a licence for you.

### E_OPENSMILE_CONSENT_MISSING

**openSMILE needs your consent first.** openSMILE's licence allows non-commercial use only, so it runs only after you accept that licence yourself. Until then its loudness values are missing. Nothing else changes.

What to do: tick openSMILE's licence box on the Components page. Or run `fluentfolio components consent opensmile` in a terminal and type `I ACCEPT`.

### E_INPUT_OPTION

**This option does not fit.** An option or a name does not fit here. For example: a swap style outside role-swap, a part name that does not exist, or a role-swap job without the person to check.

What to do: leave the option out or correct the name. `fluentfolio components list` shows the parts. For a role-swap job, add `--candidate person1` or `--candidate person2`.

### E_RESUME_OTHER_VERSION

**Made by another version.** This job was made by another version of Fluentfolio, so this version cannot continue it.

What to do: analyse the same recording again as a new job. In the app, start a new analysis. On the command line, run `fluentfolio analyze FILE --mode MODE`.

## Voices, roles and Parts

### W_ONE_VOICE

**Only one voice was found.** The mode expects two people, but only one voice was found, so no roles were given.

What to do: check the mode. A recording of one person reading the questions aloud is a solo recording.

### W_THIRD_VOICE

**A third voice was found.** Three voices each speak for a noticeable time. The third may be a real person, or one voice that was split in two.

What to do: check who speaks when in the results. If the third voice is not a person in the test, correct it in the review form.

### W_OTHER_VOICE

**A brief other voice.** A voice that speaks for under 5 percent of the time was heard. It is left out of the evidence.

What to do: nothing, unless that voice is one of the two people in the test.

### W_SIMILAR_VOICES

**The voices sound similar.** The two voices sound alike, so the points where the speaker changes are less certain.

What to do: look at who speaks when in the results, and correct a role in the review form if it is wrong.

### W_SECOND_VOICE_SOLO

**A second voice in a solo recording.** Another voice speaks for more than 5 percent of this solo recording. All speech still counts as yours.

What to do: if someone asked you the questions, analyse the recording again in interview mode.

### W_SPEAKERS_FALLBACK

**Speakers were told apart without pyannote.** pyannote could not run, so sherpa-onnx, the speaker separation that needs no account, was used. Roles are then at most medium confidence.

What to do: run the system check to see why pyannote did not run. pyannote needs your own free Hugging Face token (see `E_TOKEN_MISSING` below).

### W_WORDS_MISSING

**Some speech has no words.** The speech model gave few or no words for part of the speech, even when it read that part again in smaller pieces.

What to do: check the transcript around the gap. Measures there count fewer words than were spoken.

### W_ROLES_UNAVAILABLE

**The candidate is not known.** Fluentfolio could not tell who asked and who answered, so it made no candidate measures. It never guesses roles.

What to do: set the roles in the review form, or run the recording again as solo.

### W_FREQUENT_SWAPS

**The roles seem to change often.** Who asks and who answers seems to change more often than the swap style you chose allows, so some changes were not taken.

What to do: check the swap times in the review form, or analyse the recording again with the swap style "After each question" (on the command line, `--swap-style per-question`).

### W_ROLE_CHANGE

**The roles seem to change.** In this interview, the person asking and the person answering seem to swap for a while.

What to do: if you took turns asking, analyse the recording again in role-swap mode.

### W_LIGHT_RUN

**A light run.** The slowest optional steps were off for this recording: the MFA aligner, the phone model and openSMILE. Their measures are missing.

What to do: for every measure, run the recording again without `--light-run`.

### E_OVERRIDES_INVALID

**The saved corrections cannot be used.** The corrections saved from the review form for this job are not in the form Fluentfolio expects.

What to do: open the review form again and save the corrections once more.

### E_REVIEW_INVALID

**These corrections do not fit together.** Each time must be inside the recording, a Part must start before it ends, and Part 3 cannot start before Part 2 ends. Nothing was saved.

What to do: correct the times in the review form and save again.

## Running an analysis

### E_BUSY

**Another job is running.** Fluentfolio runs one analysis, download, self-test or folder move at a time, and one is running now.

What to do: wait for it to finish, or cancel it. In the app, History shows the running analysis. On the command line, run `fluentfolio cancel JOB`.

### E_JOB_FAILED

**The analysis failed.** A step that every analysis needs did not finish, so there are no results.

What to do: open the analysis in History. Its page says which step failed and why, and that step's own code is on this page.

### E_JOB_CANCELLED

**The analysis was cancelled.** It was stopped at your request. The finished steps are kept.

What to do: to continue, open it in History and choose Run again, or run `fluentfolio resume JOB`. It goes on from the first step that did not finish, and keeps the steps before it.

### E_JOB_INTERRUPTED

**The job was interrupted.** Fluentfolio was closed, or the computer stopped, while this job ran. The finished steps are kept.

What to do: open it in History and choose Run again, or run `fluentfolio resume JOB`.

### E_INTERRUPTED

**Stopped.** Fluentfolio was stopped with Ctrl+C or by the system. The finished steps are kept.

What to do: run `fluentfolio resume JOB` to continue from the first step that did not finish.

### E_RUNNER_NOT_STARTED

**The background run did not start.** The job was created, but its run in the background did not start within 15 seconds. Some tools, such as some AI agents' sandboxes, end the programs they start.

What to do: run the job in the foreground: `fluentfolio resume JOB`.

### E_STAGE_CRASHED

**A step stopped unexpectedly.** One step of the analysis stopped without a result.

What to do: run the job again. If it stops again, run `fluentfolio doctor --report report.txt` and share the report when you ask for help.

### E_STAGE_TIMEOUT

**A step took too long.** One step made no progress for too long, or ran far longer than expected, so it was stopped. A step that shows no progress is stopped after 10 minutes unless you changed "Stop a stuck step after (min)" in Settings, on the Advanced tab.

What to do: run the job again. On a slow computer, choose a lighter speech model in Settings.

### E_RAM_OOM

**The computer ran out of memory.** This step needed more memory than the computer had free.

What to do: close other programs and run the job again, or choose a lighter speech model in Settings.

### E_GPU_OOM

**The graphics card ran out of memory.** The graphics card did not have enough memory for this step, and the step also failed on the processor.

What to do: close other programs that use the graphics card, such as games or video editors. Or in Settings, under Device, set "Run on" to "Processor (CPU)".

### E_CUDA_RUNTIME

**The graphics card failed.** The graphics card driver reported an error, and the step also failed on the processor.

What to do: update the NVIDIA driver. Or in Settings, under Device, set "Run on" to "Processor (CPU)".

### E_GPU_UNAVAILABLE

**No graphics card to use.** "Run on" is set to "Graphics card (GPU)", but this step found no graphics card it can use.

What to do: in Settings, under Device, set "Run on" to Auto or "Processor (CPU)". With Auto, the system check says why the graphics card is not used.

### W_CPU_FALLBACK

**A step ran on the processor.** A step failed on the graphics card and ran again on the processor. Its results are shown as usual.

What to do: nothing for this analysis. If it happens often, set "Run on" to "Processor (CPU)" in Settings.

### W_GPU_NOT_USED

**The graphics card is not used.** Auto runs every step on the processor. The system check says why.

What to do: nothing is needed. A graphics card is optional; without one, analyses take longer.

### W_LOW_RAM

**Little memory.** This computer has less than 12 GB of memory, such as an 8 GB computer. Analyses work, but slowly. 16 GB is recommended.

What to do: close other programs while an analysis runs.

### W_FREE_RAM

**Less free memory than a step needs.** Before an analysis starts, Fluentfolio compares the memory that is free now with what each of its steps needs for a recording of this length. One step needs more than is free, so the analysis may stop with an out-of-memory error (E_RAM_OOM). The page before Start and the command line name the step and both amounts. The amounts come from runs on one computer, so treat them as a guide.

What to do: close other programs before you start, such as browsers with many tabs. A shorter recording needs less memory.

## Jobs, History and pages

### E_JOB_NOT_FOUND

**No such job.** There is no job with this ID in your data folder. It may have been deleted.

What to do: run `fluentfolio jobs list` to see the jobs, or open History in the app.

### E_JOB_NOT_FINISHED

**The job has no results yet.** The job has not finished, so it has no results to export or check.

What to do: wait until the analysis finishes, or run it again from History, then try again.

### E_PAGE_NOT_FOUND

**Page not found.** Fluentfolio has no page at this address. A link may be old, or an analysis may have been deleted.

What to do: open History to see the analyses on this computer.

### E_DELETE_INCOMPLETE

**Some files were not deleted.** Another program was using some files, so they were not deleted. The analysis stays in History as "Partly deleted", so you can delete it again.

What to do: close programs that may use these files, such as a media player, then delete the analysis again.

## Parts, downloads and the Hugging Face token

### E_COMPONENT_UNAVAILABLE

**A part of Fluentfolio is not available.** A step needs a part that is not installed, or that does not work on this computer.

What to do: open the Components page, or run `fluentfolio components list`. Each part's row says its status and what to do.

### E_MODEL_MISSING

**A model is not downloaded.** This step needs a model that is not on this computer yet. An analysis never downloads anything by itself.

What to do: install it on the Components page, or run `fluentfolio components install --all`. `--all` fetches only the parts your settings use, so a model you chose for one analysis with `--asr` needs its own ID, for example `fluentfolio components install whisper-large-v3-turbo`.

### E_HASH_MISMATCH

**A model file is damaged.** A model file does not match the copy Fluentfolio expects, so it is not used. A download that fails this check twice is deleted.

What to do: install it again on the Components page, or run `fluentfolio components install --all`. Only the damaged files are downloaded again.

The installer gives this code too, when the uv file it downloaded does not match the SHA-256 written in the installer. It deletes that file and stops before it installs the program. Run the installer again.

### E_DOWNLOAD_UNSAFE

**A download was refused.** A download was sent to an address without https, or a downloaded archive tried to write outside its folder or would unpack far larger than expected. Nothing was kept.

What to do: run the system check, then try the download again. If it happens again, report it: it may mean a network problem or a changed download.

### E_NETWORK

**The download did not work.** Fluentfolio could not reach the download site, or the site answered with an error. Nothing half-downloaded is used.

What to do: check your internet connection, then try again. A stopped download goes on from where it stopped.

### E_TLS_INTERCEPTED

**A secure connection was blocked.** The download site's security certificate was refused. A company proxy or an antivirus that inspects secure connections usually causes this.

What to do: try another network. Or ask your administrator to allow the sites Fluentfolio downloads from. The installer uses github.com, release-assets.githubusercontent.com, releases.astral.sh, pypi.org, files.pythonhosted.org, download.pytorch.org and download-r2.pytorch.org. The models come from huggingface.co (with cas-bridge.xethub.hf.co) and github.com, and the MFA aligner from conda.anaconda.org and github.com.

### E_TOKEN_MISSING

**No Hugging Face sign-in.** The pyannote speaker model needs your own free Hugging Face account and a read token. Solo mode and the other speaker separation, sherpa-onnx, work without it.

What to do: create a free account on huggingface.co. Create a read token there, under Settings, Access Tokens. Then open the Hugging Face token page from the Components page and paste it, or run `fluentfolio token set`. The [user guide](user-guide.md) shows each click.

### E_TOKEN_INVALID

**The Hugging Face token was refused.** Hugging Face did not accept the saved token. It may have been deleted or mistyped.

What to do: create a new read token on huggingface.co, then save it on the Hugging Face token page, or run `fluentfolio token set`.

### E_HF_TERMS_NOT_ACCEPTED

**Model terms not accepted.** Your Hugging Face account has not accepted the pyannote model's terms.

What to do: open the model page (the Hugging Face token page links to it), sign in and accept its terms. Then check the token again on that page, or run `fluentfolio token status`.

### W_TOKEN_FILE_PLACE

**The token file is in a shared place.** The Hugging Face token file is inside the app folder, the data folder or a git working tree, where it may be copied or shared.

What to do: point the `HF_TOKEN_PATH` variable at a private place, or remove the variable so the standard place is used.

### W_NOT_OFFLINE

**Not everything is downloaded yet.** A part that is switched on is not on this computer yet, so analyses run without it.

What to do: install it on the Components page, or run `fluentfolio components install --all`.

### E_SELFTEST_FAILED

**A part failed the self-test.** At least one part gave a wrong result on the test clip that comes with Fluentfolio. The system check and the Components page name it.

What to do: run the check again on the Components page, or run `fluentfolio self-test`. If a part still fails, install it again.

### E_INSTALLER_NEEDED

**This part comes with the installer.** This part is a package inside the app itself, which only the installer may add or repair, and only while Fluentfolio is closed.

What to do: close Fluentfolio and run the installer again. It repairs the install and adds this part.

### W_NOT_INSTALLED

**Not set up by the installer.** This copy of Fluentfolio was not set up by the installer, or was set up by an installer of another version.

What to do: run the installer for a normal install, or run it again to bring this one up to date.

## Folders, disk and settings

### E_PATH_RULE

**This folder cannot be used.** Some tools fail in this folder. On Windows it must be on a local NTFS disk, outside OneDrive, under 60 characters long, with no spaces and no letters outside A to Z (digits are fine). Linux has no such rule.

What to do: choose a folder such as `C:\Fluentfolio`.

### E_MOVE_FAILED

**The folder was not moved.** The new folder must be given as a full path, be empty and lie outside the old one, and the copy there must match the old folder. A data folder must also lie outside the app folder, because the uninstaller deletes the app folder with everything in it. The data folder and the models folder must lie outside each other. A folder that holds another Fluentfolio folder (the app, data or models folder) is not moved, because a move deletes the old folder with everything in it. One of these did not hold, so the old folder is still in use and nothing in it changed.

What to do: in Settings, on the Advanced tab, under Folders, type the full path of a new or empty folder, for example on another drive, and move again. If the folder you move holds another Fluentfolio folder, move that one out first.

### E_HOME_SET_BY_ENV

**The data folder is set outside Fluentfolio.** The `FLUENTFOLIO_HOME` variable chooses the data folder, so Fluentfolio cannot move it.

What to do: move the folder yourself and change `FLUENTFOLIO_HOME`. Or remove the variable and move the folder in Settings.

### E_DISK_FULL

**The disk is full.** There is not enough free space to save the result. Nothing half-written was kept.

What to do: free some space on this disk, then run the job again. Deleting old analyses in History frees space too.

### W_DISK_LOW

**Little free disk space.** Less than 5 GB is free on the drive of the data folder or the models folder.

What to do: free some space on that drive, or move the folder to another drive in Settings, on the Advanced tab, under Folders.

### E_SETTINGS_INVALID

**This setting cannot take that value.** The setting does not exist, or it does not allow that value. Nothing was changed.

What to do: run `fluentfolio settings show` to see every setting and its value. Switches take true or false.

### E_SETTINGS_READ_ONLY

**Settings from a newer version.** A newer version of Fluentfolio wrote settings.toml, so this version does not change it.

What to do: update Fluentfolio to change these settings.

### W_SETTINGS_UNKNOWN_KEY

**Unknown setting ignored.** settings.toml has a setting this version does not know. It was ignored.

What to do: open Settings and choose "Restore defaults", or remove the line from settings.toml.

### W_SETTINGS_INVALID

**A setting was reset.** A value in settings.toml is not allowed, so its default is used.

What to do: open Settings and choose the value again.

### W_SETTINGS_UNREADABLE

**Settings could not be read.** settings.toml could not be read, so Fluentfolio kept it as a backup and uses the defaults.

What to do: open Settings and choose your settings again.

### W_SETTINGS_NEWER

**Settings from a newer version.** A newer version of Fluentfolio wrote settings.toml. This version uses it read-only and never resets it.

What to do: update Fluentfolio to change these settings.

## The command line and unexpected errors

### E_USAGE

**The command was not understood.** The command has an unknown option, a missing argument or a value that is not allowed.

What to do: run `fluentfolio COMMAND --help`, for example `fluentfolio analyze --help`. The [command-line guide](cli.md) lists every command.

### E_INTERNAL

**Fluentfolio stopped on an error it did not expect.** Your recordings and results are kept.

What to do: run `fluentfolio doctor --report report.txt`, read the report, and share it when you report the problem (see [SECURITY.md](../SECURITY.md) for security problems and the issue templates for everything else).
