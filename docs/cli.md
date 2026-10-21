# The command line

Everything the app does can also be done with the `fluentfolio` command, for example from a script or by an AI agent that runs on your computer. The command and the app share the same data folder, settings and jobs.

## Where the command is

The installer does not change your PATH. The command is inside the app folder:

- Windows: `%LOCALAPPDATA%\Fluentfolio\app\<version>\.venv\Scripts\fluentfolio.exe`
- Linux: `~/.local/share/fluentfolio/app/<version>/.venv/bin/fluentfolio`

`fluentfolio doctor --json` prints the exact path as `data.executable`. The app's pages write the start of a path as `%LOCALAPPDATA%` (on Linux `~`), so a screenshot shows no user name. PowerShell does not read that form: write `$env:LOCALAPPDATA` instead, for example `& "$env:LOCALAPPDATA\Fluentfolio\app\0.1.0\.venv\Scripts\fluentfolio.exe" doctor`. The examples below write `fluentfolio` for short.

Every command has `--help` with two examples, for example `fluentfolio analyze --help`.

## A first analysis

```
fluentfolio doctor
fluentfolio analyze practice.m4a --mode solo
fluentfolio export 20261013-101500-ab12 --out ./for-my-ai
```

`analyze` shows its progress and ends with the job ID. `export --out` copies the texts for an AI chat and the ZIP into the folder you name. A job ID looks like `20261013-101500-ab12`: the start date and time in UTC and four random characters.

## Commands

### Analysing

| Command | What it does |
|---|---|
| `analyze FILE [FILE2] --mode MODE` | Analyse a recording and write the evidence and the texts for an AI chat. |
| `status JOB` | A job's state, step, time so far, outputs and limits. `--wait --timeout SEC` waits until it stops, at most SEC seconds (100 unless you say). |
| `resume JOB` | Run a job again from its first unfinished step: the first whose result is missing, or is neither `ok` nor `disabled`. The steps before it are kept; it and every later step run again, the export of the texts for an AI chat included. A completed job stays as it is. So does a job completed with gaps whose results are all finished, for example when its only gap is a step that used its fallback. In a job made by `rerun`, the steps before the one the rerun started from are kept as they are. |
| `rerun JOB --mode MODE` | Run a finished job again as another mode, in a new job. Its speech is not transcribed again. |
| `cancel JOB` | Ask a running job to stop. It stops within seconds and keeps its finished steps. |
| `jobs list` | The newest jobs first (10 unless `--limit N`): ID, start time, state, mode and length. Never file names. |
| `jobs delete JOB` | Delete one job: its copy of the recording, its results and its temporary files. It asks first; `--yes` answers yes. |

`analyze` options:

| Option | Meaning |
|---|---|
| `--mode solo\|interview\|role-swap` | Who speaks. Required, because Fluentfolio never guesses it. solo: you read each question aloud, then answer it. interview: one person asks, one answers. role-swap: two learners take turns asking. |
| `--layout mixed\|per-track\|two-files` | How the voices were recorded. mixed (the default): one track for both. per-track: one track per person. two-files: a second file, FILE2, for the other person. |
| `--offset mm:ss` | two-files: where FILE2 starts in FILE, as mm:ss or mm:ss.s, for example `01:30.5`. |
| `--swap-style once\|per-question` | role-swap: the roles change once per Part (the default) or on each question. |
| `--questions FILE` | Your question list, as a .txt or .md file, with optional Part 1/2/3 headings. |
| `--transcript FILE` | Your own transcript (.txt, .md, .vtt or .srt). The names found in its speaker labels are replaced where they are written as in the label; some labels and some forms of a name are not found, see [limitations](limitations.md). |
| `--device auto\|gpu\|cpu` | Where the analysis runs. |
| `--asr ID` | The speech recognition model for this run (a part ID from `components list`). |
| `--partner-consent` | interview and role-swap: the other person agreed to be recorded and analysed on this computer. Pass it only when they said so. |
| `--confirm-long` | Go on with a recording over 60 minutes (or the length in Settings). |
| `--light-run` | Over 90 minutes (or the length in Settings): a faster run with the MFA aligner, the phone model and openSMILE off. The results say so. |
| `--dry-run` | Check the file and show its length, the time estimate, the steps, what is missing and what a person must decide. Starts nothing, so it takes no `--detach`. |
| `--detach` | Run in the background and return at once. Follow it with `status JOB --wait`. |
| `--quiet` | Show no progress. |

`resume` and `rerun` take `--detach` and `--quiet` too. `rerun` from solo to interview or role-swap also takes `--partner-consent`, with the same meaning.

### Results and AI chats

| Command | What it does |
|---|---|
| `export JOB` | The paths of a finished job's texts for an AI chat and its ZIP. A missing one is written again. `--out DIR` copies them into DIR. `--print` prints one person's text. `--audio` also makes the audio file of each person's own answers (at most 9.5 minutes each). It is cut to their turns, so the partner can still be heard where the speakers were mixed up or both spoke at once. |
| `prompt` | Print the grading prompt on its own, without any data. `--kind pronunciation` prints the prompt for an audio file. |
| `check-answer JOB ANSWER.md` | Check an AI chat's answer, saved as a .md or .txt file, against the job's text: PASS, FAIL or N/A for each rule. |

In a role-swap job there are two texts, one per person. `--candidate person1` or `--candidate person2` chooses one for `export --print` and `check-answer`.

### Parts and the Hugging Face token

| Command | What it does |
|---|---|
| `components list` | Every part with its status, download size and licence class. |
| `components install ID ...` or `--all` | Download parts, or make MFA's environment. Every file is checked against its SHA-256. It asks before it downloads; `--yes` answers yes. |
| `components remove ID ...` | Delete downloaded parts or MFA's environment. They can be installed again at any time. It asks first; `--yes` answers yes. |
| `components consent opensmile` | Read openSMILE's non-commercial licence and accept it yourself by typing `I ACCEPT`. Only a person at a terminal can do this. |
| `token set` | Save your own Hugging Face read token after testing it. It reads the token from a hidden prompt, or from one line of standard input with `--stdin`. A saved token that works, with the pyannote terms accepted, is never replaced. |
| `token status` | Whether a token is saved, whether it works, and whether the pyannote terms are accepted. This asks huggingface.co. The token itself is never printed. |
| `token clear` | Remove the token, but only one that Fluentfolio saved. It asks first; `--yes` answers yes. |

### Checks, settings and the app

| Command | What it does |
|---|---|
| `doctor` | Check the install, the disks, the computer and each part. `--report FILE` also writes a report to share when you ask for help; read it first. |
| `self-test` | Run the 58-second clip that comes with Fluentfolio through every part in use and check each result. `--device auto\|gpu\|cpu` chooses where it runs. |
| `settings show` | Every setting and its value. |
| `settings set SECTION.KEY VALUE` | Change one setting, for example `fluentfolio settings set export.audio true`. It applies to jobs started after it. |
| `serve` | Start the app and open it in your browser. `--no-browser` prints the file to open instead. `--port N` chooses the port on 127.0.0.1. |
| `guide agents` | Print the guide for AI agents. `--write-skill DIR` writes `DIR/fluentfolio/SKILL.md` for agents that read Agent Skills. |
| `--version` | The version. With `--json`, one JSON object with the versions of Fluentfolio, Python, the system, the evidence schema and the prompt in the text for an AI chat. |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Done. Also when the result has limits: warnings never change the exit code. |
| 1 | An error Fluentfolio did not expect. |
| 2 | The command was not understood (see its `--help`). |
| 3 | Setup: a part, a download, the token or a folder needs attention. `doctor` also gives 3 when no analysis can run, and `self-test` when a part fails. |
| 4 | The input was refused, such as an unsupported file. |
| 5 | A person must decide first, such as the other person's agreement or a long recording. The message names the option or step. |
| 6 | Busy: another job, download or check is running. |
| 7 | The job failed or was cancelled. |
| 8 | The job still runs: `status --wait` reached its timeout. Ask again. |
| 130 | Stopped with Ctrl+C or by the system. `resume JOB` continues. |

For a foreground `analyze`, `resume` or `rerun`, and for `status --wait`, the exit code is the job's outcome: 0, 7 or 8. Ctrl+C during a foreground run gives 130. A foreground `analyze` gives 4 when the recording is refused during the analysis, and then leaves no job behind. Each error's code and fix are in [troubleshooting](troubleshooting.md).

## Output for scripts and agents

With `--json`, a command prints exactly one JSON object on standard output, and its progress and messages go to standard error:

```
{"schema": "fluentfolio.cli/1", "command": "...", "ok": true, "exit_code": 0, "version": "...",
 "data": {...}, "warnings": [...], "error": null}
```

On an error, `error` holds `code`, `category`, `message`, `hint`, `retryable` and `doc`. Times are in UTC, durations are whole milliseconds in fields that end in `_ms`, and paths are absolute with forward slashes. The full schema is [schemas/cli-v1.schema.json](schemas/cli-v1.schema.json).

The output holds job IDs, codes, counts, durations and paths, never transcript text, file names or people's names. Only `export --print`, `prompt` and `guide agents` print text. `export --print` prints the text for an AI chat alone and takes none of `--json`, `--out` and `--audio`. With `--json`, `prompt` and `guide agents` put their text in `data.text`; neither text holds anything you said. `check-answer` without `--json` also prints the words hidden for the audio check, after its rule results.

Fluentfolio asks a question only when it runs in a terminal. Without one, or with `FLUENTFOLIO_NO_INPUT=1` set, a missing decision ends with exit code 5 and a hint that names the option. Nothing ever waits for an answer that cannot come.

## Environment variables

| Variable | Effect |
|---|---|
| `FLUENTFOLIO_HOME` | Use this folder as the data folder. While it is set, the data folder cannot be moved in Settings. |
| `FLUENTFOLIO_APP_ROOT` | Use this folder as the app folder. |
| `FLUENTFOLIO_NO_INPUT=1` | Never ask a question, even in a terminal. |

AI agents: start with `fluentfolio guide agents`. More in [Use with AI](use-with-ai.md).
