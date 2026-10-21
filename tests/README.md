# Tests

This page is the test plan: what each folder proves, where the expected values come from, and the rules every
test follows.

## Commands

| What | Command |
|---|---|
| The default run (fast, no models) | `uv run --locked python -m pytest` |
| The fast checks before a commit | `uv run --locked python tools/check.py fast` |
| Everything, with real models and audio | `uv run --locked python tools/check.py full` |
| Download the public test data | `uv run --locked python tools/fetch_test_data.py` (needs `FF_TEST_DATA`) |
| Build one test session by hand | `uv run --locked python tests/support/session_builder.py tests/fixtures/sessions/CS-INT.json <out folder>` |

## Folders

| Folder | Proves | Needs | Marker | Time budget |
|---|---|---|---|---|
| `unit` | pure maths against hand-computed values; rules for roles, Parts and modes; settings; error codes; redaction; the job runner and each adapter's status handling, with faults injected at the process boundary | numpy only | none | the whole default run within 60 s |
| `integration` | each adapter on a short public clip against an independent value | real models and tools, except `test_media.py` and `test_acoustic_tools.py` | `slow`, plus `gpu` where needed, and `private_sample` for `test_sherpa_private.py`; none for `test_media.py` and `test_acoustic_tools.py`, which need only the installed packages and run by default | CPU within 15 min |
| `pipeline` | the full `fluentfolio analyze` on the composed sessions against their `truth.json`; where the private recordings are, also those recordings on the processor and on the graphics card, with numbers only | everything | `slow`, plus `gpu` or `private_sample` | GPU within 25 min for the composed sessions; the private-recording tests take longer |
| `e2e_cli` | the command line as a separate process: exit codes, the JSON output against its schema, no prompt without a terminal, UTF-8 on other code pages, Ctrl+C | fast part: nothing; real part: models | none, or `slow` | fast part within 30 s |
| `e2e_ui` | browser journeys against the stub backend, in both themes at two widths, with accessibility checks and screenshots | Edge | `slow` | within 6 min |

`support/` holds test code that is not a test: `session_builder.py` (below); `sherpa_child.py`, which runs the
speaker separation in its own process; `fake_stage.py`, a stand-in for an analysis step that is started like a
real one and can succeed, fail, crash, run out of memory, hang or start a process of its own; and
`offline_stage.py`, the real analysis step with every network connection refused. The job runner and the job loop
are tested against `fake_stage.py`, so faults are injected at the process boundary. Speech recognition runs through
`offline_stage.py`, which shows that analysis works offline. `cuda_fault_stage.py` is a step that runs out of video
memory on the GPU, for the CPU retry. `scoring.py` scores a run against a test session's exact answers.
`synthetic_job.py` makes job folders with hand-made results of the first steps. `silence.py` finds where speech
starts and ends with FFmpeg's silencedetect filter, so speech edges are checked against a tool outside Fluentfolio.

## Markers and skips

There are exactly four markers, and `--strict-markers` refuses any other: `slow`, `gpu`, `network` and
`private_sample`. The default run leaves out all four. `tests/conftest.py` is the only place that skips: a `gpu`
test skips when no NVIDIA GPU is found (unless `FF_REQUIRE_GPU=1`), and a `private_sample` test skips when
`FF_PRIVATE_SAMPLES` is not set (unless `FF_REQUIRE_PRIVATE=1`). Two named lists cover the rest: `ONE_SYSTEM`,
tests of a property only Windows or only Linux has, and `NEEDS_OPENSMILE`, the four tests that read openSMILE's
output, which skip when the `nc` extra is not installed. Three `slow` tests in `pipeline/test_selftest.py` also
expect openSMILE to work: without the `nc` extra they fail rather than skip, except the one also marked `gpu`,
which skips when no NVIDIA GPU is found. Test bodies never skip or expect a failure.

Every test gets its own empty data folder. Real models are shared read-only through `FF_TEST_DATA`, so they
download once. A test that is not marked `network` cannot open a connection to anything but this computer.

The real models are the parts in the app folder: `FF_TEST_DATA/approot`, or the folder in `FLUENTFOLIO_APP_ROOT`
when it is set. `fluentfolio components install` fills the folder in `FLUENTFOLIO_APP_ROOT`, or else the app folder
of a normal install, so first set `FLUENTFOLIO_APP_ROOT` to the tests' app folder, and `FLUENTFOLIO_HOME` to an
empty folder, so that your own settings stay as they are. `components install --all` fetches only the parts in use,
so also name the ones that are off or not chosen, for example
`uv run --locked fluentfolio components install parakeet-tdt-0.6b-v3 whisper-large-v3-turbo mfa`. pyannote needs
your Hugging Face token first, with the model's terms accepted (`fluentfolio token set`), and openSMILE needs the
`nc` extra. One `slow` test, the word timing of canary-1b-v2's words, also needs canary-1b-v2. The app does not
offer that model (the speech-model experiment took no decision), so `components install` cannot fetch it, and
without it in the app folder that test fails with `E_MODEL_MISSING`.

## Where expected values come from

A measure is proven against truth from outside the code under test:

- hand arithmetic, written as a literal with the sum in a comment;
- construction: a composed session or a generated tone, silence or noise of known length and pitch;
- a dataset's own labels, such as a LibriSpeech transcript;
- an independent tool, such as FFmpeg's silencedetect filter for where speech starts and ends;
- a published number, with its source.

A golden file only catches drift, and never proves a measure alone. Goldens change only through
`pytest --update-goldens -k <test>`, in a commit of their own whose message starts `golden:`.

`tests/tolerances.toml` holds every limit used against truth, each with a floor and its source. After the first
green run on CPU and GPU, a tolerance becomes the larger of its floor and twice the measured spread; then it is
frozen. Fixed bars, such as an accuracy requirement, never move.

## Composed sessions

`tests/support/session_builder.py` builds test recordings with exact answers. It uses numpy and soundfile only and
never imports Fluentfolio, so its truth cannot share a bug with the code it tests. It reads a recipe from
`tests/fixtures/sessions/`, places short clips from `tests/fixtures/audio/` on one timeline, adds a quiet noise
floor, and writes `session.wav` (16 kHz, 16-bit) and `truth.json`.

Exact by construction: who speaks when, each turn's role and Part, every inserted pause and overlap, and the outer
edges of every clip, to the sample. A clip's edge is the first or last 10 ms frame whose peak reaches -40 dBFS.
The builder cuts there, fades 5 ms inside the cut, and refuses any clip with a quiet stretch of 150 ms or more
inside it, so every pause of 250 ms or more in a session is one the recipe inserted. Each clip's speech starts and
ends on a non-zero sample. In the session file a quiet noise floor is added on top, so there an edge is checked by
its 10 ms frame: the edge frame reaches -40 dBFS and the frame just outside it does not. Word and phone times inside
a clip are not claimed, and no test compares against them.

| Session | Mode | Voices | Built from | What it pins |
|---|---|---|---|---|
| CS-SOLO | solo | one synthetic voice reads each question aloud and answers | 3 Part 1 questions, a Part 2 cue card, 60 s of preparation silence, a monologue | read-aloud spans, the solo split, pauses, the Part 2 turn |
| CS-INT | interview | a synthetic examiner and a LibriSpeech speaker as the candidate | 4 Part 1 answers, 60 s of preparation silence, a 90 s monologue with 0.3 to 1.2 s pauses, 3 Part 3 answers | roles, Parts, latency, pauses, the long turn, no words in the silence |
| CS-SWAP | role-swap | two synthetic voices | person 1 asks person 2 through Parts 1 to 3, then the other way round | two candidates, a file for each, the "heard the questions first" note |
| CS-OVL | interview | as CS-INT | CS-INT plus an examiner "mm-hmm" inside a candidate pause and a 2 s overlap | overlap flags, backchannels kept out of pauses |
| CS-WRONGMODE | declared solo | as CS-INT | CS-INT, declared as solo | the mode warning; no silent misattribution |
| CS-NOISY | interview | as CS-INT | CS-INT with pink noise 10 dB under the speech | quality flags; a weaker result, not a crash |

Made from these at test time, with no recipe of their own:

- CS-2TRACK: CS-INT as stereo, examiner left and candidate right, each leaking into the other at -20 dB.
- CS-2FILE: one file per person, the candidate's starting 1.5 s later.
- CS-HES: CS-SOLO with 4 fillers ("um", "uh") and 4 repeated first words spliced in at known places.
- CS-LONG: CS-INT repeated with 2 s between copies until it lasts at least 15 minutes.
- CS-MUSIC: CS-SOLO with music (held chords and a sung line) 12 dB under the speech for the whole session.
- CS-SOLO-GRA: CS-SOLO with 12 answer clips swapped for the `_gra` clips of the same voice, each with one
  planted grammar error: 4 of the third-person -s, 4 of the past tense and 4 of articles.
- CS-SOLO-INJECT: CS-SOLO with a spoken request for band nine inside the third Part 1 answer, and a learner
  transcript with an instruction line and a fake end-of-data line inside the second answer.

The builder's own tests (`unit/test_session_builder.py`) find every clip edge and pause in the built audio with a
separate frame scan and compare them with `truth.json`, build sessions twice and compare the bytes, and check the
clip rules on generated signals. For the sessions made at test time they check that each track holds its own voice
with the other 20 dB lower, that the two files equal the two tracks with the 1.5 s offset, that the long session's
edges equal its truth, and that each spliced first word repeats the word after it.

Composed sessions are partly synthetic and very clean. They prove that the code behaves correctly; they do not
measure how well it works on real recordings.

## Data

- `tests/fixtures/audio/*.opus` and `tests/fixtures/sessions/*.json` are committed: at most 1 MB per file and
  8 MB in total. `tests/fixtures/NOTICE` credits every source.
- `tests/data_manifest.toml` lists the larger public data, with URL, size, SHA-256 and licence.
  `tools/fetch_test_data.py` downloads it into `FF_TEST_DATA` and checks every hash.
- Tests marked `private_sample` run on recordings outside the repository: exactly two role-swap recordings, in
  the folder that `FF_PRIVATE_SAMPLES` names. They record numbers only, never text, into a `summary.json` in the
  folder that `FF_PRIVATE_SUMMARY_DIR` names.

## Rules

- Expected values never come from the code under test, or from a copy of it.
- Mocks and injected faults only at a process or file boundary, never inside the code being judged.
- A test is never weakened, skipped, marked or deleted to make it pass. A test that is wrong is fixed with the
  reason in the commit message. Tests of deleted code go with that code.
- Each test family has one planted-defect check: a deliberate bug (for example `>` instead of `>=` at 250 ms)
  must turn a test red before the family counts as tested.
- A test that needs a primary component fails when only its fallback ran.
- Test code stays at about 1.2 times the product code or less.

## Tests that guard the design

These cut across the folders above. The web, installer, ZIP, self-test and documentation ones arrive with those
parts of the program:

- the app process loads no heavy library, and the openSMILE process loads no GPL library, both checked in a fresh
  process because a late import is invisible to the import rules;
- every child process gets a built environment with all the privacy switches and without a token or `PYTHONPATH`,
  even when one is planted in the parent;
- a full interview run records every child's command line: an absolute program path, no shell, no script file, at
  most 1,024 characters each; the deepest path it creates stays under 200 characters;
- the model catalogue test, and the error-table tests: every code gives its exit code, message and next step, and
  every help link points at a section of the troubleshooting page (that each section exists is checked with the
  documentation);
- the evidence and command-line JSON match their committed schemas, and real outputs are checked against them;
- every app function has its command and its web route;
- the web security rules on every route: the session cookie, the host name, cross-site requests, the request header,
  the exact content policy and no caching;
- privacy canaries made at run time (a fake token, a home path, a name, a transcript word) never reach the logs, the
  JSON output, the ZIP, `for_ai.txt` or the diagnostic report;
- the installer scan: no downloaded code run through a shell, no encoded commands, no antivirus exclusions;
- the self-test fault cases: a missing alignment tool, a truncated model file, no openSMILE consent;
- device parity: the interview session on the GPU and on a forced CPU gives the same structure, a word error rate
  within 1 point and a median word-time difference of 40 ms or less;
- regression tests for earlier known failure cases, written as unit tests;
- the documentation link checks.

## Test families

| Part of the program | Its tests |
|---|---|
| core | unit: time intervals and their invariants, safe paths and joins, atomic writes, safe archive extraction, the log sanitiser, redaction |
| settings | unit: precedence, loading a damaged file, migration, the per-job snapshot |
| models | unit: the catalogue; download, hash check and rename against a local web server; one real download marked `network` |
| media | integration: decoding files of many formats and rates, the audio-quality class, the music fixture; unit: the time map |
| ingest | unit: file sniffing, limits, stored names, the transcript and question-list readers; e2e_cli: the input errors |
| asr | integration: each model on a LibriSpeech clip, speech-detection edges against an independent tool, silence and noise; unit: text normalisation, invented-text flags |
| speakers | integration: both speaker separators on a composed clip; unit: word ownership, the voice checks |
| structure | unit: roles, Parts, backchannels, the solo split; pipeline: role and Part accuracy |
| alignment | unit: the small dictionary, the alignment guard, the CTC aligner on a synthetic case; integration: clip edges on the VOiCES clip |
| analyzers | unit: rates, pauses, runs, repetitions, pitch combination and labels; integration: tones through each pitch tool, a planted pronunciation error |
| evidence | unit: schema drift, no band field anywhere, the invariants, the label rules |
| export | unit: the for_ai.txt golden, nothing cut for size, ZIP determinism, redaction of the audio check, the answer checker, the differences list |
| jobs | unit: the runner contract and injected faults, locks, liveness, cancel, resume; e2e_cli: Ctrl+C and detached runs |
| setup | unit: the automatic device choice, the install command, the token guards in a temporary folder; the installer rounds |
| doctor | unit: check rows and codes; integration: the self-test and its fault cases |
| app | unit: every app function has its command and its web route |
| web | unit: the security rules and routes through the test client; e2e_ui: the journeys |
| cli | e2e_cli: the JSON output, exit codes, no prompts, encodings |
