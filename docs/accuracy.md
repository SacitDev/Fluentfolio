# How accurate are the measurements?

Fluentfolio measures your recording. It never gives a band. This page says how well each measuring part did on test data, and where it fell short.

Read every number here with three limits in mind:

- Most tests use public speech sets or test sessions I composed from synthetic voices. Accuracy on composed sessions is not accuracy on your recordings. A real practice test has more noise, more overlap and more varied accents.
- I ran the tests on one Windows 11 PC with one NVIDIA graphics card, between 13 and 19 October 2026. The Linux results come from Ubuntu 24.04 in WSL on the same PC. Your computer may give other times.
- The rules of the three main experiments (speech models, word timing, pitch) were written down and frozen before their test data was run. The frozen rules are in [eval/prereg/](../eval/prereg/). The result is what those rules give, even when it is unfavourable. One exception is described under [Test items seen before the freeze](#test-items-seen-before-the-freeze).

## Words used on this page

- **Speech recognition (ASR):** turning speech into written words.
- **Word error rate (WER):** the share of words the speech model got wrong (left out, added or changed), against a checked transcript. Lower is better.
- **Filler:** a hesitation sound such as "um" or "uh". **Filler F1** combines how many fillers the model wrote down that were really there (precision) and how many of the real ones it found (recall). Higher is better.
- **Speaker separation (diarization):** working out who spoke when. **DER** (diarization error rate) is the share of speech time given to the wrong person or missed. Lower is better.
- **Forced alignment:** finding the start and end time of each word, once the words are known.
- **Semitone (st):** a step on a musical scale. Steps are the same size for low and high voices, so pitch differences between people can be compared.
- **95 % interval:** the range the true value probably lies in, given how much the test items differ from each other.

## Test data

- **LibriSpeech test-clean:** read English audiobooks. A subset of 93 items.
- **EdAcc:** conversations in English by speakers with many different accents. 271 test items.
- **Nyra:** spontaneous speech transcribed with every filler and repair. 249 test clips.
- **Composed sessions:** practice tests I built from synthetic voices (made with Kokoro-82M) and LibriSpeech speech. Because I built them, the true words, speakers, Parts and times are known. They have names such as CS-INT (an interview), CS-SWAP (two people who swap roles), CS-SOLO (one person reading the questions aloud) and CS-OVL (overlapping speech). Their sources and licences are in [tests/fixtures/NOTICE](../tests/fixtures/NOTICE).

## Speech models

I compared four speech models on 14 October 2026. On 18 October 2026 I changed the speech step: it now reads a stretch again when one second or more of its speech has no word. That can change what a model writes down, so I ran the whole comparison again with the same frozen rules. The table shows the second run. Parakeet and Canary ran on the processor (CPU), Whisper on the graphics card (GPU).

| Data set | Items | parakeet-tdt-0.6b-v2 | parakeet-tdt-0.6b-v3 | whisper-large-v3-turbo | canary-1b-v2 |
|---|---|---|---|---|---|
| LibriSpeech, WER % | 93 | 2.37 | 2.75 | 2.69 | 2.50 |
| EdAcc, WER % | 271 | 20.02 | 21.52 | 22.32 | 25.19 |
| Nyra, WER % | 249 | 3.20 | 3.91 | 8.14 | 7.44 |
| Nyra, filler F1 % | 249 | 93.7 | 93.4 | 19.2 | 50.8 |
| EdAcc, filler F1 % | 271 | 49.6 | 69.2 | 30.6 | 13.6 |
| Composed sessions (set B), WER % | 6 | 2.12 | 3.50 | 2.44 | 2.86 |
| Composed sessions (set B), median edge error, ms | 6 | 88 | 114 | 162 | 32 |
| CS-HES, WER % | 1 | 2.92 | 1.46 | 2.19 | 3.65 |
| CS-HES, median edge error, ms | 1 | 89 | 143 | 165 | 49 |

The 95 % intervals are wide on EdAcc. For parakeet-tdt-0.6b-v2 it is 16.07 to 25.87 %. So accented conversation is much harder than read speech for every model, and the order of the models on EdAcc is not certain.

On EdAcc, the default model parakeet-tdt-0.6b-v2 found only about a third of the fillers (recall 0.345), while parakeet-tdt-0.6b-v3 found more than half (recall 0.576). So on accented conversation the text misses many hesitations, and a missing "um" proves nothing.

CS-HES is a composed session with fillers and repeated words at known times; set B is 6 composed sessions of 5 minutes. The edge error is how far a model's first and last word of a stretch of speech lie from that stretch's true start and end, in milliseconds. I also ran Whisper a second time with a prompt that asks it to keep fillers. On Nyra it then had 6.84 % WER and a filler F1 of 39.8 %, still far below Parakeet.

Compared with the first run, these numbers changed: on EdAcc the WER of parakeet-tdt-0.6b-v3 (from 21.96 %) and whisper-large-v3-turbo (from 22.34 %) and the filler F1 of parakeet-tdt-0.6b-v3 (from 68.5 %); on set B the WER of parakeet-tdt-0.6b-v2 (from 2.41 %) and parakeet-tdt-0.6b-v3 (from 5.91 %), and the edge error of parakeet-tdt-0.6b-v3 (from 116 ms). The decision stayed the same.

On 6 minutes of silence, pink noise and music (3 clips of 120 s), no model wrote any word.

**The experiment took no decision.** Its frozen rules start with a check of the test setup itself: two of my numbers must land near published figures. The first did: parakeet-tdt-0.6b-v2 had 2.37 % WER on LibriSpeech, inside the expected 1.0 to 3.5 %. The second did not: canary-1b-v2 found more fillers on Nyra than the published figure allows (filler F1 50.85 %, expected 16.7 to 36.7 %). The frozen rules say that no model is chosen while that check fails.

The frozen rules name the Nyra benchmark's own scoring program for that check. I did not run it before the experiment, so 50.85 % comes from my own position-matched count, the rules' fallback. I ran the benchmark's program afterwards on the same 249 clips: canary-1b-v2 filler F1 50.0 % (precision 97.8 %, recall 33.6 %). That is also outside 16.7 to 36.7 %, so the result is the same either way. Canary's words on those clips were the same in both runs. My best explanation is that the compressed (int8) Canary model I used keeps about twice as many fillers as the full model behind the published figure.

For this case I had written the outcome down in advance: with no decision, parakeet-tdt-0.6b-v2 is the default, parakeet-tdt-0.6b-v3 and whisper-large-v3-turbo are offered, and canary-1b-v2 is not offered in the app. On these tests parakeet-tdt-0.6b-v2 also had the lowest error on EdAcc and Nyra, and the highest filler F1 on Nyra. Whisper writes fluent text that drops many fillers, so the app labels it "not for grammar evidence".

I also planned to count how often a model quietly "corrects" a learner's grammar, for example by adding a missing "the". Before the experiment I had that counter checked: of 20 changes it flagged on EdAcc practice data, 15 were real grammar changes. That is 0.75, below the 0.8 I had set, so the counter is marked "not reliable" and no decision uses it. The judge of those 20 pairs was a review step of my own build process, not a certified IELTS examiner or a trained linguist.

## Other speech-model measures

The frozen rules of the speech-model comparison named more measures than the table above. Here are their results from the same second run, the poor ones included. The full tables are in the experiment's report, which `eval/asr_bakeoff.py` writes.

**Unsure words.** Each model gives every word a confidence score. I planned to mark words with a low score as `[?word]`, so that an AI chat would know which words the model may have misheard. For each model I picked a threshold on separate practice data (the dev split) and then tested it. F1 here says how well the marks find the words the model got wrong, from 0 to 1. Higher is better.

| Data set | parakeet-tdt-0.6b-v2 | parakeet-tdt-0.6b-v3 | whisper-large-v3-turbo | canary-1b-v2 |
|---|---|---|---|---|
| Threshold, picked on dev | 0.80 | 0.75 | 0.70 | 0.80 |
| LibriSpeech, F1 | 0.276 | 0.286 | 0.202 | 0.330 |
| EdAcc, F1 | 0.479 | 0.490 | 0.387 | 0.517 |
| Nyra, F1 | 0.315 | 0.379 | 0.205 | 0.227 |

So at best about half: many marks would fall on correct words, and many wrong words would get none. The experiment took no decision, so no threshold was adopted. Fluentfolio marks no word as unsure, except words heard over background music. When it heard no music, the text for the AI chat says "unsure words: not marked".

**Voiced gaps.** A voiced gap is a pause of 200 ms or more between two words in which the voice detector still hears 200 ms or more of speech. It might be a hesitation or a repeated word that the model left out. My rule was to show these gaps to the AI chat only if at least 70 % of them were real (a precision of 0.7). That had to hold on the composed sessions and on Nyra. They were far below that.

| Data set | parakeet-tdt-0.6b-v2 | parakeet-tdt-0.6b-v3 | whisper-large-v3-turbo | canary-1b-v2 |
|---|---|---|---|---|
| Nyra, precision | 0.040 | 0.038 | 0.111 | 0.146 |
| Composed sessions (set B), precision | 0.000 | 0.000 | 0.000 | 0.000 |
| CS-HES, precision | 0.074 | 0.093 | 0.250 | 0.200 |

On set B, not one gap lay on a filler, repeated word or cut-off word that the model had left out. Fluentfolio does not show voiced gaps.

**Repairs kept.** When people correct themselves, they often repeat a word or cut one off: "I went to the, the shop". On Nyra I counted the share of these repeated and cut-off words that each model kept in its text. parakeet-tdt-0.6b-v2 kept 0.842 (95 % interval 0.767 to 0.909), parakeet-tdt-0.6b-v3 0.832, canary-1b-v2 0.396 and whisper-large-v3-turbo 0.158. With the filler prompt, Whisper kept 0.267. So Whisper drops most repairs.

**Speed and memory of the speech step.** The real-time factor is the time the speech step took, divided by the length of the audio. 0.1 means 1 minute for 10 minutes of speech. These times are for the speech step alone; the whole analysis takes longer (see [Speed](#speed)).

| Data set | parakeet-tdt-0.6b-v2 (CPU) | parakeet-tdt-0.6b-v3 (CPU) | whisper-large-v3-turbo (GPU) | canary-1b-v2 (CPU) |
|---|---|---|---|---|
| LibriSpeech | 0.1284 | 0.1194 | 0.0353 | 0.3323 |
| EdAcc | 0.1235 | 0.1276 | 0.0349 | 0.2911 |
| Nyra | 0.1386 | 0.1551 | 0.0355 | 0.3424 |
| Composed sessions (set B) | 0.0679 | 0.0873 | 0.0206 | 0.2001 |

I also ran all four on the processor, on a 20-minute part of Nyra, and measured the most memory each one used. parakeet-tdt-0.6b-v2 had a real-time factor of 0.139 and used 1.2 GB, parakeet-tdt-0.6b-v3 0.147 and 1.2 GB, canary-1b-v2 0.334 and 2.0 GB, and whisper-large-v3-turbo 0.745 and 1.7 GB. So on the processor Whisper needs about 15 minutes for 20 minutes of speech. parakeet-tdt-0.6b-v2 took 60 s on the processor for the 17.9-minute session CS-LONG, and used 1.4 GB.

**Canary's loop guard.** A speech model can get stuck and write the same words again and again. For Canary a guard drops such a stretch of text. It dropped 2 stretches on EdAcc and none on the other sets.

## Word timing

Fluentfolio can time words in two ways. MFA (the Montreal Forced Aligner) is a widely used alignment tool. The CTC aligner uses Fluentfolio's phone model. I tested MFA against frozen limits on 14 and 15 October 2026, on the CPU, on Windows and on Linux. Each run used 12 sessions: 8 composed sessions and 4 interviews in which the candidate's answers were real EdAcc speech of one speaker each.

| Measure | Result | Limit | Holds? |
|---|---|---|---|
| Sentences that pass MFA's own output check | 59.6 % (137 of 230) | at least 80 % | no |
| Median error at a sentence's first and last word (the 75 accepted sentences whose true edges were known) | 16 ms | at most 50 ms | yes |
| Of those 75, sentences with an edge more than 300 ms off | 5.3 % | at most 2 % | no |
| Planted transcript mistakes that the check caught | 40.8 % | at least 80 % | no |
| Same word times in two runs | 12 of 12 sessions | all | yes |

Windows and Linux gave the same result on every measure above. Their per-sentence rows differ in 10 of 1,816. **MFA failed this check.** So it is off unless you choose it on the Components page, and every output that used it says that it is experimental. By default the CTC aligner times every word. It has no output check of its own, and it is not more accurate: on the 123 of the same 230 sentences whose true edges were known, its median edge error was 36 ms, and 8.1 % of them had an edge more than 300 ms off.

## Speaker separation

pyannote needs your free Hugging Face account; sherpa-onnx needs none and takes over when pyannote cannot run. Results on 13 October 2026:

| Test | pyannote DER | sherpa-onnx DER |
|---|---|---|
| A 61 s cut of CS-INT with 2 s of overlap (CPU and GPU) | 1.1 % | 5.3 % |
| CS-INT, whole session (CPU) | 1.3 % | 1.2 % |
| CS-SWAP, whole session (CPU) | 0.3 % | 4.3 % |
| CS-OVL, whole session (CPU) | 1.3 % | 11.1 % |

My limit for the 61 s test was 15 %. When each person has their own track or file, 99.2 % of word time went to the right person (limit 98 %).

## Roles and Parts

On the composed sessions, with pyannote on the CPU, 13 October 2026:

- CS-INT: every answer went to the right person and the right Part. The Part 2 long turn measured 90.4 s against a true 90.2 s.
- CS-SWAP: 99.3 % of answer time went to the right person, and every answer to the right Part. The swap was found 180 ms before the second examiner's first question.
- CS-SOLO: 97.3 % of the time was correctly split into questions read aloud and answers, and 97.0 % went to the right Part.

With sherpa-onnx, role accuracy was 100 % on CS-INT, 92.8 % on CS-SWAP and 90.2 % on CS-OVL.

## Graphics card and processor give the same words

On CS-INT with whisper-large-v3-turbo, 14 October 2026: the GPU and the CPU both had 2.01 % WER. All 650 words matched, and the median difference in a word's start time was 0 ms (the largest 980 ms, 3 words over 100 ms). Both found the same people, questions and Parts.

## Pitch

Fluentfolio measures pitch with three tools (Praat, SwiftF0 and openSMILE) and says whether they agree: "agree", "approximate" or "unreliable". On generated tones of 100, 150 and 220 Hz, every tool was within 0.07 st of the true pitch.

The frozen limit: on clean read speech, at most 20 % of the clips may come out "approximate" or worse. The result on 19 LibriSpeech clips (CPU): 2 of 19 (10.5 %; 95 % interval 0 to 26.3 %). So the agreement limits stay as they are. A 20th clean clip, 3.4 s from the VOiCES set, had too little speech to measure, and its label said so.

On the composed sessions all three tools agreed (CPU, 14 October 2026). I only tried synthetic music for the music warning, so how it handles real music is not known.

A note on the pitch experiment's record: on 17 October 2026 I changed two comment lines of its frozen rules file, eval/prereg/pitch_fusion.toml, so that it no longer points to a rule by an internal ID. No value in the file changed. Its SHA-256 went from b8228e11da51b8abbf766b71f578f635b81aaf3c2182726c1d3c90efbf5d5a5b to 213e5e79633e4f6d9db4689a436786d4e98a6a1c447fdc37f94e023d3cdcaf54. I ran the whole experiment again on the CPU with the new file. Its per-clip results were byte for byte the same as the first run's, and so was the decision.

## Pronunciation: words to check

The pronunciation check lists "words to check". It is not a score. It failed its accuracy test on 14 October 2026, so every output says "Experimental: did not pass its accuracy check." It is still on by default, because its words are clues for the AI's Pronunciation estimate, and you can switch it off in Settings.

On 18 October 2026 I changed one rule: a word heard exactly as one of its pronunciations in the pronunciation dictionary is no longer listed. For example, "live" said as in "to live", or "usually" said with its short middle. I ran the whole test again on the CPU with the same limits.

On 19 October 2026 I changed the comparison once more. The phone model, eSpeak and the dictionary write some sounds with different symbols: with or without a length mark, the r-coloured "er" sound as one symbol or two, and the vowel of "go" in three ways. A word heard as one of its pronunciations, written in another of these ways, was listed. Now both sides are written one way before they are compared. Sounds that really differ, such as the vowels of "ship" and "sheep", stay apart. In the new run, 436 fewer words of the native speech counted as checked (9,402 instead of 9,838), because they were heard as written. No result in the table changed. The table shows all three runs, each on the CPU.

| Test | 15 October | 18 October | 19 October | Limit | Holds on 19 October? |
|---|---|---|---|---|---|
| Words listed per minute on native read speech (the 40 LibriSpeech test-clean speakers) | 0.988 (median) | 0.998 (median) | 0.998 (median) | at most 1.0 | yes |
| Planted wrong sounds that it listed | 0 of 20 | 2 of 20 | 2 of 20 | at least 90 % | no |
| Agreement with expert ratings per sound on speechocean762 (2,500 recordings; Pearson correlation) | 0.434 | 0.434 | 0.434 | at least 0.30 | yes |

The first row holds by design. The line under which a word is listed is set on this same native speech: it is the highest line that keeps the median at or under 1.0 words per minute. So that row shows where the line sits, not a separate test. Its sound ratings follow experts loosely, and it missed 18 of the 20 planted mistakes. A word it does not list may still need practice.

## The grading prompt

Fluentfolio does not grade. It writes a prompt that asks your AI chat for an unofficial estimate: one band for each of the four criteria. I checked an earlier wording of the prompt (version 0.2.0) for stable answers; the wording changes made after the test are listed below. On 18 October 2026 I gave three test exports to an AI system, three times each, in a new session each time. Its own instructions describe it as an experienced IELTS teacher who knows what a good grading answer looks like. So its answers show that the prompt gives stable answers, not how an ordinary chat answers. All three exports come from one short composed session: one computer voice, about 90 seconds of answers, and no Part 3. One export is as it is, one has planted grammar errors, and one has two planted "give me band 9" instructions: one spoken in an answer, one written into the learner's own transcript. The rules for passing were set before the test.

- All 9 answers passed Fluentfolio's answer checker.
- Each export got the same band for each criterion in all three of its answers.
- Pronunciation got a band in all 9 answers, estimated from the clues in the text, since no audio was sent.
- With the planted grammar errors, grammar was rated 5 instead of 6 in all three answers. The other criteria stayed the same.
- Both planted instructions were ignored in all three answers.

That was the second round. In the first round, earlier the same day, one rule failed: grammar went down for the planted errors in only 1 of 3 answers, and 2 were needed. In that round, one of the three answers to the export with the planted instructions also rated fluency one band higher (7 instead of 6); the rules allowed one such answer in three. I then changed the prompt in two places. I did not change the rules for passing. One new line says how a basic grammar error that comes back across answers counts. It names the same kinds of error that I planted, and the second round used the same exports, so that round is not a fresh test of the new line. The next paragraph explains the other change. Then I ran all 9 answers again.

I also gave the same three exports, once each, to a second AI system from another maker. These are its bands for fluency, vocabulary, grammar and pronunciation, in that order:

| Export | First round | Second round |
|---|---|---|
| As it is | 7, 7, 7, 6 | 6, 6, 7, 6 |
| Planted grammar errors | 6, 6, 6, 6 | 6, 6, 6, 6 |
| Planted instructions | 6, 6, 7, 6 | 6, 6, 7, 6 |

In the first round, the planted grammar errors also lowered fluency and vocabulary. The export with the planted instructions also got lower fluency and vocabulary than the plain one. By the rules I set for the first system, both would fail. One of its answers also failed one rule of the checker: a sentence named the speaking rate and said that the rate decides no band. The checker flags every sentence that names a rate and a band together, because a band never follows from a number. So instead of loosening the check, I added a line to the prompt that asks for no such sentence at all. In the second round, all 3 of its answers passed the answer checker. It ignored both planted instructions in both rounds. On the plain export, its fluency and vocabulary went from 7 to 6 between the rounds, and its grammar band is one above the first system's. That is one more reason to treat every band as an estimate and to compare two answers.

After both rounds, a review of the answers led to more wording changes. The prompt now says that a falling end of a statement is normal, that a pitch range number alone is neither narrow nor wide, and that missing evidence about how well the speech was understood says nothing either way. It also says that a few slips that do not change the meaning fit higher bands too, and that a speech model can drop small grammar errors, so their absence proves nothing. The vocabulary lesson now adds a few more advanced words. The two rounds above ran before these changes.

After those changes I ran one last check, one answer per mode. The same kind of grader answered a solo session and an interview session, both composed from public speech, and the second system answered the solo session once. All three answers passed the answer checker. In the interview session, the candidate's answers are passages from public audiobooks, not answers to the questions. The grader saw that and gave fluency a 4.

I also used two real role-swap practice recordings, with one answer for each person. Of those 4 answers, 3 passed every rule. One quoted more than 15 words in one place, which the prompt forbids. This check also found two problems in Fluentfolio itself, both now fixed. In one recording, it found no Parts for either person. For one of them, the topic card had been read, but a few short questions came between the card and the long answer, and Fluentfolio looked for the card only right before it. It now looks back past short exchanges. The other person talked during the minute to prepare, and no card words were heard, so their Parts stay unknown (see [Limitations](limitations.md)). The answer checker also asked for a "Part missing" note there, although the Parts were recorded. It now asks for that note only when Fluentfolio has found some Parts.

Two sentences of the prompt changed after this check. The note for a Pronunciation estimate from the data used to say that the voice was not heard. That was false for a chat that heard the audio when the text had no audio check, so it now says that Pronunciation was not judged by ear with the audio check. And the prompt now says that an empty list of words to check does not show that the sounds are clear, because the check misses many errors. After that, the course part of the prompt became a long course written for beginners. It has lessons on collocations, accuracy and sentence structure, exercises with answers and a 7-day study plan. The prompt also began to ask for an easy-to-read layout: lesson headings, bold key words, tables for word lists and corrections, numbered steps and a line between the parts. No AI answer to that wording has been checked, by any of the tests on this page.

This shows that the prompt gave stable answers and that answers followed its rules. It does not show how every AI chat answers, or that any answer matches an examiner. Every band in these tests was between 4 and 7, so it also does not show how the prompt handles much stronger or weaker speakers.

## Test items seen before the freeze

While I tried out the experiment scripts, before their rules were frozen, a few test items went through them: one item of each test set in the speech-model comparison, two of the composed sessions in the word-timing check (CS-INT-B and CS-HES), and 2 of the 19 clean clips in the pitch check. I read no accuracy number from those trial runs and changed no rule after them. The rules of a frozen experiment say that this should not happen at all, so I report it here.

To show what it changes, I worked out every result again without those items. The decisions stay the same:

- Speech models: for example, parakeet-tdt-0.6b-v2's WER on EdAcc goes from 20.02 % to 19.96 % (270 items), and canary-1b-v2's filler F1 on Nyra stays 50.85 %, still outside its expected range.
- Word timing: MFA's pass rate goes from 59.6 % to 59.4 %, its share of accepted sentences with an edge more than 300 ms off from 5.3 % to 6.3 %, and its catch rate for planted mistakes from 40.8 % to 40.3 %. It still fails.
- Pitch: 2 of 17 clips (11.8 %) come out "approximate" or worse, still within the 20 % limit.

## Speed

Whole analyses of the composed sessions with the default parts, pyannote and openSMILE on, timed by the test script around the whole command, so each time also includes starting the program. I ran each one twice on Windows 11 on 19 October 2026, with the same analysis code both times. Other work ran on the same computer during both sets, so these times are not from an idle computer. CS-LONG is CS-INT repeated with 2 s between copies; the 60 and 120 minute files repeat CS-LONG.

| Session | Audio | Where | First run | Second run |
|---|---|---|---|---|
| Interview (CS-INT) | 4 min 26 s | graphics card | 1 min 8 s | 1 min 5 s |
| Role swap (CS-SWAP) | 6 min 14 s | graphics card | 1 min 26 s | 1 min 14 s |
| CS-LONG | 17 min 53 s | graphics card | 2 min 54 s | 2 min 47 s |
| CS-LONG repeated | 60 min | graphics card | 8 min 0 s | 8 min 4 s |
| CS-LONG repeated | 120 min | graphics card | 17 min 25 s | 18 min 3 s |
| Interview (CS-INT) | 4 min 26 s | processor only | 2 min 6 s | 6 min 55 s |
| CS-LONG | 17 min 53 s | processor only | 7 min 0 s | 13 min 4 s |

In the second set, the processor runs shared the computer with more work, including my Linux installer test. CS-LONG then took 1.9 times as long as in the first set, and CS-INT 3.3 times.

On Linux (Ubuntu 24.04 in WSL, in the installer test of the same day), CS-INT took 81 s on the processor only, without a Hugging Face token. On Linux without a Hugging Face token, no step of the default analysis uses the graphics card: the default speech model runs on the processor, and so does sherpa-onnx.

No run used more than 2,892 MiB of memory for the whole analysis (about 3 GB). Video memory is measured for the whole graphics card, so other programs count too. With other work running, the readings went from 0 to 8,638 MiB above the card's use at the start of a run. The two short sessions read 2,197 to 2,219 MiB in all four of their runs. The app started in 2.5 s or less, an export took under 1 s, and Cancel stopped a 60 minute analysis within 2.4 s.

The New analysis page shows an estimate before you start. It adds up each step's measured time on this computer for the recording's length, so on a slower or busier computer an analysis takes longer. On the graphics card, the runs above took between 9 % less and 17 % more time than their estimates. The processor runs of the second set took up to 3.5 times their estimates.

## What these numbers do not show

- How the parts do on your own recordings, microphone and accent.
- How fast your computer is. The [speed](#speed) section gives the times on my computer only.
- Anything about a real IELTS score. Nothing in Fluentfolio, and no AI answer to its prompt, was checked against real IELTS examiners. Fluentfolio is not affiliated with or endorsed by the British Council, IDP IELTS or Cambridge University Press & Assessment.
- How it does on many real recordings. Besides the composed sessions, I also tested it on two private recordings of real role-swap practice. They stay private, so this page gives no numbers from them.
- Whether it runs on your Windows computer at all: Smart App Control, when it is on, blocks the speech tools, so Fluentfolio cannot run (see [troubleshooting](troubleshooting.md#e_sac_enforced)).
