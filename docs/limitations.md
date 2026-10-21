# Limitations

Fluentfolio is a practice tool. This page says plainly what it cannot do, so that you can trust what it does. The numbers behind each point are on the [accuracy](accuracy.md) page.

**Contents:** [What it is not](#what-it-is-not) | [How far the measures go](#how-far-the-measures-go) | [Recordings](#recordings) | [Computers](#computers) | [AI chats](#ai-chats)

## What it is not

- **Not an IELTS score.** Fluentfolio gives no band. A band comes only from the AI chat you choose, and it is an unofficial practice estimate.
- **Not checked against examiners.** No part of Fluentfolio, and no AI answer to its prompt, was compared with real examiners' bands.
- **Not a teacher.** It measures and points to examples. It does not correct your grammar or tell you what to say.

## How far the measures go

- **Test data is not your recording.** Most tests used public speech sets and practice sessions I composed from synthetic voices. A real practice test has more noise, more overlap and more accents, so results on your recordings can be worse.
- **Accented conversation is hard.** On accented conversations, the default speech model got about 1 word in 5 wrong; on read speech, about 1 in 40.
- **Many hesitations are missed.** On accented conversations, the default speech model wrote down only about 1 "um" or "uh" in 3. So the hesitation count is often too low, and a missing "um" in the text proves nothing.
- **Word times can be off.** With the default aligner, half of the sentence edges I could check were within 40 milliseconds, but in about 1 sentence in 12 an edge was more than 0.3 seconds off.
- **Pronunciation is limited.** The "words to check" list did not pass its accuracy test, so it is marked experimental. It is on by default, because its words give the AI clues for a Pronunciation estimate, and you can switch it off in Settings. It also lists some words that were said well: on read speech by native speakers, it listed about 1 word a minute. A word it does not list may still need practice. For example, a word said without its last sound is not listed when the pronunciation dictionary also has that shorter form, as it does for "last" and "most". Fluentfolio never rates pronunciation itself. From the text alone, the AI estimates Pronunciation from these clues without hearing your voice; a chat that listens to the optional audio file can judge it by ear.
- **The MFA aligner** did not pass its word-timing check either, so it is off by default and marked experimental.
- **The music check is simple.** It was tried only on synthetic music and can be wrong.
- **Measures have no pass mark.** A higher or lower number is not better on its own. Compare your own practice tests over time.

## Recordings

- **One or two people.** A third voice gets a warning; Fluentfolio is built for a candidate and one partner.
- **Parts need a sign.** Fluentfolio finds the three Parts from your long Part 2 answer, and only when it hears topic-card words before it or a quiet minute to prepare. A few short questions in between are fine. If neither sign is there, for example because you talk while you prepare and nobody reads the card aloud, every Part is "unknown", and the text for the AI chat says so. Saying "Part 2" alone is not enough. Two things help: read the card aloud, or add your questions with a "Part 1", "Part 2" or "Part 3" line before each Part (see the [recording guide](recording.md)). You can also set where each Part starts in the review form after the analysis.
- **The audio file can hold your partner.** The optional audio file is cut to your own turns. Where the speakers were mixed up, or you both spoke at once, your partner can still be heard in it. In one clean test recording of a role swap, under a second of a partner's question ended up in the other person's file. Upload it only if your partner agrees.
- **English only.** Speech in another language is marked in the transcript, never measured as English.
- **Lengths.** At least 10 seconds, at most 180 minutes, and at most 2 GB by default. Under 60 seconds the measures are less reliable.
- **Sound cleaning hides things.** Noise reduction in a phone or a meeting app can remove quiet words and hesitations. See the [recording guide](recording.md).
- **Two files must start together.** Fluentfolio does not line up two files by itself; on the command line, `--offset` says where the second starts.

## Computers

- **Windows and Ubuntu only,** on a 64-bit Intel or AMD processor. Windows 10 22H2 or later, Windows 11, Ubuntu 22.04 or 24.04.
- **Smart App Control** on Windows 11, when it is on, blocks Fluentfolio. See [troubleshooting](troubleshooting.md#e_sac_enforced).
- **Tested on one PC.** I tested the graphics card path on one computer with one NVIDIA card, and Linux on Ubuntu 24.04 in WSL on the same computer. Other computers may be slower or behave differently.
- **Memory.** With less than 12 GB of memory, analyses work but slowly; 16 GB is recommended. A graphics card is optional.
- **Not signed.** The installer and the speech tools are not code-signed, so an antivirus may warn about them.

## AI chats

- **Different chats, different answers.** The same text can get different bands from different chats, or from the same chat twice. Compare several runs.
- **Audio support changes.** Which chats can listen to audio, and how much, changes often. See [use with AI](use-with-ai.md).
- **Names are found by their labels.** Fluentfolio finds the names in your own transcript from the speaker labels at the start of its lines, such as "Anna Smith:" or "Maria de Souza:". It then replaces each of those names in the transcript where it is written as in its label, with the same capitals. This includes a name with 's, 'd or 'll after it, as in "Anna's", and a name in quotes. Some rarer forms keep the name, such as "Anna've", "l'Anna", "the Smiths'" or 'Anna's' in straight single quotes. A name written in lower case in the talk, such as "anna", stays. A label with a word in lower case, such as "anna:" or "Anna smith:", is not seen as a name. Small words inside a name, such as "de" or "van", are fine. A label with a time before it, such as "[00:03] Anna Smith:", is not seen either. Neither is a name that is only said in the talk, so "my friend Sarah" stays. A label in a few rare scripts, such as Adlam, is not seen at all, so its whole name stays. Check the labels before you add your transcript.
- **Headings can be taken for names.** A heading in your transcript that starts with capitals and ends with a colon, such as "Talk About Your Hometown:", looks like a speaker label. Its words are then replaced in the transcript, and the answer check may say that an AI answer names a person. Write such headings without the colon.
- **Some names cannot be looked for.** The names found in your labels are replaced before your transcript is kept. A name that is also a common English word or a word of the grading prompt, such as "An", "Will" or "Mark", is replaced too. But the answer check cannot tell it from the word, so it looks only for the rest of that name.
- **The answer check reads its quotes from the data block.** It checks the IDs, times and quotes in the report's JSON block against Fluentfolio's transcript of your recording. The IDs and times in the report's tables and text are not checked. Before you practise a correction, find one or two of them in the transcript yourself.
- **Prompt injection is contained, not prevented.** A recording can contain words that look like instructions. The prompt tells the chat to ignore them, but no prompt can guarantee it.
