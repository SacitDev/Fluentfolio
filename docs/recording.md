# Recording a practice test

Fluentfolio works with the recordings you already make: a phone voice memo, a Google Meet call, or a recording from OBS Studio. This page shows how to record so that the analysis hears you as you really spoke, can tell who speaks, and which choices to make in Fluentfolio afterwards.

## Before you record

- If someone else speaks in the recording, ask them first. Fluentfolio analyses a recording of two people only after you confirm that the other person agreed.
- Record at least a few minutes. Under 60 seconds the measures are less reliable, and under 10 seconds Fluentfolio does not analyse it (both lengths can be changed in Settings).
- Use a quiet room, turn off music and notifications, and keep the microphone at the same distance all the time. A headset microphone is better than a laptop's built-in one.
- **Turn off noise reduction, "voice isolation" and "enhance" options** in the app you record with. They clean the sound for listeners, but they can also cut quiet words, breaths and hesitation sounds such as "um", which are part of what Fluentfolio measures. A quiet room does that job better.
- Fluentfolio reads WAV, MP3, MP4, M4A, MOV, WebM, MKV, Ogg, Opus and FLAC files, audio or video. Choose the best quality your app offers.

## Seven habits for the clearest results

1. **One track per person, if you can.** When each person's voice is on their own track or in their own file, Fluentfolio knows for certain who said what. On my composed test sessions (October 2026), 99.2 % of the word time went to the right person this way. With one shared track, it has to tell the voices apart by how they sound.
2. **Keep the sound natural.** Switch off noise reduction and "enhance" options (see above), and record in a quiet room instead.
3. **Do not talk over each other.** Let the other person finish before you start. Speech where both of you talk at once is left out of your measures.
4. **Read the topic card aloud.** Fluentfolio finds the three Parts from your long Part 2 answer, and it needs a sign that this long turn starts: the words of a topic card, such as "you should say" or "one minute to prepare", or a quiet minute to prepare. So, before Part 2, the person asking reads the card aloud, as an examiner does (in a solo test, read it yourself), and you prepare in silence. Saying "Part 2" and "Part 3" as each starts helps place the edges between the Parts, but it is not enough on its own.
5. **In a solo test, read each question aloud, then pause briefly before you answer.** The pause helps Fluentfolio tell the question from your answer.
6. **Record the whole test**, all three Parts, 11 to 14 minutes as in the real test. A longer recording gives more evidence than a short one.
7. **Add the questions** you were asked (see the end of this page). They help Fluentfolio find where each answer starts. With a line "Part 1", "Part 2" or "Part 3" before each Part's questions, the Parts come from your list, even without the signs in habit 4.

## Which way to record

From the most accurate to the least:

- **Each person on their own track:** OBS Studio with two audio tracks, or Zoom with "a separate audio file for each participant" and its original sound turned on (see below).
- **One track, natural sound:** a phone's voice recorder between the two of you, or alone for a solo test, with its noise reduction off.
- **A meeting recording with its sound cleaning on**, for example Google Meet or Zoom with their default settings. This works, but quiet words and hesitations can be lost, so your hesitation measures may come out lower than they really were.

## Choose who speaks (the mode)

When you start an analysis, you choose who speaks. Fluentfolio never guesses this.

- **Solo**: only you. You read each question aloud, or play it, then answer it. The questions you read are left out of the measures.
- **Interview**: someone asks you the questions, as an examiner would, and you answer.
- **Role swap**: two learners take turns asking and answering. Choose when the roles change: once in each Part, or after each question. Each person gets their own evidence.

## Choose how it was recorded (the layout)

For interview and role swap:

- **One recording of everyone** (mixed): one file in which both voices are on the same track. Fluentfolio tells the voices apart by how they sound. This works best when the voices are clearly different and rarely speak at once.
- **One track per person** (per-track): one file with each person on their own track, either a stereo file with one person on the left and one on the right, or a file with two audio tracks. This gives the clearest result.
- **Two files**: one file for each person, which started recording at the same moment. On the command line, `--offset mm:ss` says where the second file starts in the first if they did not.

With per-track and two files, Fluentfolio works out from what each person does which one asks and which one answers. If each voice is also loud on the other track, it warns you and the mixed layout may work better.

## A phone

Record with the phone's voice recorder app, with the phone on the table between you and the other person, or held near you for a solo test. Before you start, look in the recorder's settings for options such as noise reduction, "enhance recording" or "voice focus", and switch them off; their names differ between phones. Choose the best quality the app offers. Send the file to your computer and choose **One recording of everyone**. Close the recorder app properly when you finish, so the file is saved completely.

## Google Meet

Before the call, open Meet's settings, then Audio, and switch **Noise cancellation** off if your account shows that option. Meet records the whole call into one video file, on the plans that allow recording. Choose **One recording of everyone**. If Meet made a transcript, you can add it as your own transcript.

## Zoom

With its default settings, Zoom removes background noise and evens out the sound, so quiet words and hesitation sounds can be lost. Two settings make it much better for practice, if your Zoom version has them (the names change between versions):

- In the audio settings, allow **original sound** and turn it on in the meeting. This switches off Zoom's sound cleaning.
- In the recording settings, turn on **a separate audio file for each participant**, and record to your computer.

After the call, choose **Two files** in Fluentfolio and give it your file and the other person's file. If you have only Zoom's combined file, choose **One recording of everyone**. If you saved Zoom's transcript (a `.vtt` file), you can add it as your own transcript.

## OBS Studio (two tracks)

OBS Studio can record your microphone and the other person's sound on separate tracks:

1. In Settings, Output, set the output mode to Advanced, and under Recording tick audio tracks 1 and 2. Use the MKV format, which keeps both tracks.
2. In Edit, Advanced Audio Properties, put your microphone on track 1 only and the desktop audio (the call) on track 2 only.
3. Record, then choose **One track per person** in Fluentfolio.

## Adding the questions or your own transcript (optional)

Under "Add the questions or your own transcript" on the New analysis page, or with `--questions` and `--transcript` on the command line:

- **The questions**: one question per line. Lines that say `Part 1`, `Part 2` or `Part 3` mark where each Part starts. They help Fluentfolio find where each answer starts.
- **Your own transcript**: for example from Google Meet, as a `.txt`, `.md`, `.vtt` or `.srt` file of at most 2 MB, or lines such as `A: words`. The names found in its speaker labels are replaced by Speaker A and Speaker B before it is kept, with the limits in [limitations](limitations.md). Fluentfolio uses it only to list where it differs from its own transcript; it never changes the timing or the measures.

## After the analysis

If the roles or the Parts came out wrong, open the review form on the analysis page. There you can correct who spoke and where each Part starts, and Fluentfolio analyses again, usually without transcribing the speech again.
