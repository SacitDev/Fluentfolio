"""Fixed labels for the measures: ID, kind, default reliability, source, and the plain definition of each.

Reliability comes from these rules only, never from a model. No definition calls a value good or bad:
a rate or a pause count describes how the answers were spoken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fluentfolio.core.enums import Mode

Level = Literal["high", "medium", "low"]
RANK: tuple[Level, ...] = ("low", "medium", "high")
UNSURE_LIMIT_PCT = 10.0  # above this share of unsure words, syllable-based rates are low
SPEAKING = "how the answers were spoken, not how good they were"
ASKED = (Mode.INTERVIEW, Mode.ROLE_SWAP)  # the modes where someone asks the questions
RHYTHM = "Context only: no norm says which values are better, and accents differ in rhythm."
ENDING = (
    "Heuristic thresholds, not a norm. Rising, falling and level endings are all normal in English; "
    "none is better."
)
VOICE = "Context only: measured in connected speech, not on held vowels, so clinical norms do not apply."
FORMANT = "Context only: formants depend on the voice and on the vowels spoken."
POOR_AUDIO = ("meeting_codec", "noisy")  # audio classes on which pitch tracking is unreliable
OVER_RUNS = "over the voiced stretches of 1 s or more in the answer"
BY_BASE_FORM = (
    "Names of people and places found by the grammar parser are left out, and a word counts as common when "
    'its base form is ("dishes" as "dish").'
)
BY_RANK = "wordfreq ranks of the word and its base form; names left out"
PITCH_TOOLS = "Praat, SwiftF0 and openSMILE (when enabled), combined with fixed weights"
OURS = "fluentfolio"
PITCH_AGREEMENT = (
    "Use it as it stands when the tools agree (within 1.0 semitone), only for broad statements when they "
    "are approximate (within 3.2), and not at all when they disagree."
)


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    kind: Literal["measured", "estimated", "context"]
    reliability: Level
    source: str  # a step or tool ("vad", "asr", "mfa", "praat", "nlp"...); the builder adds its version
    definition: str
    caveat: str
    how: str = ""  # how it is measured, in a few words, for the text for an AI chat
    details_only: bool = False  # kept out of the text for an AI chat
    modes: tuple[str, ...] = tuple(Mode)
    syllables: bool = False  # a rate from transcript syllables: low when many words are unsure
    text: bool = False  # read from the words alone: at most medium when the transcript is smoothed
    pitch: bool = False  # read from a pitch track: low on audio classes where pitch tracking fails


LABELS = {
    "speaking_time": Label(
        "M01", "speaking time", "measured", "high", "vad",
        "Seconds of voice-detector speech inside the answers; silences under 0.25 s count as speech.",
        "Longer is not better; it depends on the questions.",
        how="voice detector; silences under 0.25 s count as speech"),
    "speech_rate": Label(
        "M02", "speech rate", "estimated", "medium", "fluentfolio",
        "Transcript syllables divided by the total answer time, pauses included.",
        f"Faster is not better; this describes {SPEAKING}. Syllables come from spelling.",
        how="syllables counted from spelling / total answer time", syllables=True),
    "articulation_rate": Label(
        "M03", "articulation rate", "estimated", "medium", "fluentfolio",
        "Transcript syllables divided by the speaking time, pauses of 0.25 s or more left out.",
        f"Faster is not better; this describes {SPEAKING}. Syllables come from spelling.",
        how="syllables counted from spelling / speaking time", syllables=True),
    "pauses_per_min": Label(
        "M04", "silent pauses of 0.25 s or more", "measured", "high", "vad",
        "Silences of 0.25 s or more inside the answers, per minute of answer time, from the voice detector.",
        "Fewer pauses is not better: where a pause falls matters more than how many there are.",
        how="voice detector; count / minutes of answer time"),
    "mean_pause": Label(
        "M05", "mean silent pause", "measured", "high", "vad",
        "The mean length of the silent pauses of 0.25 s or more inside the answers.",
        "Shorter is not better on its own.", how="voice detector; mean of the pauses of 0.25 s or more"),
    "pauses_1s": Label(
        "M06", "pauses of 1 s or more", "measured", "high", "vad",
        "How many silent pauses inside the answers last 1 s or more.",
        "A long pause can be planning; read it with the transcript.", how="voice detector; count"),
    "pauses_500ms": Label(
        "DM01", "pauses of 0.5 s or more", "measured", "high", "vad",
        "How many silent pauses inside the answers last 0.5 s or more.",
        "A view between the 0.25 s and 1 s counts.", details_only=True),
    "pauses_in_clause": Label(
        "M07", "pauses inside a clause", "estimated", "medium", "nlp",
        "Percent of the silent pauses of 0.25 s or more whose words on both sides belong to one clause in "
        "spaCy's parse of the answer; the other pauses fall at a clause boundary.",
        "Fewer is not better on its own: a pause inside a clause can be planning or a search for a word. "
        "The parser is trained on written text, so read each pause in the transcript.",
        how="grammar parser (spaCy) on the transcript; pauses from the voice detector"),
    "internal_pauses": Label(
        "M07", "internal pauses (no parser)", "estimated", "low", "fluentfolio",
        "Percent of the silent pauses of 0.25 s or more that fall inside a sentence of the transcript's "
        "punctuation; without the grammar parser, clause boundaries are not known.",
        "Sentence ends come from the speech model's punctuation, so this says less than pauses inside a "
        "clause; read each pause in the transcript.",
        how="transcript punctuation, as the grammar parser did not run"),
    "mean_length_of_run": Label(
        "M08", "mean length of run", "estimated", "medium", "fluentfolio",
        "Mean number of syllables between silent pauses of 0.25 s or more.",
        f"Longer is not better on its own; this describes {SPEAKING}.",
        how="syllables counted from spelling between pauses of 0.25 s or more", syllables=True),
    "latency_median": Label(
        "M09a", "response latency (median)", "measured", "medium", "asr",
        "Median time from a question's end to the first word of its answer; overlapping starts left out.",
        "A short wait is not better: thinking before answering is normal.",
        how="question end to the answer's first word, from the speech model's word times", modes=ASKED),
    "latency_over_10s": Label(
        "M09b", "answers after more than 10 s", "measured", "medium", "asr",
        "How many answers started more than 10 s after their question ended; these are not in the median.",
        "Check these by listening: the question may have been unclear.",
        how="question end to the answer's first word; count", modes=ASKED),
    "filled_pauses": Label(
        "M10", "filled pauses", "estimated", "low", "asr",
        "Hesitation sounds (um, uh, er and similar) per minute of answer time, as the transcript has them.",
        "The recogniser may miss or add hesitation sounds; fewer is not better on its own.",
        how="speech model's transcript; count / minutes of answer time"),
    "words_per_minute": Label(
        "DM02", "words per minute", "estimated", "medium", "fluentfolio",
        "Transcript words other than hesitation sounds, per minute of answer time.",
        "Kept out of the text for an AI chat, which might turn it into a band rule.", details_only=True),
    "mattr": Label(
        "M16", "word variety (MATTR, 50-word windows)", "measured", "high", "fluentfolio",
        "The share of different words in each window of 50 words, averaged over all windows.",
        "Higher variety is not better on its own; topic and answer length change it.",
        how="different words / 50 in each 50-word window, averaged", text=True),
    "outside_2k": Label(
        "M17a", "content words outside the 2,000 most common", "measured", "high", "wordfreq",
        f"Percent of content words that are not among the 2,000 most frequent English words. {BY_BASE_FORM}",
        "Rare is not good on its own; a rare word can be used wrongly.", how=BY_RANK, text=True),
    "outside_5k": Label(
        "M17b", "content words outside the 5,000 most common", "measured", "high", "wordfreq",
        f"Percent of content words that are not among the 5,000 most frequent English words. {BY_BASE_FORM}",
        "Rare is not good on its own; a rare word can be used wrongly.", how=BY_RANK, text=True),
    "marker_variety": Label(
        "M18", "linking-word variety", "measured", "high", "fluentfolio",
        "Different linking words and phrases from Fluentfolio's list, divided by all uses of them.",
        "More variety is not better on its own; a marker can be used wrongly.",
        how="different linking words / all their uses (list X03)", text=True),
    "npvi_v": Label(
        "DM03", "vowel length contrast (nPVI-V)", "context", "medium", "mfa",
        "Mean difference in length between neighbouring vowel stretches, as a percent of their mean length.",
        RHYTHM, details_only=True),
    "varco_v": Label(
        "DM04", "vowel length variation (VarcoV)", "context", "medium", "mfa",
        "Standard deviation of the vowel stretch lengths, as a percent of their mean.",
        RHYTHM, details_only=True),
    "percent_v": Label(
        "DM05", "vowel share of speech (%V)", "context", "medium", "mfa",
        "Percent of the time inside words that vowels take up.",
        RHYTHM, details_only=True),
    "vowel_ms": Label(
        "DM06", "mean vowel length", "context", "medium", "mfa",
        "Mean length of one vowel sound, from MFA's phone times.",
        RHYTHM, details_only=True),
    "consonant_ms": Label(
        "DM07", "mean consonant length", "context", "medium", "mfa",
        "Mean length of one consonant sound, from MFA's phone times.",
        RHYTHM, details_only=True),
    "pitch_range": Label(
        "M11", "pitch range (p20-p80)", "context", "medium", OURS,
        "The distance in semitones between the 20th and the 80th percentile of the candidate's pitch, over "
        "the answers with 10 s of voiced speech or more. Praat, SwiftF0 and openSMILE (when enabled) each "
        "measure it, and their values are combined with fixed weights.",
        f"{PITCH_AGREEMENT} A wider range is not better on its own: voices, topics and styles differ.",
        how=PITCH_TOOLS, pitch=True),
    "median_pitch": Label(
        "M19", "median pitch", "context", "medium", OURS,
        "The middle of the candidate's pitch over the answers with 10 s of voiced speech or more, combined "
        "from the pitch tools in semitones and shown in Hz.",
        f"{PITCH_AGREEMENT} Context only: a voice's pitch says nothing about how well it speaks.",
        how=PITCH_TOOLS, pitch=True),
    "flat_stretches": Label(
        "M12", "flat stretches of 2 s or more", "context", "medium", "praat",
        "Stretches between pauses that last 2 s or more, with 1 s of pitch or more, whose pitch stays within "
        "2 semitones (10th to 90th percentile). A semitone is the smallest step on a piano.",
        "Heuristic threshold, not a norm: a list or a calm, even style can be flat.",
        how="heuristic: Praat pitch within 2 semitones for 2 s or more", pitch=True),
    "rising_endings": Label(
        "M13a", "phrase endings: rising", "context", "medium", "praat",
        "Stretches between pauses whose pitch rises by 2 semitones or more over their last 0.3 s of voice.",
        ENDING, how="heuristic: Praat pitch rises 2 semitones or more in the last 0.3 s", pitch=True),
    "falling_endings": Label(
        "M13b", "phrase endings: falling", "context", "medium", "praat",
        "Stretches between pauses whose pitch falls by 2 semitones or more over their last 0.3 s of voice.",
        ENDING, how="heuristic: Praat pitch falls 2 semitones or more in the last 0.3 s", pitch=True),
    "level_endings": Label(
        "M13c", "phrase endings: level", "context", "medium", "praat",
        "Stretches between pauses whose pitch moves by less than 2 semitones over their last 0.3 s of voice.",
        ENDING, how="heuristic: Praat pitch moves under 2 semitones in the last 0.3 s", pitch=True),
    "loudness_variation": Label(
        "M14", "loudness variation", "context", "medium", "opensmile",
        "How much the loudness of the candidate's speech varies: its standard deviation divided by its mean, "
        "pauses left out (openSMILE, eGeMAPS).",
        "Context only: the microphone, its distance and the room change it; more is not better on its own.",
        how="openSMILE eGeMAPS; standard deviation / mean, pauses left out"),
    "loudness_peaks": Label(
        "M15", "loudness peaks", "context", "medium", "opensmile",
        "Peaks in the loudness of the candidate's speech per second, pauses left out (openSMILE, eGeMAPS).",
        "Context only: it follows syllables and stress, and the recording changes it.",
        how="openSMILE eGeMAPS; peaks per second, pauses left out"),
    "f0_sd": Label(
        "DM08", "pitch variation (standard deviation)", "context", "medium", "praat",
        "Standard deviation of the pitch in semitones over the frames where Praat found a pitch.",
        "More variation is not better on its own; voices and topics differ.", details_only=True, pitch=True),
    "voiced_share": Label(
        "DM09", "voiced share", "context", "medium", "praat",
        "Percent of the candidate's speech frames in which Praat found a pitch.",
        "It depends on the sounds spoken and on the recording; context only.", details_only=True,
        pitch=True),
    "jitter": Label(
        "DM10", "jitter (local)", "context", "medium", "praat",
        f"Mean change in length between neighbouring voice cycles, as a percent of the mean, {OVER_RUNS}.",
        VOICE, details_only=True),
    "shimmer": Label(
        "DM11", "shimmer (local)", "context", "medium", "praat",
        f"Mean change in loudness between neighbouring voice cycles, as a percent of the mean, {OVER_RUNS}.",
        VOICE, details_only=True),
    "hnr": Label(
        "DM12", "harmonics-to-noise ratio", "context", "medium", "praat",
        f"How much stronger the regular part of the voice is than its noise, in dB, {OVER_RUNS}.",
        VOICE, details_only=True),
    "f1": Label(
        "DM13a", "first formant (mean)", "context", "medium", "praat",
        f"Mean frequency of the lowest resonance of the mouth and throat, {OVER_RUNS}.",
        FORMANT, details_only=True),
    "f2": Label(
        "DM13b", "second formant (mean)", "context", "medium", "praat",
        f"Mean frequency of the second resonance of the mouth and throat, {OVER_RUNS}.",
        FORMANT, details_only=True),
    "f3": Label(
        "DM13c", "third formant (mean)", "context", "medium", "praat",
        f"Mean frequency of the third resonance of the mouth and throat, {OVER_RUNS}.",
        FORMANT, details_only=True),
}  # fmt: skip


@dataclass(frozen=True)
class PointerLabel:
    id: str
    name: str
    cap: int | None  # most items shown to an AI chat; None: no cap
    details_only: bool = False


POINTERS = {
    "repetitions": PointerLabel("X01", "repetition candidates", 15),
    "self_corrections": PointerLabel("X02", "self-correction candidates", 15),
    "discourse_markers": PointerLabel("X03", "linking words and phrases used", None),
    "rare_words": PointerLabel("X04", "rare words used", 20),
    "structures": PointerLabel("X05", "sentence structures, two examples of each", None),
    "listed_phrases": PointerLabel(
        "DX01", '"like", "you know" and "I mean", to read in context', None, details_only=True
    ),
    # Every word the check lists: they are the clues for the Pronunciation estimate without audio.
    "words_to_check": PointerLabel("X06", "words to check (pronunciation)", None),
}
COVERAGE = PointerLabel("X07", "what the pronunciation check covered", None)  # a note, no items
NOT_CHECKED = "Not checked"  # how a pointer note starts when its check did not run
NONE_FLAGGED = "None flagged. "  # how the words-to-check note starts when it lists none


def reliability(key: str, role_confidence: str, style: str, unsure_pct: float | None, audio: str) -> Level:
    """The fixed downgrade rules: unsure roles, poor audio for pitch, a smoothed transcript, unsure words."""
    label = LABELS[key]
    if role_confidence == "low" or (label.pitch and audio in POOR_AUDIO):
        return "low"
    level = label.reliability
    if label.syllables and unsure_pct is not None and unsure_pct > UNSURE_LIMIT_PCT:
        level = "low"
    if label.text and style == "smoothed":
        level = min(level, "medium", key=RANK.index)
    return level
