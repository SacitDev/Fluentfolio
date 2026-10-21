"""The error catalogue, row by row, and proof that a test raises or records every code in it."""

import ast
import re
from pathlib import Path

import click
import pytest

from fluentfolio.cli import commands, setup_commands
from fluentfolio.cli.main import cli, main
from fluentfolio.core.errors import AppError, CancelledError, InputError, ResourceError, SetupError, catalogue
from fluentfolio.settings.model import Limits

# The exit code of each category, from the design's exit-code table (not read from the code).
EXIT = {"internal": 1, "setup": 3, "input": 4, "needs_user": 5, "busy": 6, "stage": 7}
OVERRIDES = {"E_INTERRUPTED": 130, "E_USAGE": 2}
UNIT = Path(__file__).parent

# Where each code is raised or recorded by a test: a file here in tests/unit, or a path under tests/.
# A new code needs a line here, so it cannot go untested.
TRIGGERED = {
    "E_INTERNAL": "test_loop.py::test_inline_steps_write_their_result_and_an_unexpected_error_is_internal",
    "E_USAGE": (
        "e2e_cli/test_contract.py::test_a_usage_error_gives_the_envelope_without_the_path_that_was_typed"
    ),
    "E_CONSENT_REQUIRED": (
        "test_job_commands.py::test_two_people_need_the_other_persons_agreement_before_any_job"
    ),
    "E_BUSY": "test_locks.py::test_a_second_holder_is_told_busy_until_the_first_process_ends",
    "E_DELETE_INCOMPLETE": (
        "test_web_pages.py::test_a_file_another_program_holds_stays_the_rest_goes_and_the_card_says_so"
    ),
    "E_JOB_NOT_FOUND": "test_jobs_store.py::test_an_unknown_or_malformed_job_id_is_not_found",
    "E_JOB_NOT_FINISHED": (
        "test_audio_check_export.py::test_an_answer_needs_a_finished_job_and_in_role_swap_the_person"
    ),
    "E_RESUME_OTHER_VERSION": "test_loop.py::test_a_job_made_by_another_version_is_not_continued",
    "E_PATH_RULE": "test_facade.py::test_the_models_folder_follows_the_path_rule",
    "E_MOVE_FAILED": "test_files.py::test_a_copy_that_does_not_match_is_removed_and_the_source_kept",
    "E_HOME_SET_BY_ENV": "test_facade.py::test_a_data_folder_chosen_by_the_environment_is_not_moved",
    "E_DISK_FULL": "test_files.py::test_a_full_disk_leaves_no_half_file_and_says_how_much_space_is_needed",
    "E_COMPONENT_UNAVAILABLE": "test_loop.py::test_an_unavailable_optional_step_gives_gaps_not_a_failure",
    "E_STAGE_CRASHED": "test_loop.py::test_a_crash_without_a_result_gives_its_exit_code_in_hex",
    "E_STAGE_TIMEOUT": "test_loop.py::test_a_silent_step_is_killed_with_its_children_by_the_watchdog",
    "E_GPU_OOM": "test_loop.py::test_a_gpu_out_of_memory_runs_once_more_on_the_cpu_and_says_so",
    "E_CUDA_RUNTIME": "test_loop.py::test_the_cpu_retry_happens_exactly_once",
    "E_GPU_UNAVAILABLE": "test_asr.py::test_whisper_sent_to_the_gpu_without_one_says_so",
    "E_RAM_OOM": "test_loop.py::test_an_uncaught_memory_error_is_reported_as_out_of_memory",
    "E_JOB_FAILED": "test_loop.py::test_a_failed_required_step_fails_the_job_with_the_childs_own_code",
    "E_JOB_CANCELLED": "test_loop.py::test_a_cancel_request_ends_the_step_and_its_children_within_seconds",
    "E_JOB_INTERRUPTED": "test_jobs_store.py::test_a_damaged_or_abandoned_job_reads_as_interrupted",
    "E_INTERRUPTED": "test_loop.py::test_ctrl_c_marks_the_job_interrupted_and_kills_the_child",
    "W_SETTINGS_UNKNOWN_KEY": "test_settings.py::test_unknown_keys_are_ignored_and_bad_values_fall_back",
    "W_SETTINGS_INVALID": "test_settings.py::test_unknown_keys_are_ignored_and_bad_values_fall_back",
    "W_SETTINGS_UNREADABLE": "test_settings.py::test_an_unreadable_file_is_backed_up_and_defaults_used",
    "W_SETTINGS_NEWER": "test_settings.py::test_a_newer_file_is_used_read_only_and_never_reset",
    "W_CPU_FALLBACK": "test_loop.py::test_a_gpu_out_of_memory_runs_once_more_on_the_cpu_and_says_so",
    "E_INPUT_UNSUPPORTED": "test_ingest.py::test_the_type_comes_from_the_first_bytes_and_playlists_are_out",
    "E_INPUT_TOO_SHORT": "test_ingest.py::test_the_length_limits_at_each_edge",
    "E_INPUT_TOO_LONG": "test_ingest.py::test_the_length_limits_at_each_edge",
    "E_CONFIRM_LONG": "test_ingest.py::test_the_length_limits_at_each_edge",
    "W_INPUT_SHORT": "test_ingest.py::test_the_length_limits_at_each_edge",
    "E_INPUT_TOO_LARGE": "test_ingest.py::test_the_size_limit_at_its_edge",
    "E_DOWNLOAD_UNSAFE": "test_files.py::test_an_unsafe_archive_is_refused_whole",
    "E_INPUT_DAMAGED": "integration/test_media.py::test_a_truncated_mp4_fails_its_step_and_the_app_goes_on",
    "E_INPUT_UNREADABLE": "test_ingest.py::test_a_file_that_cannot_be_opened_is_refused_without_its_path",
    "E_INPUT_NO_AUDIO": "integration/test_media.py::test_bad_inputs_get_their_own_error_codes",
    "E_INPUT_SILENT": "integration/test_media.py::test_silence_noise_or_a_steady_chord_means_no_speech",
    "E_MODEL_MISSING": "test_models.py::test_analysis_finds_missing_or_damaged_files_without_downloading",
    "E_HASH_MISMATCH": "test_models.py::test_a_flipped_byte_is_fetched_twice_then_fails_and_leaves_nothing",
    "E_NETWORK": "test_models.py::test_a_missing_file_on_the_server_is_a_network_error",
    "E_TLS_INTERCEPTED": "test_models.py::test_a_tls_error_retries_once_without_xet_and_restores_the_switch",
    "E_TOKEN_MISSING": "test_models.py::test_gated_model_errors_get_their_own_codes",
    "E_TOKEN_INVALID": "test_models.py::test_gated_model_errors_get_their_own_codes",
    "E_HF_TERMS_NOT_ACCEPTED": "test_models.py::test_gated_model_errors_get_their_own_codes",
    "E_INPUT_TRACKS": "integration/test_media.py::test_one_track_per_person_needs_two_channels_or_two_tracks",
    "E_INPUT_LAYOUT": "test_analysis_plan.py::test_a_layout_that_does_not_fit_is_refused",
    "W_TRACK_LENGTHS": "integration/test_media.py::test_two_files_line_up_by_the_offset_and_lengths_warn",
    "W_TRACK_BLEED": "test_speakers.py::test_heavy_bleed_on_both_tracks_suggests_the_mixed_layout",
    "W_BACKGROUND_MUSIC": (
        "integration/test_media.py::test_music_under_speech_warns_and_the_same_speech_without_it_does_not"
    ),
    "W_ONE_VOICE": "test_speakers.py::test_the_speaker_policy",
    "W_THIRD_VOICE": "test_speakers.py::test_the_speaker_policy",
    "W_OTHER_VOICE": "test_speakers.py::test_the_speaker_policy",
    "W_SIMILAR_VOICES": "test_speakers.py::test_two_voices_that_sound_alike_cap_role_confidence",
    "W_SECOND_VOICE_SOLO": "test_speakers.py::test_a_second_voice_in_solo_warns_over_5_percent_of_speech",
    "W_SPEAKERS_FALLBACK": "test_evidence.py::test_a_fallback_names_what_ran_instead_and_warns",
    "W_WORDS_MISSING": "test_asr.py::test_a_chunk_that_stays_empty_is_kept_as_heard_and_warned",
    "W_ROLES_UNAVAILABLE": "test_structure.py::test_one_voice_gives_no_roles_unless_the_user_names_it",
    "W_FREQUENT_SWAPS": "test_structure.py::test_three_overruled_stretches_raise_frequent_swaps",
    "W_ROLE_CHANGE": "test_structure.py::test_an_interview_whose_roles_change_for_good_warns",
    "E_OVERRIDES_INVALID": "test_structure.py::test_unreadable_overrides_are_refused",
    "E_INPUT_TEXT": "test_secondary.py::test_a_text_may_have_at_most_2_mb",
    "E_INPUT_OPTION": "test_analysis_plan.py::test_a_wrong_swap_style_is_refused_before_any_job",
    "E_OPENSMILE_CONSENT_MISSING": (
        "integration/test_acoustic_tools.py::test_opensmile_without_consent_is_disabled_and_runs_nothing"
    ),
    "W_LIGHT_RUN": "test_evidence.py::test_a_light_run_says_which_tools_it_left_out_and_warns",
    "E_SELFTEST_FAILED": "test_doctor.py::test_a_failed_self_test_of_a_needed_part_blocks",
    "W_NOT_INSTALLED": "test_doctor.py::test_an_empty_models_folder_blocks_on_the_chosen_speech_model",
    "W_LOW_RAM": "test_doctor.py::test_memory_under_12_gb_warns_that_analyses_are_slow",
    "W_FREE_RAM": (
        "test_job_commands.py::test_less_free_memory_than_a_steps_minimum_warns_before_the_run_and_in_its_result"
    ),
    "W_DISK_LOW": "test_doctor.py::test_free_space_limits_at_their_edges",
    "W_GPU_NOT_USED": "test_doctor.py::test_the_gpu_row_says_why_the_cpu_runs",
    "W_TOKEN_FILE_PLACE": "test_doctor.py::test_a_token_file_inside_the_app_data_or_a_git_tree_warns",
    "W_NOT_OFFLINE": "test_doctor.py::test_an_empty_models_folder_blocks_on_the_chosen_speech_model",
    "E_RUNNER_NOT_STARTED": (
        "test_job_commands.py::test_a_detached_job_no_runner_took_is_left_interrupted_and_resume_runs_it"
    ),
    "E_INSTALLER_NEEDED": "test_components.py::test_opensmile_needs_consent_then_the_installer",
    "E_CONFIRM_NEEDED": (
        "e2e_cli/test_setup_commands.py::test_without_a_terminal_nothing_waits_and_yes_never_gives_consent"
    ),
    "E_SETTINGS_INVALID": (
        "test_settings.py::test_a_setting_that_does_not_exist_is_refused_without_its_name_typed"
    ),
    "E_SETTINGS_READ_ONLY": (
        "e2e_cli/test_settings_commands.py::test_set_waits_for_a_job_and_never_rewrites_a_newer_file"
    ),
    "E_REVIEW_INVALID": "test_web_pages.py::test_review_corrections_that_do_not_fit_are_refused",
    "E_UPLOAD_GONE": "test_web_pages.py::test_a_start_without_its_upload_asks_for_the_recording_again",
    "E_NO_RECORDING": "test_web_pages.py::test_a_start_with_no_recording_chosen_asks_for_one_first",
    "E_NO_MODE": "test_web_pages.py::test_no_mode_is_chosen_before_the_first_analysis_then_the_last_one_is",
    "E_PAGE_NOT_FOUND": (
        "test_web_security.py::test_a_missing_page_leads_to_history_and_a_card_put_in_place_comes_as_200"
    ),
    # The installer scripts print these themselves.
    "E_NOT_SUPPORTED": (
        "test_installer.py::"
        "test_an_unsupported_processor_or_an_administrator_window_is_refused_before_anything_is_written"
    ),
    "E_ELEVATED": (
        "test_installer.py::"
        "test_an_unsupported_processor_or_an_administrator_window_is_refused_before_anything_is_written"
    ),
    "E_SAC_ENFORCED": (
        "test_installer.py::test_smart_app_control_on_stops_the_install_and_evaluation_mode_warns"
    ),
    "W_SAC_EVALUATION": (
        "test_installer.py::test_smart_app_control_on_stops_the_install_and_evaluation_mode_warns"
    ),
    "E_APP_RUNNING": "test_installer.py::test_a_running_app_stops_the_installer_and_the_uninstaller",
    "E_SOURCE": "test_installer.py::test_a_folder_without_a_release_is_refused_and_logged",
    "E_DISK_PREFLIGHT": "test_installer.py::test_too_little_disk_space_stops_the_install_with_both_numbers",
    "E_VCRUNTIME_MISSING": "test_installer.py::test_a_missing_visual_cpp_runtime_is_named_with_its_download",
    "E_FILE_BLOCKED": "test_installer.py::test_a_failed_tool_gets_its_code_from_its_own_words",
    "E_INSTALL_FAILED": "test_installer.py::test_a_failed_tool_gets_its_code_from_its_own_words",
    "E_UPDATE_FAILED_OLD_KEPT": (
        "test_installer.py::test_a_failed_update_removes_its_new_folder_and_keeps_the_old_version"
    ),
    "E_NOT_INSTALLED": "test_installer.py::test_uninstall_removes_the_program_and_keeps_the_learners_data",
    "E_UNINSTALL_FAILED": (
        "test_installer.py::test_the_start_menu_uninstaller_removes_the_folder_it_starts_in_and_waits_for_enter"
    ),
}


@pytest.mark.parametrize("code", sorted(catalogue()))
def test_each_code_has_its_exit_code_message_next_step_and_help_link(code: str) -> None:
    entry = catalogue()[code]
    assert entry.code == code and entry.title and entry.message and entry.hint
    assert all(text.isascii() and "\n" not in text for text in (entry.title, entry.message, entry.hint))
    assert entry.doc == f"docs/troubleshooting.md#{code.lower()}"
    assert isinstance(entry.retryable, bool)
    if code.startswith("W_"):
        assert (entry.severity, entry.category, entry.exit_code) == ("warning", None, 0)
    else:
        assert code.startswith("E_") and entry.severity == "error"
        assert entry.exit_code == OVERRIDES.get(code, EXIT[entry.category])


TROUBLESHOOTING = UNIT.parents[1] / "docs" / "troubleshooting.md"
# "### E_CODE", a blank line, then the card's title in bold: GitHub makes the anchor "#e_code" from it.
SECTION = re.compile(r"^### ([EW]_[A-Z0-9_]+)\n\n\*\*(.+?)\.?\*\*", re.MULTILINE)


def test_the_troubleshooting_page_has_one_section_per_code_under_the_title_its_card_shows() -> None:
    sections = SECTION.findall(TROUBLESHOOTING.read_text(encoding="utf-8"))
    assert sorted(code for code, _ in sections) == sorted(catalogue())
    assert all(title == catalogue()[code].title for code, title in sections)


@pytest.mark.parametrize(
    ("kind", "code", "exit_code"),
    [
        (InputError, "E_JOB_NOT_FOUND", 4),
        (SetupError, "E_DISK_FULL", 3),
        (ResourceError, "E_BUSY", 6),
        (CancelledError, "E_INTERRUPTED", 130),
    ],
)
def test_an_error_carries_its_catalogue_row(kind: type[AppError], code: str, exit_code: int) -> None:
    error = kind(code, "need 2 GB, have 1 GB")
    assert (error.code, error.entry.exit_code, error.detail) == (code, exit_code, "need 2 GB, have 1 GB")
    assert str(error) == f"{code}: {catalogue()[code].message} (need 2 GB, have 1 GB)"


# The command each hint gives, as the hint writes it; the capital words are what the learner fills in.
HINT_COMMANDS = {
    "E_INTERNAL": "fluentfolio doctor --report report.txt",
    "E_USAGE": "fluentfolio COMMAND --help",
    "E_BUSY": "fluentfolio cancel JOB",
    "E_JOB_NOT_FOUND": "fluentfolio jobs list",
    "E_RESUME_OTHER_VERSION": "fluentfolio analyze FILE --mode MODE",
    "E_COMPONENT_UNAVAILABLE": "fluentfolio components list",
    "E_OPENSMILE_CONSENT_MISSING": "fluentfolio components consent opensmile",
    "E_DOWNLOAD_UNSAFE": "fluentfolio doctor",
    "E_MODEL_MISSING": "fluentfolio components install --all",
    "E_HASH_MISMATCH": "fluentfolio components install --all",
    "E_TOKEN_MISSING": "fluentfolio token set",
    "E_TOKEN_INVALID": "fluentfolio token set",
    "E_HF_TERMS_NOT_ACCEPTED": "fluentfolio token status",
    "E_STAGE_CRASHED": "fluentfolio doctor --report report.txt",
    "E_JOB_CANCELLED": "fluentfolio resume JOB",
    "E_JOB_INTERRUPTED": "fluentfolio resume JOB",
    "E_INTERRUPTED": "fluentfolio resume JOB",
    "W_SPEAKERS_FALLBACK": "fluentfolio doctor",
    "E_INPUT_OPTION": "fluentfolio components list",
    "E_SELFTEST_FAILED": "fluentfolio self-test",
    "W_NOT_OFFLINE": "fluentfolio components install --all",
    "E_RUNNER_NOT_STARTED": "fluentfolio resume JOB",
    "E_SETTINGS_INVALID": "fluentfolio settings show",
}
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z-]*")
# What follows a whole command in a hint: its end, a stop, or a plain word (not a flag or a capital word).
COMMAND_END = re.compile(r"$|[.,;]|\s(?!-)(?![A-Z]{2})")


class Reached(Exception):
    """Any use of the application: the command got past its own checks of what was typed."""


class NoApp:
    def __getattr__(self, name: str) -> object:
        raise Reached(name)


def every_flag(command: click.Command = cli) -> set[str]:
    found = {o for p in command.params for o in p.opts if o.startswith("--")} | {"--help"}
    for sub in command.commands.values() if isinstance(command, click.Group) else []:
        found |= every_flag(sub)
    return found


def test_each_hint_that_gives_a_command_is_listed_with_its_whole_command() -> None:
    known = every_flag()
    for code, entry in catalogue().items():
        assert entry.hint.count("fluentfolio ") == (code in HINT_COMMANDS), code
        assert set(FLAG.findall(entry.hint)) <= known, code
        if code in HINT_COMMANDS:
            at = entry.hint.index(HINT_COMMANDS[code]) + len(HINT_COMMANDS[code])
            assert COMMAND_END.match(entry.hint, at), code


@pytest.mark.parametrize("code", sorted(HINT_COMMANDS))
def test_each_command_a_hint_gives_runs_without_a_usage_error(
    code: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "practice.m4a").write_bytes(b"\0")
    filled = {
        "JOB": "20261017-120000-ab12", "FILE": str(tmp_path / "practice.m4a"), "MODE": "solo",
        "COMMAND": "analyze",
    }  # fmt: skip
    monkeypatch.setattr(commands, "facade", NoApp())
    monkeypatch.setattr(setup_commands, "facade", NoApp())
    args = [filled.get(word, word) for word in HINT_COMMANDS[code].split()[1:]]
    assert main(args) != OVERRIDES["E_USAGE"], HINT_COMMANDS[code]


def test_the_long_recording_hint_offers_a_light_run_only_where_one_is_accepted() -> None:
    # A light run is refused at or under limits.long_min, so the hint must not offer it from confirm_min on.
    limits, hint = Limits(), catalogue()["E_CONFIRM_LONG"].hint
    assert limits.long_min > limits.confirm_min
    assert f"Over {limits.long_min} minutes (or the length set in Settings), --light-run" in hint, hint


def test_a_choice_a_message_names_in_quotes_is_the_label_the_pages_show() -> None:
    # A learner looks for the quoted words on the page, so they must be the page's own label, word for word.
    templates = UNIT.parents[1] / "src" / "fluentfolio" / "web" / "templates"
    pages = " ".join(p.read_text(encoding="utf-8") for p in templates.rglob("*.html"))
    quoted = [
        (code, words)
        for code, entry in catalogue().items()
        for text in (entry.title, entry.message, entry.hint)
        for words in re.findall(r'"([^"]+)"', text)
    ]
    assert quoted, "no quoted label left to check"
    assert [(code, words) for code, words in quoted if words not in pages] == []


def test_every_code_is_raised_or_recorded_by_a_named_test() -> None:
    assert set(TRIGGERED) == set(catalogue())
    for where in TRIGGERED.values():
        file, name = where.split("::")
        source = (UNIT / file if "/" not in file else UNIT.parent / file).read_text(encoding="utf-8")
        tests = {node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)}
        assert name in tests, where
