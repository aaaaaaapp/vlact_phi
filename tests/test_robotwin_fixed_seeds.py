import pytest

from examples.Robotwin.eval_files.eval_policy_fixed_seeds import (
    label_video_file,
    parse_fixed_test_seeds,
    validate_video_file,
)


def test_parse_fixed_test_seeds_requires_exact_unique_integer_list():
    assert parse_fixed_test_seeds("[100000, 100002, 100003]", 3) == [
        100000,
        100002,
        100003,
    ]

    with pytest.raises(ValueError, match="does not match"):
        parse_fixed_test_seeds("[1, 2]", 3)
    with pytest.raises(ValueError, match="unique"):
        parse_fixed_test_seeds("[1, 1]", 2)
    with pytest.raises(ValueError, match="integer"):
        parse_fixed_test_seeds("[1, true]", 2)


def test_validate_video_file_records_nonempty_video(tmp_path):
    video = tmp_path / "episode0.mp4"
    video.write_bytes(b"x" * 1024)

    assert validate_video_file(video, 0) == {
        "file": "episode0.mp4",
        "size_bytes": 1024,
    }


def test_validate_video_file_rejects_failed_or_empty_output(tmp_path):
    video = tmp_path / "episode0.mp4"
    video.write_bytes(b"")

    with pytest.raises(RuntimeError, match="too small"):
        validate_video_file(video, 0)
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        validate_video_file(video, 1)


@pytest.mark.parametrize(
    ("succeeded", "expected_name"),
    [(True, "episode0_success.mp4"), (False, "episode0_failure.mp4")],
)
def test_label_video_file_adds_outcome_suffix(
    tmp_path, succeeded, expected_name
):
    video = tmp_path / "episode0.mp4"
    video.write_bytes(b"x" * 1024)

    labeled = label_video_file(video, succeeded)

    assert labeled.name == expected_name
    assert labeled.is_file()
    assert not video.exists()
