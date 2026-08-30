import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from topoppi.file_utils import git_worktree_state


class GitWorktreeStateTests(unittest.TestCase):
    def test_skips_git_outside_a_worktree(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch("topoppi.file_utils.subprocess.run") as run:
            self.assertEqual(git_worktree_state(tmpdir), (None, None))

        run.assert_not_called()

    def test_reads_revision_and_dirty_state_in_a_worktree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, ".git").mkdir()
            completed = [
                subprocess.CompletedProcess([], 0, "abc123\n", ""),
                subprocess.CompletedProcess([], 0, " M README.md\n", ""),
            ]
            with mock.patch("topoppi.file_utils.subprocess.run", side_effect=completed) as run:
                self.assertEqual(git_worktree_state(tmpdir), ("abc123", True))

        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
