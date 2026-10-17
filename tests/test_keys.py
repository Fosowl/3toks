"""Tests for the raw key reader, driven through a pipe instead of a tty."""
import os
import unittest

from threetoks import keys


@unittest.skipIf(keys.IS_WINDOWS, "msvcrt path is not covered here")
class KeyboardDecodeTest(unittest.TestCase):
    def read_all(self, data: bytes, count: int) -> list:
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        os.write(write_end, data)
        os.close(write_end)
        board = keys.Keyboard.__new__(keys.Keyboard)
        board._fd, board._saved, board._pending = read_end, None, ""
        return [board.read() for _ in range(count)]

    def test_arrows_and_paging_keys(self):
        got = self.read_all(b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[Z\x1b[5~", 6)
        self.assertEqual(got, ["up", "down", "right", "left", "btab", "pgup"])

    def test_two_quick_escapes_stay_two_keys(self):
        got = self.read_all(b"\x1b\x1b\x1b[A", 3)
        self.assertEqual(got, ["esc", "esc", "up"])

    def test_multibyte_character_is_one_key(self):
        self.assertEqual(self.read_all("\u00b2\u00e9".encode(), 2),
                         ["\u00b2", "\u00e9"])

    def test_control_keys_have_names(self):
        self.assertEqual(self.read_all(b"\r\x7f\x13\t", 4),
                         ["enter", "backspace", "ctrl-s", "tab"])

    def test_unknown_sequence_is_unknown_not_fatal(self):
        self.assertEqual(self.read_all(b"\x1b[9z", 1), ["unknown"])


if __name__ == "__main__":
    unittest.main()
