"""Offline tests for source decoding and byte-faithful encoding.

Line endings (LF / CRLF / CR), the trailing newline run, BOM and coding
cookies. No model, no network.
"""
import unittest

from threetoks.code.edit import encoding


class EncodeLikeTest(unittest.TestCase):
    def test_crlf_without_trailing_newline_round_trips(self):
        original = b"a = 1\r\nb = 2"
        text = original.decode().replace("\r\n", "\n")
        self.assertEqual(encoding.encode_like(original, text), original)
        self.assertEqual(encoding.encode_like(original, "a = 1\nb = 3\n"),
                         b"a = 1\r\nb = 3")

    def test_lf_with_trailing_newline_gets_exactly_one(self):
        self.assertEqual(encoding.encode_like(b"x = 1\n", "x = 2"), b"x = 2\n")
        self.assertEqual(encoding.encode_like(b"x = 1\n", "x = 2\n"),
                         b"x = 2\n")

    def test_trailing_newline_run_is_kept(self):
        self.assertEqual(encoding.encode_like(b"a = 1\n\n", "a = 2\n\n"),
                         b"a = 2\n\n")
        self.assertEqual(encoding.encode_like(b"a = 1\r\n\r\n", "a = 2\n"),
                         b"a = 2\r\n\r\n")

    def test_cr_only_file_round_trips(self):
        original = b"a = 1\rb = 2\r"
        text = encoding.decode_source(original)
        self.assertEqual(text, "a = 1\nb = 2\n")
        self.assertEqual(encoding.encode_like(original, text), original)
        self.assertEqual(encoding.encode_like(original, "a = 1\nb = 3\n"),
                         b"a = 1\rb = 3\r")

    def test_utf8_bom_is_kept(self):
        self.assertEqual(encoding.encode_like(b"\xef\xbb\xbfx = 1\n",
                                              "x = 'é'\n"),
                         b"\xef\xbb\xbfx = '\xc3\xa9'\n")


class DecodeSourceTest(unittest.TestCase):
    def test_bom_is_dropped_and_crlf_becomes_lf(self):
        raw = b"\xef\xbb\xbfx = '\xc3\xa9'\r\ny = 2\r\n"
        self.assertEqual(encoding.decode_source(raw), "x = 'é'\ny = 2\n")
        text = encoding.decode_source(raw)
        self.assertEqual(encoding.encode_like(raw, text), raw)

    def test_coding_cookie_is_honoured_both_ways(self):
        raw = b"# -*- coding: latin-1 -*-\nx = '\xe9'\n"
        text = encoding.decode_source(raw)
        self.assertEqual(text, "# -*- coding: latin-1 -*-\nx = 'é'\n")
        self.assertEqual(encoding.encode_like(raw, text), raw)
        with self.assertRaises(UnicodeEncodeError):
            encoding.encode_like(raw, text.replace("é", "—"))

    def test_undecodable_bytes_give_none(self):
        self.assertIsNone(encoding.decode_source(b"x = '\xff'\n"))
        self.assertIsNone(
            encoding.decode_source(b"x = 1\ny = 2\nz = '\xff'\n"))
        self.assertIsNone(encoding.decode_source(b"# coding: no-such-codec\n"))


if __name__ == "__main__":
    unittest.main()
