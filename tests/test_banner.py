"""Offline tests for the startup banner: the fish-3 art and its shadow."""
import random
import re
import unittest

from threetoks import banner, theme


class ArtShapeTest(unittest.TestCase):
    def test_every_art_row_has_one_width(self):
        self.assertEqual(len({len(row) for row in banner.FISH}), 1)
        self.assertEqual(len({len(row) for row in banner.WORDMARK}), 1)

    def test_shadow_mirrors_the_fish_row_for_row(self):
        shadow = banner.reflect(banner.FISH, banner.FISH_SHADES)
        self.assertEqual(len(shadow), len(banner.FISH))
        # The waterline rows: the tail sinks first, the eye last.
        self.assertIn("◥", shadow[0])
        self.assertIn("○", shadow[-2])
        for sky, sea in zip(reversed(banner.FISH), shadow):
            self.assertEqual(len(sky), len(sea))
            self.assertEqual([i for i, c in enumerate(sky) if c != " "],
                             [i for i, c in enumerate(sea) if c != " "])

    def test_shadow_holds_no_solid_glyphs_and_fades_with_depth(self):
        shadow = banner.reflect(banner.FISH, banner.FISH_SHADES)
        for row in shadow:
            self.assertFalse(set(row) & set(banner.FULL + banner.HALF), row)
        wordmark = banner.reflect(banner.WORDMARK, banner.WORDMARK_SHADES)
        self.assertEqual(set(wordmark[-1]), {" "})

    def test_thin_edges_sink_one_shade_lighter_than_solids(self):
        self.assertEqual(banner.reflect(("█▀",), "▓"), ["▓▒"])
        self.assertEqual(banner.reflect(("▀",), " "), [" "])
        self.assertEqual(banner.reflect(("m◉",), "▒"), ["▒○"])

    def test_accent_cells_draw_their_glyph_in_their_own_color(self):
        plain = banner._paint("m◉█", enabled=False, ramp=(1, 2))
        self.assertEqual(plain, "█◉█")
        colored = banner._paint("m◉█", enabled=True, ramp=(1, 2))
        self.assertIn(theme.fg(banner.ACCENTS["m"][1], "█"), colored)
        self.assertIn(theme.fg(banner.ACCENTS["◉"][1], "◉"), colored)
        self.assertTrue(colored.endswith(theme.fg(2, "█")))


class BuildBannerTest(unittest.TestCase):
    def test_names_model_and_agent_count_between_two_hairlines(self):
        text = banner.build_banner("tinyllama", 3, enabled=False)
        self.assertIn("tinyllama", text)
        self.assertIn("3 agents online", text)
        self.assertEqual(text.count(theme.NODE), 2)

    def test_every_line_fits_the_banner_width(self):
        text = banner.build_banner("m", 2, enabled=False, rng=random.Random(5))
        self.assertEqual(max(len(line) for line in text.splitlines()),
                         banner.WIDTH)

    def test_wordmark_stands_on_the_horizon(self):
        lines = banner.build_banner("m", 2, enabled=False).splitlines()
        horizon = next(i for i, line in enumerate(lines)
                       if set(line.strip()) == {banner.HORIZON})
        above = lines[horizon - len(banner.WORDMARK):horizon]
        for art, line in zip(banner.WORDMARK, above):
            self.assertTrue(line.endswith(art))
        self.assertEqual(horizon, 1 + len(banner.FISH))  # hairline, then sky

    def test_splash_lands_only_in_blank_columns(self):
        sky, _, _ = banner.banner_rows(random.Random(2))
        for index, row in enumerate(banner.FISH):
            drawn = sky[index][:len(row)]
            for art, char in zip(row, drawn):
                if art != " ":
                    self.assertEqual(char, art)
                else:
                    self.assertIn(char, " " + banner.SPLASH)

    def test_same_seed_reproduces_the_banner(self):
        one = banner.build_banner("m", 2, enabled=True, rng=random.Random(7))
        two = banner.build_banner("m", 2, enabled=True, rng=random.Random(7))
        self.assertEqual(one, two)

    def test_colors_come_from_one_ramp_and_water_from_its_dark_end(self):
        text = banner.build_banner("m", 2, enabled=True, rng=random.Random(1))
        accents = {color for _, color in banner.ACCENTS.values()}
        art_codes = _codes(text) - {theme.GRAY, theme.GRAY_LIGHT} - accents
        ramp = next(r for r in banner.BANNER_RAMPS if art_codes <= set(r))
        sea = text[text.index(banner.HORIZON):text.index(banner.TAGLINE)]
        self.assertTrue(_codes(sea) - {theme.GRAY} <= set(ramp[:banner.WATER_DEPTH]))


def _codes(chunk: str) -> set:
    """The xterm-256 foreground codes used in ``chunk``."""
    return {int(code) for code in re.findall(r"38;5;(\d+)m", chunk)}


if __name__ == "__main__":
    unittest.main()
