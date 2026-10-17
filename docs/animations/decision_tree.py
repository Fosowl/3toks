"""Render an illustrative ThreeToks web-research trace with Manim 0.19.2.

Run: python3 animations/decision_tree.py
The scene is scripted, not a recorded model run. No network calls are made.
"""
from pathlib import Path

import numpy as np
from manim import (
    AnimationGroup, Circle, Create, CubicBezier, Dot, FadeIn, FadeOut,
    Flash, Indicate, Line, MoveAlongPath, RoundedRectangle, Scene, Text,
    VGroup, smooth, tempconfig,
)

BACKGROUND = "#0B1020"
SURFACE = "#121C30"
WHITE = "#EDF4FF"
MUTED = "#91A3BF"
EDGE = "#33445E"
TEAL = "#57E0CB"
GOLD = "#FFC875"
FONT = "Avenir Next"
OUTPUT = Path(__file__).resolve().parent


def label(words, size=24, color=WHITE):
    """Build scene text from storyboard words.

    Args:
        words (str): Text from this scene's storyboard.
        size (int): Font size in Manim units.
        color (str): Hex color from the scene palette.
    Returns:
        Text: Text with a portable font fallback.
    """
    return Text(words, font=FONT, font_size=size, color=color)


def vertex(words, position, color=EDGE):
    """Build a labeled decision point for the graph.

    Args:
        words (str): Stage name from the storyboard.
        position (tuple): Scene coordinates for the point.
        color (str): Hex color for its ring.
    Returns:
        VGroup: Ring, center, and stage name in that order.
    """
    ring = Circle(radius=0.105, color=color, stroke_width=2.5)
    ring.set_fill(BACKGROUND, opacity=1).move_to(position)
    center = Dot(position, radius=0.035, color=color)
    directions = {"route": (1, 0, 0), "web": (1, 0, 0),
                  "search results": (-1, 0, 0), "page A": (-1, 0, 0), "page B": (1, 0, 0)}
    direction = np.array(directions.get(words, (0, -1, 0)))
    text = label(words, 21, MUTED).next_to(ring, direction, buff=0.15)
    return VGroup(ring, center, text)


def exchange(old, new):
    """Replace scene text without overlapping readable letters.

    Args:
        old (VGroup): Existing card or caption from the scene.
        new (VGroup or Text): Replacement built by the scene.
    Returns:
        AnimationGroup: Quick sequential fades followed by a reading hold.
    """
    return AnimationGroup(FadeOut(old), FadeIn(new), lag_ratio=1,
                          rate_func=lambda time: smooth(min(1, time * 3)))


def branch(start, end):
    """Connect graph points with a smooth path.

    Args:
        start (array): Source point from a scene vertex.
        end (array): Destination point from a scene vertex.
    Returns:
        CubicBezier: Path behind the labeled points.
    """
    bend = np.array([0, (end[1] - start[1]) * 0.5, 0])
    return CubicBezier(start, start + bend, end - bend, end,
                       stroke_color=EDGE, stroke_width=2).set_z_index(-1)


class DecisionTree(Scene):
    """Show a compressed web trace with a rejected and a revised answer."""

    def construct(self):
        """Run the storyboard for Manim. Args: none. Returns: None."""
        self.camera.background_color = BACKGROUND
        self.make_layout()
        self.introduce()
        self.route_request()
        self.open_result()
        self.select_evidence()
        self.reject_draft()
        self.refine_search()
        self.read_second_page()
        self.accept_answer()
        self.close_scene()

    def make_layout(self):
        """Build the scene from fixed coordinates. Args: none. Returns: None."""
        points = {
            "route": ("route", (-4.1, 1.55, 0)),
            "code": ("code", (-6.15, 0.55, 0)),
            "web": ("web", (-4.1, 0.55, 0)),
            "files": ("files", (-2.05, 0.55, 0)),
            "search": ("search results", (-4.1, -0.55, 0)),
            "page_a": ("page A", (-5.7, -1.7, 0)),
            "page_b": ("page B", (-2.5, -1.7, 0)),
            "notes": ("evidence notes", (-4.1, -2.85, 0)),
        }
        self.nodes = {key: vertex(*value) for key, value in points.items()}
        pairs = ("route/code", "route/web", "route/files", "web/search",
                 "search/page_a", "search/page_b", "page_a/notes", "page_b/notes")
        self.edges = {pair: branch(*(self.nodes[key][0].get_center()
                      for key in pair.split("/"))) for pair in pairs}
        self.cursor = Dot(radius=0.065, color=TEAL).set_z_index(5)
        self.panel = VGroup()
        self.caption = VGroup()

    def introduce(self):
        """Show the question and trace label. Args: none. Returns: None."""
        brand = label("THREETOKS", 17, TEAL).move_to((-5.85, 3.55, 0))
        title = label("Small choices. Deep research.", 39).move_to((0, 2.98, 0))
        question = label("Why do leaves change color in autumn?", 25, MUTED)
        question.move_to((0, 2.35, 0))
        self.header = VGroup(brand, title, question)
        self.footer = label("ILLUSTRATIVE TRACE  /  TIME COMPRESSED", 13, MUTED)
        self.footer.move_to((0, -3.76, 0))
        self.divider = Line((1.05, 1.65, 0), (1.05, -3.2, 0), color=EDGE)
        self.play(FadeIn(self.header, shift=np.array([0, 0.12, 0])), run_time=0.8)
        self.play(FadeIn(self.footer), Create(self.divider), run_time=0.6)
        self.wait(0.6)

    def set_panel(self, heading, rows):
        """Prepare the right-hand detail card for a transition.

        Args:
            heading (str): Card title from the storyboard.
            rows (list): Pairs of text and palette color.
        Returns:
            AnimationGroup: Old card out and new card in.
        """
        box = RoundedRectangle(width=5.1, height=4.7, corner_radius=0.18,
                               stroke_color=EDGE, stroke_width=1)
        box.set_fill(SURFACE, opacity=0.85).move_to((3.95, -0.66, 0))
        title = label(heading, 18, TEAL).move_to((1.72, 1.28, 0), aligned_edge=(-1, 0, 0))
        body = VGroup(*(label(text, 23, color) for text, color in rows))
        body.arrange((0, -1, 0), aligned_edge=(-1, 0, 0), buff=0.31)
        if body.width > 4.45:
            body.scale_to_fit_width(4.45)
        body.move_to((1.72, 0.66, 0), aligned_edge=(-1, 1, 0))
        new = VGroup(box, title, body)
        old, self.panel = self.panel, new
        return exchange(old, new)

    def set_caption(self, words):
        """Prepare the graph caption for a transition.

        Args:
            words (str): Short explanation of the current action.
        Returns:
            AnimationGroup: Caption replacement.
        """
        new = label(words, 19, WHITE).move_to((-3.65, 1.99, 0))
        old, self.caption = self.caption, new
        return exchange(old, new)

    def travel(self, key, color=TEAL):
        """Prepare a glowing traversal on one graph edge.

        Args:
            key (str): Edge name from the fixed graph.
            color (str): Hex color for this traversal.
        Returns:
            AnimationGroup: Path highlight and moving cursor.
        """
        path = self.edges[key]
        self.cursor.move_to(path.get_start()).set_color(color)
        self.add(self.cursor)
        return AnimationGroup(path.animate.set_stroke(color, width=3),
                              MoveAlongPath(self.cursor, path, rate_func=smooth))

    def light_node(self, key, color=TEAL):
        """Prepare an active vertex style.

        Args:
            key (str): Vertex name from the fixed graph.
            color (str): Hex color for the active vertex.
        Returns:
            AnimationGroup: Ring, center, and label highlights.
        """
        return AnimationGroup(*(part.animate.set_color(color) for part in self.nodes[key]))

    def route_request(self):
        """Route the example question to web. Args: none. Returns: None."""
        self.play(FadeIn(self.nodes["route"]), self.set_caption("The model chooses. Python acts."),
                  self.set_panel("01 / ROUTE", [("1   casual", MUTED), ("2   web", TEAL),
                  ("3   files", MUTED), ("4   code", MUTED),
                  ("MENU  →  2", TEAL)]), run_time=0.7)
        self.play(*(Create(self.edges[f"route/{key}"]) for key in ("code", "web", "files")),
                  *(FadeIn(self.nodes[key]) for key in ("code", "web", "files")), run_time=0.5)
        self.play(self.travel("route/web"), self.light_node("web"), run_time=0.8)

    def open_result(self):
        """Search and open a selected page. Args: none. Returns: None."""
        self.play(FadeIn(self.nodes["search"]), self.travel("web/search"),
                  self.set_panel("02 / SEARCH + OPEN", [("1   Photo gallery", MUTED),
                  ("2   Leaf pigments", TEAL), ("3   Autumn walks", MUTED),
                  ("MENU  →  open result 2", TEAL)]), run_time=0.65)
        self.play(*(Create(self.edges[f"search/{key}"]) for key in ("page_a", "page_b")),
                  *(FadeIn(self.nodes[key]) for key in ("page_a", "page_b")), run_time=0.55)
        self.play(self.travel("search/page_a"), self.light_node("page_a"),
                  self.set_caption("Explore one branch, not every branch."), run_time=0.8)

    def select_evidence(self):
        """Copy selected sentences into sourced notes. Args: none. Returns: None."""
        self.play(self.set_panel("03 / READ NUMBERED SENTENCES", [
            ("1   A guide to autumn leaves", MUTED),
            ("2   Chlorophyll makes leaves green.", TEAL),
            ("3   Related articles", MUTED),
            ("4   Other pigments can show through.", TEAL),
            ("PICK  →  2, 4", GOLD)]), run_time=0.65)
        self.play(FadeIn(self.nodes["notes"]), self.travel("page_a/notes"),
                  self.light_node("notes"), run_time=0.65)
        self.play(self.set_caption("Point at evidence. Keep its source."),
                  self.set_panel("NOTES / COPIED, NOT REWRITTEN", [
                  ("Chlorophyll makes leaves green.", WHITE), ("page A · sentence 2", MUTED),
                  ("Other pigments can show through.", WHITE), ("page A · sentence 4", MUTED)]),
                  run_time=1.1)

    def reject_draft(self):
        """Reject a vague draft before another round. Args: none. Returns: None."""
        self.play(self.set_panel("04 / DRAFT → JUDGE", [("“Pigments change.”", WHITE),
                  ("DIRECT?    Not specific enough.", GOLD),
                  ("Why does autumn matter?", MUTED)]), run_time=0.6)
        self.play(self.set_caption("A weak answer is not the end."),
                  Indicate(self.nodes["notes"], color=GOLD, scale_factor=1.06), run_time=0.7)
        self.wait(0.7)

    def refine_search(self):
        """Keep notes and return with a new query. Args: none. Returns: None."""
        self.retry = CubicBezier((-3.9, -2.85, 0), (1.05, -2.85, 0),
                                 (0.8, -0.55, 0), (-3.92, -0.55, 0),
                                 stroke_color=GOLD, stroke_width=3).set_z_index(-1)
        self.play(self.set_panel("05 / REFINE + SEARCH AGAIN", [
                  ("Keep the evidence.", WHITE), ("Update the search angle.", WHITE),
                  ("QUERY", GOLD), ("shorter days chlorophyll autumn", GOLD)]), run_time=0.65)
        self.cursor.set_color(GOLD)
        self.play(Create(self.retry), MoveAlongPath(self.cursor, self.retry),
                  self.set_caption("New angle. Same evidence store."), run_time=0.75)
        self.play(self.light_node("search", GOLD), run_time=0.6)

    def read_second_page(self):
        """Read new evidence on the second branch. Args: none. Returns: None."""
        self.play(self.travel("search/page_b", GOLD), self.light_node("page_b", GOLD),
                  self.set_panel("06 / OPEN A NEW SOURCE", [
                  ("1   Shorter days help trigger", GOLD), ("     chlorophyll loss.", GOLD),
                  ("3   Yellow and orange pigments", GOLD), ("     become visible.", GOLD),
                  ("PICK  →  1, 3", TEAL)]), run_time=0.6)
        self.play(self.travel("page_b/notes", GOLD),
                  self.set_caption("New evidence fills the gap."), run_time=0.7)
        self.play(Flash(self.nodes["notes"][0], color=TEAL, flash_radius=0.25),
                  self.set_panel("07 / SYNTHESIZE FROM NOTES", [
                  ("Shorter days help trigger", WHITE), ("chlorophyll loss, revealing", WHITE),
                  ("yellow and orange pigments.", WHITE)]), run_time=0.8)

    def accept_answer(self):
        """Show all three judging lenses approving. Args: none. Returns: None."""
        self.play(self.set_panel("08 / THREE JUDGING LENSES", [
                  ("DIRECT       Answers the question", TEAL),
                  ("CLEAN        Readable, not fragments", TEAL),
                  ("USEFUL      Gives an explanation", TEAL),
                  ("All approve → return answer", WHITE)]), run_time=0.6)
        self.play(self.set_caption("Check the answer. Then return it."),
                  self.light_node("notes"), run_time=0.6)
        self.wait(0.8)

    def close_scene(self):
        """Close with the division of responsibilities. Args: none. Returns: None."""
        graph = VGroup(*self.nodes.values(), *self.edges.values(), self.retry, self.cursor)
        title = label("Menus choose. Code executes.", 42).move_to((0, 0.65, 0))
        detail = label("Tiny decisions. An evidence-guided path.", 27, MUTED)
        detail.move_to((0, -0.2, 0))
        token = label("~3 output tokens per menu choice · longer text for answers", 19, TEAL)
        token.move_to((0, -1.0, 0))
        self.play(FadeOut(graph), FadeOut(self.panel), FadeOut(self.caption),
                  FadeOut(self.divider), FadeIn(VGroup(title, detail, token)), run_time=0.7)
        self.wait(1.8)


if __name__ == "__main__":
    with tempconfig({"pixel_width": 1920, "pixel_height": 1080, "frame_rate": 60,
                     "media_dir": str(OUTPUT / ".render"),
                     "output_file": str(OUTPUT / "decision_tree.mp4"),
                     "disable_caching": True, "verbosity": "WARNING"}):
        scene = DecisionTree()
        scene.render()
        assert 18.9 <= scene.time < 20, scene.time
        print(f"Rendered {OUTPUT / 'decision_tree.mp4'} ({scene.time:.2f}s)")
