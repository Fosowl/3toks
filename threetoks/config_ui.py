"""The interactive config screen: terminal loop, save, and live apply.

``/config`` in the TUI and the first start call into this module. The
screen is drawn on the terminal's alternate buffer, so the chat scrollback
comes back untouched when it closes. Keys, saving, and the probe sweep are
injected, so tests drive the whole loop with scripted keys.
"""
import functools
import shutil
import sys
from dataclasses import dataclass

from threetoks.config import ThreetoksConfig, load_config, resolve_path
from threetoks.fields import APPLY_LIVE, changed_fields
from threetoks.hints import LEVEL_BAD, LEVEL_OK
from threetoks.ini_edit import save_changes
from threetoks.keys import Keyboard, is_interactive
from threetoks.probes import sweep
from threetoks.screen_state import (ACTION_PROBE, ACTION_QUIT, ACTION_SAVE,
                                    Screen, handle_key, unsaved)
from threetoks.screen_view import render
from threetoks.theme import AMBER, color_enabled, dim, fg

_ENTER_SCREEN = "\x1b[?1049h\x1b[?25l"
_LEAVE_SCREEN = "\x1b[?25h\x1b[?1049l"
_HOME = "\x1b[H"
_CLEAR_LINE = "\x1b[K"
_CLEAR_BELOW = "\x1b[J"
_DEFAULT_COLUMNS, _DEFAULT_ROWS = 80, 24
_SWEEP_NOTICE = "  sonar sweep…"


@dataclass
class Session:
    """One run of the screen: its state and the things it talks to.

    ``keyboard`` has ``read()``; ``out`` has ``write``/``flush``;
    ``save(config)`` writes the file and returns its path; ``probe(config)``
    returns fresh Probes. ``saved_config`` holds the last saved config.
    """

    screen: Screen
    keyboard: object
    out: object
    save: object = save_changes
    probe: object = sweep
    enabled: bool = True
    saved_config: ThreetoksConfig | None = None

    def _draw(self) -> None:
        """Paint one frame from the top-left corner."""
        size = shutil.get_terminal_size((_DEFAULT_COLUMNS, _DEFAULT_ROWS))
        lines = render(self.screen, self.enabled, size.columns - 1)
        frame = (_CLEAR_LINE + "\n").join(lines)
        self.out.write(_HOME + frame + _CLEAR_LINE + _CLEAR_BELOW)
        self.out.flush()

    def _do_save(self) -> None:
        """Write the working copy to the file and update the notice."""
        screen = self.screen
        count = len(unsaved(screen))
        try:
            path = self.save(screen.config)
        except OSError as error:
            screen.level, screen.message = LEVEL_BAD, f"cannot save: {error}"
            return
        screen.saved, screen.path = screen.config, path
        self.saved_config = screen.config
        screen.level = LEVEL_OK
        done = f"saved {count} change(s)" if count else "already in sync"
        screen.message = done

    def _do_probe(self) -> None:
        """Ask the probes again, with the working copy as input."""
        self.screen.probes = self.probe(self.screen.config)

    def _act(self, action: str | None) -> bool:
        """Run one action; return True when the loop should stop."""
        if action == ACTION_SAVE:
            self._do_save()
        elif action == ACTION_PROBE:
            self._do_probe()
        return action == ACTION_QUIT

    def _loop(self) -> None:
        """Draw, read a key, act; repeat until an action says quit."""
        self._draw()
        while not self._act(handle_key(self.screen, self.keyboard.read())):
            self._draw()

    def run(self) -> ThreetoksConfig | None:
        """Show the screen until the user leaves.

        Returns: the last saved config, or None when nothing was saved.
        Ctrl-C leaves at once and keeps what was already saved.
        """
        self.out.write(_ENTER_SCREEN)
        try:
            self._loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.out.write(_LEAVE_SCREEN)
            self.out.flush()
        return self.saved_config


def live_commands(changes: tuple, config: ThreetoksConfig) -> list:
    """Turn saved changes into the slash commands that apply them now.

    Used after the screen closes. Args: changes (tuple of Field): what
    was saved; config (ThreetoksConfig): the saved values. Returns: list
    of command lines such as ``/model qwen3.5:2b``.
    """
    names = {item.dotted for item in changes}
    commands = []
    if "llm.model" in names:
        commands.append(f"/model {config.llm.model}")
    if names & {"browser.mode", "browser.visible"}:
        commands.append(_browser_command(config))
    if "research.max_rounds" in names:
        commands.append(f"/deep {config.research.max_rounds}")
    return commands


def _browser_command(config: ThreetoksConfig) -> str:
    """The ``/browser`` line that matches the saved browser section."""
    if config.browser.mode == "http":
        return "/browser http"
    window = "on-screen" if config.browser.visible else "headless"
    return f"/browser {config.browser.mode} {window}"


def restart_notice(changes: tuple) -> str:
    """Name the saved settings that a running session cannot take yet.

    Args: changes (tuple of Field). Returns: a one-line text, or "" when
    every change is already live.
    """
    waiting = [item.dotted for item in changes if item.apply != APPLY_LIVE]
    if not waiting:
        return ""
    return "next start / next /voice on: " + ", ".join(waiting)


def run_screen(enabled: bool, path: str = None) -> tuple:
    """Open the screen on the real terminal for the active config file.

    Used by ``/config``, the first start, and ``threetoks config``.
    Args: enabled (bool): colors on; path (str): file to edit and save,
    default the file the loader reads (or would create). Returns: (old
    config, saved config or None).
    """
    target = path or resolve_path()
    base = load_config(target)
    print(_SWEEP_NOTICE, end="\r", flush=True)
    screen = Screen(base, base, sweep(base), target)
    save = functools.partial(save_changes, path=target)
    with Keyboard() as keyboard:
        session = Session(screen, keyboard, sys.stdout, save, enabled=enabled)
        return base, session.run()


def _apply_saved(state, base: ThreetoksConfig, saved) -> str:
    """Apply what the screen saved to the live session; report it.

    Args: state (ReplState): the live TUI state; base (ThreetoksConfig):
    config before; saved (ThreetoksConfig or None): config after.
    Returns: styled text for the REPL to print.
    """
    from threetoks.tui import handle_command
    if saved is None or saved == base:
        return dim("config closed — nothing changed", state.enabled)
    changes = changed_fields(base, saved)
    lines = [dim(f"config saved ({len(changes)} change(s))", state.enabled)]
    for command in live_commands(changes, saved):
        lines.append(handle_command(state, command))
    notice = restart_notice(changes)
    return "\n".join(lines + ([dim(notice, state.enabled)] if notice else []))


def config_command(state) -> str:
    """The ``/config`` command: run the screen, then apply live changes.

    Nothing the screen does may end the chat session: Ctrl-C while the
    probes run, or any failure, comes back as a one-line message.
    Args: state (ReplState): the live TUI state. Returns: styled text for
    the REPL to print.
    """
    try:
        base, saved = run_screen(state.enabled)
    except KeyboardInterrupt:
        return dim("config closed — nothing changed", state.enabled)
    except Exception as error:  # noqa: BLE001 - never unwind the REPL
        return fg(AMBER, f"config screen failed: {error}", state.enabled)
    return _apply_saved(state, base, saved)


def first_run(enabled: bool) -> None:
    """Open the screen on a first start, with nothing saved yet.

    Leaving without a save (Esc or Ctrl-C) keeps the built-in defaults
    and writes nothing. Args: enabled (bool): colors on. Returns: None.
    """
    try:
        _, saved = run_screen(enabled)
    except KeyboardInterrupt:
        saved = None
    if saved is None:
        print(dim("no config saved — running on defaults; /config opens "
                  "this screen any time", enabled))


def _wizard_start(enabled: bool) -> bool:
    """Run the line wizard for a piped first start; False on Ctrl-C."""
    from threetoks.onboard import run_onboarding
    try:
        run_onboarding()
    except EOFError:  # piped/closed stdin: run on defaults, save nothing
        print(dim("setup skipped (no interactive input) — using "
                  "built-in defaults", enabled))
    except KeyboardInterrupt:
        print()
        print(dim("setup cancelled — nothing saved", enabled))
        return False
    return True


def first_start() -> bool:
    """Set up a configless first start; False when the wizard was cancelled.

    Used by ``tui.main``. A real terminal gets the config screen, and
    leaving it never stops the start. A piped terminal gets the line
    wizard, where Ctrl-C does stop it. Returns: bool, True to go on.
    """
    enabled = color_enabled()
    if not is_interactive():
        return _wizard_start(enabled)
    first_run(enabled)
    return True


if __name__ == "__main__":
    import io

    class _Keys:
        """Scripted keyboard for the smoke check."""

        def __init__(self, keys):
            self.keys = list(keys)

        def read(self):
            return self.keys.pop(0)

    stored = {}
    from threetoks.probes import Probes
    demo = Screen(ThreetoksConfig(), ThreetoksConfig(), Probes(), "demo.ini")
    keys = _Keys(["tab", "right", "s", "esc"])
    session = Session(demo, keys, io.StringIO(), enabled=False,
                      save=lambda cfg: stored.setdefault("cfg", cfg) and "x")
    assert session.run().browser.mode == "plain"
    assert live_commands(changed_fields(ThreetoksConfig(), stored["cfg"]),
                         stored["cfg"]) == ["/browser plain headless"]
    print("smoke OK")
