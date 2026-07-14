"""Drive one GPIO relay from a Raspberry Pi (optional `relay` extra).

The relay switches a desk lamp on and off. Pins use BCM numbering. Wiring
is ACTIVE-LOW: the relay energizes (lamp ON) when the GPIO line is driven
LOW and de-energizes (lamp OFF) when driven HIGH.

RPi.GPIO is a Pi-only dependency, so it is imported LAZILY inside
``Relay.__init__``; importing this module needs only the stdlib. Callers
gate on ``relay_available()`` before constructing a Relay.
"""
import importlib.util
import platform
import sys

DEFAULT_PIN = 17            # BCM pin wired to the relay input
ACTIVE_LOW = True           # relay energized (lamp ON) when the pin is LOW
_ARM_MACHINES = ("aarch64", "armv")
_GPIO_PACKAGE = "RPi.GPIO"


def _import_gpio():
    """Import RPi.GPIO lazily; let ImportError propagate to the caller."""
    import RPi.GPIO as gpio  # optional `relay` extra, Raspberry Pi only
    return gpio


class Relay:
    """One active-low GPIO relay driving a desk lamp.

    Inputs: ``pin`` (BCM number) and ``gpio`` (an injected RPi.GPIO-like
    module for tests; None imports the real one). Lamp state is tracked in
    the bool ``self.on``. The string methods return spoken "on"/"off" for
    an answer sentence — never boolean-negate them.
    """

    def __init__(self, pin: int = DEFAULT_PIN, gpio=None):
        """Configure the pin as an output held at the OFF level."""
        self._gpio = gpio if gpio is not None else _import_gpio()
        self._pin = pin
        self.on = False
        self._gpio.setwarnings(False)
        self._gpio.setmode(self._gpio.BCM)
        self._gpio.setup(pin, self._gpio.OUT)
        self._gpio.output(pin, self._level(self.on))

    def _level(self, on: bool):
        """GPIO level for a lamp state under active-low wiring."""
        if ACTIVE_LOW:
            return self._gpio.LOW if on else self._gpio.HIGH
        return self._gpio.HIGH if on else self._gpio.LOW

    def set_light(self, on: bool) -> str:
        """Drive the lamp to ``on``; return the new spoken state."""
        self.on = bool(on)
        self._gpio.output(self._pin, self._level(self.on))
        return self.get_status()

    def get_status(self) -> str:
        """Return the current spoken lamp state, "on" or "off"."""
        return "on" if self.on else "off"

    def toggle(self) -> str:
        """Flip the lamp; return the new spoken state."""
        return self.set_light(not self.on)

    def close(self) -> None:
        """Release the GPIO pins."""
        self._gpio.cleanup()

    def __enter__(self) -> "Relay":
        """Enter the context manager; return self."""
        return self

    def __exit__(self, *exc) -> None:
        """Exit the context manager, releasing the pins."""
        self.close()


def relay_available() -> bool:
    """True only on ARM Linux with RPi.GPIO importable; a pure check.

    No GPIO import side effects: it inspects the platform, the machine
    architecture, and the package spec. ``find_spec`` can raise when the
    parent package is missing, so that call is guarded.
    """
    if not sys.platform.startswith("linux"):
        return False
    if not platform.machine().startswith(_ARM_MACHINES):
        return False
    try:
        return importlib.util.find_spec(_GPIO_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


if __name__ == "__main__":
    class _FakeGPIO:
        """Inline fake so the smoke test needs no RPi.GPIO."""
        BCM, OUT, LOW, HIGH = "BCM", "OUT", 0, 1

        def __init__(self):
            self.calls = []

        def setwarnings(self, flag):
            self.calls.append(("setwarnings", flag))

        def setmode(self, mode):
            self.calls.append(("setmode", mode))

        def setup(self, pin, direction):
            self.calls.append(("setup", pin, direction))

        def output(self, pin, level):
            self.calls.append(("output", pin, level))

        def cleanup(self, *pins):
            self.calls.append(("cleanup", pins))

    gpio = _FakeGPIO()
    with Relay(gpio=gpio) as relay:
        assert relay.get_status() == "off"                 # starts off
        assert ("output", DEFAULT_PIN, gpio.HIGH) in gpio.calls  # off = HIGH
        assert relay.set_light(True) == "on"               # string, not bool
        assert ("output", DEFAULT_PIN, gpio.LOW) in gpio.calls   # on = LOW
        assert relay.toggle() == "off"
    assert ("cleanup", ()) in gpio.calls                   # context closed
    assert relay_available() is False                      # not a Pi here
    print("smoke OK")
