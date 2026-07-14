"""Offline tests for the GPIO relay; no RPi.GPIO is imported.

A FakeGPIO records every driver call and exposes BCM/OUT/HIGH/LOW as
distinct sentinels, so the tests can assert exact pin levels under the
active-low wiring without a Raspberry Pi.
"""
import unittest

from threetoks.relay import DEFAULT_PIN, Relay, relay_available


class FakeGPIO:
    """In-memory GPIO double; constants are distinct sentinel strings."""

    BCM = "BCM"
    OUT = "OUT"
    HIGH = "HIGH"
    LOW = "LOW"

    def __init__(self):
        self.mode = None
        self.warnings = None
        self.setup_calls = []
        self.output_calls = []
        self.cleanup_calls = 0

    def setwarnings(self, flag):
        """Record the last warnings flag."""
        self.warnings = flag

    def setmode(self, mode):
        """Record the numbering mode."""
        self.mode = mode

    def setup(self, pin, direction):
        """Record a pin direction setup."""
        self.setup_calls.append((pin, direction))

    def output(self, pin, level):
        """Record a driven pin level."""
        self.output_calls.append((pin, level))

    def cleanup(self, *pins):
        """Count cleanup calls."""
        self.cleanup_calls += 1


class RelayInitTest(unittest.TestCase):
    def test_init_sets_bcm_output_and_off_level(self):
        gpio = FakeGPIO()
        Relay(gpio=gpio)
        self.assertEqual(gpio.warnings, False)
        self.assertEqual(gpio.mode, gpio.BCM)
        self.assertEqual(gpio.setup_calls, [(DEFAULT_PIN, gpio.OUT)])
        # Off under active-low is the HIGH level.
        self.assertEqual(gpio.output_calls, [(DEFAULT_PIN, gpio.HIGH)])

    def test_custom_pin_is_honored(self):
        gpio = FakeGPIO()
        Relay(pin=23, gpio=gpio)
        self.assertEqual(gpio.setup_calls, [(23, gpio.OUT)])
        self.assertEqual(gpio.output_calls, [(23, gpio.HIGH)])


class RelayDriveTest(unittest.TestCase):
    def test_set_light_on_drives_low_and_returns_string(self):
        gpio = FakeGPIO()
        relay = Relay(gpio=gpio)
        self.assertEqual(relay.set_light(True), "on")
        self.assertTrue(relay.on)
        self.assertEqual(gpio.output_calls[-1], (DEFAULT_PIN, gpio.LOW))

    def test_set_light_off_drives_high(self):
        gpio = FakeGPIO()
        relay = Relay(gpio=gpio)
        relay.set_light(True)
        self.assertEqual(relay.set_light(False), "off")
        self.assertEqual(gpio.output_calls[-1], (DEFAULT_PIN, gpio.HIGH))

    def test_get_status_tracks_state(self):
        relay = Relay(gpio=FakeGPIO())
        self.assertEqual(relay.get_status(), "off")
        relay.set_light(True)
        self.assertEqual(relay.get_status(), "on")

    def test_toggle_flips_state_and_return_value(self):
        gpio = FakeGPIO()
        relay = Relay(gpio=gpio)
        self.assertEqual(relay.toggle(), "on")
        self.assertEqual(gpio.output_calls[-1], (DEFAULT_PIN, gpio.LOW))
        self.assertEqual(relay.toggle(), "off")
        self.assertEqual(gpio.output_calls[-1], (DEFAULT_PIN, gpio.HIGH))

    def test_sequence_leaves_expected_final_state(self):
        relay = Relay(gpio=FakeGPIO())
        relay.set_light(True)
        relay.toggle()   # -> off
        relay.toggle()   # -> on
        self.assertEqual(relay.get_status(), "on")


class RelayLifecycleTest(unittest.TestCase):
    def test_close_calls_cleanup(self):
        gpio = FakeGPIO()
        Relay(gpio=gpio).close()
        self.assertEqual(gpio.cleanup_calls, 1)

    def test_context_manager_closes(self):
        gpio = FakeGPIO()
        with Relay(gpio=gpio) as relay:
            self.assertEqual(relay.get_status(), "off")
        self.assertEqual(gpio.cleanup_calls, 1)


class RelayAvailableTest(unittest.TestCase):
    def test_returns_false_on_this_machine_without_raising(self):
        # macOS dev box: not ARM Linux, so unavailable and never raises.
        self.assertIs(relay_available(), False)


if __name__ == "__main__":
    unittest.main()
