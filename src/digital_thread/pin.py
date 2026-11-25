"""Pin input on the Raspberry Pi (production, section 3).

A normally open mechanical switch, pressed by the pin on the mandrel shaft,
connects GPIO2 (physical pin 3) to GND (pin 6); the pull-ups (the Pi's
internal one, and the 1.8 kOhm one on the board, as GPIO2 is the I2C data line)
hold the input high while the contact is open. I2C must stay disabled. Each closing of the contact is one pin pulse, which fires
the camera by software trigger (the frame's timestamp still comes from the
camera clock).

A closing only counts after the contact has been open for at least
`DEBOUNCE_S`: this drops the chatter of the contact both when the pin presses
the switch and when it leaves it. A switch held closed (mandrel stopped at the
pin) gives no further pulses.
"""

import time

GPIO_PIN = 2  # physical pin 3 (GPIO26 drives the fan)
DEBOUNCE_S = 0.02


class GpioPin:
    """Calls `on_pulse()` (from gpiozero's thread) once per closing of the switch."""

    def __init__(self, on_pulse, gpio=GPIO_PIN, debounce_s=DEBOUNCE_S):
        self.on_pulse, self.debounce_s = on_pulse, debounce_s
        try:
            from gpiozero import Button
            from gpiozero.exc import GPIOZeroError

            try:
                self._button = Button(gpio, pull_up=True)
            except GPIOZeroError as error:
                raise RuntimeError(
                    f"No GPIO{gpio} on this computer ({error}); choose the camera "
                    "input line as pin input, or run the app on the Raspberry Pi."
                ) from error
        except ImportError as error:
            raise RuntimeError(f"No GPIO{gpio}: gpiozero is not installed") from error
        # Open since "ever", unless the pin already rests on the switch.
        self._opened = None if self._button.is_pressed else -float("inf")
        self._button.when_pressed = self._closed
        self._button.when_released = self._released

    def _released(self):
        self._opened = time.monotonic()

    def _closed(self):
        opened, self._opened = self._opened, None
        if opened is not None and time.monotonic() - opened >= self.debounce_s:
            self.on_pulse()

    def close(self):
        self._button.close()
