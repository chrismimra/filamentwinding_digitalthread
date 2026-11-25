"""Pin input on the Raspberry Pi (section 3): a normally open switch on GPIO2.

gpiozero's mock pins stand in for the Pi's GPIO, so the tests run anywhere.
"""

import time

import pytest

from digital_thread import pin
from digital_thread.camera import ReplayCamera


def _settle():
    time.sleep(2 * pin.DEBOUNCE_S)


def test_each_closing_of_the_switch_is_one_pulse(gpio):
    pulses = []
    switch = pin.GpioPin(lambda: pulses.append(1))
    try:
        for _ in range(3):  # contact chatter as the pin presses the switch
            gpio.drive_low()
            gpio.drive_high()
        gpio.drive_low()
        assert len(pulses) == 1
        _settle()  # the pin rests on the switch: no further pulse
        assert len(pulses) == 1
        for _ in range(3):  # chatter as the pin leaves the switch
            gpio.drive_high()
            gpio.drive_low()
        gpio.drive_high()
        assert len(pulses) == 1
        _settle()
        gpio.drive_low()  # the next revolution
        assert len(pulses) == 2
    finally:
        switch.close()


def test_without_gpio_the_pin_input_reports_it(no_gpio):
    with pytest.raises(RuntimeError, match="GPIO2"):
        pin.GpioPin(lambda: None)


def test_replay_camera_serves_a_frame_per_software_trigger(cylinder_geometry):
    run = cylinder_geometry[0]
    camera = ReplayCamera(run.folder)
    camera.set_trigger("software")
    assert camera.grab(timeout_s=0.05) is None  # no trigger, no frame
    camera.software_trigger()
    _, stamp = camera.grab(timeout_s=0.05)
    assert stamp == run.frames[0].timestamp_s
    assert camera.grab(timeout_s=0.05) is None
