#!/usr/bin/env python3
#
# Copyright 2026 boat1470
# Licensed under the Apache License, Version 2.0

"""The level crossing bar, read with an HC-SR04 on the front of the robot (ARX).

The camera detector (detect_level.py) was tuned on the simulator's bar. The
real one is an Arduino-driven MG996R that rests DOWN, across the road, and
lifts for 5 s when its own sensor sees the robot inside 20 cm. All the robot
has to know is whether something is standing across the lane in front of it,
and how far away - which is what an ultrasonic sensor measures directly.

It publishes the same two topics as detect_level, with the same meaning, so
mission_control does not know or care which one is running:

    /detect/level_bar     UInt8    0 nothing, 1 up, 2 down
    /detect/level_range   Float32  metres to the bar, 9.9 when it is not down

Down is "the median of the last few readings is nearer than down_m". Up is
anything else, including no echo at all - with the bar lifted there is
nothing in front of the sensor to reflect.

The median is what keeps the robot out of the bar. A single lost echo reads
as far away, which is "up", and wait_bar leaves after five ticks of up. One
odd reading in five cannot move a median of five.

Wired straight to the Pi's GPIO (a 3.3 V HC-SR04, so no divider on Echo):

    VCC   3.3 V   pin 1
    Trig  GPIO23  pin 16
    Echo  GPIO24  pin 18
    GND   GND     pin 6

The echo is timed from lgpio's edge alerts, which carry the kernel's own
timestamps, rather than by polling the pin from Python - a Python loop on a
busy Pi misses edges by enough to be off by centimetres.

Like every detector here it idles until mission_control arms mission 5. To
look at the reading on its own:

    ros2 launch arx_ultrasonic_bringup ultrasonic.launch.py always_on:=true
    ros2 topic echo /detect/level_range
"""

import statistics
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import UInt8

try:
    import lgpio
except ImportError:  # not a Pi, or python3-lgpio not installed
    lgpio = None

# Must match mission_control.py and detect_level.py.
MISSION_LEVEL = 5

BAR_NONE = 0
BAR_UP = 1
BAR_DOWN = 2
BAR_NAMES = {BAR_NONE: 'nothing', BAR_UP: 'up', BAR_DOWN: 'down'}

# Large rather than zero, so `range <= stop_m` does not read silence as
# "arrived". The same value detect_level uses.
RANGE_NONE = 9.9

SPEED_OF_SOUND = 343.0  # m/s, at about 20 C


class DetectLevelUltrasonic(Node):

    def __init__(self):
        super().__init__('detect_level_ultrasonic')

        self.declare_parameter('level.ultrasonic.gpio_chip', 0)
        self.declare_parameter('level.ultrasonic.trig_pin', 23)
        self.declare_parameter('level.ultrasonic.echo_pin', 24)
        self.declare_parameter('level.ultrasonic.rate_hz', 15.0)
        self.declare_parameter('level.ultrasonic.down_m', 0.45)
        self.declare_parameter('level.ultrasonic.max_m', 2.0)
        self.declare_parameter('level.ultrasonic.window', 5)
        self.declare_parameter('level.always_on', False)

        self.armed = 0
        self.bar = None
        self.readings = []
        self.rise_ns = None
        self.echo_ns = None
        self.waiting = False

        self.pub_bar = self.create_publisher(UInt8, '/detect/level_bar', 10)
        self.pub_range = self.create_publisher(Float32, '/detect/level_range', 10)
        self.create_subscription(UInt8, '/arx/armed_mission', self.on_armed, 1)

        self.chip = None
        self.alert = None
        if not self.fnOpenGpio():
            return

        rate = self.get_parameter('level.ultrasonic.rate_hz').value
        self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().info('DetectLevelUltrasonic Node Initialized')

    def fnOpenGpio(self):
        """Claim Trig and Echo. Says why and gives up if it cannot."""
        if lgpio is None:
            self.get_logger().error(
                'python3-lgpio is not installed - nothing will be measured. '
                'sudo apt install python3-lgpio')
            return False
        chip = self.get_parameter('level.ultrasonic.gpio_chip').value
        self.trig = self.get_parameter('level.ultrasonic.trig_pin').value
        self.echo = self.get_parameter('level.ultrasonic.echo_pin').value
        try:
            self.chip = lgpio.gpiochip_open(chip)
            lgpio.gpio_claim_output(self.chip, self.trig, 0)
            lgpio.gpio_claim_alert(self.chip, self.echo, lgpio.BOTH_EDGES)
            self.alert = lgpio.callback(
                self.chip, self.echo, lgpio.BOTH_EDGES, self.on_edge)
        except lgpio.error as e:
            self.get_logger().error(
                f'cannot open GPIO (chip {chip}, trig {self.trig}, '
                f'echo {self.echo}): {e} - nothing will be measured. Is the '
                f'user allowed /dev/gpiochip{chip}? On an old Pi 5 kernel the '
                f'chip is 4.')
            self.fnCloseGpio()
            return False
        self.get_logger().info(
            f'HC-SR04 on gpiochip{chip}: trig {self.trig}, echo {self.echo}')
        return True

    def fnCloseGpio(self):
        if self.alert is not None:
            self.alert.cancel()
            self.alert = None
        if self.chip is not None:
            lgpio.gpiochip_close(self.chip)
            self.chip = None

    def on_armed(self, msg):
        """Remember which mission the sequencer says is running."""
        if msg.data == self.armed:
            return
        self.armed = msg.data
        self.get_logger().info(f'armed mission -> {self.armed}')

    def fnArmed(self):
        """Whether this detector should be measuring at all."""
        if self.get_parameter('level.always_on').value:
            return True
        return self.armed == MISSION_LEVEL

    def on_edge(self, chip, gpio, level, tick):
        """Echo edges, from lgpio's thread. `tick` is in nanoseconds."""
        if level == 1:
            self.rise_ns = tick
        elif level == 0 and self.rise_ns is not None:
            self.echo_ns = tick - self.rise_ns
            self.rise_ns = None

    def on_timer(self):
        """Read the ping sent last tick, then send the next one.

        One period (66 ms at 15 Hz) is longer than the 38 ms an HC-SR04 takes
        to give up on an echo, so the last ping has always finished by now and
        nothing here has to wait.
        """
        if not self.fnArmed():
            self.waiting = False
            self.readings = []
            return

        if self.waiting:
            self.fnTakeReading()

        self.rise_ns = None
        self.echo_ns = None
        lgpio.gpio_write(self.chip, self.trig, 1)
        time.sleep(10e-6)
        lgpio.gpio_write(self.chip, self.trig, 0)
        self.waiting = True

    def fnTakeReading(self):
        max_m = self.get_parameter('level.ultrasonic.max_m').value
        echo_ns = self.echo_ns
        if echo_ns is None:
            dist = max_m
        else:
            dist = min(echo_ns * 1e-9 * SPEED_OF_SOUND / 2.0, max_m)

        window = self.get_parameter('level.ultrasonic.window').value
        self.readings.append(dist)
        del self.readings[:-window]
        median = statistics.median(self.readings)

        down_m = self.get_parameter('level.ultrasonic.down_m').value
        bar = BAR_DOWN if median < down_m else BAR_UP

        self.pub_bar.publish(UInt8(data=bar))
        self.pub_range.publish(
            Float32(data=float(median) if bar == BAR_DOWN else RANGE_NONE))

        if bar != self.bar:
            self.bar = bar
            self.get_logger().info(
                f'level bar -> {BAR_NAMES[bar]} ({median:.2f} m, '
                f'median of {len(self.readings)})')

    def destroy_node(self):
        self.fnCloseGpio()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectLevelUltrasonic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
