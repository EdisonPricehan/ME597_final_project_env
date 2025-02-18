#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
import argparse


def area_from_pgm(map_name: str):
    import yaml
    import os

    # specify file paths by map name, make sure your map files are in ../maps folder
    map_dir_path = os.path.dirname(__file__) + '/../maps/'
    map_pgm_path = map_dir_path + map_name + '.pgm'
    map_yaml_path = map_dir_path + map_name + '.yaml'

    # read resolution from .yaml file
    with open(map_yaml_path, 'r') as f:
        map_yaml = yaml.safe_load(f)
        resolution = map_yaml['resolution']
        print(f"Map resolution is {resolution}")

    # read pixel values from .pgm file
    with open(map_pgm_path, 'rb') as f:
        assert f.readline() == b'P5\n'

        t = f.readline()
        while t[0] == '#':
            t = f.readline()

        (width, height) = map(int, f.readline().split())
        depth = int(f.readline())
        assert depth <= 255

        raster = []
        for _ in range(height * width):
            raster.append(ord(f.read(1)))
        print(f"All cells count is {len(raster)}")
        cells_count = sum([1 for i in raster if i > 250])
        area = cells_count * resolution * resolution
        return area


class Coverage:
    def __init__(self, area: float):
        rospy.init_node('coverage_metric_node', anonymous=True)
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, callback=self.coverage_callback, queue_size=1)
        self.OBSTACLE_THR: int = 65
        self.FREE_THR: int = 25
        self.area: float = area
        self.covered_area: int = 0
        self.coverage_rate: float = 0
        rospy.loginfo(f"Total area is {self.area} m2, coverage metric starts running ...")
        rospy.spin()

    def coverage_callback(self, msg: OccupancyGrid):
        map_data = msg.data
        res = msg.info.resolution
        cell_counts = sum([1 for i in map_data if 0 <= i < self.FREE_THR])
        self.covered_area = res * res * cell_counts
        self.coverage_rate = self.covered_area / self.area
        rospy.loginfo_throttle(1, f'Your current coverage rate is %.1f%%', self.coverage_rate * 100)


if __name__ == '__main__':
    # Example code to get the area statically from .pgm and .yaml files
    # map_name = 'map'
    # # map_name = 'map_unseen'
    # area = area_from_pgm(map_name)
    # print(f'Map area is {area} m2.')

    # Code to give out real-time coverage rate while you map the environment
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--area", help="Area of the house in m2 for calculating coverage rate", type=float)
    args, _ = parser.parse_known_args()

    try:
        coverage_node = Coverage(args.area)
    except rospy.ROSInterruptException:
        pass
    finally:
        coverage_rate = coverage_node.coverage_rate
        rospy.loginfo(f'Your final coverage rate is %.1f%%', coverage_rate * 100)
        final_score = 36 if coverage_rate >= 0.85 else coverage_rate / 0.85 * 36
        rospy.loginfo("Your final score for task 1 is %d", final_score)

