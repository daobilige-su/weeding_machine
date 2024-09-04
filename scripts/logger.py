#! /usr/bin/env python
import numpy as np
import rospy, rospkg
from std_msgs.msg import String
import sys
import yaml
from datetime import datetime
import csv

class logger:
    def __init__(self):
        # locate ros pkg
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('weeding_machine') + '/'

        # load key params
        params_filename = rospy.get_param('param_file')  # self.pkg_path + 'cfg/' + 'param.yaml'
        with open(params_filename, 'r') as file:
            self.param = yaml.safe_load(file)

        self.log_msg_sub = rospy.Subscriber('/log_msg', String, self.log_msg_cb)

        cur_datetime = datetime.now()
        cur_datetime_str = cur_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        self.log_file = self.pkg_path + 'log/' + cur_datetime_str + '.csv'

        self.file = None
        self.csv_writer = None
        with open(self.log_file, 'w', newline='') as self.file:
            self.csv_writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rospy.spin()

        return

    # sub for weeder weeder speed and pos, simply store it in member var
    def log_msg_cb(self, msg):
        str_data = msg.data
        t, txt = str_data.split(':')

        self.csv_writer.writerow([t] + [txt])


# the main function entrance
def main(args):
    rospy.init_node('logger_node', anonymous=True)
    logger_obj = logger()


# the main entrance
if __name__ == '__main__':
    main(sys.argv)
