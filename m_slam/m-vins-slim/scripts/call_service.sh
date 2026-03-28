#!/bin/bash

source ~/.bashrc

sleep 1

action=$1

str='{"action":'$action'}'

echo $str

ros2 service call /slam_service mirobot_msgs/srv/SlamService $str


