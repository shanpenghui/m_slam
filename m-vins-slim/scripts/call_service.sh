#!/bin/bash

source ~/.bashrc
action=$1

if [ -z "$action" ]; then
  echo "usage: ./call_service.sh start|stop|reset_pose|save_map|remove_map|get_docker_pose"
  exit 1
fi

ros2 service call /slam_service mirobot_msgs/srv/SlamService "{action: '$action'}"
