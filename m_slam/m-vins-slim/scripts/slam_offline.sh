#!/usr/bin/env bash

DIR=/root/nav_test

CFG_PATH=$1
MAP_PATH=$2
LOG_PATH=$3

if [ ! -d "${LOG_PATH}" ];then
  mkdir -p ${LOG_PATH}
  echo "dir-log has NOT existed"
else
  echo "dir-log has existed"
fi

if [ ! -d "${MAP_PATH}" ];then
  mkdir -p ${MAP_PATH}
  echo "dir-map has NOT existed"
else
  echo "dir-map has existed"
fi

${DIR}/install/bin/m_vins/m_vins_node \
  --slam_yaml_config=${CFG_PATH} \
  --v=1 \
