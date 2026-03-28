#!/usr/bin/env bash

DIR=/userdata/nav

#sh ${DIR}/install/scripts/m_vins/slam_online_log_clear.sh
#sleep 1

CFG_PATH=${DIR}/install/cfg/m_vins
MAP_PATH=${DIR}/data/m_vins/map
LOG_PATH=${DIR}/log/m_vins

# ulimit -c unlimited

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

export LD_LIBRARY_PATH=/userdata/nav/install/lib/m_vins/:/userdata/nav/install/lib:/userdata/nav/install/lib/m_vins/third_lib:/opt/ros/foxy/lib:{LD_LIBRARY_PATH}

${DIR}/install/bin/m_vins/m_vins_node \
  --slam_yaml_config=${CFG_PATH} \
  --v=0 \
