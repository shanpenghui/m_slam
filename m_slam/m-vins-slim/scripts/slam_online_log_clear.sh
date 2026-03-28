#!/bin/bash

DIR_SLAM=/userdata/nav/log/m_vins

echo "slam log path " ${DIR_SLAM}

# no time to consider
find ${DIR_SLAM} -name "2*"  -exec rm -rf {} \;
find ${DIR_SLAM} -name "m_vins*"  -exec rm -rf {} \;
#find ${DIR_SLAM} -mtime +0 -name "m_vins*"  -exec rm -rf {} \;

#sudo find /var/log -size +500M -exec rm -rf {} \;
