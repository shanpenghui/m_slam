do_date=`date "+%Y-%m-%d %H:%M:%S"`

YDLIDAR_NODE=`ps -ef | grep ydlidar_node | grep -v grep | awk '{print $2}' | xargs`

if [ -n "$YDLIDAR_NODE" ];then
	echo "kill ydlidar_node with pid $YDLIDAR_NODE"
	kill -9 $YDLIDAR_NODE
else
	echo "ydlidar_node not found "
fi
