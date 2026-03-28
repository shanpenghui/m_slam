do_date=`date "+%Y-%m-%d %H:%M:%S"`

M_VINS_NODE=`ps -ef | grep m_vins | grep -v grep | awk '{print $2}' | xargs`

if [ -n "$M_VINS_NODE" ];then
	echo "kill M_VINS_NODE with pid $M_VINS_NODE"
	kill -9 $M_VINS_NODE
else
	echo "M_VINS_NODE not found "
fi
