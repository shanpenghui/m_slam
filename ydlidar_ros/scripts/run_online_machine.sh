PRJ_NAME=ydlidar_ros

DIR=/userdata/nav/install
LOG_DIR=${DIR}/../log/${PRJ_NAME}/
CFG_FILE_PATH=${DIR}/cfg/${PRJ_NAME}/config.yaml
SCRIPT_DIR=${DIR}/scripts/${PRJ_NAME}

if [ ! -d "${LOG_DIR}" ];then
  mkdir -p ${LOG_DIR}
fi

sh ${SCRIPT_DIR}/kill_all.sh                                     
sleep 1

${DIR}/bin/${PRJ_NAME}/ydlidar_node \
  --log_file_dir=${LOG_DIR} \
  --cfg_file_path=${CFG_FILE_PATH} > ${LOG_DIR}sdk_log &
