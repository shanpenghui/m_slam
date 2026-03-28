# /asr_tts_interfaces/msg/audio_data.msg
# liangjunjie
# 以int16数组传输音频流数据，对应应publisher中16bit的设置。
# 更改成其他8bit，32bit等设置时需要改成int8[]，int32[]等。
# ros2本身不支持字节数组的传递，所以需要事先把原始的字节数组转换成int数组?
uint8[] data

