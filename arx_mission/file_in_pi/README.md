# ไฟล์สำหรับลงหุ่น

แพ็กเกจในโฟลเดอร์นี้ใช้บน Pi เท่านั้น **ไม่ได้ build บนเครื่อง dev** แต่ copy ไปที่หุ่นแล้ว build บนนั้น

มีสองอย่างที่กันไม่ให้แพ็กเกจเหล่านี้ถูก build บนเครื่อง dev และตั้งใจทำไว้ทั้งคู่:

1. โฟลเดอร์นี้อยู่ใน `arx_mission/` และ colcon จะหยุดค้นลงไปข้างในทันทีที่เจอว่าโฟลเดอร์ไหนเป็นแพ็กเกจ
   `package.xml` ที่อยู่ลึกลงไปจึงไม่ถูกเจอเลย
2. ไฟล์ `COLCON_IGNORE` บอกเรื่องเดียวกันอย่างชัดเจน การซ่อนแพ็กเกจจึงไม่ต้องพึ่งพฤติกรรมข้อ 1
   ซึ่งอาจเปลี่ยนได้ และไม่ต้องพึ่งว่าโฟลเดอร์นี้จะยังอยู่ที่เดิม

ตรวจแล้ว: `colcon list --names-only` แสดง `arx_mission` แต่ไม่แสดง `arx_camera_bringup`

## วิธี copy ไปที่หุ่น

บนหุ่นเก็บแพ็กเกจเหล่านี้ไว้ใน workspace แยกของตัวเองชื่อ `~/arx_mission_pi`
ไม่ปนกับสอง workspace ที่มีอยู่เดิม

รันจากเครื่อง dev:

    rsync -av --delete \
        ~/turtlebot3_ws/src/turtlebot3_autorace/arx_mission/file_in_pi/arx_camera_bringup \
        iotcentral@192.168.0.101:~/arx_mission_pi/src/

แล้ว build บน Pi:

    cd ~/arx_mission_pi && colcon build --symlink-install

ถ้าเคย build ด้วย `--symlink-install` แล้ว และรอบนี้แก้แค่ `camera.launch.py` ให้ rsync อย่างเดียวก็พอ
ต้อง build ใหม่เฉพาะตอนเพิ่มไฟล์ใหม่หรือแก้ `setup.py`

`--delete` จะลบไฟล์บน Pi ที่ไม่มีในเครื่อง dev ทิ้ง อย่าแก้ไฟล์ในโฟลเดอร์นี้บน Pi โดยตรง
เพราะ rsync รอบหน้าจะทับหมด

## ลำดับการ source บน Pi

บน Pi มีสาม workspace ที่ซ้อนกัน (overlay) ลำดับการ source จึงสำคัญ
workspace หนึ่งจะมองเห็นได้เฉพาะ workspace ที่ source ไว้ก่อนหน้า

    source /opt/ros/jazzy/setup.bash
    source ~/camera_ws/install/setup.bash       # camera_ros
    source ~/turtlebot3_ws/install/setup.bash   # turtlebot3_bringup, drivers
    source ~/arx_mission_pi/install/setup.bash  # แพ็กเกจในโฟลเดอร์นี้

`arx_camera_bringup` โหลด composable node ของ `camera_ros` จึงต้อง source `camera_ws` ไว้ก่อน
ถ้าลำดับผิด container จะขึ้นมาแต่หา component ไม่เจอ อาการจะดูเหมือนแพ็กเกจหาย
ทั้งที่จริงแค่ source ผิดลำดับหรือขาดไปบรรทัดเดียว

## ในโฟลเดอร์นี้มีอะไร

### `arx_camera_bringup`

`camera.launch.py` fork มาจากไฟล์ของ `turtlebot3_bringup` เพราะไฟล์เดิม hardcode sensor mode
ของกล้องรุ่นอื่นไว้ เหตุผลและค่าที่วัดได้อยู่ใน docstring ของไฟล์

### `camera_ros_patches/`

แก้ bug ของ `camera_ros` (เวอร์ชัน c7bb8ef, rel 0.5.1) ที่อยู่บน Pi ใน `~/camera_ws`
ไม่ได้อยู่ใน repo นี้ **ถ้าติดตั้ง `camera_ros` ใหม่ หรือ flash SD card ใหม่ ต้อง apply อีกครั้ง**

`exposure_override.patch` แก้ปัญหาที่ `ExposureTime` ที่ส่งตอน launch ถูกทับเป็น 20000
(ค่า default ของกล้อง) เสมอ ส่วน `AnalogueGain` ไม่โดน สาเหตุอธิบายไว้ใน `camera.launch.py`
**`camera.launch.py` ต้องพึ่ง patch นี้** ถ้าไม่มี patch ค่า `exposure_us` ที่ส่งไปจะไม่ถูกใช้
และจะไม่มีอะไรเตือน หลังติดตั้งใหม่ทุกครั้งให้ตรวจด้วย `ros2 param get /camera ExposureTime`

รันจากเครื่อง dev:

    scp arx_mission/file_in_pi/camera_ros_patches/exposure_override.patch \
        iotcentral@192.168.0.101:~/

แล้วบน Pi (ใน terminal ที่ source แค่ `/opt/ros/jazzy/setup.bash`):

    cd ~/camera_ws/src/camera_ros && git apply ~/exposure_override.patch
    cd ~/camera_ws && colcon build --packages-select camera_ros

ตรวจผลโดยรัน `camera_ros` ตรงๆ:

    ros2 run camera_ros camera_node --ros-args \
        -p camera:=0 -p sensor_mode:=1296:972 -p width:=320 -p height:=240 -p format:=RGB888 \
        -p ExposureTimeMode:=1 -p ExposureTime:=10000
    ros2 param get /camera ExposureTime   # ต้องได้ 10000 ถ้าไม่มี patch จะได้ 20000

### `camera_info/`

ค่า calibration intrinsic ที่วัดเมื่อ 2026-10-04 ไฟล์นี้เป็นแค่สำเนาสำรอง ไฟล์ที่ใช้งานจริงอยู่บนหุ่นที่

    ~/.ros/camera_info/ov5647__base_soc_i2c0mux_i2c_1_ov5647_36_320x240_1296x972_SGBRG10_CSI2P_RAW.yaml

`camera_ros` อ่านไฟล์จาก path นี้เองตอนเริ่มทำงาน แล้วส่งค่าที่อ่านได้ออกที่ `/camera/camera_info`
จึงไม่มี launch file ไหนอ้างถึงไฟล์นี้ ถ้า flash SD card ใหม่ ให้ copy กลับไปด้วยคำสั่งนี้
(รันจากโฟลเดอร์ `turtlebot3_autorace/` บนเครื่อง dev):

    scp arx_mission/file_in_pi/camera_info/*.yaml \
        iotcentral@192.168.0.101:~/.ros/camera_info/

ถ้า calibrate ใหม่ ทิศทางจะกลับกัน เพราะการกด COMMIT ใน `cameracalibrator` เขียนไฟล์ลงหุ่นโดยตรง
ให้ดึงไฟล์ใหม่กลับมาเก็บเป็นสำเนาในโฟลเดอร์นี้:

    scp iotcentral@192.168.0.101:~/.ros/camera_info/*.yaml \
        arx_mission/file_in_pi/camera_info/

ชื่อไฟล์ไม่ได้ตั้งไว้สวยๆ เฉยๆ `camera_ros` สร้างชื่อนี้จากชื่อเซนเซอร์ ที่อยู่ I2C ขนาดภาพที่ส่งออก
และ sensor mode แล้วจะโหลดไฟล์ก็ต่อเมื่อชื่อตรงกับที่มันสร้างได้ในตอนนั้นทุกตัวอักษร
ถ้าเปลี่ยน `width`, `height` หรือ `sensor_mode` ใน `camera.launch.py` ไฟล์นี้จะหาไม่เจอทันทีโดยไม่มี
error ใดๆ `camera_info` จะกลับเป็นศูนย์ทั้งหมด และการแก้ความบิดเบือนของภาพจะไม่ทำงานไปเงียบๆ
ซึ่งเป็นสภาพเดียวกับก่อนที่จะวัดค่านี้

วัดด้วย `cameracalibrator` บนเครื่อง dev ใช้ 196 ภาพ กระดานมีจุดตัดภายใน 8x6 จุด
ใช้โมเดล pinhole ไม่ใช่ fisheye เลนส์ขายเป็น 160 องศา แต่นั่นคือมุมของเลนส์เปล่า
เซนเซอร์ขนาด 1/4 นิ้วเห็นจริงประมาณ 112 องศา ตรวจภายหลังด้วยภาพกระดานหนึ่งภาพ
ความโค้งของแถวและคอลัมน์กระดานลดจาก 2.27 px เหลือ 0.56 px และรูปแบบความคลาดเคลื่อนที่เป็นระบบหายไป

ภาพดิบเก็บไว้นอก git ที่ `notes/calibrationdata_2026-10-04.tar.gz` จึงคำนวณใหม่ได้
(เช่น ลองโมเดล fisheye) โดยไม่ต้องตั้งกระดานถ่ายภาพใหม่
