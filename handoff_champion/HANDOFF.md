# ส่งต่องาน: จับเวลาทีมแชมป์

## วางไฟล์
แตก zip นี้ที่ root ของโปรเจค turtlebot3_autorace ไฟล์จะลงที่ docs/ และ notes/champion_video/
(docs/12, docs/13, curve_map.png อาจมีอยู่แล้ว เขียนทับได้ เป็นฉบับเดียวกัน)

## ข้อความสั่ง Claude (แก้บรรทัด "งานต่อ")
อ่าน docs/14.จับเวลาทีมแชมป์.html ก่อน (ผลจับเวลาทีมแชมป์จากคลิป
https://www.youtube.com/watch?v=6t6cyFiGLvs ช่วง START → ลานจอด → ออก 18B)
ประกอบกับ docs/12.การทดลองความเร็ว.html, docs/13.ข้อระวัง_เพดานความเร็ว.html
และ docs/curve_map.png

สรุป: แชมป์ใช้ ~63.5 s เราประมาณ ~70 s ความเร็วสูงสุดของแชมป์ ~0.21 m/s เท่ากับเรา
เขาได้เวลาจากการจอดแบบขับหน้าเข้าแล้วหมุน 180° ออก (~4 s)
และช่วงตามเลนอีก ~5 s ที่ยังอธิบายไม่ได้

งานต่อ: ออกแบบการจอดแบบขับหน้าเข้าใน arx_mission/mission_control.py
และพารามิเตอร์ bay.park ใน arx_mission/param/mission.yaml

## ถ้าจะวัดจากคลิปต่อ (notes/champion_video/)
ต้องติดตั้ง: pip install yt-dlp imageio-ffmpeg pillow numpy
1. ดาวน์โหลดช่วง 2:00-5:20:
   python -m yt_dlp -f "bv*[height<=720]" --download-sections "*120-320" -o vid/run.mp4 <URL>
2. แตกเฟรม 2 fps ลง full/0001.jpg... (ffmpeg -i vid/run.mp4 -vf fps=2 -q:v 3 full/%04d.jpg)
   เฟรม n อยู่ที่เวลาในคลิป 120 + (n-1)*0.5 วินาที
3. sheet2.py = ทำชีตเฟรม, trk.py = ชีตพร้อมกริดพิกเซลสำหรับอ่านตำแหน่ง,
   homo.py = homography (มุมลานจอด 4 จุด) + คำนวณความเร็วช่วง 3:13-3:17.5
   ตำแหน่งพิกเซลใน homo.py อ่านด้วยตาจากเฟรม 0147 (3:13) ใช้ได้เฉพาะช่วงที่กล้องนิ่งเท่านั้น
