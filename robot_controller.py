from robomaster import robot


class RobotController:
    def __init__(self):
        self.ep_robot   = robot.Robot()
        self.ep_chassis = None
        self.ep_gimbal  = None
        self.ep_camera  = None
        self.ep_sensor  = None
        self.is_connected   = False
        self.distance_mm    = 0.0
        self._camera_active = False

    def connect(self, gimbal_pitch_deg=15):
        self.ep_robot.initialize(conn_type="ap")
        if not self.ep_robot.get_version():
            raise RuntimeError("Не удалось получить версию робота")

        self.ep_chassis = self.ep_robot.chassis
        self.ep_gimbal  = self.ep_robot.gimbal
        self.ep_sensor  = self.ep_robot.sensor

        self.ep_gimbal.resume()
        self.ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
        self.ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        self.ep_gimbal.recenter().wait_for_completed()
        self.ep_gimbal.moveto(pitch=gimbal_pitch_deg, yaw=0).wait_for_completed()
        self.ep_sensor.sub_distance(freq=5, callback=self._tof_callback)

        self.is_connected = True

    def disconnect(self):
        try:
            if self.ep_sensor:
                self.ep_sensor.unsub_distance()
            self.stop_wheels()
            self.stop_camera()
            self.ep_robot.close()
        except Exception:
            pass
        finally:
            self.is_connected   = False
            self.ep_chassis     = None
            self.ep_gimbal      = None
            self.ep_sensor      = None
            self.ep_camera      = None
            self.distance_mm    = 0.0

    def set_gimbal_pitch(self, pitch_deg):
        """Перемещает гимбал на заданный угол наклона (0–35°)."""
        if self.ep_gimbal and self.is_connected:
            self.ep_gimbal.moveto(pitch=pitch_deg, yaw=0).wait_for_completed()

    def start_camera(self):
        """Запускает стрим и возвращает первый кадр для определения разрешения."""
        if not self.is_connected or self._camera_active:
            return None
        self.ep_camera = self.ep_robot.camera
        self.ep_camera.start_video_stream(display=False)
        self._camera_active = True
        return self.ep_camera.read_cv2_image(strategy="newest")

    def stop_camera(self):
        if self.ep_camera and self._camera_active:
            try:
                self.ep_camera.stop_video_stream()
            except Exception:
                pass
        self._camera_active = False

    def read_frame(self):
        if self.ep_camera and self._camera_active:
            return self.ep_camera.read_cv2_image(strategy="newest")
        return None

    def drive_wheels(self, w1, w2, w3, w4):
        if self.ep_chassis:
            self.ep_chassis.drive_wheels(w1=w1, w2=w2, w3=w3, w4=w4)

    def stop_wheels(self):
        self.drive_wheels(0, 0, 0, 0)

    def _tof_callback(self, data):
        self.distance_mm = data[0]
