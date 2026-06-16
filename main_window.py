from tkinter import *
import cv2
from robomaster import robot
from ultralytics import YOLO
import numpy as np
import time
import math
import threading
from pynput import keyboard


class RobomasterPersonFollower:
    def __init__(self):
        # Инициализация главного окна
        self.window = Tk()
        self.window.title("Окно управления")

        # Состояние приложения
        self.is_connected = False
        self.is_tracking = False
        self.selected_person_id = None
        self.tracking_thread = None

        # Данные с датчиков и видения
        self.current_distance_mm = 0.0
        self.tracks = {}
        self.frame_width = 0
        self.frame_height = 0

        # Новые переменные для управления поиском
        self.target_lost_time = None
        self.search_start_time = None
        self.is_searching = False
        self.search_direction = 1  # 1 для вращения вправо, -1 для вращения влево
        self.last_seen_time = {}  # Время последнего обнаружения для каждого трека

        # Флаги для управления потоком трекинга
        self.stop_tracking_flag = threading.Event()

        # Настройки управления
        self.MIN_DISTANCE_MM = 1500
        self.MAX_DISTANCE_MM = 3000
        self.MAX_ROTATION_SPEED = 80
        self.MAX_MOVE_SPEED = 100

        # Новые настройки поиска
        self.SEARCH_TIMEOUT = 20.0  # секунды поиска перед остановкой
        self.SEARCH_ROTATION_SPEED = 20 # скорость вращения при поиске
        self.TRACK_TIMEOUT = 2.0  # время, после которого считаем трек потерянным

        # Инициализация компонентов
        self.ep_robot = robot.Robot()
        self.ep_chassis = None
        self.ep_gimbal = None
        self.ep_camera = None
        self.ep_sensor = None
        self.model = YOLO('yolo11n.pt')

        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()

        self._setup_ui()

    def on_press(self,key):
        if self.ep_chassis is not None:
            if hasattr(key, 'char'):
                if key.char == 'w':
                    self.ep_chassis.drive_wheels(w1=150, w2=150, w3=150, w4=150)
                    pass
                if key.char == 'a':
                    self.ep_chassis.drive_wheels(w1=100, w2=-100, w3=-100, w4=100)
                    pass
                if key.char == 'd':
                    self.ep_chassis.drive_wheels(w1=-100, w2=100, w3=100, w4=-100)
                    pass
                if key.char == 's':
                    self.ep_chassis.drive_wheels(w1=-150, w2=-150, w3=-150, w4=-150)
                    pass

    def on_release(self, key):
        if self.ep_chassis is not None:
            if hasattr(key, 'char'):
                if key.char in ['w', 'd', 'a','s']:
                    self.ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        pass

    def _setup_ui(self):
        """Создает и размещает элементы управления в интерфейсе."""
        # Кнопки управления
        self.btn_connect = Button(self.window, text="Подключение", command=self._connect_to_robot)
        self.btn_reset = Button(self.window, text="Сброс цели", command=self._reset_target)
        self.btn_track = Button(self.window, text="Запуск трекинга", command=self._start_tracking)
        self.btn_stop = Button(self.window, text="Остановить", command=self._stop_tracking)

        # Размещение кнопок
        self.btn_connect.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.btn_reset.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.btn_track.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.btn_stop.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Метка и текстовое поле для логов
        Label(self.window, text="Сообщения лога:").grid(row=2, column=0, columnspan=2, padx=10, pady=(10, 0),
                                                        sticky="w")

        text_frame = Frame(self.window)
        text_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.log_text = Text(text_frame, height=15, width=70, wrap=WORD, yscrollcommand=scrollbar.set, state="disabled")
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Настройка расширения строк и колонок
        self.window.grid_rowconfigure(3, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=1)

    def _log_message(self, message):
        """Добавляет сообщение в лог с временной меткой."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        self.log_text.config(state="normal")
        self.log_text.insert(END, formatted_message + "\n")
        self.log_text.see(END)
        self.log_text.config(state="disabled")

        # Ограничение лога
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, f"{lines - 100}.0")
            self.log_text.config(state="disabled")

    def _tof_callback(self, data):
        """Callback-функция для получения данных с TOF-датчика."""
        self.current_distance_mm = data[0]

    def _connect_to_robot(self):
        """Устанавливает соединение с роботом."""
        if self.is_connected:
            self._log_message("Уже подключено")
            return

        try:
            self._log_message("Попытка подключения к роботу...")
            self.ep_robot.initialize(conn_type="ap")

            if not self.ep_robot.get_version():
                raise Exception("Не удалось получить версию робота")

            self.ep_chassis = self.ep_robot.chassis
            self.ep_gimbal = self.ep_robot.gimbal
            self.ep_sensor = self.ep_robot.sensor

            self.ep_gimbal.resume()
            self.ep_robot.set_robot_mode(mode=robot.CHASSIS_LEAD)
            self.ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

            self.ep_gimbal.recenter().wait_for_completed()
            self.ep_gimbal.moveto(pitch=15, yaw=0).wait_for_completed()

            self.ep_sensor.sub_distance(freq=5, callback=self._tof_callback)

            self.is_connected = True
            self._log_message("✅ Успешно подключено к роботу!")

        except Exception as e:
            self._log_message(f"❌ Ошибка подключения: {str(e)}")
            self._cleanup_robot()

    def _cleanup_robot(self):
        """Корректно закрывает соединение с роботом."""
        try:
            if self.ep_sensor:
                self.ep_sensor.unsub_distance()
            if self.ep_camera:
                self.ep_camera.stop_video_stream()
            self.ep_robot.close()
        except:
            pass
        finally:
            self.is_connected = False
            self.is_tracking = False

    def _reset_target(self):
        """Сбрасывает текущую цель для трекинга."""
        self.selected_person_id = None
        self.target_lost_time = None
        self.is_searching = False
        self._stop_movement()
        self._log_message("Цель сброшена")

    def _stop_movement(self):
        """Останавливает движение шасси."""
        if self.ep_chassis:
            try:
                self.ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            except:
                pass

    def _start_search_movement(self):
        """Начинает вращение для поиска потерянной цели."""
        if not self.is_searching:
            self.is_searching = True
            self.search_start_time = time.time()
            self._log_message("Цель потеряна. Начинаю поиск...")

        # Вращение на месте
        w1 = -self.SEARCH_ROTATION_SPEED * self.search_direction
        w2 = self.SEARCH_ROTATION_SPEED * self.search_direction
        w3 = self.SEARCH_ROTATION_SPEED * self.search_direction
        w4 = -self.SEARCH_ROTATION_SPEED * self.search_direction

        return w1, w2, w3, w4

    def _check_search_timeout(self):
        """Проверяет, не превышено ли время поиска."""
        if self.is_searching and time.time() - self.search_start_time > self.SEARCH_TIMEOUT:
            self._log_message("Поиск завершен: цель не найдена")
            self.is_searching = False
            self._reset_target()
            return True
        return False

    def _is_target_really_lost(self):
        """Проверяет, действительно ли цель потеряна (не видна дольше TRACK_TIMEOUT)."""
        if self.selected_person_id is None:
            return False

        current_time = time.time()

        # Если трек есть в текущих обнаружениях - обновляем время последнего обнаружения
        if self.selected_person_id in self.tracks:
            self.last_seen_time[self.selected_person_id] = current_time
            return False

        # Если трека нет в текущих обнаружениях, проверяем как давно мы его видели
        if self.selected_person_id in self.last_seen_time:
            time_since_seen = current_time - self.last_seen_time[self.selected_person_id]
            return time_since_seen > self.TRACK_TIMEOUT

        return True

    def _start_tracking(self):
        """Запускает процесс трекинга в отдельном потоке."""
        if not self.is_connected:
            self._log_message("Сначала подключитесь к роботу")
            return

        if self.is_tracking:
            self._log_message("Трекинг уже запущен")
            return

        try:
            # Сбрасываем флаг остановки
            self.stop_tracking_flag.clear()

            # Инициализируем камеру
            self.ep_camera = self.ep_robot.camera
            self.ep_camera.start_video_stream(display=False)

            # Получаем размер кадра
            test_frame = self.ep_camera.read_cv2_image(strategy="newest")
            if test_frame is not None:
                self.frame_height, self.frame_width = test_frame.shape[:2]
            else:
                raise Exception("Не удалось получить кадр с камеры")

            self.is_tracking = True
            self._log_message("🎯 Начинаю трекинг. Кликните на человека для отслеживания")

            # Запускаем поток для обработки видео
            self.tracking_thread = threading.Thread(target=self._tracking_thread, daemon=True)
            self.tracking_thread.start()

        except Exception as e:
            self._log_message(f"Ошибка запуска трекинга: {str(e)}")
            self.is_tracking = False
            try:
                if self.ep_camera:
                    self.ep_camera.stop_video_stream()
            except:
                pass

    def _tracking_thread(self):
        """Поток для обработки видео и трекинга."""
        # Создаем окно OpenCV
        cv2.namedWindow("Person Tracking")
        cv2.setMouseCallback("Person Tracking", self._mouse_callback)

        try:
            while self.is_tracking and not self.stop_tracking_flag.is_set():
                # Чтение кадра
                frame = self.ep_camera.read_cv2_image(strategy="newest")
                if frame is None:
                    time.sleep(0.03)
                    continue

                # Обработка кадра
                processed_frame = self._process_frame(frame)

                # Управление движением
                if self.selected_person_id and self.selected_person_id in self.tracks:
                    # Цель найдена - сбрасываем флаги поиска
                    if self.is_searching:
                        self.is_searching = False
                        self._log_message("Цель найдена! Продолжаю трекинг")

                    target_x, target_y = self.tracks[self.selected_person_id]
                    self._log_message(f"Координаты цели {int(target_x)},{int(target_y)},ID={self.selected_person_id}")
                    w1, w2, w3, w4 = self._calculate_movement_speeds(target_x, target_y)
                    try:
                        self.ep_chassis.drive_wheels(w1=w1, w2=w2, w3=w3, w4=w4)
                    except Exception as e:
                        print('Ошибка при управлении колесами', e)
                else:
                    # Проверяем, действительно ли цель потеряна
                    if self.selected_person_id and self._is_target_really_lost():
                        # Цель действительно потеряна - начинаем поиск
                        if not self.is_searching:
                            self._log_message("Цель потеряна! Начинаю поиск...")
                            self.target_lost_time = time.time()
                            self.search_direction = 1  # Начинаем с вращения вправо

                        # В режиме поиска
                        if not self._check_search_timeout():
                            w1, w2, w3, w4 = self._start_search_movement()
                            try:
                                self.ep_chassis.drive_wheels(w1=w1, w2=w2, w3=w3, w4=w4)
                            except Exception as e:
                                print('Ошибка при управлении колесами в режиме поиска', e)
                        else:
                            self._stop_movement()
                    else:
                        # Цель временно не видна, но еще не считаем ее потерянной
                        self._stop_movement()

                # Отображение информации
                cv2.putText(processed_frame, f"Distance: {self.current_distance_mm}mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Определение статуса для отображения
                status_text = "OFF"
                status_color = (0, 0, 255)  # Красный по умолчанию

                if self.selected_person_id:
                    if self.selected_person_id in self.tracks:
                        status_text = "ON"
                        status_color = (0, 255, 0)  # Зеленый
                    elif self.is_searching:
                        status_text = "SEARCHING"
                        status_color = (0, 165, 255)  # Оранжевый для поиска
                        # Отображаем оставшееся время поиска
                        time_left = self.SEARCH_TIMEOUT - (time.time() - self.search_start_time)
                        cv2.putText(processed_frame, f"Search: {time_left:.1f}s", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    else:
                        status_text = "LOST"
                        status_color = (0, 0, 255)  # Красный для потерянной цели

                cv2.putText(processed_frame, f"Tracking: {status_text}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Отображаем время с последнего обнаружения выбранной цели
                if self.selected_person_id and self.selected_person_id in self.last_seen_time:
                    time_since_seen = time.time() - self.last_seen_time[self.selected_person_id]
                    cv2.putText(processed_frame, f"Last seen: {time_since_seen:.1f}s", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Показ кадра
                cv2.imshow("Person Tracking", processed_frame)

                # Проверка нажатия клавиши или закрытия окна
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty("Person Tracking", cv2.WND_PROP_VISIBLE) < 1:
                    break

                time.sleep(0.03)

        except Exception as e:
            self._log_message(f"Ошибка в потоке трекинга: {str(e)}")
        finally:
            # Всегда закрываем окно и останавливаем трекинг при выходе из потока
            try:
                cv2.destroyWindow("Person Tracking")
            except:
                pass

            # Важно: сбрасываем флаг is_tracking в конце потока
            self.is_tracking = False
            self._stop_movement()

            try:
                if self.ep_camera:
                    self.ep_camera.stop_video_stream()
            except:
                pass

    def _mouse_callback(self, event, x, y, flags, param):
        """Обрабатывает клики мыши для выбора человека."""
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_track_id = None
            min_distance = float('inf')

            for track_id, (center_x, center_y) in self.tracks.items():
                distance = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    closest_track_id = track_id

            if closest_track_id is not None:
                self.selected_person_id = closest_track_id
                self.is_searching = False  # Сбрасываем поиск при выборе новой цели
                self.last_seen_time[closest_track_id] = time.time()  # Обновляем время обнаружения
                self._log_message(f"Выбрана цель: ID {closest_track_id}")

    def _stop_tracking(self):
        """Останавливает трекинг по команде пользователя."""
        if not self.is_tracking:
            self._log_message("Трекинг не запущен")
            return

        self._log_message("Останавливаю трекинг...")

        # Устанавливаем флаг остановки
        self.stop_tracking_flag.set()
        self.is_tracking = False

        # Ждем завершения потока
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2.0)

        # Сбрасываем флаг для следующего запуска
        self.stop_tracking_flag.clear()

        self._log_message("Трекинг остановлен")

    def _scale_value(self, value, old_range, new_range):
        """Масштабирует значение из одного диапазона в другой."""
        old_min, old_max = old_range
        new_min, new_max = new_range

        if old_max - old_min == 0:
            return new_min

        return ((value - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min

    def _calculate_movement_speeds(self, target_x, target_y):
        """Рассчитывает скорости колес на основе позиции цели и расстояния."""
        center_x = self.frame_width / 2
        error_x = target_x - center_x

        error_x_range = [-center_x, center_x]

        rotation_speed = self._scale_value(
            error_x,
            error_x_range,
            [-self.MAX_ROTATION_SPEED, self.MAX_ROTATION_SPEED]
        )

        if math.fabs(error_x) > center_x / 2:
            move_speed = 0
        else:
            if self.current_distance_mm < self.MIN_DISTANCE_MM:
                move_speed = -self._scale_value(
                    self.current_distance_mm,
                    [0, self.MIN_DISTANCE_MM],
                    [self.MAX_MOVE_SPEED * 2, 0]
                )
            elif self.current_distance_mm > self.MAX_DISTANCE_MM:
                move_speed = self.MAX_MOVE_SPEED * 2
            else:
                move_speed = self._scale_value(
                    self.current_distance_mm,
                    [self.MIN_DISTANCE_MM, self.MAX_DISTANCE_MM],
                    [0, self.MAX_MOVE_SPEED * 1.5]
                )

        w1 = -rotation_speed + move_speed
        w2 = rotation_speed + move_speed
        w3 = rotation_speed + move_speed
        w4 = -rotation_speed + move_speed

        return w1, w2, w3, w4

    def _process_frame(self, frame):
        """Обрабатывает кадр: детекция, трекинг, отрисовка."""
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker="custom_track.yaml",
                classes=[0],
                conf=0.3
            )

            current_time = time.time()
            current_tracks = {}

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                confidences = results[0].boxes.conf.float().cpu().tolist()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    if cls_id == 0 and conf > 0.3:  # person class with confidence threshold
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        current_tracks[track_id] = (center_x, center_y)
                        self.last_seen_time[track_id] = current_time  # Обновляем время последнего обнаружения

                        color = (0, 0, 255) if track_id == self.selected_person_id else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, color, -1)
                        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Обновляем tracks только текущими обнаружениями
            self.tracks = current_tracks

        except Exception as e:
            self._log_message(f"Ошибка обработки кадра: {str(e)}")

        return frame

    def run(self):
        """Запускает главный цикл приложения."""
        try:
            self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.window.mainloop()
        finally:
            self._cleanup_robot()

    def _on_closing(self):
        """Обрабатывает закрытие окна."""
        self._stop_tracking()
        self._cleanup_robot()
        self.window.destroy()


if __name__ == "__main__":
    app = RobomasterPersonFollower()
    app.run()