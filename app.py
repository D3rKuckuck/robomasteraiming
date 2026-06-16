import collections
import threading
import time

import cv2
import pygame

from motion import calculate_movement_speeds, search_speeds
from robot_controller import RobotController
from tracker import PersonTracker

# ── Размеры окна ──────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720
SIDEBAR_W = 320
VIDEO_W = WIN_W - SIDEBAR_W   # 960
VIDEO_H = WIN_H               # 720

# ── Цветовая схема (тёмная) ───────────────────────────────────────────────────
C = {
    "bg":          (22, 24, 32),
    "sidebar":     (28, 32, 44),
    "divider":     (50, 55, 70),
    "btn":         (52, 76, 128),
    "btn_hover":   (70, 100, 165),
    "btn_off":     (42, 42, 55),
    "btn_danger":  (120, 42, 42),
    "btn_dng_h":   (155, 58, 58),
    "text":        (220, 225, 235),
    "text_dim":    (120, 128, 145),
    "green":       (72, 210, 115),
    "red":         (215, 72, 72),
    "orange":      (225, 160, 42),
    "blue":        (80, 150, 230),
    "log_bg":      (16, 18, 26),
    "video_bg":    (10, 12, 18),
    "overlay_bg":  (0, 0, 0),
}

MAX_LOG = 60


def _find_font(size, bold=False):
    for name in ("dejavusans", "ubuntu", "freesans", "liberationsans", "sans"):
        f = pygame.font.SysFont(name, size, bold=bold)
        if f:
            return f
    return pygame.font.Font(None, size)


# ── Компонент кнопки ─────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, enabled=True, danger=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.enabled = enabled
        self.danger = danger
        self._hover = False

    def update(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.enabled and self.rect.collidepoint(event.pos)):
            return True
        return False

    def draw(self, surf, font):
        if not self.enabled:
            bg = C["btn_off"]
            fg = C["text_dim"]
        elif self._hover:
            bg = C["btn_dng_h"] if self.danger else C["btn_hover"]
            fg = C["text"]
        else:
            bg = C["btn_danger"] if self.danger else C["btn"]
            fg = C["text"]
        pygame.draw.rect(surf, bg, self.rect, border_radius=7)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


# ── Главное приложение ────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Система слежения")

        self.font_title = _find_font(17, bold=True)
        self.font_med   = _find_font(15)
        self.font_small = _find_font(13)
        self.font_tiny  = _find_font(11)
        self.clock = pygame.time.Clock()

        self.robot   = RobotController()
        self.tracker = PersonTracker()

        # Сырой кадр с камеры (обновляется потоком камеры)
        self._raw_lock   = threading.Lock()
        self._raw_frame  = None

        # Аннотированный кадр (обновляется потоком YOLO)
        self._frame_lock = threading.Lock()
        self._cur_frame  = None

        self._frame_w = 0
        self._frame_h = 0

        # Поток камеры (запускается при подключении)
        self._camera_thread    = None
        self._camera_stop_flag = threading.Event()

        # Поток YOLO-трекинга (запускается кнопкой)
        self.is_tracking   = False
        self._track_thread = None
        self._stop_flag    = threading.Event()

        self.log: collections.deque = collections.deque(maxlen=MAX_LOG)

        # Видео: масштаб и смещение в пределах VIDEO_W x VIDEO_H
        self._vscale = 1.0
        self._voff_x = 0
        self._voff_y = 0

        # FPS-счётчик трекингового потока
        self._track_fps = 0.0
        self._fps_times: collections.deque = collections.deque(maxlen=30)

        # Флаг ручного управления колёсами (WASD)
        self._wasd_moving = False

        self._build_buttons()

    # ── UI-кнопки ─────────────────────────────────────────────────────────────
    def _build_buttons(self):
        x = VIDEO_W + 12
        w = SIDEBAR_W - 24
        self.btn_connect = Button((x, 130, w, 36), "Подключиться")
        self.btn_track   = Button((x, 174, w, 36), "Запуск трекинга", enabled=False)
        self.btn_stop    = Button((x, 218, w, 36), "Остановить",      enabled=False, danger=True)
        self.btn_reset   = Button((x, 262, w, 36), "Сброс цели",      enabled=False)
        self._buttons = [self.btn_connect, self.btn_track, self.btn_stop, self.btn_reset]

    def _refresh_buttons(self):
        c = self.robot.is_connected
        t = self.is_tracking
        h = self.tracker.selected_id is not None
        self.btn_connect.enabled = not c
        self.btn_track.enabled   = c and not t
        self.btn_stop.enabled    = t
        self.btn_reset.enabled   = t and h

    # ── Лог ───────────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    # ── Подключение к роботу ──────────────────────────────────────────────────
    def _connect(self):
        if self.robot.is_connected:
            self._log("Уже подключено")
            return
        try:
            self._log("Подключение...")
            self.robot.connect()
            self._log("Подключено успешно")
            self._start_camera()
        except Exception as e:
            self._log(f"Ошибка: {e}")
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self._refresh_buttons()

    # ── Поток камеры (сырое видео, без обработки) ─────────────────────────────
    def _start_camera(self):
        first = self.robot.start_camera()
        if first is None:
            self._log("Не удалось получить кадр с камеры")
            return
        h, w = first.shape[:2]
        self._frame_w, self._frame_h = w, h
        self._compute_video_layout(w, h)
        with self._raw_lock:
            self._raw_frame = first
        self._camera_stop_flag.clear()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        self._log("Видео запущено")

    def _camera_loop(self):
        while not self._camera_stop_flag.is_set():
            frame = self.robot.read_frame()
            if frame is not None:
                with self._raw_lock:
                    self._raw_frame = frame
            time.sleep(0.02)

    def _stop_camera(self):
        self._camera_stop_flag.set()
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
        self.robot.stop_camera()
        with self._raw_lock:
            self._raw_frame = None

    # ── Запуск YOLO-трекинга ─────────────────────────────────────────────────
    def _start_tracking(self):
        if not self.robot.is_connected or self.is_tracking:
            return
        if self._raw_frame is None:
            self._log("Видеопоток не готов")
            return
        self._stop_flag.clear()
        self.is_tracking = True
        self._log("Трекинг запущен. Кликните на цель")
        self._track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._track_thread.start()
        self._refresh_buttons()

    # ── Остановка YOLO-трекинга ───────────────────────────────────────────────
    def _stop_tracking(self):
        if not self.is_tracking:
            return
        self._stop_flag.set()
        self.is_tracking = False
        if self._track_thread:
            self._track_thread.join(timeout=3.0)
        self.robot.stop_wheels()
        with self._frame_lock:
            self._cur_frame = None
        self._log("Трекинг остановлен")
        self._refresh_buttons()

    # ── Сброс цели ────────────────────────────────────────────────────────────
    def _reset_target(self):
        self.tracker.reset()
        self.robot.stop_wheels()
        self._log("Цель сброшена")
        self._refresh_buttons()

    # ── Вычисление размещения видео ───────────────────────────────────────────
    def _compute_video_layout(self, fw, fh):
        sx = VIDEO_W / fw
        sy = VIDEO_H / fh
        self._vscale = min(sx, sy)
        dw = int(fw * self._vscale)
        dh = int(fh * self._vscale)
        self._voff_x = (VIDEO_W - dw) // 2
        self._voff_y = (VIDEO_H - dh) // 2

    def _screen_to_frame(self, sx, sy):
        return (sx - self._voff_x) / self._vscale, (sy - self._voff_y) / self._vscale

    # ── Клик по видеообласти (выбор цели) ────────────────────────────────────
    def _handle_video_click(self, sx, sy):
        if not self.is_tracking or self._frame_w == 0:
            return
        fx, fy = self._screen_to_frame(sx, sy)
        if not (0 <= fx <= self._frame_w and 0 <= fy <= self._frame_h):
            return
        tid = self.tracker.select_nearest(fx, fy)
        if tid is not None:
            self._log(f"Цель выбрана: ID {tid}")
        else:
            self._log("Цель не найдена рядом с кликом")
        self._refresh_buttons()

    # ── Ручное управление WASD (только вне трекинга) ─────────────────────────
    def _handle_manual_drive(self):
        if not self.robot.is_connected or self.is_tracking:
            if self._wasd_moving:
                self._wasd_moving = False
            return
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.robot.drive_wheels(150, 150, 150, 150)
            self._wasd_moving = True
        elif keys[pygame.K_s]:
            self.robot.drive_wheels(-150, -150, -150, -150)
            self._wasd_moving = True
        elif keys[pygame.K_a]:
            self.robot.drive_wheels(100, -100, -100, 100)
            self._wasd_moving = True
        elif keys[pygame.K_d]:
            self.robot.drive_wheels(-100, 100, 100, -100)
            self._wasd_moving = True
        else:
            if self._wasd_moving:
                self.robot.stop_wheels()
                self._wasd_moving = False

    # ── Поток YOLO-трекинга ───────────────────────────────────────────────────
    def _tracking_loop(self):
        while not self._stop_flag.is_set():
            t0 = time.time()

            with self._raw_lock:
                raw = self._raw_frame
            if raw is None:
                time.sleep(0.02)
                continue
            frame = raw.copy()  # копируем, чтобы не портить сырой кадр

            annotated = self.tracker.process_frame(frame)
            self._drive_from_tracking()

            # Оверлеи на кадр (OpenCV)
            self._draw_cv_hud(annotated)

            with self._frame_lock:
                self._cur_frame = annotated

            self._fps_times.append(time.time())
            if len(self._fps_times) >= 2:
                span = self._fps_times[-1] - self._fps_times[0]
                if span > 0:
                    self._track_fps = (len(self._fps_times) - 1) / span

            elapsed = time.time() - t0
            sleep = max(0, 0.033 - elapsed)
            time.sleep(sleep)

        self.is_tracking = False

    def _drive_from_tracking(self):
        tr = self.tracker
        if tr.selected_id is None:
            self.robot.stop_wheels()
            return

        if tr.selected_id in tr.tracks:
            if tr.is_searching:
                tr.is_searching = False
                self._log("Цель найдена")
            tx, _ = tr.tracks[tr.selected_id]
            w1, w2, w3, w4 = calculate_movement_speeds(
                tx, self._frame_w, self.robot.distance_mm
            )
            try:
                self.robot.drive_wheels(w1, w2, w3, w4)
            except Exception:
                pass
        elif tr.is_target_lost():
            if not tr.is_searching:
                tr.begin_search()
                self._log("Цель потеряна — поиск...")
            if not tr.search_timed_out():
                w1, w2, w3, w4 = search_speeds(tr.search_direction, tr.SEARCH_ROTATION_SPEED)
                try:
                    self.robot.drive_wheels(w1, w2, w3, w4)
                except Exception:
                    pass
            else:
                self._log("Поиск завершён: цель не найдена")
                tr.reset()
                self.robot.stop_wheels()
                self._refresh_buttons()
        else:
            self.robot.stop_wheels()

    def _draw_cv_hud(self, frame):
        dist = int(self.robot.distance_mm)
        cv2.putText(frame, f"Dist: {dist} mm", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if self.tracker.is_searching:
            left = self.tracker.search_time_left()
            cv2.putText(frame, f"Поиск: {left:.1f}s", (10, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # ── Рендер ────────────────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(C["bg"])
        self._draw_video_panel()
        self._draw_sidebar()
        pygame.display.flip()

    def _draw_video_panel(self):
        panel = pygame.Surface((VIDEO_W, VIDEO_H))
        panel.fill(C["video_bg"])

        # Приоритет: аннотированный кадр (YOLO) → сырой кадр → заглушка
        if self.is_tracking:
            with self._frame_lock:
                frame = self._cur_frame
        else:
            frame = None

        if frame is None:
            with self._raw_lock:
                frame = self._raw_frame

        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fh, fw = rgb.shape[:2]
                surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
                dw = int(fw * self._vscale)
                dh = int(fh * self._vscale)
                scaled = pygame.transform.scale(surf, (dw, dh))
                panel.blit(scaled, (self._voff_x, self._voff_y))
            except Exception:
                pass
        else:
            self._draw_no_signal(panel)

        # FPS
        if self.is_tracking and self._track_fps > 0:
            fps_surf = self.font_small.render(
                f"{self._track_fps:.1f} fps", True, C["text_dim"]
            )
            panel.blit(fps_surf, (VIDEO_W - fps_surf.get_width() - 8, 8))

        self.screen.blit(panel, (0, 0))

    def _draw_no_signal(self, surf):
        msg = "Нет подключения" if not self.robot.is_connected else "Ожидание камеры..."
        txt = self.font_title.render(msg, True, C["text_dim"])
        surf.blit(txt, txt.get_rect(center=(VIDEO_W // 2, VIDEO_H // 2)))

    def _draw_sidebar(self):
        sb = pygame.Surface((SIDEBAR_W, WIN_H))
        sb.fill(C["sidebar"])

        y = 14
        # Заголовок
        title = self.font_title.render("V0.2 PYGAME", True, C["blue"])
        sb.blit(title, (SIDEBAR_W // 2 - title.get_width() // 2, y))
        y += title.get_height() + 4
        sub = self.font_tiny.render("Система слежения", True, C["text_dim"])
        sb.blit(sub, (SIDEBAR_W // 2 - sub.get_width() // 2, y))
        y += sub.get_height() + 10
        pygame.draw.line(sb, C["divider"], (8, y), (SIDEBAR_W - 8, y))
        y += 10

        # Статус
        self._draw_status_block(sb, y)
        y = 310

        # Лог
        pygame.draw.line(sb, C["divider"], (8, y), (SIDEBAR_W - 8, y))
        y += 6
        lbl = self.font_small.render("ЛОГ", True, C["text_dim"])
        sb.blit(lbl, (10, y))
        y += lbl.get_height() + 4

        log_h = WIN_H - y - 80
        self._draw_log(sb, y, log_h)

        # Управление клавиатурой
        y = WIN_H - 74
        pygame.draw.line(sb, C["divider"], (8, y), (SIDEBAR_W - 8, y))
        y += 6
        hints = [
            ("W/S", "вперёд / назад"),
            ("A/D", "стрейф влево / вправо"),
        ]
        if self.is_tracking:
            note = self.font_tiny.render("Ручное управление: только вне трекинга", True, C["text_dim"])
            sb.blit(note, (10, y))
        else:
            for key, desc in hints:
                line = self.font_tiny.render(f"{key}  —  {desc}", True, C["text_dim"])
                sb.blit(line, (10, y))
                y += line.get_height() + 2

        self.screen.blit(sb, (VIDEO_W, 0))

        # Кнопки (поверх sidebar, в глобальных координатах)
        for btn in self._buttons:
            btn.draw(self.screen, self.font_med)

    def _draw_status_block(self, surf, y):
        tr = self.tracker

        def indicator(label, value_text, color):
            nonlocal y
            pygame.draw.circle(surf, color, (18, y + 8), 6)
            lbl_s = self.font_small.render(f"{label}:", True, C["text_dim"])
            val_s = self.font_small.render(value_text, True, color)
            surf.blit(lbl_s, (30, y))
            surf.blit(val_s, (30 + lbl_s.get_width() + 4, y))
            y += lbl_s.get_height() + 6

        # Связь
        if self.robot.is_connected:
            indicator("Связь", "Подключено", C["green"])
        else:
            indicator("Связь", "Отключено", C["red"])

        # Трекинг
        if not self.is_tracking:
            indicator("Трекинг", "ВЫКЛ", C["red"])
        elif tr.selected_id is None:
            indicator("Трекинг", "ВКЛ — цель не выбрана", C["orange"])
        elif tr.is_searching:
            indicator("Трекинг", "ПОИСК", C["orange"])
        elif tr.selected_id in tr.tracks:
            indicator("Трекинг", "СЛЕЖЕНИЕ", C["green"])
        else:
            indicator("Трекинг", "ПОТЕРЯ", C["red"])

        # Цель
        tid_text = str(tr.selected_id) if tr.selected_id is not None else "нет"
        indicator("Цель ID", tid_text, C["text"])

        # Расстояние
        dist_text = f"{int(self.robot.distance_mm)} мм"
        dist_color = C["green"]
        d = self.robot.distance_mm
        if d < 1500 or d > 3000:
            dist_color = C["orange"]
        indicator("Дист.", dist_text, dist_color)

        # Оставшееся время поиска
        if tr.is_searching:
            left = tr.search_time_left()
            indicator("Поиск", f"{left:.1f} с", C["orange"])
            y += 4

    def _draw_log(self, surf, y_start, height):
        log_rect = pygame.Rect(8, y_start, SIDEBAR_W - 16, height)
        pygame.draw.rect(surf, C["log_bg"], log_rect, border_radius=4)

        line_h = self.font_tiny.get_height() + 2
        visible = height // line_h
        lines = list(self.log)[-visible:]

        for i, line in enumerate(lines):
            # Метку времени выделяем тусклым, текст — обычным
            if line.startswith("[") and "] " in line:
                bracket_end = line.index("] ") + 1
                ts_part  = line[:bracket_end + 1]
                msg_part = line[bracket_end + 1:]
                ts_surf  = self.font_tiny.render(ts_part,  True, C["text_dim"])
                msg_surf = self.font_tiny.render(msg_part, True, C["text"])
                surf.blit(ts_surf,  (12, y_start + i * line_h + 3))
                surf.blit(msg_surf, (12 + ts_surf.get_width(), y_start + i * line_h + 3))
            else:
                s = self.font_tiny.render(line, True, C["text"])
                surf.blit(s, (12, y_start + i * line_h + 3))

    # ── Главный цикл ─────────────────────────────────────────────────────────
    def run(self):
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

                    # Клик в область видео
                    if (event.type == pygame.MOUSEBUTTONDOWN
                            and event.button == 1
                            and event.pos[0] < VIDEO_W):
                        self._handle_video_click(*event.pos)

                    # Кнопки
                    if self.btn_connect.update(event):
                        threading.Thread(target=self._connect, daemon=True).start()
                    if self.btn_track.update(event):
                        threading.Thread(target=self._start_tracking, daemon=True).start()
                    if self.btn_stop.update(event):
                        threading.Thread(target=self._stop_tracking, daemon=True).start()
                    if self.btn_reset.update(event):
                        self._reset_target()

                self._handle_manual_drive()
                self._draw()
                self.clock.tick(60)
        finally:
            self._cleanup()

    def _cleanup(self):
        if self.is_tracking:
            self._stop_flag.set()
            self.is_tracking = False
            if self._track_thread:
                self._track_thread.join(timeout=3.0)
        self._stop_camera()
        try:
            self.robot.disconnect()
        except Exception:
            pass
        pygame.quit()
