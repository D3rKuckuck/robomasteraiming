import collections
import threading
import time

import cv2
import pygame

from motion import calculate_movement_speeds, search_speeds, wasd_rpm
from robot_controller import RobotController
from tracker import PersonTracker, resolve_device

# ── Фиксированные ширины боковых панелей ─────────────────────────────────────
LEFT_W    = 320
SIDEBAR_W = 320
WIN_W_MIN = LEFT_W + 480 + SIDEBAR_W   # минимальная ширина окна

# ── Цветовая схема ────────────────────────────────────────────────────────────
C = {
    "bg":         (22,  24,  32),
    "sidebar":    (28,  32,  44),
    "left":       (24,  28,  40),
    "divider":    (50,  55,  70),
    "btn":        (52,  76,  128),
    "btn_hover":  (70,  100, 165),
    "btn_off":    (42,  42,  55),
    "btn_danger": (120, 42,  42),
    "btn_dng_h":  (155, 58,  58),
    "text":       (220, 225, 235),
    "text_dim":   (120, 128, 145),
    "green":      (72,  210, 115),
    "red":        (215, 72,  72),
    "orange":     (225, 160, 42),
    "blue":       (80,  150, 230),
    "log_bg":     (16,  18,  26),
    "video_bg":   (10,  12,  18),
    "val_bg":     (18,  20,  30),
}

MAX_LOG = 60
PAD = 12   # горизонтальный отступ внутри панелей


def _find_font(size, bold=False):
    for name in ("dejavusans", "ubuntu", "freesans", "liberationsans", "sans"):
        f = pygame.font.SysFont(name, size, bold=bold)
        if f:
            return f
    return pygame.font.Font(None, size)


# ── Кнопка ────────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, enabled=True, danger=False):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.enabled = enabled
        self.danger  = danger
        self._hover  = False

    def update(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.enabled and self.rect.collidepoint(event.pos)):
            return True
        return False

    def draw(self, surf, font):
        if not self.enabled:
            bg, fg = C["btn_off"], C["text_dim"]
        elif self._hover:
            bg = C["btn_dng_h"] if self.danger else C["btn_hover"]
            fg = C["text"]
        else:
            bg = C["btn_danger"] if self.danger else C["btn"]
            fg = C["text"]
        pygame.draw.rect(surf, bg, self.rect, border_radius=7)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


# ── Числовой спиннер ──────────────────────────────────────────────────────────
class SpinRow:
    """Ряд [−] значение [+] в глобальных экранных координатах."""

    BTN_W = 26

    def __init__(self, x, y, w, h, value, min_val, max_val, step, fmt="{:.1f}"):
        self.value   = value
        self.min_val = min_val
        self.max_val = max_val
        self.step    = step
        self.fmt     = fmt
        self.enabled = True
        bw = self.BTN_W
        self.btn_minus   = pygame.Rect(x,          y, bw,      h)
        self.btn_plus    = pygame.Rect(x + w - bw, y, bw,      h)
        self.display     = pygame.Rect(x + bw,     y, w-2*bw,  h)

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_minus.collidepoint(event.pos):
                self.value = round(max(self.min_val, self.value - self.step), 4)
                return True
            if self.btn_plus.collidepoint(event.pos):
                self.value = round(min(self.max_val, self.value + self.step), 4)
                return True
        return False

    def draw(self, surf, font):
        mp = pygame.mouse.get_pos()
        for rect, label in ((self.btn_minus, "−"), (self.btn_plus, "+")):
            hover = self.enabled and rect.collidepoint(mp)
            bg = C["btn_hover"] if hover else (C["btn"] if self.enabled else C["btn_off"])
            pygame.draw.rect(surf, bg, rect, border_radius=4)
            t = font.render(label, True, C["text"] if self.enabled else C["text_dim"])
            surf.blit(t, t.get_rect(center=rect.center))
        pygame.draw.rect(surf, C["val_bg"], self.display)
        vt = font.render(self.fmt.format(self.value), True,
                         C["text"] if self.enabled else C["text_dim"])
        surf.blit(vt, vt.get_rect(center=self.display.center))


# ── Пара переключателей ───────────────────────────────────────────────────────
class TogglePair:
    """Два взаимоисключающих переключателя (как radio buttons)."""

    def __init__(self, x, y, w, h, options):
        """options: [(label_a, value_a), (label_b, value_b)]"""
        self.options  = options
        self.selected = 0
        self.enabled  = True
        half = (w - 4) // 2
        self.rects = [
            pygame.Rect(x,          y, half,       h),
            pygame.Rect(x + half + 4, y, w-half-4, h),
        ]

    def handle_event(self, event):
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(self.rects):
                if rect.collidepoint(event.pos) and self.selected != i:
                    self.selected = i
                    return True
        return False

    @property
    def value(self):
        return self.options[self.selected][1]

    def draw(self, surf, font):
        mp = pygame.mouse.get_pos()
        for i, (rect, (label, _)) in enumerate(zip(self.rects, self.options)):
            active = (i == self.selected)
            hover  = self.enabled and rect.collidepoint(mp) and not active
            if not self.enabled:
                bg, fg = C["btn_off"], C["text_dim"]
            elif active:
                bg, fg = C["green"], C["bg"]
            elif hover:
                bg, fg = C["btn_hover"], C["text"]
            else:
                bg, fg = C["btn"], C["text_dim"]
            pygame.draw.rect(surf, bg, rect, border_radius=5)
            t = font.render(label, True, fg)
            surf.blit(t, t.get_rect(center=rect.center))


# ── Главное приложение ────────────────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        start_w = max(WIN_W_MIN, info.current_w)
        start_h = max(480,       info.current_h)
        self.screen = pygame.display.set_mode(
            (start_w, start_h), pygame.RESIZABLE
        )
        pygame.display.set_caption("Система слежения")

        self.font_title = _find_font(17, bold=True)
        self.font_med   = _find_font(15)
        self.font_small = _find_font(13)
        self.font_tiny  = _find_font(11)
        self.clock = pygame.time.Clock()

        self.robot = RobotController()

        # Создаём трекер с дефолтными настройками; пересоздаётся при смене конфига
        self._cur_model  = 'yolo11n.pt'
        self._cur_device = 'cpu'
        self.tracker = PersonTracker(self._cur_model, self._cur_device)

        # Кадры
        self._raw_lock   = threading.Lock()
        self._raw_frame  = None
        self._frame_lock = threading.Lock()
        self._cur_frame  = None
        self._frame_w = 0
        self._frame_h = 0

        # Потоки
        self._camera_thread    = None
        self._camera_stop_flag = threading.Event()
        self.is_tracking       = False
        self._track_thread     = None
        self._stop_flag        = threading.Event()

        self.log: collections.deque = collections.deque(maxlen=MAX_LOG)

        # Видео: масштаб и смещение внутри видеопанели
        self._vscale = 1.0
        self._voff_x = 0
        self._voff_y = 0

        # FPS трекингового потока
        self._track_fps  = 0.0
        self._fps_times: collections.deque = collections.deque(maxlen=30)

        self._wasd_moving = False

        self._build_ui()

    # ── Динамические размеры ──────────────────────────────────────────────────
    @property
    def _vw(self):
        """Текущая ширина видеопанели (меняется при ресайзе)."""
        return self.screen.get_width() - LEFT_W - SIDEBAR_W

    @property
    def _wh(self):
        """Текущая высота окна."""
        return self.screen.get_height()

    def _on_resize(self):
        """Вызывается при изменении размера окна."""
        self._reposition_sidebar_buttons()
        if self._frame_w > 0:
            self._compute_video_layout(self._frame_w, self._frame_h)

    def _reposition_sidebar_buttons(self):
        """Пересчитывает X-позиции кнопок правой панели."""
        rx = LEFT_W + self._vw + PAD
        rw = SIDEBAR_W - 2 * PAD
        for i, btn in enumerate([self.btn_connect, self.btn_track, self.btn_stop,
                                   self.btn_reset, self.btn_disconnect]):
            btn.rect = pygame.Rect(rx, 130 + i * 44, rw, 36)

    # ── Построение UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # --- Правая панель: кнопки управления ---
        rx = LEFT_W + self._vw + PAD
        rw = SIDEBAR_W - 2 * PAD

        self.btn_connect    = Button((rx, 130, rw, 36), "Подключиться")
        self.btn_track      = Button((rx, 174, rw, 36), "Запуск трекинга",  enabled=False)
        self.btn_stop       = Button((rx, 218, rw, 36), "Остановить",       enabled=False, danger=True)
        self.btn_reset      = Button((rx, 262, rw, 36), "Сброс цели",       enabled=False)
        self.btn_disconnect = Button((rx, 306, rw, 36), "Отключиться",      enabled=False, danger=True)
        self._buttons = [
            self.btn_connect, self.btn_track, self.btn_stop,
            self.btn_reset, self.btn_disconnect,
        ]

        # --- Левая панель: параметры ---
        # Все Y-позиции вычислены вручную с учётом высоты меток (13px), отступов
        # и разделителей. Схема на один блок:
        #   divider(1) + gap(4) + section_title(13) + gap(4) = 22px
        #   label(13) + gap(2) + spinrow(24) + gap(4) = 43px
        lx = PAD
        lw = LEFT_W - 2 * PAD
        ROW_H = 24

        def spin(y, val, lo, hi, step, fmt="{:.1f}"):
            return SpinRow(lx, y, lw, ROW_H, val, lo, hi, step, fmt)

        # ── Скорости вращения ─────────────────────────────────── y=8
        # label at 30, spin at 45
        self.spin_max_rot = spin(45,  1.0, 0.1, 3.0, 0.1)   # bottom=69
        # label at 73, spin at 88
        self.spin_min_rot = spin(88,  0.1, 0.1, 3.0, 0.1)   # bottom=112

        # ── Скорость вперёд ───────────────────────────────────── y=120
        # label at 142, spin at 157
        self.spin_max_fwd = spin(157, 1.0, 0.1, 8.0, 0.1)   # bottom=181
        # label at 185, spin at 200
        self.spin_min_fwd = spin(200, 0.1, 0.1, 8.0, 0.1)   # bottom=224

        # ── Скорость назад ────────────────────────────────────── y=232
        # label at 254, spin at 269
        self.spin_max_bwd = spin(269, 0.5, 0.1, 8.0, 0.1)   # bottom=293
        # label at 297, spin at 312
        self.spin_min_bwd = spin(312, 0.1, 0.1, 8.0, 0.1)   # bottom=336

        # ── Дистанция ─────────────────────────────────────────── y=344
        # label at 366, spin at 381
        self.spin_min_dist = spin(381, 1500, 200,  4000, 50, "{:.0f}")  # bottom=405
        # label at 409, spin at 424
        self.spin_max_dist = spin(424, 3000, 500,  6000, 50, "{:.0f}")  # bottom=448

        # ── Модель YOLO ───────────────────────────────────────── y=456
        # toggle at 478, bottom=506; hint1=508, hint2=521
        self.toggle_model = TogglePair(
            lx, 478, lw, 26,
            [("yolo11n", "yolo11n.pt"), ("yolo26n", "yolo26n.pt")],
        )

        # ── Устройство ────────────────────────────────────────── y=536
        # toggle at 558, bottom=584; hint=586
        self.toggle_device = TogglePair(
            lx, 558, lw, 26,
            [("CPU", "cpu"), ("GPU (CUDA)", "cuda")],
        )

        # ── Гимбал ────────────────────────────────────────────── y=606
        # label at 622, spin at 637, bottom=661; btn=665
        self.spin_gimbal = spin(637, 15, 0, 35, 1, "{:.0f}°")
        self.btn_gimbal  = Button((lx, 665, lw, 26), "Применить угол", enabled=False)

        self._speed_spins = [
            self.spin_max_rot, self.spin_min_rot,
            self.spin_max_fwd, self.spin_min_fwd,
            self.spin_max_bwd, self.spin_min_bwd,
        ]
        self._dist_spins  = [self.spin_min_dist, self.spin_max_dist]
        self._left_spins  = self._speed_spins + self._dist_spins + [self.spin_gimbal]

        self._reposition_sidebar_buttons()

    def _refresh_buttons(self):
        c = self.robot.is_connected
        t = self.is_tracking
        h = self.tracker.selected_id is not None

        self.btn_connect.enabled    = not c
        self.btn_track.enabled      = c and not t
        self.btn_stop.enabled       = t
        self.btn_reset.enabled      = t and h
        self.btn_disconnect.enabled = c and not t

        # Дистанции — только до трекинга
        for s in self._dist_spins:
            s.enabled = not t

        # Модель и устройство — только до подключения
        self.toggle_model.enabled  = not c
        self.toggle_device.enabled = not c

        # Гимбал — подключены, не в трекинге
        self.spin_gimbal.enabled = c and not t
        self.btn_gimbal.enabled  = c and not t

    # ── Лог ───────────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    # ── Подключение ───────────────────────────────────────────────────────────
    def _connect(self):
        if self.robot.is_connected:
            self._log("Уже подключено")
            return
        try:
            self._log("Подключение...")
            self.robot.connect(gimbal_pitch_deg=int(self.spin_gimbal.value))
            self._log("Подключено успешно")
            self._start_camera()
        except Exception as e:
            self._log(f"Ошибка подключения: {e}")
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self._refresh_buttons()

    # ── Отключение ────────────────────────────────────────────────────────────
    def _disconnect(self):
        if not self.robot.is_connected:
            return
        if self.is_tracking:
            self._stop_tracking()
        self._stop_camera()
        try:
            self.robot.disconnect()
            self._log("Отключено")
        except Exception as e:
            self._log(f"Ошибка отключения: {e}")
        self._refresh_buttons()

    # ── Гимбал ────────────────────────────────────────────────────────────────
    def _apply_gimbal(self):
        angle = int(self.spin_gimbal.value)
        self._log(f"Гимбал → {angle}°...")
        try:
            self.robot.set_gimbal_pitch(angle)
            self._log(f"Гимбал установлен: {angle}°")
        except Exception as e:
            self._log(f"Ошибка гимбала: {e}")

    # ── Поток камеры ──────────────────────────────────────────────────────────
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

    # ── Трекинг ───────────────────────────────────────────────────────────────
    def _start_tracking(self):
        if not self.robot.is_connected or self.is_tracking:
            return
        if self._raw_frame is None:
            self._log("Видеопоток не готов")
            return

        # Пересоздаём трекер если конфиг поменялся
        new_model  = self.toggle_model.value
        new_device = resolve_device(self.toggle_device.value)
        if new_model != self._cur_model or new_device != self._cur_device:
            self._log(f"Загрузка {new_model} на {new_device}...")
            self.tracker    = PersonTracker(new_model, new_device)
            self._cur_model  = new_model
            self._cur_device = new_device
            self._log("Модель загружена")

        self._stop_flag.clear()
        self.is_tracking = True
        self._log("Трекинг запущен. Кликните на цель")
        self._track_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._track_thread.start()
        self._refresh_buttons()

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

    def _reset_target(self):
        self.tracker.reset()
        self.robot.stop_wheels()
        self._log("Цель сброшена")
        self._refresh_buttons()

    # ── Видео: масштаб ────────────────────────────────────────────────────────
    def _compute_video_layout(self, fw, fh):
        sx = self._vw / fw
        sy = self._wh  / fh
        self._vscale = min(sx, sy)
        dw = int(fw * self._vscale)
        dh = int(fh * self._vscale)
        self._voff_x = (self._vw - dw) // 2
        self._voff_y = (self._wh  - dh) // 2

    def _screen_to_frame(self, sx, sy):
        """Перевод глобальных координат клика в координаты кадра камеры."""
        lx = sx - LEFT_W - self._voff_x
        ly = sy - self._voff_y
        return lx / self._vscale, ly / self._vscale

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

    # ── Ручное управление WASD ────────────────────────────────────────────────
    def _handle_manual_drive(self):
        if not self.robot.is_connected or self.is_tracking:
            if self._wasd_moving:
                self._wasd_moving = False
            return
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.robot.drive_wheels(*wasd_rpm(forward=0.8))
            self._wasd_moving = True
        elif keys[pygame.K_s]:
            self.robot.drive_wheels(*wasd_rpm(forward=-0.8))
            self._wasd_moving = True
        elif keys[pygame.K_a]:
            self.robot.drive_wheels(*wasd_rpm(strafe=-0.5))
            self._wasd_moving = True
        elif keys[pygame.K_d]:
            self.robot.drive_wheels(*wasd_rpm(strafe=0.5))
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
            frame = raw.copy()

            annotated = self.tracker.process_frame(frame)
            self._drive_from_tracking()
            self._draw_cv_hud(annotated)

            with self._frame_lock:
                self._cur_frame = annotated

            self._fps_times.append(time.time())
            if len(self._fps_times) >= 2:
                span = self._fps_times[-1] - self._fps_times[0]
                if span > 0:
                    self._track_fps = (len(self._fps_times) - 1) / span

            elapsed = time.time() - t0
            time.sleep(max(0, 0.033 - elapsed))

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
                tx, self._frame_w, self.robot.distance_mm,
                max_rotation_rev_s = self.spin_max_rot.value,
                min_rotation_rev_s = self.spin_min_rot.value,
                max_fwd_m_s        = self.spin_max_fwd.value,
                min_fwd_m_s        = self.spin_min_fwd.value,
                max_bwd_m_s        = self.spin_max_bwd.value,
                min_bwd_m_s        = self.spin_min_bwd.value,
                min_dist_mm        = self.spin_min_dist.value,
                max_dist_mm        = self.spin_max_dist.value,
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
                w1, w2, w3, w4 = search_speeds(tr.search_direction,
                                                tr.SEARCH_ROTATION_SPEED)
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
        self._draw_left_panel()
        self._draw_video_panel()
        self._draw_sidebar()
        pygame.display.flip()

    # ── Левая панель ──────────────────────────────────────────────────────────
    def _draw_left_panel(self):
        panel = pygame.Surface((LEFT_W, self._wh))
        panel.fill(C["left"])
        self.screen.blit(panel, (0, 0))

        lx = PAD

        def section(title, y):
            """Рисует разделитель + заголовок секции; возвращает y нижней границы."""
            pygame.draw.line(self.screen, C["divider"], (lx, y), (LEFT_W - lx, y))
            y += 5
            t = self.font_tiny.render(title, True, C["blue"])
            self.screen.blit(t, (lx, y))
            return y + t.get_height() + 4

        def lbl(text, y):
            """Рисует метку; возвращает y нижней границы."""
            t = self.font_tiny.render(text, True, C["text_dim"])
            self.screen.blit(t, (lx, y))
            return y + t.get_height() + 2

        def hint(text, y):
            t = self.font_tiny.render(text, True, C["text_dim"])
            self.screen.blit(t, (lx + 4, y))
            return y + t.get_height() + 2

        # ── Скорости вращения ──────── y=8
        section("СКОРОСТИ ВРАЩЕНИЯ (об/с)", 8)
        lbl("Макс. скорость поворота",  30)
        self.spin_max_rot.draw(self.screen, self.font_small)
        lbl("Мин. скорость (мёртвая зона)", 73)
        self.spin_min_rot.draw(self.screen, self.font_small)

        # ── Скорость вперёд ────────── y=120
        section("СКОРОСТЬ ВПЕРЁД (м/с)", 120)
        lbl("Макс.", 142)
        self.spin_max_fwd.draw(self.screen, self.font_small)
        lbl("Мин.", 185)
        self.spin_min_fwd.draw(self.screen, self.font_small)

        # ── Скорость назад ─────────── y=232
        section("СКОРОСТЬ НАЗАД (м/с)", 232)
        lbl("Макс.", 254)
        self.spin_max_bwd.draw(self.screen, self.font_small)
        lbl("Мин.", 297)
        self.spin_min_bwd.draw(self.screen, self.font_small)

        # ── Дистанция ──────────────── y=344
        section("ДИСТАНЦИЯ (мм)", 344)
        lbl("Мин. (ближе — ехать назад)", 366)
        self.spin_min_dist.draw(self.screen, self.font_small)
        lbl("Макс. (дальше — полный вперёд)", 409)
        self.spin_max_dist.draw(self.screen, self.font_small)

        # ── Модель YOLO ────────────── y=456
        section("МОДЕЛЬ YOLO", 456)
        self.toggle_model.draw(self.screen, self.font_small)
        hint("yolo11n — быстрее на GPU",        508)
        hint("yolo26n — эффективнее на CPU",    521)

        # ── Устройство ─────────────── y=536
        section("УСТРОЙСТВО", 536)
        self.toggle_device.draw(self.screen, self.font_small)
        hint("GPU — только при наличии CUDA",   586)

        # ── Гимбал ─────────────────── y=606
        section("ГИМБАЛ (0–35°)", 606)
        lbl("Угол наклона башни", 622)
        self.spin_gimbal.draw(self.screen, self.font_small)
        self.btn_gimbal.draw(self.screen, self.font_small)

    # ── Видеопанель ───────────────────────────────────────────────────────────
    def _draw_video_panel(self):
        panel = pygame.Surface((self._vw, self._wh))
        panel.fill(C["video_bg"])

        frame = None
        if self.is_tracking:
            with self._frame_lock:
                frame = self._cur_frame
        if frame is None:
            with self._raw_lock:
                frame = self._raw_frame

        if frame is not None:
            try:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fh, fw = rgb.shape[:2]
                surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
                dw   = int(fw * self._vscale)
                dh   = int(fh * self._vscale)
                scaled = pygame.transform.scale(surf, (dw, dh))
                panel.blit(scaled, (self._voff_x, self._voff_y))
            except Exception:
                pass
        else:
            msg = "Нет подключения" if not self.robot.is_connected else "Ожидание камеры..."
            txt = self.font_title.render(msg, True, C["text_dim"])
            panel.blit(txt, txt.get_rect(center=(self._vw // 2, self._wh // 2)))

        if self.is_tracking and self._track_fps > 0:
            fps_s = self.font_small.render(f"{self._track_fps:.1f} fps", True, C["text_dim"])
            panel.blit(fps_s, (self._vw - fps_s.get_width() - 8, 8))

        self.screen.blit(panel, (LEFT_W, 0))

    # ── Правая панель ─────────────────────────────────────────────────────────
    def _draw_sidebar(self):
        sb = pygame.Surface((SIDEBAR_W, self._wh))
        sb.fill(C["sidebar"])

        y = 14
        title = self.font_title.render("V0.3", True, C["blue"])
        sb.blit(title, (SIDEBAR_W // 2 - title.get_width() // 2, y))
        y += title.get_height() + 4
        sub = self.font_tiny.render("Система слежения", True, C["text_dim"])
        sb.blit(sub, (SIDEBAR_W // 2 - sub.get_width() // 2, y))
        y += sub.get_height() + 10
        pygame.draw.line(sb, C["divider"], (8, y), (SIDEBAR_W - 8, y))
        y += 10

        self._draw_status_block(sb, y)
        y = 360

        pygame.draw.line(sb, C["divider"], (8, y), (SIDEBAR_W - 8, y))
        y += 6
        lbl = self.font_small.render("ЛОГ", True, C["text_dim"])
        sb.blit(lbl, (10, y))
        y += lbl.get_height() + 4

        log_h = self._wh - y - 80
        self._draw_log(sb, y, log_h)

        y = self._wh - 74
        pygame.draw.line(sb, C["divider"], (8, y), (SIDEBAR_W - 8, y))
        y += 6
        if self.is_tracking:
            note = self.font_tiny.render("Ручное управление: только вне трекинга", True, C["text_dim"])
            sb.blit(note, (10, y))
        else:
            for key, desc in [("W/S", "вперёд / назад"), ("A/D", "стрейф влево / вправо")]:
                line = self.font_tiny.render(f"{key}  —  {desc}", True, C["text_dim"])
                sb.blit(line, (10, y))
                y += line.get_height() + 2

        self.screen.blit(sb, (LEFT_W + self._vw, 0))

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

        indicator("Связь",  "Подключено" if self.robot.is_connected else "Отключено",
                  C["green"] if self.robot.is_connected else C["red"])

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

        indicator("Цель ID", str(tr.selected_id) if tr.selected_id is not None else "нет",
                  C["text"])

        d = self.robot.distance_mm
        dist_color = C["green"] if self.spin_min_dist.value <= d <= self.spin_max_dist.value else C["orange"]
        indicator("Дист.", f"{int(d)} мм", dist_color)

        indicator("Модель", self._cur_model, C["text_dim"])
        indicator("Устр.", self._cur_device, C["text_dim"])

        if tr.is_searching:
            left = tr.search_time_left()
            indicator("Поиск", f"{left:.1f} с", C["orange"])

    @staticmethod
    def _wrap_text(font, text, max_width):
        """Разбивает text на строки шириной не более max_width пикселей."""
        if not text:
            return ['']
        words, lines, current = text.split(' '), [], ''
        for word in words:
            candidate = (current + ' ' + word).lstrip()
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or ['']

    def _draw_log(self, surf, y_start, height):
        log_rect = pygame.Rect(8, y_start, SIDEBAR_W - 16, height)
        pygame.draw.rect(surf, C["log_bg"], log_rect, border_radius=4)

        line_h = self.font_tiny.get_height() + 2
        text_x = 12
        max_w  = SIDEBAR_W - text_x - 12   # доступная ширина для текста

        # Строим список визуальных строк: каждая — список (surface, x_offset)
        rows = []
        for entry in self.log:
            if entry.startswith("[") and "] " in entry:
                end     = entry.index("] ") + 1
                ts_surf = self.font_tiny.render(entry[:end+1], True, C["text_dim"])
                ts_w    = ts_surf.get_width()
                msg_lines = self._wrap_text(self.font_tiny, entry[end+1:], max_w - ts_w)
                # Первая строка: метка времени + начало сообщения
                rows.append([(ts_surf, 0),
                              (self.font_tiny.render(msg_lines[0], True, C["text"]), ts_w)])
                # Продолжение: с отступом под метку
                for ml in msg_lines[1:]:
                    rows.append([(self.font_tiny.render(ml, True, C["text"]), ts_w)])
            else:
                for ml in self._wrap_text(self.font_tiny, entry, max_w):
                    rows.append([(self.font_tiny.render(ml, True, C["text"]), 0)])

        visible = height // line_h
        for i, row in enumerate(rows[-visible:]):
            y = y_start + i * line_h + 3
            for s, xoff in row:
                surf.blit(s, (text_x + xoff, y))

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
                    if event.type == pygame.VIDEORESIZE:
                        w = max(WIN_W_MIN, event.w)
                        h = max(480, event.h)
                        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
                        self._on_resize()

                    # Клик по видеообласти
                    if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                            and LEFT_W <= event.pos[0] < LEFT_W + self._vw):
                        self._handle_video_click(*event.pos)

                    # Правая панель: кнопки
                    if self.btn_connect.update(event):
                        threading.Thread(target=self._connect, daemon=True).start()
                    if self.btn_track.update(event):
                        threading.Thread(target=self._start_tracking, daemon=True).start()
                    if self.btn_stop.update(event):
                        threading.Thread(target=self._stop_tracking, daemon=True).start()
                    if self.btn_reset.update(event):
                        self._reset_target()
                    if self.btn_disconnect.update(event):
                        threading.Thread(target=self._disconnect, daemon=True).start()
                    if self.btn_gimbal.update(event):
                        threading.Thread(target=self._apply_gimbal, daemon=True).start()

                    # Левая панель: спиннеры и переключатели
                    for spin in self._left_spins:
                        spin.handle_event(event)

                    if self.toggle_model.handle_event(event):
                        pass  # применится при следующем старте трекинга
                    if self.toggle_device.handle_event(event):
                        pass

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
