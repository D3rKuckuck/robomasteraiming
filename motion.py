import math

# Геометрия колёсной базы Robomaster S1
WHEEL_RADIUS_M = 0.05   # диаметр 10 см → радиус 5 см
L_EFF_M = 0.20          # (расст. м/у осями)/2 + (расст. м/у роликами)/2 = 0.10 + 0.10

# Коэффициенты пересчёта физических единиц → RPM
_ROT_SCALE  = L_EFF_M * 60.0 / (WHEEL_RADIUS_M * 2 * math.pi)  # рад/с вокруг оси → RPM колёс ≈ 38.20
_MOVE_SCALE = 60.0 / (2 * math.pi * WHEEL_RADIUS_M)             # м/с → RPM колёс ≈ 190.99


def _scale(value, old_range, new_range):
    old_min, old_max = old_range
    new_min, new_max = new_range
    if old_max == old_min:
        return new_min
    return (value - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def calculate_movement_speeds(target_x, frame_width, distance_mm,
                               max_rotation_rad_s=1.0,
                               max_fwd_m_s=1.0,  min_fwd_m_s=0.1,
                               max_bwd_m_s=0.5,  min_bwd_m_s=0.1,
                               min_dist_mm=1500,  max_dist_mm=3000,
                               rotation_deadzone_px=15,
                               verbose=False):
    """Возвращает (w1, w2, w3, w4) в RPM для Mecanum-привода.

    max_rotation_rad_s     — максимальная угловая скорость (рад/с)
    max_fwd_m_s / min_fwd_m_s — диапазон скорости вперёд (м/с)
    max_bwd_m_s / min_bwd_m_s — диапазон скорости назад (м/с)
    min_dist_mm / max_dist_mm — целевой диапазон дистанции (мм)
    rotation_deadzone_px   — мёртвая зона поворота по ошибке в пикселях
    """
    center_x = frame_width / 2
    error_x  = target_x - center_x

    # Угловое воздействие: мёртвая зона по пикселям, затем масштаб → рад/с → RPM
    if abs(error_x) < rotation_deadzone_px:
        rot_rad_s = 0.0
    else:
        rot_rad_s = _scale(error_x, [-center_x, center_x], [-max_rotation_rad_s, max_rotation_rad_s])
    rotation = rot_rad_s * _ROT_SCALE  # RPM

    # Линейное воздействие: дистанция → м/с → RPM
    if abs(error_x) > frame_width / 11.3: # При ширине цгла обзора 113 градусов это ~10
        move_m_s = 0.0  # сначала довернуть, движение заблокировано
    elif distance_mm < min_dist_mm:
        move_m_s = _scale(distance_mm, [0, min_dist_mm], [-max_bwd_m_s, 0.0])
        if abs(move_m_s) < min_bwd_m_s:
            move_m_s = 0.0
    elif distance_mm > max_dist_mm:
        move_m_s = max_fwd_m_s
    else:
        move_m_s = _scale(distance_mm, [min_dist_mm, max_dist_mm], [0.0, max_fwd_m_s])
        if 0 < move_m_s < min_fwd_m_s:
            move_m_s = 0.0
    move = move_m_s * _MOVE_SCALE  # RPM

    w1 = -rotation + move
    w2 =  rotation + move
    w3 =  rotation + move
    w4 = -rotation + move

    if verbose:
        print(
            f"error_x={error_x:+.1f}px  "
            f"rot={rot_rad_s:+.3f}rad/s ({rotation:+.1f}RPM)  "
            f"dist={distance_mm:.0f}mm  "
            f"move={move_m_s:+.3f}m/s ({move:+.1f}RPM)  "
            f"wheels=[{w1:.1f}, {w2:.1f}, {w3:.1f}, {w4:.1f}]"
        )

    return w1, w2, w3, w4


def search_speeds(direction, speed_rad_s=0.3):
    """Вращение на месте при поиске цели.

    direction   — 1 или -1
    speed_rad_s — скорость вращения платформы (рад/с)
    """
    s = direction * speed_rad_s * _ROT_SCALE
    return -s, s, s, -s


def wasd_rpm(forward=0.0, strafe=0.0):
    """RPM колёс для ручного WASD-управления.

    forward — скорость вперёд (м/с), отрицательная = назад
    strafe  — скорость стрейфа вправо (м/с), отрицательная = влево
    """
    fwd = forward * _MOVE_SCALE
    lat = strafe  * _MOVE_SCALE
    return (
        fwd - lat,   # w1
        fwd + lat,   # w2
        fwd + lat,   # w3
        fwd - lat,   # w4
    )
