import math


def _scale(value, old_range, new_range):
    old_min, old_max = old_range
    new_min, new_max = new_range
    if old_max == old_min:
        return new_min
    return ((value - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min


def calculate_movement_speeds(target_x, frame_width, distance_mm,
                               max_rotation=80, max_move=100,
                               min_dist=1500, max_dist=3000):
    """Возвращает (w1, w2, w3, w4) для Mecanum-привода по ошибке X и дистанции."""
    center_x = frame_width / 2
    error_x = target_x - center_x

    rotation = _scale(error_x, [-center_x, center_x], [-max_rotation, max_rotation])

    if math.fabs(error_x) > center_x / 2:
        move = 0
    elif distance_mm < min_dist:
        move = -_scale(distance_mm, [0, min_dist], [max_move * 2, 0])
    elif distance_mm > max_dist:
        move = max_move * 2
    else:
        move = _scale(distance_mm, [min_dist, max_dist], [0, max_move * 1.5])

    return (
        -rotation + move,
         rotation + move,
         rotation + move,
        -rotation + move,
    )


def search_speeds(direction, speed=20):
    """Скорости колёс для вращения на месте при поиске цели."""
    s = speed * direction
    return -s, s, s, -s
