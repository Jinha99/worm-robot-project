"""
Worm Robot Simulation - Utility Functions
유틸리티 함수 모음
"""

from config import GRID_MIN, GRID_MAX


def in_bounds(pos):
    """
    위치가 격자 범위 내에 있는지 확인

    Args:
        pos: (x, y) 튜플

    Returns:
        bool: 범위 내에 있으면 True, 아니면 False
    """
    x, y = pos
    return GRID_MIN <= x <= GRID_MAX and GRID_MIN <= y <= GRID_MAX


def add_pos(pos1, pos2):
    """
    두 위치 좌표를 더함

    Args:
        pos1: (x1, y1) 튜플
        pos2: (x2, y2) 튜플

    Returns:
        tuple: (x1+x2, y1+y2)
    """
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])


def get_sensor_area(head_pos):
    """
    앞발 위치 기준 3x3 센서 영역 반환

    Args:
        head_pos: 앞발 위치 (x, y) 튜플

    Returns:
        list: 센서 영역 내 모든 좌표 리스트
    """
    sensors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            sensors.append(add_pos(head_pos, (dx, dy)))
    return sensors
