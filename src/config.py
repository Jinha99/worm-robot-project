"""
Worm Robot Simulation - Configuration
상수 및 설정값 관리
"""

import random

# ========================================
# 격자 설정
# ========================================

GRID_SIZE = 7
GRID_MIN = -3
GRID_MAX = 3


# ========================================
# 방향 정의
# ========================================

# 방향: 0=동(1,0), 1=남(0,-1), 2=서(-1,0), 3=북(0,1)
DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
DIR_NAMES = ["동", "남", "서", "북"]


# ========================================
# 행동 타입
# ========================================

ACTION_MOVE = "move"              # 전진 (5초)
ACTION_ROTATE_CW = "rotate_cw"    # 시계방향 회전 (3초)
ACTION_ROTATE_CCW = "rotate_ccw"  # 반시계방향 회전 (3초)


# ========================================
# 행동 소요 시간
# ========================================

ACTION_TIMES = {
    ACTION_MOVE: 5,
    ACTION_ROTATE_CW: 3,
    ACTION_ROTATE_CCW: 3,
}


# ========================================
# 게임 상태
# ========================================

STATUS_RUNNING = "running"
STATUS_WIN = "win"
STATUS_PARTIAL_WIN = "partial_win"  # 일부 로봇만 성공
STATUS_FAIL = "fail"


# ========================================
# 목적지 설정
# ========================================

# 4개의 목적지 (십자 패턴)
GOAL_POSITIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ========================================
# 로봇 초기 설정
# ========================================

NUM_ROBOTS = 1  # 커리큘럼 학습 1단계: 1개로 시작 → 2개 → 4개

# 커리큘럼 학습을 위한 로봇 간 최소 거리
# 1단계 (1개): 0 (거리 제한 없음)
# 2단계 (2개): 6 (서로 멀리 배치)
# 3단계 (4개): 0 (거리 제한 없음)
MIN_ROBOT_DISTANCE = 0  # 현재 1단계: 거리 제한 없음


def generate_random_robot_configs(num_robots=NUM_ROBOTS, min_distance=None):
    """
    격자 내에서 랜덤한 위치에 로봇들을 배치하고, 각 로봇에게 서로 다른 목적지를 할당합니다.
    로봇끼리 겹치지 않도록 보장합니다.

    Args:
        num_robots: 로봇 수 (최대 4개)
        min_distance: 로봇 간 최소 맨해튼 거리 (None이면 MIN_ROBOT_DISTANCE 사용)
                     예: min_distance=6이면 로봇들이 최소 6칸 이상 떨어짐

    Returns:
        list: 로봇 설정 리스트 [{"id": int, "head": tuple, "tail": tuple, "dir": int, "goal": tuple}, ...]

    Raises:
        ValueError: 로봇 수가 목적지 수보다 많을 경우
        RuntimeError: 로봇 배치에 실패한 경우
    """
    # min_distance 기본값 설정
    if min_distance is None:
        min_distance = MIN_ROBOT_DISTANCE

    # 로봇 수 검증
    if num_robots > len(GOAL_POSITIONS):
        raise ValueError(
            f"로봇 수({num_robots})가 목적지 수({len(GOAL_POSITIONS)})보다 많습니다. "
            f"로봇 수는 최대 {len(GOAL_POSITIONS)}개까지 가능합니다."
        )
    
    # 목적지를 랜덤하게 섞어서 각 로봇에게 할당
    available_goals = GOAL_POSITIONS.copy()
    random.shuffle(available_goals)
    assigned_goals = available_goals[:num_robots]
    
    occupied_cells = set()
    robot_configs = []
    
    max_attempts = 1000  # 무한 루프 방지
    
    for robot_id in range(num_robots):
        placed = False
        
        for _ in range(max_attempts):
            # 랜덤 방향 선택 (0=동, 1=남, 2=서, 3=북)
            direction = random.randint(0, 3)
            dx, dy = DIRECTIONS[direction]
            
            # 랜덤 head 위치 선택
            head_x = random.randint(GRID_MIN, GRID_MAX)
            head_y = random.randint(GRID_MIN, GRID_MAX)
            head = (head_x, head_y)
            
            # tail 위치 계산 (head 반대 방향)
            tail_x = head_x - dx
            tail_y = head_y - dy
            tail = (tail_x, tail_y)
            
            # tail이 격자 범위 내에 있는지 확인
            if not (GRID_MIN <= tail_x <= GRID_MAX and GRID_MIN <= tail_y <= GRID_MAX):
                continue
            
            # head와 tail 모두 비어있는지 확인
            if head not in occupied_cells and tail not in occupied_cells:
                # min_distance 체크 (다른 로봇들과의 거리)
                if min_distance > 0:
                    too_close = False
                    for existing_config in robot_configs:
                        # 다른 로봇의 head, tail과의 거리 계산
                        dist_to_head = abs(head[0] - existing_config["head"][0]) + abs(head[1] - existing_config["head"][1])
                        dist_to_tail = abs(head[0] - existing_config["tail"][0]) + abs(head[1] - existing_config["tail"][1])
                        dist_from_tail = abs(tail[0] - existing_config["head"][0]) + abs(tail[1] - existing_config["head"][1])

                        # 모든 부분 간 최소 거리 체크
                        if min(dist_to_head, dist_to_tail, dist_from_tail) < min_distance:
                            too_close = True
                            break

                    if too_close:
                        continue  # 다시 시도

                # 배치 성공
                occupied_cells.add(head)
                occupied_cells.add(tail)

                robot_configs.append({
                    "id": robot_id,
                    "head": head,
                    "tail": tail,
                    "dir": direction,
                    "goal": assigned_goals[robot_id]  # 목적지 할당
                })

                placed = True
                break
        
        if not placed:
            raise RuntimeError(
                f"로봇 {robot_id}를 배치할 수 없습니다. "
                f"격자 크기({GRID_SIZE}x{GRID_SIZE})에 비해 로봇 수({num_robots})가 너무 많을 수 있습니다."
            )
    
    return robot_configs


def get_initial_robot_configs():
    """
    매 시뮬레이션마다 새로운 랜덤 로봇 배치를 생성합니다.
    
    Returns:
        list: 로봇 설정 리스트
    """
    return generate_random_robot_configs(NUM_ROBOTS)


# ========================================
# 시뮬레이션 설정
# ========================================

SIMULATION_TERMINATION_TIME = 500  # 시뮬레이션 최대 실행 시간 (초)
