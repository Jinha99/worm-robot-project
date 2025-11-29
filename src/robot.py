"""
Worm Robot Simulation - Robot Model
로봇 DEVS 모델 정의
"""

from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import (
    DIR_NAMES,
    DIRECTIONS,
    ACTION_MOVE,
    ACTION_ROTATE_CW,
    ACTION_ROTATE_CCW,
    ACTION_STOP,
    ACTION_TIMES,
)
from utils import add_pos, in_bounds


# ========================================
# Robot 상태 클래스
# ========================================

class RobotState:
    """로봇의 내부 상태를 표현하는 클래스"""

    def __init__(self, robot_id, head_pos, tail_pos, direction):
        """
        Args:
            robot_id: 로봇 고유 ID
            head_pos: 앞발 위치 (x, y)
            tail_pos: 뒷발 위치 (x, y)
            direction: 방향 (0-3)
        """
        self.robot_id = robot_id
        self.head_pos = head_pos      # 앞발 위치
        self.tail_pos = tail_pos      # 뒷발 위치
        self.direction = direction    # 방향 (0-3)
        self.phase = "IDLE"           # 상태: IDLE, EXECUTING
        self.current_action = None    # 현재 실행 중인 행동

    def __str__(self):
        return (
            f"Robot{self.robot_id}["
            f"상태:{self.phase},"
            f"앞발:{self.head_pos},"
            f"뒷발:{self.tail_pos},"
            f"방향:{DIR_NAMES[self.direction]}]"
        )


# ========================================
# Robot 모델 (Atomic DEVS)
# ========================================

class Robot(AtomicDEVS):
    """2관절 로봇의 DEVS 모델"""

    def __init__(self, robot_id, initial_head, initial_tail, initial_direction):
        """
        Args:
            robot_id: 로봇 고유 ID
            initial_head: 초기 앞발 위치
            initial_tail: 초기 뒷발 위치
            initial_direction: 초기 방향
        """
        AtomicDEVS.__init__(self, f"Robot{robot_id}")
        self.robot_id = robot_id
        self.state = RobotState(robot_id, initial_head, initial_tail, initial_direction)

        # 포트 정의
        self.action_in = self.addInPort("action_in")           # 입력: 행동 명령
        self.action_done_out = self.addOutPort("action_done_out")  # 출력: 행동 완료 신호

    def timeAdvance(self):
        """시간 진행 함수"""
        if self.state.phase == "IDLE":
            return INFINITY  # 명령 대기 중
        elif self.state.phase == "EXECUTING":
            # 현재 행동의 소요 시간 반환
            return ACTION_TIMES[self.state.current_action]
        else:
            return INFINITY

    def intTransition(self):
        """내부 전이 함수 - 행동 완료"""
        if self.state.phase == "EXECUTING":
            # 행동 완료 -> IDLE로 전환
            self.state.phase = "IDLE"
            self.state.current_action = None
        return self.state

    def extTransition(self, inputs):
        """외부 전이 함수 - 행동 명령 수신 (격자 이탈 방지)"""
        action = inputs.get(self.action_in)

        if action and self.state.phase == "IDLE":
            action_type = action["type"]

            if action_type == ACTION_MOVE:
                # 전진: 앞발이 방향으로 1칸 이동, 뒷발은 앞발의 이전 위치로
                direction_vec = DIRECTIONS[self.state.direction]
                new_head = add_pos(self.state.head_pos, direction_vec)
                new_tail = self.state.head_pos

                # 격자 범위 체크: 새로운 위치가 모두 범위 내에 있어야 함
                if in_bounds(new_head) and in_bounds(new_tail):
                    self.state.head_pos = new_head
                    self.state.tail_pos = new_tail
                # else: 격자를 벗어나면 위치 변경 없음 (제자리 유지)

            elif action_type == ACTION_ROTATE_CW:
                # 시계방향 회전: 뒷발 고정, 앞발만 회전
                new_direction = (self.state.direction + 1) % 4
                direction_vec = DIRECTIONS[new_direction]
                new_head = add_pos(self.state.tail_pos, direction_vec)

                # 격자 범위 체크: 새로운 앞발 위치가 범위 내에 있어야 함
                if in_bounds(new_head):
                    self.state.direction = new_direction
                    self.state.head_pos = new_head
                # else: 격자를 벗어나면 회전하지 않음

            elif action_type == ACTION_ROTATE_CCW:
                # 반시계방향 회전: 뒷발 고정, 앞발만 회전
                new_direction = (self.state.direction - 1) % 4
                direction_vec = DIRECTIONS[new_direction]
                new_head = add_pos(self.state.tail_pos, direction_vec)

                # 격자 범위 체크: 새로운 앞발 위치가 범위 내에 있어야 함
                if in_bounds(new_head):
                    self.state.direction = new_direction
                    self.state.head_pos = new_head
                # else: 격자를 벗어나면 회전하지 않음

            elif action_type == ACTION_STOP:
                # 정지: 위치 변경 없음 (대기)
                pass

            # EXECUTING 상태로 전환 (행동이 유효하든 아니든 시간은 소모)
            self.state.phase = "EXECUTING"
            self.state.current_action = action_type

        return self.state

    def outputFnc(self):
        """출력 함수 - 행동 완료 신호 전송"""
        if self.state.phase == "EXECUTING":
            return {
                self.action_done_out: {
                    "robot_id": self.robot_id,
                    "head": self.state.head_pos,
                    "tail": self.state.tail_pos,
                    "direction": self.state.direction,
                    "action": self.state.current_action
                }
            }
        return {}
