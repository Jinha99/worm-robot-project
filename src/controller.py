"""
Worm Robot Simulation - Controller Model
컨트롤러 DEVS 모델 정의 (강화학습 연동 지점)
"""

import random
from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import (
    STATUS_RUNNING,
    ACTION_MOVE,
    ACTION_ROTATE_CW,
    ACTION_ROTATE_CCW,
)


# ========================================
# Controller 상태 클래스
# ========================================

class ControllerState:
    """컨트롤러의 내부 상태를 표현하는 클래스"""

    def __init__(self):
        self.observations = {}
        self.status = STATUS_RUNNING
        self.step = 0
        self.phase = "IDLE"  # 상태: IDLE, DECIDING

    def __str__(self):
        return (
            f"Controller["
            f"상태:{self.phase},"
            f"스텝:{self.step},"
            f"게임상태:{self.status}]"
        )


# ========================================
# Controller 모델 (Atomic DEVS)
# ========================================

class Controller(AtomicDEVS):
    """
    로봇들의 행동을 결정하는 컨트롤러 DEVS 모델

    강화학습 연동 지점:
    - _select_action() 메서드를 수정하여 RL 에이전트 통합 가능
    """

    def __init__(self, num_robots=4, rl_agent=None):
        """
        Args:
            num_robots: 로봇 수
            rl_agent: (선택) 강화학습 에이전트 인스턴스
        """
        AtomicDEVS.__init__(self, "Controller")
        self.num_robots = num_robots
        self.state = ControllerState()
        self.rl_agent = rl_agent  # 강화학습 에이전트 (None이면 휴리스틱 사용)

        # 입력 포트
        self.obs_in = self.addInPort("obs_in")          # 관찰 데이터
        self.status_in = self.addInPort("status_in")    # 게임 상태

        # 출력 포트 (로봇들로)
        self.action_out = [self.addOutPort(f"action{i}_out") for i in range(num_robots)]

    def timeAdvance(self):
        """시간 진행 함수"""
        if self.state.phase == "IDLE":
            return INFINITY  # 관찰 데이터 대기
        elif self.state.phase == "DECIDING":
            return 0  # 즉시 행동 결정
        return INFINITY

    def intTransition(self):
        """내부 전이 함수 - 행동 결정 완료"""
        if self.state.phase == "DECIDING":
            self.state.phase = "IDLE"
        return self.state

    def extTransition(self, inputs):
        """외부 전이 함수 - 관찰 데이터 수신"""
        # 관찰 데이터 수신
        obs = inputs.get(self.obs_in)
        if obs:
            self.state.observations = obs

        # 게임 상태 수신
        status = inputs.get(self.status_in)
        if status:
            self.state.status = status["status"]
            self.state.step = status["step"]

        # 게임이 진행 중이고 관찰 데이터가 있으면 결정 시작
        if self.state.observations and self.state.status == STATUS_RUNNING:
            self.state.phase = "DECIDING"

        return self.state

    def outputFnc(self):
        """출력 함수 - 각 로봇에 행동 명령 전송"""
        if self.state.phase == "DECIDING":
            actions = {}
            for rid in range(self.num_robots):
                if rid in self.state.observations:
                    action = self._select_action(rid, self.state.observations[rid])
                    actions[self.action_out[rid]] = action
            return actions
        return {}

    def _select_action(self, rid, obs):
        """
        행동 선택 정책 - 강화학습 연동 지점

        현재는 간단한 휴리스틱 사용:
        - 목표와 거리가 멀면 주로 전진
        - 목표와 가까우면 신중하게 회전도 고려

        Args:
            rid: 로봇 ID
            obs: 관찰 데이터 딕셔너리

        Returns:
            dict: {"type": action_type}

        강화학습 연동 예시:
        ----------------------
        if self.rl_agent is not None:
            # RL 에이전트를 사용하여 행동 선택
            state = self._observation_to_state(obs)
            action = self.rl_agent.get_action(state)
            return {"type": action}
        else:
            # 휴리스틱 사용 (아래 기본 정책)
            ...
        """
        if self.rl_agent is not None:
            # TODO: RL 에이전트 연동 구현
            # state = self._observation_to_state(obs)
            # action = self.rl_agent.get_action(state)
            # return {"type": action}
            pass

        # 기본 휴리스틱 정책
        distance = obs["distance_to_goal"]

        # 간단한 휴리스틱: 거리가 멀면 전진, 가까우면 다양한 행동
        if distance > 2:
            if random.random() < 0.7:
                return {"type": ACTION_MOVE}
            else:
                return {"type": random.choice([ACTION_ROTATE_CW, ACTION_ROTATE_CCW])}
        else:
            # 목표 근처에서는 더 신중하게
            return {"type": random.choice([ACTION_MOVE, ACTION_ROTATE_CW, ACTION_ROTATE_CCW])}

    def _observation_to_state(self, obs):
        """
        관찰 데이터를 RL 에이전트가 사용할 상태 표현으로 변환

        Args:
            obs: 관찰 데이터

        Returns:
            강화학습 상태 표현 (예: numpy array, dict 등)

        TODO: 강화학습 프레임워크에 맞게 구현
        """
        # 예시: 상태 벡터 구성
        # state = np.array([
        #     obs["own_head"][0], obs["own_head"][1],
        #     obs["own_tail"][0], obs["own_tail"][1],
        #     obs["own_direction"],
        #     obs["distance_to_goal"],
        #     len(obs["detected_robots"]),
        #     ...
        # ])
        # return state
        pass
