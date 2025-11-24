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
        self.prev_observations = {}  # 이전 스텝의 관찰 데이터
        self.current_actions = {}    # 현재 스텝에서 선택한 행동 {robot_id: action_idx}

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
        self.reward_in = self.addInPort("reward_in")    # 보상 데이터

        # 출력 포트 (로봇들로)
        self.action_out = [self.addOutPort(f"action{i}_out") for i in range(num_robots)]

        # 스텝별 경험 데이터 저장 (학습용)
        self.step_experiences = []  # [(state, action, reward, next_state, done), ...]
        self.current_rewards = {}   # 현재 스텝의 보상

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
        """외부 전이 함수 - 관찰 데이터 및 보상 수신"""
        # 보상 데이터 수신
        rewards = inputs.get(self.reward_in)
        if rewards:
            self.current_rewards = rewards

        # 관찰 데이터 수신
        obs = inputs.get(self.obs_in)
        if obs:
            # 경험 데이터 수집 (이전 관찰이 있고, 행동을 취한 경우)
            if self.state.prev_observations and self.state.current_actions:
                self._collect_experiences(self.current_rewards, self.state.status)

            # 이전 관찰 데이터 저장 (다음 스텝에서 사용)
            if self.state.observations:
                self.state.prev_observations = self.state.observations.copy()
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
            self.state.current_actions = {}  # 현재 스텝의 행동 초기화

            for rid in range(self.num_robots):
                if rid in self.state.observations:
                    action_dict, action_idx = self._select_action(rid, self.state.observations[rid])
                    actions[self.action_out[rid]] = action_dict
                    self.state.current_actions[rid] = action_idx  # 행동 인덱스 저장
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
            tuple: (action_dict, action_idx)
                - action_dict: {"type": action_type} - 로봇에 전송할 행동
                - action_idx: int (0-2) - 학습에 사용할 행동 인덱스

        강화학습 연동 예시:
        ----------------------
        if self.rl_agent is not None:
            # RL 에이전트를 사용하여 행동 선택
            state = self._observation_to_state(obs)
            action = self.rl_agent.get_action(state)
            return ({"type": action_types[action]}, action)
        else:
            # 휴리스틱 사용 (아래 기본 정책)
            ...
        """
        action_types = [ACTION_MOVE, ACTION_ROTATE_CW, ACTION_ROTATE_CCW]

        if self.rl_agent is not None:
            # RL 에이전트 연동
            state = self._observation_to_state(obs)
            action_idx = self.rl_agent.get_action(state, training=True)
            return ({"type": action_types[action_idx]}, action_idx)

        # 기본 휴리스틱 정책
        goal_pos = obs["goal_position"]
        tail_pos = obs["own_tail"]

        # 목적지까지의 맨해튼 거리 계산
        distance = abs(goal_pos[0] - tail_pos[0]) + abs(goal_pos[1] - tail_pos[1])

        # 간단한 휴리스틱: 거리가 멀면 전진, 가까우면 다양한 행동
        if distance > 2:
            if random.random() < 0.7:
                action_idx = 0  # MOVE
            else:
                action_idx = random.choice([1, 2])  # ROTATE_CW or ROTATE_CCW
        else:
            # 목표 근처에서는 더 신중하게
            action_idx = random.choice([0, 1, 2])

        return ({"type": action_types[action_idx]}, action_idx)

    def _observation_to_state(self, obs):
        """
        관찰 데이터를 RL 에이전트가 사용할 상태 표현으로 변환

        Args:
            obs: 관찰 데이터

        Returns:
            강화학습 상태 표현 (numpy array)
        """
        import numpy as np

        # 자신의 위치 (정규화: -3~3 → -1~1)
        own_head = obs["own_head"]
        own_tail = obs["own_tail"]

        # 목표 위치
        goal_position = obs["goal_position"]

        # 목표까지 벡터 계산
        vector_to_goal_head = (goal_position[0] - own_head[0], goal_position[1] - own_head[1])
        vector_to_goal_tail = (0 - own_tail[0], 0 - own_tail[1])  # 뒷발은 항상 (0,0)

        # 방향 (0~3)
        direction = obs["own_direction"]

        # 주변 로봇 정보 (간단하게: 개수와 가장 가까운 로봇까지 거리)
        detected = obs["detected_robots"]
        num_nearby = len(detected)

        closest_dist = 10.0  # 기본값 (멀리 있음)
        if detected:
            for robot in detected:
                dist = abs(robot["head"][0] - own_head[0]) + abs(robot["head"][1] - own_head[1])
                closest_dist = min(closest_dist, dist)

        # 상태 벡터 구성 (13차원)
        state = np.array([
            own_head[0] / 3.0,          # -1 ~ 1
            own_head[1] / 3.0,          # -1 ~ 1
            own_tail[0] / 3.0,          # -1 ~ 1
            own_tail[1] / 3.0,          # -1 ~ 1
            direction / 3.0,            # 0 ~ 1
            vector_to_goal_head[0] / 6.0,  # -1 ~ 1
            vector_to_goal_head[1] / 6.0,  # -1 ~ 1
            vector_to_goal_tail[0] / 6.0,  # -1 ~ 1
            vector_to_goal_tail[1] / 6.0,  # -1 ~ 1
            goal_position[0] / 3.0,     # -1 ~ 1
            goal_position[1] / 3.0,     # -1 ~ 1
            num_nearby / 3.0,           # 0 ~ 1
            closest_dist / 10.0         # 0 ~ 1
        ], dtype=np.float32)

        return state

    def get_step_experiences(self):
        """
        현재까지 수집된 스텝 경험 데이터를 반환하고 초기화

        Returns:
            list: [(state, action, reward, next_state, done), ...]
        """
        experiences = self.step_experiences.copy()
        self.step_experiences = []
        return experiences

    def _collect_experiences(self, rewards, env_status):
        """
        현재 스텝의 경험 데이터 수집 (내부 메서드)

        Args:
            rewards: {robot_id: reward} - 각 로봇의 보상
            env_status: 환경 상태 (STATUS_RUNNING, STATUS_WIN, etc.)
        """
        # 이전 관찰과 현재 관찰이 모두 있어야 함
        if not self.state.prev_observations or not self.state.observations:
            return

        # 각 로봇의 경험 데이터 수집
        for rid in self.state.current_actions.keys():
            if rid not in self.state.prev_observations or rid not in self.state.observations:
                continue

            # 이전 상태
            prev_state = self._observation_to_state(self.state.prev_observations[rid])

            # 행동
            action = self.state.current_actions[rid]

            # 보상
            reward = rewards.get(rid, 0.0)

            # 다음 상태
            next_state = self._observation_to_state(self.state.observations[rid])

            # 종료 여부
            done = (env_status != STATUS_RUNNING)

            # 경험 저장
            self.step_experiences.append((prev_state, action, reward, next_state, done))

