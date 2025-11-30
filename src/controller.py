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
    ACTION_STOP,
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
        self.last_actions = {}       # 이전 스텝의 행동 {robot_id: action_idx} - 회전 반복 방지용

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
        self.step_experiences = []  # [(state, action, reward, next_state, done), ...] - DQN용
        self.current_rewards = {}   # 현재 스텝의 보상

        # PPO/MAPPO용 trajectory 데이터
        self.ppo_trajectory = {
            'states': [],
            'global_states': [],  # MAPPO: 전역 상태
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }

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
            # 이전 스텝의 행동 저장 (회전 반복 방지용)
            self.state.last_actions = self.state.current_actions.copy()
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
        행동 선택 정책 - PPO 에이전트 연동 + 정책 마스킹

        정책 (PPO용 확률 조정):
        1. 센서 범위에 다른 로봇 탐지 시 충돌 회피
        2. head가 센터(0,0)로 향하는 회전 우대
        3. 목표 방향 기준 회전 확률 조정
        4. 격자 밖으로 나가는 forward 금지
        5. 조건부 행동: 위험 시 회전, 목표 방향 있으면 전진, 아니면 STOP

        Args:
            rid: 로봇 ID
            obs: 관찰 데이터 딕셔너리

        Returns:
            tuple: (action_dict, action_idx)
                - action_dict: {"type": action_type} - 로봇에 전송할 행동
                - action_idx: int (0-3) - 학습에 사용할 행동 인덱스
                    0: MOVE, 1: ROTATE_CW, 2: ROTATE_CCW, 3: STOP
        """
        action_types = [ACTION_MOVE, ACTION_ROTATE_CW, ACTION_ROTATE_CCW, ACTION_STOP]

        if self.rl_agent is not None and hasattr(self.rl_agent, 'select_action'):
            # PPO/MAPPO 에이전트 연동
            state = self._observation_to_state(obs)
            action_mask = self._compute_action_mask(obs, rid=rid)

            # MAPPO: global state 생성 (모든 로봇의 관찰값 사용)
            global_state = self._observation_to_global_state()

            # MAPPO 에이전트인지 확인 (global_state_dim 속성 존재 여부)
            if hasattr(self.rl_agent, 'global_state_dim'):
                # MAPPO: global state 전달
                action_idx, log_prob, value = self.rl_agent.select_action(state, global_state, action_mask)
            else:
                # PPO: local state만 사용
                action_idx, log_prob, value = self.rl_agent.select_action(state, action_mask)

            # 현재 스텝의 정보 저장 (학습용)
            if not hasattr(self, 'step_data'):
                self.step_data = {}
            self.step_data[rid] = {
                'state': state,
                'global_state': global_state,
                'action': action_idx,
                'log_prob': log_prob,
                'value': value
            }

            return ({"type": action_types[action_idx]}, action_idx)
        elif self.rl_agent is not None:
            # DQN 에이전트 (기존 방식)
            state = self._observation_to_state(obs)
            action_idx = self.rl_agent.get_action(state, training=True)

            # 기본 충돌 회피
            own_head = obs["own_head"]
            own_direction = obs["own_direction"]
            detected_robots = obs["detected_robots"]

            from config import DIRECTIONS
            direction_vec = DIRECTIONS[own_direction]
            front_position = (own_head[0] + direction_vec[0], own_head[1] + direction_vec[1])

            robot_in_front = False
            for robot in detected_robots:
                if robot["head"] == front_position or robot["tail"] == front_position:
                    robot_in_front = True
                    break

            if robot_in_front and action_idx == 0:
                action_idx = self._select_smart_rotation(obs)

            return ({"type": action_types[action_idx]}, action_idx)

        # 기본 휴리스틱 정책 (RL 없을 때)
        if robot_in_front:
            # 앞에 로봇이 있으면 회전만 (스마트 회전)
            action_idx = self._select_smart_rotation(obs)
        else:
            # 앞이 비어있으면 전진 우선
            goal_pos = obs["goal_position"]
            tail_pos = obs["own_tail"]
            distance = abs(goal_pos[0] - tail_pos[0]) + abs(goal_pos[1] - tail_pos[1])

            if distance > 2:
                if random.random() < 0.7:
                    action_idx = 0  # MOVE
                else:
                    action_idx = self._select_smart_rotation(obs)
            else:
                # 목표 근처에서는 더 신중하게
                action_idx = random.choice([0, 1, 2])

        return ({"type": action_types[action_idx]}, action_idx)

    def _compute_action_mask(self, obs, rid=None):
        """
        PPO용 행동 확률 마스크 계산

        사용자 정책:
        1. 센서 범위에 다른 로봇 탐지 시:
           - head 앞쪽에 있으면: 전진 확률 낮추고, 회전은 탐지 로봇으로부터 먼 쪽으로
           - tail 쪽에 있으면: 전진 허용
        2. head가 센터(0,0)로 향하는 회전 우대
        3. 목표 방향 기준 회전 확률 조정
        4. 격자 밖으로 나가는 forward 금지
        5. 조건부 행동: 위험 시 회전 우대, 목표 방향 있으면 전진, 아니면 STOP
        6. 목표 근처에서 회전 반복 방지

        Args:
            obs: 관찰 데이터
            rid: 로봇 ID (최근 행동 확인용)

        Returns:
            list: [forward_prob, cw_prob, ccw_prob, stop_prob] - 각 행동의 확률 가중치
        """
        from config import DIRECTIONS, GRID_SIZE

        own_head = obs["own_head"]
        own_tail = obs["own_tail"]
        own_direction = obs["own_direction"]
        detected_robots = obs["detected_robots"]
        goal_position = obs["goal_position"]

        # 기본 확률 (모두 동일)
        forward_prob = 1.0
        cw_prob = 1.0
        ccw_prob = 1.0
        stop_prob = 1.0

        # 현재 방향 벡터
        direction_vec = DIRECTIONS[own_direction]

        # 정책 4: 격자 밖으로 나가는 forward 금지
        next_head = (own_head[0] + direction_vec[0], own_head[1] + direction_vec[1])
        if not (0 <= next_head[0] < GRID_SIZE and 0 <= next_head[1] < GRID_SIZE):
            forward_prob = 0.0  # 완전 금지

        # 정책 1: 센서 범위에 다른 로봇 탐지 시 충돌 회피
        if detected_robots:
            front_position = next_head

            # 각 방향별 로봇 위치 분석
            robot_in_front = False
            robot_in_back = False
            robot_positions = []

            for robot in detected_robots:
                robot_positions.extend([robot["head"], robot["tail"]])

                # 정면에 로봇이 있는지
                if robot["head"] == front_position or robot["tail"] == front_position:
                    robot_in_front = True

                # 후방에 로봇이 있는지 (tail 방향)
                back_vec = (-direction_vec[0], -direction_vec[1])
                back_position = (own_head[0] + back_vec[0], own_head[1] + back_vec[1])
                if robot["head"] == back_position or robot["tail"] == back_position:
                    robot_in_back = True

            # head 앞쪽에 로봇이 있으면 전진 확률 대폭 감소, 회전 확률 증가
            if robot_in_front:
                forward_prob *= 0.1  # 전진 확률 90% 감소

                # 회전 방향 선택: 탐지된 로봇으로부터 먼 방향으로
                # 시계 방향으로 회전했을 때와 반시계 방향으로 회전했을 때를 계산
                cw_direction_idx = (own_direction + 1) % 4
                ccw_direction_idx = (own_direction - 1) % 4

                cw_vec = DIRECTIONS[cw_direction_idx]
                ccw_vec = DIRECTIONS[ccw_direction_idx]

                cw_next = (own_head[0] + cw_vec[0], own_head[1] + cw_vec[1])
                ccw_next = (own_head[0] + ccw_vec[0], own_head[1] + ccw_vec[1])

                # 각 회전 방향에 로봇이 있는지 확인
                cw_blocked = cw_next in robot_positions
                ccw_blocked = ccw_next in robot_positions

                if cw_blocked and not ccw_blocked:
                    cw_prob *= 0.3  # CW 쪽에 로봇 있으면 CCW 우대
                    ccw_prob *= 2.0
                elif ccw_blocked and not cw_blocked:
                    ccw_prob *= 0.3  # CCW 쪽에 로봇 있으면 CW 우대
                    cw_prob *= 2.0
                else:
                    # 둘 다 막혔거나 둘 다 비었으면 회전 확률 증가
                    cw_prob *= 1.5
                    ccw_prob *= 1.5

            # tail 쪽에만 로봇이 있으면 전진 허용 (오히려 권장)
            if robot_in_back and not robot_in_front:
                forward_prob *= 1.3  # 전진 확률 증가

        # 정책 2: head가 센터(0,0)로 향하는 회전 우대
        center = (0, 0)
        to_center_vec = (center[0] - own_head[0], center[1] - own_head[1])

        if to_center_vec != (0, 0):  # 이미 센터에 있지 않은 경우
            # 외적으로 센터로 향하는 회전 방향 계산
            cross = direction_vec[0] * to_center_vec[1] - direction_vec[1] * to_center_vec[0]

            if cross > 0:
                # 센터가 왼쪽 → CCW가 센터로 향함
                ccw_prob *= 1.5
            elif cross < 0:
                # 센터가 오른쪽 → CW가 센터로 향함
                cw_prob *= 1.5

        # 정책 3: 목표 방향 기준 회전 확률 조정
        to_goal_vec = (goal_position[0] - own_head[0], goal_position[1] - own_head[1])

        if to_goal_vec != (0, 0):
            # 현재 방향과 목표 방향의 외적
            cross_goal = direction_vec[0] * to_goal_vec[1] - direction_vec[1] * to_goal_vec[0]

            if cross_goal > 0:
                # 목표가 왼쪽 → CCW 우대
                ccw_prob *= 1.8
            elif cross_goal < 0:
                # 목표가 오른쪽 → CW 우대
                cw_prob *= 1.8

            # 목표와 현재 방향이 얼마나 일치하는지 (내적)
            dot = direction_vec[0] * to_goal_vec[0] + direction_vec[1] * to_goal_vec[1]
            if dot > 0:
                # 목표 방향으로 향하고 있으면 전진 우대
                forward_prob *= 1.5

        # 정책 5: 조건부 행동 로직
        # 목표까지 거리 계산
        distance_to_goal = abs(to_goal_vec[0]) + abs(to_goal_vec[1])

        # 위험하거나 탐지된 로봇이 있으면 → 회전 우대, STOP 억제
        if detected_robots or forward_prob < 0.5:
            cw_prob *= 1.5
            ccw_prob *= 1.5
            stop_prob *= 0.3
        # 목표 방향이 정해져 있고 거리가 멀면 → 전진 우대, STOP 억제
        elif distance_to_goal > 1:
            forward_prob *= 1.3
            stop_prob *= 0.5
        # 거리가 매우 가까우면 → STOP 우대 (목표 주변에서 대기)
        else:
            stop_prob *= 2.0
            forward_prob *= 0.7

        # 정책 6: 목표 근처에서 회전 반복 방지
        if distance_to_goal <= 2 and rid is not None:
            # 최근 행동이 회전이었는지 확인
            last_action = self.state.last_actions.get(rid, None)
            if last_action in [1, 2]:  # CW 또는 CCW
                # 회전 확률 감소, 전진/STOP 우대
                cw_prob *= 0.4
                ccw_prob *= 0.4
                forward_prob *= 1.5
                stop_prob *= 1.5

        # 정규화는 PPO 에이전트에서 수행
        return [forward_prob, cw_prob, ccw_prob, stop_prob]

    def _select_smart_rotation(self, obs):
        """
        목표 방향 기반 스마트 회전 선택

        Args:
            obs: 관찰 데이터

        Returns:
            int: 1 (시계방향) 또는 2 (반시계방향)
        """
        from config import DIRECTIONS

        own_head = obs["own_head"]
        own_direction = obs["own_direction"]
        goal_position = obs["goal_position"]

        # 목표 벡터 (뒷발이 가야할 곳)
        goal_vec = (goal_position[0] - own_head[0], goal_position[1] - own_head[1])

        # 현재 방향 벡터
        current_vec = DIRECTIONS[own_direction]

        # 외적으로 시계/반시계 판단
        # cross = current_x * goal_y - current_y * goal_x
        cross = current_vec[0] * goal_vec[1] - current_vec[1] * goal_vec[0]

        if cross > 0:
            # 목표가 왼쪽에 있음 → 반시계 회전
            return 2  # ACTION_ROTATE_CCW
        elif cross < 0:
            # 목표가 오른쪽에 있음 → 시계 회전
            return 1  # ACTION_ROTATE_CW
        else:
            # 정확히 같은 방향이거나 반대 방향 → 랜덤
            return random.choice([1, 2])

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

        # 목표까지 벡터 계산 (앞발은 (0,0), 뒷발은 목표 위치)
        vector_to_goal_head = (0 - own_head[0], 0 - own_head[1])  # 앞발은 항상 (0,0)
        vector_to_goal_tail = (goal_position[0] - own_tail[0], goal_position[1] - own_tail[1])

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

    def _observation_to_global_state(self):
        """
        모든 로봇의 관찰 데이터를 합쳐 전역 상태 표현으로 변환 (MAPPO용)

        Returns:
            강화학습 전역 상태 표현 (numpy array)
            차원: num_robots * 13 (각 로봇의 local state를 연결)
        """
        import numpy as np

        global_state_list = []

        # 모든 로봇의 관찰값을 순서대로 추가
        for rid in range(self.num_robots):
            if rid in self.state.observations:
                # 각 로봇의 local state 생성
                local_state = self._observation_to_state(self.state.observations[rid])
                global_state_list.append(local_state)
            else:
                # 관찰이 없으면 0으로 채움
                global_state_list.append(np.zeros(13, dtype=np.float32))

        # 모든 local state를 연결하여 global state 생성
        global_state = np.concatenate(global_state_list, axis=0)

        return global_state

    def get_step_experiences(self):
        """
        현재까지 수집된 스텝 경험 데이터를 반환하고 초기화 (DQN용)

        Returns:
            list: [(state, action, reward, next_state, done), ...]
        """
        experiences = self.step_experiences.copy()
        self.step_experiences = []
        return experiences

    def get_ppo_trajectory(self):
        """
        PPO/MAPPO trajectory 데이터 가져오기 (Trainer에서 호출)

        Returns:
            dict: {'states': [...], 'global_states': [...], 'actions': [...],
                   'log_probs': [...], 'values': [...], 'rewards': [...], 'dones': [...]}
        """
        trajectory = {
            'states': self.ppo_trajectory['states'].copy(),
            'global_states': self.ppo_trajectory['global_states'].copy(),
            'actions': self.ppo_trajectory['actions'].copy(),
            'log_probs': self.ppo_trajectory['log_probs'].copy(),
            'values': self.ppo_trajectory['values'].copy(),
            'rewards': self.ppo_trajectory['rewards'].copy(),
            'dones': self.ppo_trajectory['dones'].copy()
        }
        # 초기화
        self.ppo_trajectory = {
            'states': [],
            'global_states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
        return trajectory

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

            # DQN용 경험 저장
            self.step_experiences.append((prev_state, action, reward, next_state, done))

            # PPO/MAPPO용 trajectory 저장
            if hasattr(self, 'step_data') and rid in self.step_data:
                step_info = self.step_data[rid]
                self.ppo_trajectory['states'].append(step_info['state'])
                self.ppo_trajectory['global_states'].append(step_info['global_state'])
                self.ppo_trajectory['actions'].append(step_info['action'])
                self.ppo_trajectory['log_probs'].append(step_info['log_prob'])
                self.ppo_trajectory['values'].append(step_info['value'])
                self.ppo_trajectory['rewards'].append(reward)
                self.ppo_trajectory['dones'].append(done)

