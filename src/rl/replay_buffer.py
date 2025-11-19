"""
Worm Robot Simulation - Replay Buffer
경험 재생 버퍼 구현 (템플릿)
"""

import random
from collections import deque


class ReplayBuffer:
    """
    경험 재생 버퍼 (Experience Replay Buffer)

    DQN 등의 알고리즘에서 사용되는 경험 저장소
    """

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: 버퍼 최대 크기
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        경험 추가

        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        배치 샘플링

        Args:
            batch_size: 샘플링할 경험의 개수

        Returns:
            (states, actions, rewards, next_states, dones) 튜플
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """버퍼 내 경험 개수 반환"""
        return len(self.buffer)

    def clear(self):
        """버퍼 초기화"""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    우선순위 경험 재생 버퍼 (Prioritized Experience Replay)

    중요한 경험을 더 자주 샘플링하는 고급 버퍼
    """

    def __init__(self, capacity=10000, alpha=0.6):
        """
        Args:
            capacity: 버퍼 최대 크기
            alpha: 우선순위 강도 (0: 균등 샘플링, 1: 완전 우선순위)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, priority=None):
        """
        경험 추가 (우선순위 포함)

        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
            priority: 우선순위 (None이면 최대 우선순위 사용)
        """
        super().add(state, action, reward, next_state, done)

        if priority is None:
            # 새 경험은 최대 우선순위로 설정
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_priority)
        else:
            self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        """
        우선순위 기반 배치 샘플링

        Args:
            batch_size: 샘플링할 경험의 개수
            beta: 중요도 샘플링 보정 강도

        Returns:
            (states, actions, rewards, next_states, dones, weights, indices) 튜플
        """
        # TODO: 우선순위 기반 샘플링 구현
        # 현재는 기본 샘플링 사용
        return super().sample(batch_size)

    def update_priorities(self, indices, priorities):
        """
        샘플링된 경험들의 우선순위 업데이트

        Args:
            indices: 업데이트할 경험의 인덱스 리스트
            priorities: 새로운 우선순위 리스트
        """
        # TODO: 우선순위 업데이트 구현
        pass
