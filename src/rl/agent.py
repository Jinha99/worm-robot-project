"""
Worm Robot Simulation - RL Agent
강화학습 에이전트 구현 (템플릿)
"""


class RLAgent:
    """
    강화학습 에이전트 기본 클래스

    향후 DQN, PPO, A3C 등 다양한 알고리즘 구현 가능
    """

    def __init__(self, state_dim, action_dim, **kwargs):
        """
        Args:
            state_dim: 상태 공간 차원
            action_dim: 행동 공간 차원
            **kwargs: 추가 하이퍼파라미터
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_action(self, state, training=True):
        """
        주어진 상태에서 행동 선택

        Args:
            state: 현재 상태
            training: 학습 모드 여부 (탐색 vs 활용)

        Returns:
            선택된 행동
        """
        raise NotImplementedError("get_action() must be implemented")

    def update(self, state, action, reward, next_state, done):
        """
        에이전트 업데이트 (학습)

        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        raise NotImplementedError("update() must be implemented")

    def save(self, path):
        """모델 저장"""
        raise NotImplementedError("save() must be implemented")

    def load(self, path):
        """모델 로드"""
        raise NotImplementedError("load() must be implemented")


# TODO: DQN, PPO 등 구체적인 알고리즘 구현
# class DQNAgent(RLAgent):
#     def __init__(self, state_dim, action_dim, **kwargs):
#         super().__init__(state_dim, action_dim, **kwargs)
#         # DQN 네트워크 초기화
#         ...
