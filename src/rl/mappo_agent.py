"""
MAPPO (Multi-Agent PPO) Agent for Worm Robot

중앙집중식 Critic (Centralized Critic):
- 모든 에이전트의 상태를 입력으로 받아 가치 함수 학습
- 학습 시에만 사용, 실행 시에는 불필요

분산식 Actor (Decentralized Actor):
- 각 에이전트는 자신의 관찰값만 사용하여 행동 선택
- 실행 시 사용
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os


class DecentralizedActor(nn.Module):
    """분산식 정책 네트워크 (각 에이전트의 로컬 관찰만 사용)"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DecentralizedActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Args:
            state: 개별 에이전트의 로컬 관찰 (state_dim,)

        Returns:
            action_probs: 각 행동의 확률 분포
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class CentralizedCritic(nn.Module):
    """중앙집중식 가치 네트워크 (모든 에이전트의 상태 사용)"""

    def __init__(self, global_state_dim, hidden_dim=128):
        super(CentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, global_state):
        """
        Args:
            global_state: 모든 에이전트의 상태를 합친 전역 상태

        Returns:
            value: 전역 상태 가치 V(s_global)
        """
        x = F.relu(self.fc1(global_state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPOAgent:
    """MAPPO 에이전트"""

    def __init__(
        self,
        state_dim,
        global_state_dim,
        action_dim,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dim=128,
        device="cpu"
    ):
        """
        Args:
            state_dim: 개별 에이전트의 로컬 상태 차원
            global_state_dim: 전역 상태 차원 (모든 에이전트 정보 포함)
            action_dim: 행동 차원
            learning_rate: 학습률
            gamma: 할인율
            gae_lambda: GAE lambda 파라미터
            clip_epsilon: PPO clip 범위
            value_coef: 가치 손실 계수
            entropy_coef: 엔트로피 보너스 계수
            max_grad_norm: 그래디언트 클리핑 최대값
            hidden_dim: 은닉층 차원
            device: 디바이스 (cpu/cuda)
        """
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # 네트워크 초기화
        self.actor = DecentralizedActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CentralizedCritic(global_state_dim, hidden_dim).to(self.device)

        # 옵티마이저
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

    def select_action(self, state, global_state, action_mask=None):
        """
        정책에 따라 행동 선택

        Args:
            state: 현재 로컬 상태 (numpy array)
            global_state: 현재 전역 상태 (numpy array)
            action_mask: 행동 마스크 (None이면 모든 행동 허용)

        Returns:
            action: 선택된 행동
            log_prob: 행동의 로그 확률
            value: 전역 상태 가치
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(global_state_tensor)

            # 행동 마스크 적용
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                action_probs = action_probs * mask_tensor
                # 재정규화
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            # 확률 분포에서 샘플링
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states, global_states, actions):
        """
        주어진 상태-행동 쌍에 대한 평가

        Args:
            states: 로컬 상태 배치
            global_states: 전역 상태 배치
            actions: 행동 배치

        Returns:
            log_probs: 행동의 로그 확률
            values: 전역 상태 가치
            entropy: 정책 엔트로피
        """
        action_probs = self.actor(states)
        values = self.critic(global_states)

        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def compute_gae(self, rewards, values, dones, next_value):
        """
        GAE (Generalized Advantage Estimation) 계산

        Args:
            rewards: 보상 리스트
            values: 가치 리스트
            dones: 종료 플래그 리스트
            next_value: 다음 상태 가치

        Returns:
            advantages: 어드밴티지
            returns: 리턴 (타겟 가치)
        """
        advantages = []
        gae = 0

        # 역순으로 계산
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]

            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]

            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)

        return advantages, returns

    def update(self, states, global_states, actions, old_log_probs, returns, advantages, epochs=4, batch_size=64):
        """
        MAPPO 업데이트

        Args:
            states: 로컬 상태 배치
            global_states: 전역 상태 배치
            actions: 행동 배치
            old_log_probs: 이전 정책의 로그 확률
            returns: 리턴 (타겟 가치)
            advantages: 어드밴티지
            epochs: 업데이트 에폭 수
            batch_size: 미니배치 크기

        Returns:
            mean_loss: 평균 손실
        """
        states = torch.FloatTensor(states).to(self.device)
        global_states = torch.FloatTensor(global_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # 어드밴티지 정규화 (std가 0이 아닐 때만)
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        total_loss = 0
        num_updates = 0

        # 여러 에폭 반복
        for _ in range(epochs):
            # 미니배치로 나누어 학습
            for i in range(0, len(states), batch_size):
                batch_indices = slice(i, min(i + batch_size, len(states)))

                batch_states = states[batch_indices]
                batch_global_states = global_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 현재 정책 평가
                log_probs, values, entropy = self.evaluate_actions(
                    batch_states, batch_global_states, batch_actions
                )

                # PPO clip 손실
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 가치 손실 (중앙집중식 Critic 사용)
                value_loss = F.mse_loss(values, batch_returns)

                # 엔트로피 보너스 (탐험 장려)
                entropy_loss = -entropy.mean()

                # 전체 손실
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        mean_loss = total_loss / num_updates if num_updates > 0 else 0
        return mean_loss

    def save(self, path):
        """모델 저장"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else "models", exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"💾 모델 저장 완료: {path}")

    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"📂 모델 로드 완료: {path}")
