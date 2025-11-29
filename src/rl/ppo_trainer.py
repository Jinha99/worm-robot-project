"""
PPO Trainer for Worm Robot

Trajectory 수집 및 PPO 업데이트 관리
"""

import os
from pypdevs.simulator import Simulator

from config import STATUS_WIN, STATUS_PARTIAL_WIN, STATUS_FAIL, STATUS_RUNNING

# TensorBoard 지원 (선택)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard를 사용하려면 설치하세요: pip3 install tensorboard")


class PPOTrainer:
    """
    PPO 학습 루프를 관리하는 클래스

    에피소드 단위로 trajectory 수집 후 PPO 업데이트
    커리큘럼 학습 지원
    """

    def __init__(
        self,
        agent,
        create_system_fn,
        num_episodes=1000,
        termination_time=100,
        update_epochs=4,
        batch_size=64,
        log_interval=10,
        save_interval=100,
        model_path="models/ppo_worm_robot.pth",
        use_tensorboard=True,
        tensorboard_dir="runs/worm_robot_ppo",
        curriculum_stages=None,
        progression_threshold=0.7,
        progression_window=100
    ):
        """
        Args:
            agent: PPO 에이전트
            create_system_fn: WormRobotSystem을 생성하는 함수
            num_episodes: 학습 에피소드 수
            termination_time: 시뮬레이션 최대 시간 (초)
            update_epochs: PPO 업데이트 에폭 수
            batch_size: 미니배치 크기
            log_interval: 로그 출력 간격
            save_interval: 모델 저장 간격
            model_path: 모델 저장 경로
            use_tensorboard: TensorBoard 사용 여부
            tensorboard_dir: TensorBoard 로그 디렉토리
            curriculum_stages: 커리큘럼 학습 단계 리스트
            progression_threshold: 다음 단계로 진행하기 위한 성공률 임계값
            progression_window: 성공률 계산에 사용할 최근 에피소드 수
        """
        self.agent = agent
        self.create_system_fn = create_system_fn
        self.num_episodes = num_episodes
        self.termination_time = termination_time
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.model_path = model_path

        # 커리큘럼 학습 설정
        self.curriculum_stages = curriculum_stages
        self.progression_threshold = progression_threshold
        self.progression_window = progression_window
        self.current_stage_idx = 0
        self.stage_start_episode = 0

        # TensorBoard
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(tensorboard_dir)
            print(f"📊 TensorBoard 로깅 활성화: {tensorboard_dir}")
            print(f"   실행: tensorboard --logdir=runs")

        # 통계
        self.stats = {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_losses": [],
            "episode_results": [],  # 각 에피소드 결과
            "success_count": 0,
            "partial_success_count": 0,
            "fail_count": 0,
            "timeout_count": 0
        }

    def train(self):
        """학습 루프 실행 (커리큘럼 학습 지원)"""
        print("=" * 60)
        print("PPO 학습 시작")
        print("=" * 60)
        print(f"에피소드 수: {self.num_episodes}")
        print(f"시뮬레이션 시간: {self.termination_time}초")
        print(f"PPO 업데이트 에폭: {self.update_epochs}")
        print(f"배치 크기: {self.batch_size}")

        # 커리큘럼 학습 정보 출력
        if self.curriculum_stages:
            print(f"\n📚 커리큘럼 학습 활성화:")
            for i, stage in enumerate(self.curriculum_stages):
                print(f"   {i+1}. {stage['name']}: {stage['num_robots']}개 로봇, 최소거리 {stage['min_distance']}")
            print(f"   진행 조건: 성공률 {self.progression_threshold*100:.0f}% (최근 {self.progression_window} 에피소드)")

            # 첫 번째 단계 설정
            import config
            first_stage = self.curriculum_stages[0]
            config.NUM_ROBOTS = first_stage["num_robots"]
            config.MIN_ROBOT_DISTANCE = first_stage["min_distance"]
            print(f"\n🚀 시작 단계: {first_stage['name']}")

        print("=" * 60)

        for episode in range(self.num_episodes):
            # 에피소드 실행 및 trajectory 수집
            episode_reward, episode_steps, episode_status, trajectory = self._run_episode()

            # 통계 업데이트
            self.stats["episode_rewards"].append(episode_reward)
            self.stats["episode_steps"].append(episode_steps)
            self.stats["episode_results"].append(episode_status)

            if episode_status == STATUS_WIN:
                self.stats["success_count"] += 1
            elif episode_status == STATUS_PARTIAL_WIN:
                self.stats["partial_success_count"] += 1
            elif episode_status == STATUS_FAIL:
                self.stats["fail_count"] += 1
            else:  # STATUS_RUNNING (시간 초과)
                self.stats["timeout_count"] += 1

            # PPO 업데이트
            if len(trajectory['states']) > 0:
                loss = self._update_policy(trajectory)
                self.stats["episode_losses"].append(loss)
            else:
                self.stats["episode_losses"].append(0.0)

            # 커리큘럼 단계 진행 체크
            if self._check_stage_progression(episode):
                self._progress_to_next_stage(episode)

            # TensorBoard 로깅
            if self.writer is not None:
                self.writer.add_scalar('Reward/episode', episode_reward, episode)
                self.writer.add_scalar('Steps/episode', episode_steps, episode)
                self.writer.add_scalar('Loss/episode', self.stats["episode_losses"][-1], episode)
                self.writer.add_scalar('Success/total', self.stats["success_count"], episode)
                self.writer.add_scalar('Success/partial', self.stats["partial_success_count"], episode)
                self.writer.add_scalar('Fail/total', self.stats["fail_count"], episode)
                self.writer.add_scalar('Timeout/total', self.stats["timeout_count"], episode)

                if episode_status == STATUS_WIN:
                    self.writer.add_scalar('Result/win', 1, episode)
                    self.writer.add_scalar('Result/partial', 0, episode)
                    self.writer.add_scalar('Result/fail', 0, episode)
                    self.writer.add_scalar('Result/timeout', 0, episode)
                elif episode_status == STATUS_PARTIAL_WIN:
                    self.writer.add_scalar('Result/win', 0, episode)
                    self.writer.add_scalar('Result/partial', 1, episode)
                    self.writer.add_scalar('Result/fail', 0, episode)
                    self.writer.add_scalar('Result/timeout', 0, episode)
                elif episode_status == STATUS_FAIL:
                    self.writer.add_scalar('Result/win', 0, episode)
                    self.writer.add_scalar('Result/partial', 0, episode)
                    self.writer.add_scalar('Result/fail', 1, episode)
                    self.writer.add_scalar('Result/timeout', 0, episode)
                else:
                    self.writer.add_scalar('Result/win', 0, episode)
                    self.writer.add_scalar('Result/partial', 0, episode)
                    self.writer.add_scalar('Result/fail', 0, episode)
                    self.writer.add_scalar('Result/timeout', 1, episode)

            # 로그 출력
            if (episode + 1) % self.log_interval == 0:
                recent = self.log_interval
                avg_reward = sum(self.stats["episode_rewards"][-recent:]) / recent
                avg_steps = sum(self.stats["episode_steps"][-recent:]) / recent
                avg_loss = sum(self.stats["episode_losses"][-recent:]) / recent

                print(
                    f"Ep {episode + 1:4d}/{self.num_episodes} | "
                    f"Reward: {avg_reward:6.1f} | "
                    f"Steps: {avg_steps:4.1f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"W: {self.stats['success_count']:3d} | "
                    f"P: {self.stats['partial_success_count']:3d} | "
                    f"F: {self.stats['fail_count']:3d} | "
                    f"T: {self.stats['timeout_count']:3d}"
                )

            # 모델 저장
            if (episode + 1) % self.save_interval == 0:
                self._save_model()

        # 최종 모델 저장
        self._save_model()

        # TensorBoard writer 종료
        if self.writer is not None:
            self.writer.close()
            print("\n📊 TensorBoard 로그 저장 완료")

        print("\n" + "=" * 60)
        print("학습 완료!")
        print(f"총 완전 성공: {self.stats['success_count']} ({self.stats['success_count'] / self.num_episodes * 100:.1f}%)")
        print(f"총 부분 성공: {self.stats['partial_success_count']} ({self.stats['partial_success_count'] / self.num_episodes * 100:.1f}%)")
        print(f"총 충돌 실패: {self.stats['fail_count']} ({self.stats['fail_count'] / self.num_episodes * 100:.1f}%)")
        print(f"총 시간 초과: {self.stats['timeout_count']} ({self.stats['timeout_count'] / self.num_episodes * 100:.1f}%)")
        combined_success = self.stats['success_count'] + self.stats['partial_success_count']
        print(f"전체 성공률: {combined_success / self.num_episodes * 100:.1f}%")
        print("=" * 60)

        return self.stats

    def _run_episode(self):
        """
        단일 에피소드 실행 및 trajectory 수집

        Returns:
            tuple: (total_reward, step_count, final_status, trajectory)
        """
        # 새로운 시스템 생성
        system = self.create_system_fn(rl_agent=self.agent)

        # trajectory 저장용
        trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }

        # 시뮬레이터 실행
        sim = Simulator(system)
        sim.setClassicDEVS()
        sim.setTerminationTime(self.termination_time)
        sim.simulate()

        # 최종 상태 수집
        final_status = system.environment.state.status
        step_count = system.environment.state.step_count

        # Controller에서 PPO trajectory 수집
        controller = system.controller
        trajectory = controller.get_ppo_trajectory()

        # 총 보상 계산
        total_reward = sum(trajectory['rewards']) if trajectory['rewards'] else 0.0

        return total_reward, step_count, final_status, trajectory

    def _update_policy(self, trajectory):
        """
        PPO 정책 업데이트

        Args:
            trajectory: 에피소드 trajectory

        Returns:
            float: 평균 손실
        """
        if len(trajectory['states']) == 0:
            return 0.0

        import torch

        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        old_log_probs = trajectory['log_probs']
        values = trajectory['values']

        # 마지막 상태의 가치 계산 (부트스트래핑용)
        if len(states) > 0:
            import numpy as np
            last_state = torch.FloatTensor(np.array(states[-1])).unsqueeze(0).to(self.agent.device)
            with torch.no_grad():
                next_value = self.agent.critic(last_state).item()
        else:
            next_value = 0.0

        # GAE 계산
        advantages, returns = self.agent.compute_gae(rewards, values, dones, next_value)

        # states를 numpy array로 변환 (경고 방지)
        import numpy as np
        states_array = np.array(states, dtype=np.float32)

        # PPO 업데이트
        loss = self.agent.update(
            states=states_array,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
            epochs=self.update_epochs,
            batch_size=self.batch_size
        )

        return loss

    def _save_model(self):
        """모델 저장"""
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else "models", exist_ok=True)
        self.agent.save(self.model_path)

    def _check_stage_progression(self, episode):
        """현재 단계에서 다음 단계로 진행할 준비가 되었는지 확인"""
        if not self.curriculum_stages or self.current_stage_idx >= len(self.curriculum_stages) - 1:
            return False

        stage_episodes = episode - self.stage_start_episode
        if stage_episodes < self.progression_window:
            return False

        recent_results = self.stats["episode_results"][-self.progression_window:]
        success_count = sum(1 for r in recent_results if r == STATUS_WIN)
        partial_success_count = sum(1 for r in recent_results if r == STATUS_PARTIAL_WIN)
        combined_success_rate = (success_count + partial_success_count * 0.5) / len(recent_results)

        if combined_success_rate >= self.progression_threshold:
            return True

        return False

    def _progress_to_next_stage(self, episode):
        """다음 커리큘럼 단계로 진행"""
        current_stage = self.curriculum_stages[self.current_stage_idx]
        stage_model_path = self.model_path.replace(".pth", f"_{current_stage['name']}.pth")
        os.makedirs(os.path.dirname(stage_model_path) if os.path.dirname(stage_model_path) else "outputs", exist_ok=True)
        self.agent.save(stage_model_path)

        print("\n" + "=" * 60)
        print(f"🎓 커리큘럼 진행: {current_stage['name']} 완료!")
        print(f"   성공률: {self.stats['success_count'] / len(self.stats['episode_rewards']) * 100:.1f}%")
        print(f"   모델 저장: {stage_model_path}")

        self.current_stage_idx += 1
        next_stage = self.curriculum_stages[self.current_stage_idx]
        self.stage_start_episode = episode

        import config
        config.NUM_ROBOTS = next_stage["num_robots"]
        config.MIN_ROBOT_DISTANCE = next_stage["min_distance"]

        print(f"🚀 다음 단계 시작: {next_stage['name']}")
        print(f"   로봇 수: {next_stage['num_robots']}")
        print(f"   최소 거리: {next_stage['min_distance']}")
        print("=" * 60 + "\n")
