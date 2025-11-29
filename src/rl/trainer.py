"""
Worm Robot Simulation - RL Trainer
DQN 학습 루프 구현 (간단한 버전)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdevs.simulator import Simulator
from rl.replay_buffer import ReplayBuffer
from config import STATUS_RUNNING, STATUS_WIN, STATUS_PARTIAL_WIN, STATUS_FAIL

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard를 사용하려면 설치하세요: pip3 install tensorboard")


class DQNTrainer:
    """
    DQN 학습 루프를 관리하는 클래스

    주의: 현재는 에피소드 전체를 실행 후 보상 계산하는 간단한 버전
    커리큘럼 학습 지원 (단계적 난이도 증가)
    """

    def __init__(
        self,
        agent,
        create_system_fn,
        num_episodes=1000,
        termination_time=100,
        batch_size=32,
        buffer_size=10000,
        log_interval=10,
        save_interval=100,
        model_path="models/dqn_worm_robot.pth",
        use_tensorboard=True,
        tensorboard_dir="runs/worm_robot_dqn",
        curriculum_stages=None,
        progression_threshold=0.7,
        progression_window=100
    ):
        """
        Args:
            agent: DQN 에이전트
            create_system_fn: WormRobotSystem을 생성하는 함수
            num_episodes: 학습 에피소드 수
            termination_time: 시뮬레이션 최대 시간 (초)
            batch_size: 배치 크기
            buffer_size: Replay Buffer 크기
            log_interval: 로그 출력 간격
            save_interval: 모델 저장 간격
            model_path: 모델 저장 경로
            use_tensorboard: TensorBoard 사용 여부
            tensorboard_dir: TensorBoard 로그 디렉토리
            curriculum_stages: 커리큘럼 학습 단계 리스트 (None이면 단일 단계)
                               예: [{"name": "Stage1", "num_robots": 1, "min_distance": 0},
                                    {"name": "Stage2", "num_robots": 2, "min_distance": 6}]
            progression_threshold: 다음 단계로 진행하기 위한 성공률 임계값 (0.0~1.0)
            progression_window: 성공률 계산에 사용할 최근 에피소드 수
        """
        self.agent = agent
        self.create_system_fn = create_system_fn
        self.num_episodes = num_episodes
        self.termination_time = termination_time
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.model_path = model_path

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

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
            "episode_results": [],  # 각 에피소드 결과 (STATUS_WIN, STATUS_PARTIAL_WIN, etc.)
            "success_count": 0,
            "partial_success_count": 0,
            "fail_count": 0,
            "timeout_count": 0  # 시간 초과 카운트 추가
        }

    def train(self):
        """학습 루프 실행 (커리큘럼 학습 지원)"""
        print("=" * 60)
        print("DQN 학습 시작")
        print("=" * 60)
        print(f"에피소드 수: {self.num_episodes}")
        print(f"시뮬레이션 시간: {self.termination_time}초")
        print(f"배치 크기: {self.batch_size}")
        print(f"초기 Epsilon: {self.agent.epsilon:.3f}")

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
            episode_reward, episode_steps, episode_status, num_experiences = self._run_episode()

            # 통계 업데이트
            self.stats["episode_rewards"].append(episode_reward)
            self.stats["episode_steps"].append(episode_steps)
            self.stats["episode_results"].append(episode_status)  # 에피소드 결과 저장

            if episode_status == STATUS_WIN:
                self.stats["success_count"] += 1
            elif episode_status == STATUS_PARTIAL_WIN:
                self.stats["partial_success_count"] += 1
            elif episode_status == STATUS_FAIL:
                self.stats["fail_count"] += 1
            else:  # STATUS_RUNNING (시간 초과)
                self.stats["timeout_count"] += 1
            
            # 학습 (배치가 충분히 쌓이면)
            if len(self.replay_buffer) >= self.batch_size:
                total_loss = 0.0
                # 여러 번 학습
                for _ in range(5):
                    # ReplayBuffer.sample()은 (states, actions, rewards, next_states, dones) 반환
                    # agent.train()은 [(s,a,r,s',d), ...] 형태 기대
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                    batch = list(zip(states, actions, rewards, next_states, dones))
                    loss = self.agent.train(batch)
                    total_loss += loss
                self.stats["episode_losses"].append(total_loss / 5)
            else:
                self.stats["episode_losses"].append(0.0)
            
            # Epsilon 감소
            self.agent.update_epsilon()

            # 커리큘럼 단계 진행 체크
            if self._check_stage_progression(episode):
                self._progress_to_next_stage(episode)

            # TensorBoard 로깅
            if self.writer is not None:
                self.writer.add_scalar('Reward/episode', episode_reward, episode)
                self.writer.add_scalar('Steps/episode', episode_steps, episode)
                self.writer.add_scalar('Loss/episode', self.stats["episode_losses"][-1], episode)
                self.writer.add_scalar('Epsilon', self.agent.epsilon, episode)
                self.writer.add_scalar('Success/total', self.stats["success_count"], episode)
                self.writer.add_scalar('Success/partial', self.stats["partial_success_count"], episode)
                self.writer.add_scalar('Fail/total', self.stats["fail_count"], episode)
                self.writer.add_scalar('Timeout/total', self.stats["timeout_count"], episode)
                self.writer.add_scalar('Experiences/episode', num_experiences, episode)

                # 성공/부분성공/실패/시간초과를 0 또는 1로 기록
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
                else:  # STATUS_RUNNING (시간 초과)
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
                    f"ε: {self.agent.epsilon:.3f} | "
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
        단일 에피소드 실행 (스텝별 경험 수집 버전)

        Returns:
            tuple: (total_reward, step_count, final_status, num_experiences)
        """
        # 새로운 시스템 생성 (랜덤 초기화)
        system = self.create_system_fn(rl_agent=self.agent)

        # 시뮬레이터 설정 및 실행
        sim = Simulator(system)
        sim.setClassicDEVS()
        sim.setTerminationTime(self.termination_time)
        sim.simulate()

        # 최종 상태 수집
        final_status = system.environment.state.status
        step_count = system.environment.state.step_count

        # Controller에서 스텝별 경험 데이터 가져오기
        experiences = system.controller.get_step_experiences()

        # Replay Buffer에 경험 추가
        for exp in experiences:
            state, action, reward, next_state, done = exp
            self.replay_buffer.add(state, action, reward, next_state, float(done))

        # 에피소드 총 보상 계산 (모든 경험의 보상 합계)
        total_reward = sum(exp[2] for exp in experiences) if experiences else 0.0

        # 에피소드 종료 시 추가 보상 (전체 결과에 따라)
        if final_status == STATUS_WIN:
            # 완전 성공 보너스
            total_reward += 500.0
        elif final_status == STATUS_PARTIAL_WIN:
            # 부분 성공 보너스
            total_reward += 200.0
        elif final_status == STATUS_FAIL:
            # 실패 페널티
            total_reward -= 100.0

        return total_reward, step_count, final_status, len(experiences)

    def _save_model(self):
        """모델 저장"""
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else "models", exist_ok=True)
        self.agent.save(self.model_path)

    def evaluate(self, num_episodes=10):
        """학습된 에이전트 평가"""
        print("\n" + "=" * 60)
        print(f"평가 시작 ({num_episodes} 에피소드)")
        print("=" * 60)

        success_count = 0
        total_rewards = []
        total_steps = []

        # 원래 epsilon 저장
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # 평가 시에는 탐험 안 함

        for episode in range(num_episodes):
            reward, steps, status, _ = self._run_episode()
            total_rewards.append(reward)
            total_steps.append(steps)

            if status == STATUS_WIN:
                success_count += 1

        # Epsilon 복원
        self.agent.epsilon = original_epsilon

        avg_reward = sum(total_rewards) / num_episodes
        avg_steps = sum(total_steps) / num_episodes
        success_rate = success_count / num_episodes * 100

        print(f"평균 보상: {avg_reward:.2f}")
        print(f"평균 스텝: {avg_steps:.1f}")
        print(f"성공률: {success_rate:.1f}%")
        print("=" * 60)

        return {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps
        }

    def _check_stage_progression(self, episode):
        """
        현재 단계에서 다음 단계로 진행할 준비가 되었는지 확인

        Args:
            episode: 현재 에피소드 번호

        Returns:
            bool: 다음 단계로 진행 가능 여부
        """
        # 커리큘럼 학습이 설정되지 않았거나 마지막 단계인 경우
        if not self.curriculum_stages or self.current_stage_idx >= len(self.curriculum_stages) - 1:
            return False

        # 충분한 에피소드가 진행되었는지 확인 (최소 window 크기만큼)
        stage_episodes = episode - self.stage_start_episode
        if stage_episodes < self.progression_window:
            return False

        # 최근 progression_window 에피소드의 성공률 계산
        recent_results = self.stats["episode_results"][-self.progression_window:]

        # 성공 및 부분 성공 카운트
        success_count = sum(1 for r in recent_results if r == STATUS_WIN)
        partial_success_count = sum(1 for r in recent_results if r == STATUS_PARTIAL_WIN)

        # 성공률 계산 (부분 성공은 0.5로 계산)
        combined_success_rate = (success_count + partial_success_count * 0.5) / len(recent_results)

        if combined_success_rate >= self.progression_threshold:
            return True

        return False

    def _progress_to_next_stage(self, episode):
        """
        다음 커리큘럼 단계로 진행

        Args:
            episode: 현재 에피소드 번호
        """
        # 현재 단계 모델 저장
        current_stage = self.curriculum_stages[self.current_stage_idx]
        stage_model_path = self.model_path.replace(".pth", f"_{current_stage['name']}.pth")
        os.makedirs(os.path.dirname(stage_model_path) if os.path.dirname(stage_model_path) else "outputs", exist_ok=True)
        self.agent.save(stage_model_path)

        print("\n" + "=" * 60)
        print(f"🎓 커리큘럼 진행: {current_stage['name']} 완료!")
        print(f"   성공률: {self.stats['success_count'] / len(self.stats['episode_rewards']) * 100:.1f}%")
        print(f"   모델 저장: {stage_model_path}")

        # 다음 단계로 이동
        self.current_stage_idx += 1
        next_stage = self.curriculum_stages[self.current_stage_idx]
        self.stage_start_episode = episode

        # config 업데이트
        import config
        config.NUM_ROBOTS = next_stage["num_robots"]
        config.MIN_ROBOT_DISTANCE = next_stage["min_distance"]

        # 통계 리셋 (선택적)
        # self.stats = {...}  # 리셋하지 않고 누적

        print(f"🚀 다음 단계 시작: {next_stage['name']}")
        print(f"   로봇 수: {next_stage['num_robots']}")
        print(f"   최소 거리: {next_stage['min_distance']}")
        print("=" * 60 + "\n")
