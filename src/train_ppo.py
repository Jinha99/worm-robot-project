"""
Worm Robot PPO 학습 스크립트

PPO (Proximal Policy Optimization)를 사용한 멀티 로봇 협력 학습
"""

from system import WormRobotSystem
from rl.ppo_agent import PPOAgent
from rl.ppo_trainer import PPOTrainer
import config


def create_system(rl_agent=None):
    """
    시스템 생성 헬퍼 함수

    Args:
        rl_agent: PPO 에이전트 인스턴스

    Returns:
        WormRobotSystem: 초기화된 시스템
    """
    # 시스템 생성 (내부에서 자동으로 랜덤 초기 위치 생성)
    system = WormRobotSystem(rl_agent=rl_agent)

    return system


def main():
    """메인 함수"""
    print("=" * 60)
    print("Worm Robot PPO 학습 (커리큘럼 학습 + 정책 마스킹)")
    print("=" * 60)

    # 하이퍼파라미터
    STATE_DIM = 13  # controller._observation_to_state에서 정의한 차원
    ACTION_DIM = 3  # 전진, 시계방향, 반시계방향

    # PPO 에이전트 생성
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=3e-4,      # PPO 표준 학습률
        gamma=0.99,
        gae_lambda=0.95,         # GAE lambda
        clip_epsilon=0.2,        # PPO clip 범위
        value_coef=0.5,          # 가치 손실 계수
        entropy_coef=0.01,       # 엔트로피 보너스 (탐험 장려)
        max_grad_norm=0.5,       # 그래디언트 클리핑
        hidden_dim=128,          # 은닉층 크기
        device="cpu"
    )

    # 커리큘럼 학습 단계 정의
    curriculum_stages = [
        {
            "name": "Stage1_1Robot",
            "num_robots": 1,
            "min_distance": 0
        },
        {
            "name": "Stage2_2Robots",
            "num_robots": 2,
            "min_distance": 6  # 로봇들을 멀리 배치
        }
    ]

    # 트레이너 생성
    trainer = PPOTrainer(
        agent=agent,
        create_system_fn=create_system,
        num_episodes=5000,           # 전체 에피소드 수
        termination_time=200,        # 시뮬레이션 최대 시간
        update_epochs=4,             # PPO 업데이트 에폭 수
        batch_size=64,               # 미니배치 크기
        log_interval=10,             # 10 에피소드마다 로그 출력
        save_interval=50,            # 50 에피소드마다 모델 저장
        model_path="outputs/ppo_worm_robot.pth",
        curriculum_stages=curriculum_stages,  # 커리큘럼 단계 전달
        progression_threshold=0.7,   # 70% 성공률로 다음 단계 진행
        progression_window=100       # 최근 100 에피소드 기준
    )

    # 학습 실행
    try:
        stats = trainer.train()

        # 학습 완료 후 최종 평가 (선택)
        # print("\n최종 평가 시작...")
        # eval_stats = trainer.evaluate(num_episodes=100)

    except KeyboardInterrupt:
        print("\n\n⚠️  학습 중단됨 (Ctrl+C)")
        print("현재까지 학습한 모델을 저장합니다...")
        trainer._save_model()
        print("✅ 모델 저장 완료!")


if __name__ == "__main__":
    main()
