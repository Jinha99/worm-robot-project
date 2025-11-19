"""
Worm Robot Simulation - Main Entry Point
시뮬레이션 실행 진입점
"""

from pypdevs.simulator import Simulator

from config import (
    GRID_SIZE,
    GRID_MIN,
    GRID_MAX,
    DIR_NAMES,
    SIMULATION_TERMINATION_TIME,
)
from system import WormRobotSystem


def print_simulation_info():
    """시뮬레이션 정보 출력"""
    print("=" * 60)
    print("Worm Robot Simulation - DEVS Model")
    print("=" * 60)
    print(f"격자: {GRID_SIZE}x{GRID_SIZE} ({GRID_MIN} ~ {GRID_MAX})")
    print("로봇: 4대 (2관절: 앞발-뒷발)")
    print("목표: 모든 뒷발을 (0,0)에 모으고, 앞발로 십자 패턴 형성")
    print("행동: 전진(5초), 시계방향 회전(3초), 반시계방향 회전(3초)")
    print("=" * 60)


def print_simulation_results(system):
    """시뮬레이션 결과 출력"""
    print("\n" + "=" * 60)
    print("시뮬레이션 완료")
    print("최종 환경 상태:")
    print(f"  게임 상태: {system.environment.state.status}")
    print(f"  진행 스텝: {system.environment.state.step_count}")
    print("\n최종 로봇 위치:")
    for rid, pos in system.environment.state.robot_positions.items():
        print(
            f"  Robot {rid}: "
            f"앞발={pos['head']}, "
            f"뒷발={pos['tail']}, "
            f"방향={DIR_NAMES[pos['direction']]}"
        )
    print("=" * 60)


def run_simulation(rl_agent=None, verbose=False, termination_time=None):
    """
    시뮬레이션 실행

    Args:
        rl_agent: (선택) 강화학습 에이전트 인스턴스
        verbose: 상세 로그 출력 여부
        termination_time: 시뮬레이션 종료 시간 (초)

    Returns:
        WormRobotSystem: 시뮬레이션이 완료된 시스템 인스턴스
    """
    # 시스템 생성
    system = WormRobotSystem(rl_agent=rl_agent)

    # 시뮬레이터 설정
    sim = Simulator(system)
    if verbose:
        sim.setVerbose()

    termination = termination_time or SIMULATION_TERMINATION_TIME
    sim.setTerminationTime(termination)
    sim.setClassicDEVS()

    # 시뮬레이션 실행
    print("\n시뮬레이션 시작...\n")
    sim.simulate()

    return system


def main():
    """메인 함수"""
    # 시뮬레이션 정보 출력
    print_simulation_info()

    # 시뮬레이션 실행
    # RL 에이전트를 사용하려면: run_simulation(rl_agent=your_agent)
    system = run_simulation(verbose=False)

    # 결과 출력
    print_simulation_results(system)


if __name__ == "__main__":
    main()
