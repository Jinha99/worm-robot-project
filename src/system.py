"""
Worm Robot Simulation - System Model
전체 시스템 통합 DEVS 모델
"""

from pypdevs.DEVS import CoupledDEVS

from config import get_initial_robot_configs, NUM_ROBOTS
from robot import Robot
from environment import Environment
from controller import Controller


# ========================================
# 결합 모델: Worm Robot System
# ========================================

class WormRobotSystem(CoupledDEVS):
    """
    2관절 로봇 시스템 전체를 통합하는 결합 DEVS 모델

    구성 요소:
    - Robot: 개별 로봇 모델 (NUM_ROBOTS개)
    - Environment: 환경 모델 (1개)
    - Controller: 컨트롤러 모델 (1개)
    """

    def __init__(self, rl_agent=None):
        """
        Args:
            rl_agent: (선택) 강화학습 에이전트 인스턴스
        """
        CoupledDEVS.__init__(self, "WormRobotSystem")

        # 매 시뮬레이션마다 새로운 랜덤 위치 생성
        robot_configs = get_initial_robot_configs()

        # 로봇 생성
        self.robots = []
        for config in robot_configs:
            robot = Robot(
                robot_id=config["id"],
                initial_head=config["head"],
                initial_tail=config["tail"],
                initial_direction=config["dir"]
            )
            self.robots.append(self.addSubModel(robot))

        # 환경 생성 (목적지 정보 포함)
        robot_goals = {config["id"]: config["goal"] for config in robot_configs}
        self.environment = self.addSubModel(
            Environment(num_robots=NUM_ROBOTS, initial_positions=robot_configs, robot_goals=robot_goals)
        )

        # 컨트롤러 생성
        self.controller = self.addSubModel(
            Controller(num_robots=NUM_ROBOTS, rl_agent=rl_agent)
        )

        # 포트 연결: 로봇 -> 환경
        for i, robot in enumerate(self.robots):
            self.connectPorts(robot.action_done_out, self.environment.robot_done_in[i])

        # 포트 연결: 환경 -> 컨트롤러
        self.connectPorts(self.environment.obs_out, self.controller.obs_in)
        self.connectPorts(self.environment.status_out, self.controller.status_in)

        # 포트 연결: 컨트롤러 -> 로봇
        for i, robot in enumerate(self.robots):
            self.connectPorts(self.controller.action_out[i], robot.action_in)

    def select(self, imm):
        """
        동시 발생 이벤트 우선순위 결정

        우선순위: 환경 > 컨트롤러 > 로봇

        Args:
            imm: 동시에 발생한 이벤트 리스트

        Returns:
            우선순위가 가장 높은 모델
        """
        if self.environment in imm:
            return self.environment
        if self.controller in imm:
            return self.controller
        return imm[0]
