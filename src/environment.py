"""
Worm Robot Simulation - Environment Model
환경 DEVS 모델 정의
"""

from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import DIR_NAMES, STATUS_RUNNING, STATUS_WIN, STATUS_FAIL
from utils import in_bounds, get_sensor_area


# ========================================
# Environment 상태 클래스
# ========================================

class EnvironmentState:
    """환경의 내부 상태를 표현하는 클래스"""

    def __init__(self):
        self.robot_positions = {}  # {robot_id: {"head": (x,y), "tail": (x,y), "direction": int}}
        self.status = STATUS_RUNNING
        self.step_count = 0
        self.phase = "INIT"        # 상태: INIT, IDLE, PROCESSING
        self.pending_updates = []  # 대기 중인 로봇 업데이트

    def __str__(self):
        return (
            f"Environment["
            f"상태:{self.phase},"
            f"게임상태:{self.status},"
            f"스텝:{self.step_count},"
            f"로봇수:{len(self.robot_positions)}]"
        )


# ========================================
# Environment 모델 (Atomic DEVS)
# ========================================

class Environment(AtomicDEVS):
    """로봇들이 활동하는 환경의 DEVS 모델"""

    def __init__(self, num_robots=4, initial_positions=None):
        """
        Args:
            num_robots: 로봇 수
            initial_positions: 초기 로봇 위치 리스트
        """
        AtomicDEVS.__init__(self, "Environment")
        self.num_robots = num_robots
        self.state = EnvironmentState()
        self.initial_positions = initial_positions or []

        # 로봇 초기 위치 설정
        for pos_data in self.initial_positions:
            self.state.robot_positions[pos_data["id"]] = {
                "head": pos_data["head"],
                "tail": pos_data["tail"],
                "direction": pos_data["dir"]
            }

        # 입력 포트 (로봇들로부터)
        self.robot_done_in = [self.addInPort(f"robot{i}_done_in") for i in range(num_robots)]

        # 출력 포트 (컨트롤러로)
        self.obs_out = self.addOutPort("obs_out")          # 관찰 데이터
        self.status_out = self.addOutPort("status_out")    # 게임 상태

    def timeAdvance(self):
        """시간 진행 함수"""
        if self.state.phase == "INIT":
            return 0  # 즉시 초기화
        elif self.state.phase == "IDLE":
            if self.state.status != STATUS_RUNNING:
                return INFINITY  # 게임 종료
            return INFINITY  # 로봇 행동 대기
        elif self.state.phase == "PROCESSING":
            return 0  # 즉시 처리
        return INFINITY

    def intTransition(self):
        """내부 전이 함수"""
        if self.state.phase == "INIT":
            # 초기 관찰 데이터를 보낸 후 IDLE로 전환
            self.state.phase = "IDLE"
            return self.state

        elif self.state.phase == "PROCESSING":
            # 로봇 위치 업데이트 및 승패 판정
            self._update_environment()
            self.state.pending_updates = []
            self.state.phase = "IDLE"
            return self.state

        return self.state

    def extTransition(self, inputs):
        """외부 전이 함수 - 로봇 행동 완료 신호 수신"""
        if self.state.phase == "IDLE":
            # 로봇들의 행동 완료 신호 수집
            for i, port in enumerate(self.robot_done_in):
                update = inputs.get(port)
                if update:
                    self.state.pending_updates.append(update)

            # 모든 로봇이 완료했으면 처리 시작
            if len(self.state.pending_updates) > 0:
                self.state.phase = "PROCESSING"

        return self.state

    def outputFnc(self):
        """출력 함수 - 관찰 데이터 및 상태 전송"""
        # INIT이나 PROCESSING 상태에서만 출력 (IDLE에서는 출력 안함)
        if self.state.phase in ["INIT", "PROCESSING"]:
            obs = self._generate_observations()
            return {
                self.obs_out: obs,
                self.status_out: {
                    "status": self.state.status,
                    "step": self.state.step_count
                }
            }
        return {}

    def _update_environment(self):
        """환경 업데이트: 로봇 위치 갱신 및 승패 판정"""
        # 로봇 위치 갱신
        for update in self.state.pending_updates:
            rid = update["robot_id"]
            self.state.robot_positions[rid] = {
                "head": update["head"],
                "tail": update["tail"],
                "direction": update["direction"]
            }

        self.state.step_count += 1

        # 승패 판정
        if self._check_fail():
            self.state.status = STATUS_FAIL
        elif self._check_win():
            self.state.status = STATUS_WIN

    def _check_fail(self):
        """실패 조건 확인: 격자 이탈 또는 충돌"""
        positions = self.state.robot_positions

        # 격자 범위 이탈 확인
        for rid, pos_data in positions.items():
            if not in_bounds(pos_data["head"]) or not in_bounds(pos_data["tail"]):
                print(f"[실패] Robot {rid}가 격자를 벗어났습니다!")
                return True

        # 충돌 확인
        occupied = {}
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]

            # 앞발 충돌 체크
            if head in occupied:
                print(f"[실패] {head} 위치에서 Robot {rid}와 Robot {occupied[head]}가 충돌!")
                return True
            occupied[head] = rid

            # 뒷발 충돌 체크 (단, (0,0)은 예외)
            if tail != (0, 0):
                if tail in occupied:
                    print(f"[실패] {tail} 위치에서 Robot {rid}와 Robot {occupied[tail]}가 충돌!")
                    return True
                occupied[tail] = rid

        return False

    def _check_win(self):
        """승리 조건 확인: 모든 뒷발이 (0,0)에 있고, 앞발이 십자 패턴"""
        if len(self.state.robot_positions) < self.num_robots:
            return False

        target_heads = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        actual_heads = set()

        for rid, pos_data in self.state.robot_positions.items():
            # 뒷발이 중앙에 있는지 확인
            if pos_data["tail"] != (0, 0):
                return False
            actual_heads.add(pos_data["head"])

        # 앞발이 십자 패턴을 이루는지 확인
        if actual_heads == target_heads:
            print(f"[승리] 모든 로봇이 중앙에 성공적으로 모였습니다!")
            return True

        return False

    def _generate_observations(self):
        """각 로봇의 센서 관찰 데이터 생성"""
        observations = {}

        for rid, pos_data in self.state.robot_positions.items():
            head = pos_data["head"]
            sensor_area = get_sensor_area(head)

            # 센서 범위 내 다른 로봇 감지
            detected = []
            for other_id, other_pos in self.state.robot_positions.items():
                if other_id == rid:
                    continue
                if other_pos["head"] in sensor_area or other_pos["tail"] in sensor_area:
                    detected.append({
                        "robot_id": other_id,
                        "head": other_pos["head"],
                        "tail": other_pos["tail"]
                    })

            observations[rid] = {
                "own_head": head,
                "own_tail": pos_data["tail"],
                "own_direction": pos_data["direction"],
                "detected_robots": detected,
                "distance_to_goal": abs(pos_data["tail"][0]) + abs(pos_data["tail"][1])
            }

        return observations
