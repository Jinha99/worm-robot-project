"""
Worm Robot Simulation - Environment Model
환경 DEVS 모델 정의
"""

from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import DIR_NAMES, STATUS_RUNNING, STATUS_WIN, STATUS_PARTIAL_WIN, STATUS_FAIL
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
        self.rewards = {}          # {robot_id: reward} - 각 로봇의 보상
        self.prev_distances = {}   # {robot_id: distance} - 이전 스텝의 목표까지 거리
        self.robot_success = {}    # {robot_id: bool} - 각 로봇의 성공 여부

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

    def __init__(self, num_robots=4, initial_positions=None, robot_goals=None):
        """
        Args:
            num_robots: 로봇 수
            initial_positions: 초기 로봇 위치 리스트
            robot_goals: 각 로봇의 목적지 딕셔너리 {robot_id: goal_position}
        """
        AtomicDEVS.__init__(self, "Environment")
        self.num_robots = num_robots
        self.state = EnvironmentState()
        self.initial_positions = initial_positions or []
        self.robot_goals = robot_goals or {}  # 각 로봇의 목적지

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
        self.reward_out = self.addOutPort("reward_out")    # 보상 데이터

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
        """출력 함수 - 관찰 데이터, 상태, 보상 전송"""
        # INIT이나 PROCESSING 상태에서만 출력 (IDLE에서는 출력 안함)
        if self.state.phase in ["INIT", "PROCESSING"]:
            obs = self._generate_observations()
            return {
                self.obs_out: obs,
                self.status_out: {
                    "status": self.state.status,
                    "step": self.state.step_count
                },
                self.reward_out: self.state.rewards.copy()  # 보상 전달
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

        # 보상 계산 (승패 판정 전에 수행)
        self._calculate_rewards()

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
        """승리 조건 확인: 로봇별 성공 여부 체크 및 전체/부분 성공 판정"""
        if len(self.state.robot_positions) < self.num_robots:
            return False

        # 각 로봇의 성공 여부 확인
        success_count = 0
        for rid, pos_data in self.state.robot_positions.items():
            # 뒷발이 중앙 (0,0)에 있고, 앞발이 목적지에 있으면 성공
            goal_position = self.robot_goals.get(rid)
            if (pos_data["tail"] == (0, 0) and
                goal_position is not None and
                pos_data["head"] == goal_position):
                self.state.robot_success[rid] = True
                success_count += 1
            else:
                self.state.robot_success[rid] = False

        # 전체 성공
        if success_count == self.num_robots:
            print(f"[완전 승리] 모든 로봇({self.num_robots}개)이 목적지에 도착했습니다!")
            return True

        # 부분 성공 (1개 이상 로봇이 성공)
        if success_count > 0:
            print(f"[부분 승리] {success_count}/{self.num_robots}개 로봇이 목적지에 도착했습니다!")
            self.state.status = STATUS_PARTIAL_WIN
            return False  # 계속 진행

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
                "goal_position": self.robot_goals.get(rid, (0, 0))  # 목적지 좌표 전달
            }

        return observations
    
    def _calculate_rewards(self):
        """각 로봇의 보상 계산 (개선된 버전 - 충돌 회피 및 무효 행동 페널티 포함)"""
        for rid, pos_data in self.state.robot_positions.items():
            reward = 0.0

            # 현재 위치
            tail = pos_data["tail"]
            head = pos_data["head"]
            goal_head = self.robot_goals.get(rid, (0, 0))

            # 목표까지 거리
            tail_dist = abs(tail[0]) + abs(tail[1])
            head_dist = abs(head[0] - goal_head[0]) + abs(head[1] - goal_head[1])
            total_dist = tail_dist + head_dist

            # 1. 거리 기반 보상 (정규화: 가까울수록 높은 보상)
            # 최대 거리 12 (7x7 격자 대각선) → -1 ~ +1로 정규화
            distance_reward = (12 - total_dist) / 6.0 - 1.0  # -1 ~ +1
            reward += distance_reward

            # 2. 거리 변화 보상 (가까워지면 큰 보너스!)
            if rid in self.state.prev_distances:
                prev_dist = self.state.prev_distances[rid]
                dist_change = prev_dist - total_dist

                if dist_change > 0:
                    # 가까워지면 큰 보너스 (변화량에 비례)
                    reward += dist_change * 10.0
                elif dist_change < 0:
                    # 멀어지면 페널티 (더 강하게)
                    reward += dist_change * 5.0
                else:
                    # 거리 변화 없음 (무효 행동) - 페널티
                    reward -= 3.0

            # 현재 거리 저장
            self.state.prev_distances[rid] = total_dist

            # 3. 충돌 회피 페널티 (다른 로봇과의 거리)
            for other_rid, other_pos in self.state.robot_positions.items():
                if other_rid == rid:
                    continue

                # 다른 로봇의 머리/꼬리와의 거리 계산
                dist_to_other_head = abs(head[0] - other_pos["head"][0]) + abs(head[1] - other_pos["head"][1])
                dist_to_other_tail = abs(head[0] - other_pos["tail"][0]) + abs(head[1] - other_pos["tail"][1])
                min_dist = min(dist_to_other_head, dist_to_other_tail)

                if min_dist == 1:
                    # 매우 가까움 (충돌 직전) - 큰 페널티
                    reward -= 15.0
                elif min_dist == 2:
                    # 가까움 - 중간 페널티
                    reward -= 5.0

            # 4. 중간 목표 달성 보너스
            tail_at_center = (tail == (0, 0))
            head_at_goal = (head == goal_head)

            if tail_at_center and head_at_goal:
                # 완전 성공! (최고 보상)
                reward += 100.0
                if rid not in self.state.robot_success or not self.state.robot_success.get(rid, False):
                    # 처음 성공한 경우 추가 보너스
                    reward += 50.0
            elif tail_at_center:
                # 뒷발만 중앙에 도달
                reward += 30.0
            elif head_at_goal:
                # 앞발만 목표에 도달
                reward += 30.0

            # 5. 매우 가까운 거리 보너스 (거의 다 왔음!)
            if total_dist <= 2:
                reward += 10.0
            elif total_dist <= 4:
                reward += 5.0

            self.state.rewards[rid] = reward
    
    def get_rewards(self):
        """현재 보상 반환 (RL 학습용)"""
        return self.state.rewards.copy()
