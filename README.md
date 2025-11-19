# Worm Robot Simulation

2관절 로봇의 협력 경로 탐색 시뮬레이션 프로젝트입니다. DEVS(Discrete Event System Specification) 모델링을 사용하여 구현되었으며, 향후 강화학습 통합을 위해 모듈화되었습니다.

## 설치 방법

먼저 PyPDEVS 라이브러리를 설치해야 합니다

```bash
cd src
python setup.py install --user
```

## 프로젝트 개요

### 시뮬레이션 목표
- **격자**: 7x7 그리드 (-3 ~ 3)
- **로봇**: 4대의 2관절 로봇 (앞발-뒷발 구조)
- **목표**: 모든 로봇의 뒷발을 중앙 (0,0)에 모으고, 앞발로 십자 패턴 형성
- **행동**: 전진(5초), 시계방향 회전(3초), 반시계방향 회전(3초)

### 특징
- DEVS 기반 이벤트 기반 시뮬레이션
- 모듈화된 구조로 강화학습 통합 준비 완료
- 충돌 감지 및 승패 판정 시스템

## 실행 방법
파이썬 3.11 버전에서만 실행 가능합니다.

```bash
cd src
python3.11 main.py
```

## 프로젝트 구조

```
src/
├── config.py              # 상수 및 설정값
├── utils.py               # 유틸리티 함수
├── robot.py               # Robot DEVS 모델
├── environment.py         # Environment DEVS 모델
├── controller.py          # Controller DEVS 모델
├── system.py              # 시스템 통합 모델
├── main.py                # 실행 진입점
└── rl/                    # 강화학습 모듈 (템플릿)
    ├── __init__.py
    ├── agent.py
    ├── trainer.py
    └── replay_buffer.py
```

## 파일 설명

### 핵심 모듈

#### `config.py`
시뮬레이션의 모든 설정값과 상수를 관리합니다.

- **격자 설정**: `GRID_SIZE`, `GRID_MIN`, `GRID_MAX`
- **방향 정의**: `DIRECTIONS`, `DIR_NAMES` (동/남/서/북)
- **행동 타입**: `ACTION_MOVE`, `ACTION_ROTATE_CW`, `ACTION_ROTATE_CCW`
- **행동 소요 시간**: 각 행동의 시뮬레이션 시간 정의
- **게임 상태**: `STATUS_RUNNING`, `STATUS_WIN`, `STATUS_FAIL`
- **로봇 초기 설정**: `INITIAL_ROBOT_CONFIGS` - 4대 로봇의 초기 위치 및 방향

#### `utils.py`
재사용 가능한 유틸리티 함수 모음입니다.

- `in_bounds(pos)`: 위치가 격자 범위 내에 있는지 확인
- `add_pos(pos1, pos2)`: 두 좌표를 더하는 함수
- `get_sensor_area(head_pos)`: 앞발 위치 기준 3x3 센서 영역 반환

#### `robot.py`
개별 로봇의 DEVS 모델을 정의합니다.

- **RobotState 클래스**: 로봇의 내부 상태 (위치, 방향, 행동)
- **Robot 클래스**: 2관절 로봇의 Atomic DEVS 모델
  - `timeAdvance()`: 행동 실행 시간 정의
  - `intTransition()`: 행동 완료 시 상태 전이
  - `extTransition()`: 행동 명령 수신 및 위치 업데이트
  - `outputFnc()`: 행동 완료 신호 전송
- **지원 행동**:
  - 전진: 앞발이 현재 방향으로 1칸 이동, 뒷발은 앞발의 이전 위치로 이동
  - 시계방향 회전: 뒷발 고정, 앞발만 회전
  - 반시계방향 회전: 뒷발 고정, 앞발만 회전

#### `environment.py`
로봇들이 활동하는 환경의 DEVS 모델입니다.

- **EnvironmentState 클래스**: 환경 상태 (로봇 위치, 게임 상태, 스텝 수)
- **Environment 클래스**: 환경 Atomic DEVS 모델
  - `_update_environment()`: 로봇 위치 갱신 및 승패 판정
  - `_check_fail()`: 실패 조건 확인 (격자 이탈 또는 충돌)
  - `_check_win()`: 승리 조건 확인 (십자 패턴 완성)
  - `_generate_observations()`: 각 로봇의 센서 관찰 데이터 생성
- **관찰 데이터**: 자신의 위치, 방향, 감지된 다른 로봇, 목표까지 거리

#### `controller.py`
로봇들의 행동을 결정하는 컨트롤러 DEVS 모델입니다. **강화학습 연동의 핵심 지점**입니다.

- **ControllerState 클래스**: 컨트롤러 상태 (관찰 데이터, 게임 상태)
- **Controller 클래스**: 컨트롤러 Atomic DEVS 모델
  - `__init__(rl_agent=None)`: RL 에이전트를 선택적으로 연결
  - `_select_action()`: **행동 선택 정책** - 현재는 휴리스틱, RL 에이전트 연동 가능
  - `_observation_to_state()`: 관찰 데이터를 RL 상태로 변환 (TODO)
- **현재 정책**: 거리 기반 휴리스틱 (목표에서 멀면 전진, 가까우면 다양한 행동)

#### `system.py`
전체 시스템을 통합하는 결합 DEVS 모델입니다.

- **WormRobotSystem 클래스**: Coupled DEVS 모델
  - 4대의 Robot, 1개의 Environment, 1개의 Controller 생성
  - 포트 연결: Robot ↔ Environment ↔ Controller
  - `select()`: 동시 이벤트 발생 시 우선순위 결정 (Environment > Controller > Robot)

#### `main.py`
시뮬레이션 실행 진입점입니다.

- `print_simulation_info()`: 시뮬레이션 설정 정보 출력
- `print_simulation_results()`: 시뮬레이션 결과 출력
- `run_simulation()`: 시뮬레이션 실행 (RL 에이전트 선택 가능)
- `main()`: 메인 함수

### 강화학습 모듈 (템플릿)

향후 강화학습 구현을 위한 템플릿 파일들입니다.

#### `rl/__init__.py`
강화학습 모듈 초기화 파일입니다.

#### `rl/agent.py`
강화학습 에이전트 기본 클래스 템플릿입니다.

- **RLAgent 클래스**: RL 에이전트 추상 클래스
  - `get_action(state, training)`: 상태에서 행동 선택
  - `update()`: 에이전트 학습
  - `save()`, `load()`: 모델 저장/로드
- **TODO**: DQN, PPO, A3C 등 구체적인 알고리즘 구현

#### `rl/trainer.py`
강화학습 학습 루프 템플릿입니다.

- **RLTrainer 클래스**: 학습 루프 관리
  - `train()`: 학습 루프 실행
  - `evaluate()`: 학습된 에이전트 평가
- **TODO**: DEVS 시뮬레이션과 RL 통합 구현

#### `rl/replay_buffer.py`
경험 재생 버퍼 구현 템플릿입니다.

- **ReplayBuffer 클래스**: 기본 경험 재생 버퍼
  - `add()`: 경험 추가
  - `sample()`: 배치 샘플링
- **PrioritizedReplayBuffer 클래스**: 우선순위 기반 버퍼
  - `update_priorities()`: 우선순위 업데이트
- **TODO**: 우선순위 샘플링 로직 구현

## 강화학습 통합 방법

### 1. RL 에이전트 구현
`src/rl/agent.py`에서 `RLAgent` 클래스를 상속받아 구체적인 알고리즘 구현:

```python
class DQNAgent(RLAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        # 네트워크 초기화
        ...
```

### 2. Controller와 연동
`src/controller.py`의 `_select_action()` 메서드 수정:

```python
def _select_action(self, rid, obs):
    if self.rl_agent is not None:
        state = self._observation_to_state(obs)
        action = self.rl_agent.get_action(state)
        return {"type": action}
    # 기본 휴리스틱
    ...
```

### 3. 학습 실행
`src/main.py`에서 RL 에이전트와 함께 실행:

```python
from rl.agent import DQNAgent

agent = DQNAgent(state_dim=..., action_dim=3)
system = run_simulation(rl_agent=agent)
```


