# 전략 스테이지 상수 — 단일 정의, 전체 공유
# 변경 시 이 파일만 수정하면 됩니다.

STAGE_RATIOS: dict[int, tuple[float, float]] = {
    0: (1.0, 0.0),   # S0: Base 100%, Leverage   0%
    1: (0.7, 0.3),   # S1: Base  70%, Leverage  30%
    2: (0.3, 0.7),   # S2: Base  30%, Leverage  70%
    3: (0.0, 1.0),   # S3: Base   0%, Leverage 100%
}

STAGE_LABELS: dict[int, str] = {
    0: 'S0 (Base 100%)',
    1: 'S1 (Base 70%+Lev 30%)',
    2: 'S2 (Base 30%+Lev 70%)',
    3: 'S3 (Lev 100%)',
}

STAGE_ICONS: dict[int, str] = {
    0: '🔵',
    1: '🟡',
    2: '🟠',
    3: '🔴',
}

STAGE_COLORS: dict[int, str] = {
    0: 'blue',
    1: 'orange',
    2: 'orange',
    3: 'red',
}

def stage_label(stage: int) -> str:
    return STAGE_LABELS.get(stage, f'S{stage}')

def stage_icon(stage: int) -> str:
    return STAGE_ICONS.get(stage, '⚪')

def stage_ratio(stage: int) -> tuple[float, float]:
    """(base_ratio, leverage_ratio) 반환"""
    return STAGE_RATIOS.get(stage, (1.0, 0.0))
