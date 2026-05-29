from __future__ import annotations


def overhead_record(
    logical_dim: tuple[int, ...],
    physical_dim: int,
    used_slots: int,
    total_slots: int,
    matrix_multiply_count: int,
    add_count: int,
    polynomial_degree: int,
    mask_count: int = 0,
    block_count: int = 1,
) -> dict:
    return {
        "logical_dim": list(logical_dim),
        "physical_dim": physical_dim,
        "used_slots": used_slots,
        "total_slots": total_slots,
        "packing_utilization": used_slots / total_slots,
        "matrix_multiply_count": matrix_multiply_count,
        "add_count": add_count,
        "polynomial_degree": polynomial_degree,
        "mask_count": mask_count,
        "block_count": block_count,
    }


def padding_ratio(logical: int, physical: int) -> float:
    return (physical - logical) / physical
