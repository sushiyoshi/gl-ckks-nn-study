from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MLPBlockSchedule:
    n_in: int
    n_hidden: int
    n_out: int
    n_samples: int
    block_size: int = 32
    shape: tuple[int, int, int] = (256, 32, 32)

    @property
    def capacity_per_ciphertext(self) -> int:
        return self.shape[0] * self.shape[2]

    @property
    def input_blocks(self) -> int:
        return math.ceil(self.n_in / self.block_size)

    @property
    def hidden_blocks(self) -> int:
        return math.ceil(self.n_hidden / self.block_size)

    @property
    def output_blocks(self) -> int:
        return math.ceil(self.n_out / self.block_size)

    @property
    def linear1_block_matmuls(self) -> int:
        return self.input_blocks * self.hidden_blocks

    @property
    def linear2_block_matmuls(self) -> int:
        return self.hidden_blocks * self.output_blocks

    @property
    def total_linear_block_matmuls(self) -> int:
        return self.linear1_block_matmuls + self.linear2_block_matmuls

    @property
    def activation_blocks(self) -> int:
        return self.hidden_blocks

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "n_samples": self.n_samples,
            "shape": list(self.shape),
            "capacity_per_ciphertext": self.capacity_per_ciphertext,
            "input_blocks": self.input_blocks,
            "hidden_blocks": self.hidden_blocks,
            "output_blocks": self.output_blocks,
            "linear1_block_matmuls": self.linear1_block_matmuls,
            "linear2_block_matmuls": self.linear2_block_matmuls,
            "total_linear_block_matmuls": self.total_linear_block_matmuls,
            "activation_blocks": self.activation_blocks,
            "ciphertext_counts": {
                "input": self.input_blocks,
                "W1": self.linear1_block_matmuls,
                "hidden": self.hidden_blocks,
                "W2": self.linear2_block_matmuls,
                "output": self.output_blocks,
                "total_live_logical": self.input_blocks
                + self.linear1_block_matmuls
                + self.hidden_blocks
                + self.linear2_block_matmuls
                + self.output_blocks,
            },
        }


def mlp_block_schedule(
    n_in: int,
    n_hidden: int,
    n_out: int,
    n_samples: int,
    block_size: int = 32,
    shape: tuple[int, int, int] = (256, 32, 32),
) -> dict[str, Any]:
    return MLPBlockSchedule(
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        n_samples=n_samples,
        block_size=block_size,
        shape=shape,
    ).to_dict()


def ffn_block_matmuls(n_in: int, n_hidden: int, n_out: int, block_size: int = 32) -> dict[str, int]:
    input_blocks = math.ceil(n_in / block_size)
    hidden_blocks = math.ceil(n_hidden / block_size)
    output_blocks = math.ceil(n_out / block_size)
    linear1 = input_blocks * hidden_blocks
    linear2 = hidden_blocks * output_blocks
    return {
        "input_blocks": input_blocks,
        "hidden_blocks": hidden_blocks,
        "output_blocks": output_blocks,
        "linear1_block_matmuls": linear1,
        "linear2_block_matmuls": linear2,
        "total_block_matmuls": linear1 + linear2,
        "activation_blocks": hidden_blocks,
    }


def swiglu_block_matmuls(hidden: int, intermediate: int, block_size: int = 32) -> dict[str, int]:
    hidden_blocks = math.ceil(hidden / block_size)
    intermediate_blocks = math.ceil(intermediate / block_size)
    gate = hidden_blocks * intermediate_blocks
    up = hidden_blocks * intermediate_blocks
    down = intermediate_blocks * hidden_blocks
    return {
        "input_blocks": hidden_blocks,
        "intermediate_blocks": intermediate_blocks,
        "output_blocks": hidden_blocks,
        "gate_proj_block_matmuls": gate,
        "up_proj_block_matmuls": up,
        "down_proj_block_matmuls": down,
        "total_block_matmuls": gate + up + down,
        "activation_blocks": intermediate_blocks,
    }
