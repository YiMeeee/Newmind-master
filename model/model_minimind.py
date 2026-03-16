# Backward-compatibility shim — re-exports from new location
from model.lite.rnewmind_base import (
    RNewMindConfig as MiniMindConfig,
    RNewMindForCausalLM as MiniMindForCausalLM,
    RNewMindModel as MiniMindModel,
    RNewMindBlock as MiniMindBlock,
)

__all__ = ["MiniMindConfig", "MiniMindForCausalLM", "MiniMindModel", "MiniMindBlock"]
