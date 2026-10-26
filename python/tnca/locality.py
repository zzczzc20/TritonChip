"""Graph-level storage, alias and lifetime consumer for Dynamic Cache Orchestration.

Implements the standard-path consumer design from evidence/r41/stage1/locality_design.md:
- Resolves views, slices and transposes to underlying storage tokens.
- Refuses bypass/dead hints on unsafe aliases, overlapping slices, or incomplete readers.
- Orchestrates TMU preamble/epilogue via public Session without dedicated numeric kernels.
- Propagates session poison on fault and keeps policy default-off when no gain exists.
"""
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Set, Tuple

# TMU memory-mapped register window (rtl/nuca/tnca_tmu.v)
TMU_BASE = 0x7FFF0000
REG_POLICY = 0x000
REG_RESET = 0x018
REG_TILE = 0x040
REG_BASE = 0x100
REG_SIZE = 0x108
REG_DESC = 0x110

MIN_GRANULE_LSB = 6  # 64-byte line


class LocalityRefusal(Exception):
    """Raised when lifetime analysis cannot establish safe tensor bypass or replacement."""


@dataclass
class StorageDescriptor:
    storage_id: int
    base_address: int
    size_bytes: int
    read_count: int = 0
    write_count: int = 0
    is_graph_input: bool = False
    is_graph_output: bool = False
    is_temporary: bool = False
    has_unsafe_alias: bool = False
    bypass_eligible: bool = False
    granule_lsb: int = MIN_GRANULE_LSB

    @property
    def desc_word(self) -> int:
        nacc = min(self.read_count, 0xFFFFFFFF)
        bypass = 1 if self.bypass_eligible else 0
        tile_lsb = self.granule_lsb & 0x3F
        ident = (self.storage_id & 0xF)
        return (nacc & 0xFFFFFFFF) | (bypass << 32) | (ident << 36) | (tile_lsb << 40)


@dataclass
class GraphLocalityPlan:
    schema: str = 'tnca-graph-locality-plan/1'
    storages: Dict[int, StorageDescriptor] = field(default_factory=dict)
    policy_flags: int = 0  # bit 0: replacement assist, bit 1: bypass assist
    refusals: List[str] = field(default_factory=list)
    is_safe: bool = True


class StorageAnalyzer:
    """Analyzes PyTorch graph tensors to establish safe storage boundaries and lifetimes."""

    def __init__(self):
        self.storages: Dict[int, StorageDescriptor] = {}
        self.tensor_intervals: Dict[int, List[Tuple[int, int]]] = {}

    def _get_storage_id(self, tensor: Any) -> int:
        if hasattr(tensor, 'untyped_storage'):
            return tensor.untyped_storage().data_ptr()
        if hasattr(tensor, 'storage'):
            return tensor.storage().data_ptr()
        if hasattr(tensor, 'data_ptr'):
            return tensor.data_ptr()
        return id(tensor)

    def register_tensor(self, tensor: Any, *, is_input: bool = False,
                        is_output: bool = False, is_read: bool = False,
                        is_write: bool = False, byte_offset: int = 0,
                        num_bytes: Optional[int] = None) -> StorageDescriptor:
        storage_id = self._get_storage_id(tensor)
        nbytes = num_bytes if num_bytes is not None else (tensor.nelement() * tensor.element_size()
                                                          if hasattr(tensor, 'nelement') else 64)
        data_ptr = tensor.data_ptr() if hasattr(tensor, 'data_ptr') else byte_offset

        if storage_id not in self.storages:
            self.storages[storage_id] = StorageDescriptor(
                storage_id=len(self.storages),
                base_address=data_ptr,
                size_bytes=nbytes,
                is_graph_input=is_input,
                is_graph_output=is_output,
                is_temporary=(not is_input and not is_output)
            )
            self.tensor_intervals[storage_id] = []

        desc = self.storages[storage_id]
        if is_input:
            desc.is_graph_input = True
            desc.is_temporary = False
        if is_output:
            desc.is_graph_output = True
            desc.is_temporary = False
        if is_read:
            desc.read_count += 1
        if is_write:
            desc.write_count += 1

        # Check for overlapping slice aliases
        start = data_ptr
        end = data_ptr + nbytes
        for prev_start, prev_end in self.tensor_intervals[storage_id]:
            # If two distinct intervals overlap within the same underlying storage
            if not (end <= prev_start or start >= prev_end):
                desc.has_unsafe_alias = True
        self.tensor_intervals[storage_id].append((start, end))

        return desc

    def finalize_plan(self) -> GraphLocalityPlan:
        plan = GraphLocalityPlan(storages=self.storages)
        for s_id, desc in self.storages.items():
            # Rule 1: Graph I/O or persistent state lifetime extends beyond graph
            if desc.is_graph_input or desc.is_graph_output:
                desc.bypass_eligible = False
            # Rule 2: Unsafe alias or overlapping slices refuse bypass and dead hints
            elif desc.has_unsafe_alias:
                desc.bypass_eligible = False
                plan.refusals.append(f'storage_{desc.storage_id}: overlapping slice or unsafe alias')
            # Rule 3: Proven internal temporary read once -> eligible for bypass
            elif desc.is_temporary and desc.read_count == 1:
                desc.bypass_eligible = True
            # Rule 4: Read multiple times -> ordinary caching, assist replacement
            elif desc.read_count > 1:
                desc.bypass_eligible = False
            else:
                desc.bypass_eligible = False

        if any(d.bypass_eligible for d in self.storages.values()):
            plan.policy_flags |= 0x2  # Bypass assist
        if any(d.read_count > 1 for d in self.storages.values()):
            plan.policy_flags |= 0x1  # Replacement assist

        return plan


class LocalityController:
    """Manages the lifecycle of TMU locality configurations on the device."""

    def __init__(self, session: Any):
        self.session = session
        self.prior_policy = 0
        self.active_epoch = False

    def emit_preamble(self, plan: GraphLocalityPlan) -> List[Tuple[int, int]]:
        """Generate MMIO control sequence for TMU configuration."""
        writes = []
        # 1. Reset previous TMU metadata
        writes.append((TMU_BASE + REG_RESET, 0))
        # 2. Set default tile granule (64 bytes = 2^6)
        writes.append((TMU_BASE + REG_TILE, MIN_GRANULE_LSB))
        # 3. Register each storage descriptor
        for desc in plan.storages.values():
            if desc.has_unsafe_alias:
                continue  # Refuse registration of unsafe aliases
            writes.append((TMU_BASE + REG_BASE, desc.base_address))
            writes.append((TMU_BASE + REG_SIZE, desc.size_bytes))
            writes.append((TMU_BASE + REG_DESC, desc.desc_word))
        # 4. Set policy bits
        writes.append((TMU_BASE + REG_POLICY, plan.policy_flags))
        return writes

    def emit_cleanup(self) -> List[Tuple[int, int]]:
        """Generate MMIO control sequence to clear TMU metadata."""
        return [
            (TMU_BASE + REG_RESET, 0),
            (TMU_BASE + REG_POLICY, self.prior_policy)
        ]

    def execute_graph_with_locality(self, plan: GraphLocalityPlan, graph_callable: Any, *args, **kwargs) -> Any:
        """Execute graph with exclusive locality epoch and poison guard."""
        if getattr(self.session, 'poisoned', False):
            raise LocalityRefusal('Cannot execute locality on poisoned session')

        enabled = os.environ.get('TNCA_LOCALITY_DCO', '0') == '1'
        self.active_epoch = True
        try:
            if enabled and plan.is_safe:
                preamble = self.emit_preamble(plan)
                if hasattr(self.session, 'mmio_write_sequence'):
                    self.session.mmio_write_sequence(preamble)
            res = graph_callable(*args, **kwargs)
            return res
        except Exception as e:
            if hasattr(self.session, 'poisoned'):
                self.session.poisoned = True
            raise e
        finally:
            if enabled and plan.is_safe and not getattr(self.session, 'poisoned', False):
                cleanup = self.emit_cleanup()
                if hasattr(self.session, 'mmio_write_sequence'):
                    self.session.mmio_write_sequence(cleanup)
            self.active_epoch = False
