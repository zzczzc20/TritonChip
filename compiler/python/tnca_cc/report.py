"""
Per-kernel resource and bottleneck report.

compiler_overview.md sec 16.1 asks for this as a formal output of the backend,
not a debugging aid, and says why:

    这比只输出"寄存器用了多少"更能指导硬件修改。

So the report records the plans that were *rejected* alongside the one that
was chosen. "Four contexts did not fit because the accumulator file is too
small" is a sentence a hardware team can act on; "16 accumulators used" is
not. The fallback order of sec 10.3 is what produces those rejections, and
every step of it is kept.
"""
from dataclasses import dataclass, field

from .diagnostics import ResourceError


@dataclass
class RejectedPlan:
    """A configuration that was tried and did not fit."""
    description: str
    reason: str

    def __str__(self):
        return f"rejected {self.description}:\n    {self.reason}"


@dataclass
class KernelPlan:
    """What a compiled kernel asks of the machine.

    Whole-program resources (sec 10.1) and the per-role costs that decide
    issue pressure (sec 10.4). Populated by the backend; this module only
    formats and checks it.
    """
    name: str
    active_contexts: int = 1
    rf_stages: int = 1
    shared_windows: int = 0
    private_vector_bytes: int = 0
    accumulator_slots: int = 0
    barriers: int = 0
    max_async_commands: int = 0

    # sec 4.3 / sec 7.3: a layout conversion is real traffic, so it is
    # reported rather than hidden inside a pass.
    layout_conversion_bytes: int = 0

    # sec 8.2: what the source-level transfers expanded into. The operation
    # counts in mb.arm are taken after this, so a report that showed only the
    # source count would not explain the barrier arming.
    dma_commands_source: int = 0
    dma_commands_expanded: int = 0

    # sec 10.4: accumulated per shared execution unit, because two roles that
    # look independent in the IR still contend for one SIMD pipe.
    issue_cycles_by_unit: dict = field(default_factory=dict)

    source_release_policy: str = 'command_completion'
    rejected: list = field(default_factory=list)

    @property
    def dma_expansion_factor(self):
        if not self.dma_commands_source:
            return 0.0
        return self.dma_commands_expanded / self.dma_commands_source

    @property
    def predicted_issue_pressure(self):
        """The busiest shared unit, and how busy.

        The bottleneck is the maximum over units, not the sum: the units run
        concurrently, and reporting the sum would name a number no unit ever
        sees.
        """
        if not self.issue_cycles_by_unit:
            return (None, 0)
        unit = max(self.issue_cycles_by_unit,
                   key=lambda k: self.issue_cycles_by_unit[k])
        return (unit, self.issue_cycles_by_unit[unit])


def check_plan(plan, target):
    """Fit a plan to a target, or raise naming what to shrink."""
    target.check_budget(barriers=plan.barriers,
                        accumulators=plan.accumulator_slots,
                        windows=plan.shared_windows,
                        rctx=plan.active_contexts)
    total_vector_bytes = target.phys_vregs * target.vlen * 4
    if plan.private_vector_bytes > total_vector_bytes:
        raise ResourceError(
            f"plan needs {plan.private_vector_bytes} bytes of vector registers,"
            f" target {target.name!r} has {total_vector_bytes}",
            because='there is no scratchpad to spill into; RF is the only '
                    'level above NUCA',
            remedy='a smaller tile, or fewer live stages',
            source='compiler_overview.md sec 10.1')
    return True


def try_plans(candidates, target):
    """Take the first plan that fits, recording why the others did not.

    sec 10.3 orders the fallbacks. Applying them silently would throw away
    the most useful output the compiler produces: sec 16.1 wants the rejected
    four-context plan and its reason in the report, because that is what tells
    a hardware team the accumulator file is the thing to grow.
    """
    rejected = []
    for plan in candidates:
        try:
            check_plan(plan, target)
        except ResourceError as e:
            rejected.append(RejectedPlan(
                description=f"{plan.active_contexts}-context plan",
                reason=e.message))
            continue
        plan.rejected = rejected
        return plan
    raise ResourceError(
        f"no candidate plan fits target {target.name!r}",
        because='; '.join(r.reason for r in rejected) or 'no candidates given',
        remedy='see compiler_overview.md sec 10.3 for the fallback order',
        source='compiler_overview.md sec 10.3')


def format_report(plan, target):
    """The report of sec 16.1."""
    unit, cycles = plan.predicted_issue_pressure
    pressure = f"{unit} {cycles} cycles" if unit else "not modelled"

    lines = [
        f"kernel: {plan.name}",
        f"target: {target.name}",
        "",
        f"active contexts:          {plan.active_contexts} "
        f"(of {target.num_rctx})",
        f"RF stages:                {plan.rf_stages}",
        f"shared RF windows:        {plan.shared_windows} "
        f"(of {target.num_windows})",
        f"private vector bytes:     {plan.private_vector_bytes}",
        f"accumulator slots:        {plan.accumulator_slots} "
        f"(of {target.num_accumulators})",
        f"barriers:                 {plan.barriers} "
        f"(of {target.num_barriers})",
        f"max async commands:       {plan.max_async_commands}",
        "",
        f"layout conversion bytes:  {plan.layout_conversion_bytes}",
        f"DMA command expansion:    {plan.dma_commands_source} -> "
        f"{plan.dma_commands_expanded}"
        + (f" (x{plan.dma_expansion_factor:.1f})"
           if plan.dma_commands_source else ""),
        f"predicted issue pressure: {pressure}",
        f"source-release policy:    {plan.source_release_policy}",
    ]

    if not target.has_separate_source_completion:
        lines.append("    (the target has one completion covering source and "
                     "destination, so this is the conservative bound)")

    if plan.rejected:
        lines.append("")
        for r in plan.rejected:
            lines.append(str(r))

    return '\n'.join(lines)
