"""
Compiler diagnostics.

compiler_overview.md sec 12 is specific about the shape these take. When the
target lacks the SFU, the answer is a capability diagnostic or an explicit
software-math lowering -- never an SFU opcode that will trap at run time. When
it lacks atomics, the answer is a diagnostic -- never a plain load and store
that silently drops the atomicity. Sec 11.1 says the same about resources:
"资源不够可以先明确报错，不能为了快速'支持'而偷偷破坏异步生命周期."

So the failure modes are values here, with enough context to act on, rather
than assertions that merely say something went wrong.
"""


class Diagnostic(Exception):
    """A refusal the user can act on."""

    def __init__(self, message, *, because=None, remedy=None, source=None):
        self.message = message
        self.because = because
        self.remedy = remedy
        self.source = source
        super().__init__(str(self))

    def __str__(self):
        out = [self.message]
        if self.because:
            out.append(f"  because: {self.because}")
        if self.remedy:
            out.append(f"  remedy:  {self.remedy}")
        if self.source:
            out.append(f"  see:     {self.source}")
        return '\n'.join(out)


class CapabilityError(Diagnostic):
    """The target cannot execute this natively, and there is no honest fallback.

    Raised instead of emitting an encoding whose CAPABILITY0 bit is clear.
    ISA sec 11: a clear bit means ILLEGAL_INSTRUCTION, not "slower by another
    route", so emitting it anyway converts a compile-time error into a runtime
    trap.
    """


class ResourceError(Diagnostic):
    """The kernel needs more of a whole-program resource than the target has.

    Barriers, accumulators, RF windows and descriptors are allocated across the
    whole program (sec 10.1). Running out is a hard error: the fallback order
    of sec 10.3 must be applied deliberately, not by quietly reusing a slot an
    async command is still reading from.
    """
