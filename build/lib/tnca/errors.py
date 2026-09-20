"""
Stable error codes -- doc/phase2_tasks_and_acceptance.md R2-12 OBS-02.

    "诊断给 source/IR/role/resource 位置及稳定错误码；正常用户错误无裸 traceback"

Two things follow from "stable", and both are the point of this module.

A code is part of the public interface. A caller writes `except TncaError as e:
if e.code == 'BUFFER_BUSY'` and that must keep working across versions, so a
code is never renamed or reused for a different meaning; it is deprecated and a
new one added. The message beside it is for a person and may change freely.

And an error a user can cause is not a crash. The contract asks for no bare
traceback on an ordinary user error, which means every boundary -- CLI,
Session, compile -- raises one of these with a code, a message, and where it
came from, rather than letting a KeyError out of an internal dict.
"""


class TncaError(Exception):
    """Base of every error this package raises across a public boundary."""

    code = 'INTERNAL'

    def __init__(self, message, *, code=None, where=None, detail=None,
                 remedy=None):
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.where = where          # source location, IR op, role, resource
        self.detail = detail or {}
        self.remedy = remedy

    def __str__(self):
        parts = [f"[{self.code}] {self.message}"]
        if self.where:
            parts.append(f"  at: {self.where}")
        for k in sorted(self.detail):
            parts.append(f"  {k}: {self.detail[k]}")
        if self.remedy:
            parts.append(f"  try: {self.remedy}")
        return '\n'.join(parts)

    def as_dict(self):
        return dict(code=self.code, message=self.message, where=self.where,
                    detail=self.detail, remedy=self.remedy)


# --- compilation -----------------------------------------------------------
class CompileError(TncaError):
    code = 'COMPILE_FAILED'


class UnsupportedError(TncaError):
    """A construct outside the frozen support boundary.

    Deliberately its own class. R2-04 freezes a narrow subset of general
    control flow and requires everything outside it to be refused by name; a
    caller that wants to know "is this a bug or a boundary?" needs to be able
    to tell, and a single COMPILE_FAILED for both does not let them.
    """
    code = 'UNSUPPORTED'


class SignatureError(TncaError):
    code = 'SIGNATURE_MISMATCH'


# --- artifacts and loading -------------------------------------------------
class ArtifactError(TncaError):
    code = 'ARTIFACT_INVALID'


class VersionError(TncaError):
    code = 'VERSION_UNSUPPORTED'


class ProfileMismatch(TncaError):
    code = 'PROFILE_MISMATCH'


class CapabilityError(TncaError):
    code = 'CAPABILITY_MISSING'


# --- arguments and buffers -------------------------------------------------
class ArgumentError(TncaError):
    code = 'ARGUMENT_INVALID'


class BufferError(TncaError):
    code = 'BUFFER_INVALID'


class BufferBusy(TncaError):
    code = 'BUFFER_BUSY'


class AliasError(TncaError):
    code = 'ALIAS_UNDECLARED'


# --- session, queue and execution ------------------------------------------
class QueueFull(TncaError):
    code = 'QUEUE_FULL'


class SessionBusy(TncaError):
    code = 'SESSION_BUSY'


class SessionPoisoned(TncaError):
    code = 'SESSION_POISONED'


class DeviceLost(TncaError):
    code = 'DEVICE_LOST'


class DependencyFailed(TncaError):
    code = 'DEPENDENCY_FAILED'


class AdmissionError(TncaError):
    """Refused before any role started. No RUNNING event, no side effect."""
    code = 'ADMISSION_REFUSED'


class KernelFault(TncaError):
    code = 'KERNEL_FAULT'


class Timeout(TncaError):
    """A host wait gave up. NOT a verdict on the task.

    R2-08 RT-04: "普通 host 等待超时只表示还没观察到终态，不能把任务标为成功或
    自动取消". The task keeps running; this says only that the caller stopped
    looking.
    """
    code = 'WAIT_TIMEOUT'


# Every code this package may raise, so the CLI and the schema can enumerate
# them and a test can assert that nothing raises an unlisted one.
ALL_CODES = sorted({
    cls.code for cls in (
        TncaError, CompileError, UnsupportedError, SignatureError,
        ArtifactError, VersionError, ProfileMismatch, CapabilityError,
        ArgumentError, BufferError, BufferBusy, AliasError,
        QueueFull, SessionBusy, SessionPoisoned, DeviceLost,
        DependencyFailed, AdmissionError, KernelFault, Timeout)
})
