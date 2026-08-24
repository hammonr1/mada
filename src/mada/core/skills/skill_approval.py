"""
Approval abstractions for manifest-based skill scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Tuple, Callable


@dataclass(frozen=True)
class SkillScriptApprovalRequest:
    """Approval request for one manifest-discovered skill script invocation."""

    skill_name: str
    script_name: str
    script_path: Path
    runner: str
    args: Tuple[str, ...] = ()
    timeout_seconds: int | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SkillScriptApprovalDecision:
    """Approval decision for one skill script invocation."""

    approved: bool
    reason: str = ""


class SkillScriptApprover(Protocol):
    """Interface for deciding whether one skill script may run."""

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        """Return an approval decision for one skill script invocation."""


class DenyAllSkillScriptApprover:
    """Default approver that denies all skill script execution."""

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        return SkillScriptApprovalDecision(
            approved=False,
            reason=(
                f"Skill script '{request.script_name}' for skill "
                f"'{request.skill_name}' was denied by the default approval policy."
            ),
        )


class PolicyBasedSkillScriptApprover:
    """Config-driven approver for skill-specific script approval policies."""

    def __init__(
        self,
        default_mode: str = "deny",
        skill_modes: dict[str, str] | None = None,
    ):
        self.default_mode = self._normalize_mode(default_mode) or "deny"
        self.skill_modes = skill_modes or {}

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        candidates = (
            f"{request.skill_name}:{request.script_name}",
            request.skill_name,
            "*",
        )

        mode = None
        for key in candidates:
            normalized = self._normalize_mode(self.skill_modes.get(key))
            if normalized is not None:
                mode = normalized
                break

        if mode is None:
            mode = self.default_mode

        if mode == "approve":
            return SkillScriptApprovalDecision(
                approved=True,
                reason=(
                    f"Skill script '{request.script_name}' for skill "
                    f"'{request.skill_name}' was approved by policy."
                ),
            )

        return SkillScriptApprovalDecision(
            approved=False,
            reason=(
                f"Skill script '{request.script_name}' for skill "
                f"'{request.skill_name}' was denied by policy."
            ),
        )

    @staticmethod
    def _normalize_mode(mode: str | None) -> str | None:
        if mode is None:
            return None
        normalized = str(mode).strip().lower()
        if normalized in {"approve", "deny"}:
            return normalized
        return None

class PromptingSkillScriptApprover:
    """Interactive approver that asks a user to authorize each skill script."""

    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ):
        self.input_func = input_func
        self.output_func = output_func

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        args_display = " ".join(request.args) if request.args else "(none)"
        self.output_func("")
        self.output_func("Skill script approval required:")
        self.output_func(f"  Skill: {request.skill_name}")
        self.output_func(f"  Script: {request.script_name}")
        self.output_func(f"  Args: {args_display}")
        self.output_func(f"  Path: {Path(request.script_path)}")

        response = self.input_func("Approve script execution? [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            return SkillScriptApprovalDecision(
                approved=True,
                reason="Skill script was approved by the user.",
            )
        return SkillScriptApprovalDecision(
            approved=False,
            reason="Skill script was denied by the user.",
        )

def build_skill_script_approver(
    config: Any,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> SkillScriptApprover:
    """
    Build the skill script approver described by runtime configuration.

    A default mode of "prompt" selects an interactive approver that asks the
    user to authorize each script. Any other mode selects a policy approver
    driven by `skill_script_approval_modes`.

    Args:
        config: Skill runtime configuration.
        input_func: Callable used to read a user's approval response.
        output_func: Callable used to display approval prompts.

    Returns:
        An approver implementing the `SkillScriptApprover` protocol.
    """
    default_mode = (
        str(getattr(config, "default_skill_script_approval_mode", "prompt"))
        .strip()
        .lower()
    )
    skill_modes = dict(getattr(config, "skill_script_approval_modes", {}) or {})

    if default_mode == "prompt":
        return PromptingSkillScriptApprover(
            input_func=input_func,
            output_func=output_func,
        )

    return PolicyBasedSkillScriptApprover(
        default_mode=default_mode,
        skill_modes=skill_modes,
    )