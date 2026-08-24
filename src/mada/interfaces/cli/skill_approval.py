"""
CLI approval adapter for manifest-based skill script execution.
"""

from pathlib import Path
from typing import Callable

from mada.core.skills.skill_approval import (
    SkillScriptApprovalDecision,
    SkillScriptApprovalRequest,
)


class CLISkillScriptApprover:
    """Synchronous terminal approver for manifest skill script execution."""

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
                reason="Skill script was approved by the CLI user.",
            )
        return SkillScriptApprovalDecision(
            approved=False,
            reason="Skill script was denied by the CLI user.",
        )
