"""
Coarse Gradio approval policies for manifest-based skill script execution.
"""

from mada.core.skills.skill_approval import (
    SkillScriptApprovalDecision,
    SkillScriptApprovalRequest,
)


class GradioPolicySkillScriptApprover:
    """Coarse app-level Gradio policy approver with explicit deny/approve modes."""

    def __init__(self, mode: str = "deny"):
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"deny", "approve"}:
            raise ValueError("Gradio script approval mode must be 'deny' or 'approve'.")
        self.mode = normalized_mode

    def approve_skill_script(
        self,
        request: SkillScriptApprovalRequest,
    ) -> SkillScriptApprovalDecision:
        if self.mode == "approve":
            return SkillScriptApprovalDecision(
                approved=True,
                reason=(
                    f"Skill script '{request.script_name}' for skill '{request.skill_name}' "
                    "was approved by the Gradio policy approver."
                ),
            )

        return SkillScriptApprovalDecision(
            approved=False,
            reason=(
                f"Skill script '{request.script_name}' for skill '{request.skill_name}' "
                "was denied by the Gradio policy approver."
            ),
        )
