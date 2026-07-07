"""
Approval abstractions for manifest-based skill scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple
from typing import Any, Mapping, Protocol, Tuple

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
class SkillScriptApprovalRequest:
      """Approval request for one manifest-discovered skill script
      invocation."""

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
      """Config-driven approver for skill-specific script approval
      policies."""

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
                      f"Skill script '{request.script_name}' for skill"
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
