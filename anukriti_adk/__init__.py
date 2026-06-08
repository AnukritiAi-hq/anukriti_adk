"""anukriti_adk — Track-2 (Optimize) ADK + Gemini PGx safety agent.

ADK's `adk web` / `adk eval` discover `root_agent` from the package's `agent`
module. We re-export the OPTIMIZED agent as the package default.
"""

from .agent import root_agent

__all__ = ["root_agent"]
