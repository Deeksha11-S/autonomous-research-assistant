from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class SummaryMemory:
    def __init__(self, max_entries: int = 100):
        self.memory = []
        self.max_entries = max_entries
        self.current_summary = ""

    def add(self, agent: str, action: str, result: Dict[str, Any],
            confidence: float = 0.5) -> None:
        """Add an entry to memory"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "result": result,
            "confidence": confidence
        }

        self.memory.append(entry)

        # Keep memory size limited
        if len(self.memory) > self.max_entries:
            self.memory = self.memory[-self.max_entries:]

        # Update summary
        self._update_summary()

    def _update_summary(self) -> None:
        """Update the summary of recent activities"""
        if not self.memory:
            self.current_summary = "No activities recorded."
            return

        # Get recent entries (last 10)
        recent_entries = self.memory[-10:]

        summary_parts = []
        for entry in recent_entries:
            time_str = datetime.fromisoformat(entry["timestamp"]).strftime("%H:%M")
            summary_parts.append(
                f"[{time_str}] {entry['agent']}: {entry['action'][:50]}..."
            )

        self.current_summary = "\n".join(summary_parts)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent memory entries"""
        return self.memory[-n:] if self.memory else []

    def get_by_agent(self, agent: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get entries by specific agent"""
        agent_entries = [e for e in self.memory if e["agent"] == agent]
        return agent_entries[-limit:] if agent_entries else []

    def get_by_time(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get entries from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            e for e in self.memory
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]

    def get_summary(self) -> str:
        """Get current summary"""
        return self.current_summary

    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each agent"""
        if not self.memory:
            return {}

        agent_stats = {}

        for entry in self.memory:
            agent = entry["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "count": 0,
                    "total_confidence": 0,
                    "actions": set(),
                    "last_active": entry["timestamp"]
                }

            stats = agent_stats[agent]
            stats["count"] += 1
            stats["total_confidence"] += entry.get("confidence", 0.5)
            stats["actions"].add(entry["action"])
            stats["last_active"] = max(stats["last_active"], entry["timestamp"])

        # Calculate averages and format
        for agent, stats in agent_stats.items():
            stats["avg_confidence"] = stats["total_confidence"] / stats["count"]
            stats["unique_actions"] = len(stats["actions"])
            stats["actions"] = list(stats["actions"])[:5]  # Top 5 actions
            del stats["total_confidence"]

        return agent_stats

    def clear(self) -> None:
        """Clear all memory"""
        self.memory = []
        self.current_summary = ""

    def save_to_file(self, filepath: str) -> bool:
        """Save memory to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "memory": self.memory,
                    "summary": self.current_summary,
                    "saved_at": datetime.now().isoformat()
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load memory from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.memory = data.get("memory", [])
                self.current_summary = data.get("summary", "")

            # Ensure memory size limit
            if len(self.memory) > self.max_entries:
                self.memory = self.memory[-self.max_entries:]

            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False

    def export_for_llm(self) -> str:
        """Export memory in format suitable for LLM context"""
        if not self.memory:
            return "No memory available."

        # Get recent entries and format for LLM
        recent = self.get_recent(20)

        lines = ["# Research Memory Summary", ""]

        for entry in recent:
            time_ago = self._time_ago(entry["timestamp"])
            lines.append(f"## {entry['agent']} ({time_ago} ago)")
            lines.append(f"**Action:** {entry['action']}")

            # Include key results
            result = entry.get("result", {})
            if isinstance(result, dict):
                for key, value in list(result.items())[:2]:  # First 2 items
                    if isinstance(value, (str, int, float)):
                        lines.append(f"- {key}: {value}")
                    elif isinstance(value, list):
                        lines.append(f"- {key}: {len(value)} items")

            lines.append(f"*Confidence: {entry.get('confidence', 0.5):.0%}*")
            lines.append("")

        return "\n".join(lines)

    def _time_ago(self, timestamp: str) -> str:
        """Calculate human-readable time difference"""
        then = datetime.fromisoformat(timestamp)
        now = datetime.now()
        diff = now - then

        if diff.days > 0:
            return f"{diff.days}d"
        elif diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h"
        elif diff.seconds >= 60:
            return f"{diff.seconds // 60}m"
        else:
            return f"{diff.seconds}s"