from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardGameCallback(BaseCallback):
    """Loguje highest_level z epizodu do TensorBoard (wymaga Monitor + info_keywords)."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if ep is None:
                continue
            level = ep.get("highest_level")
            if level is not None:
                self.logger.record("game/highest_level", float(level))
        return True
