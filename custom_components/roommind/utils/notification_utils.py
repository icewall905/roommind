"""Notification utilities for RoomMind.

Provides throttled notification sending (mold and drying mode) to HA notify entities with
per-device home/away filtering and persistent-notification fallback.
"""

from __future__ import annotations

import logging
import time

from homeassistant.components.persistent_notification import (
    async_create,
    async_dismiss,
)
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

NOTIF_PREFIX = "roommind_mold_"
DRYING_NOTIF_PREFIX = "roommind_drying_"


class NotificationThrottler:
    """Per-key in-memory throttling to prevent notification spam."""

    def __init__(self) -> None:
        self._last_sent: dict[str, float] = {}

    def should_send(self, key: str, cooldown_seconds: float) -> bool:
        """Return True if enough time has passed since the last send for *key*."""
        now = time.time()
        last = self._last_sent.get(key, 0.0)
        return (now - last) >= cooldown_seconds

    def record_sent(self, key: str) -> None:
        """Record that a notification was just sent for *key*."""
        self._last_sent[key] = time.time()

    def clear(self, key: str) -> None:
        """Clear throttle state for *key* (allows immediate re-send)."""
        self._last_sent.pop(key, None)


def _is_person_home(hass: HomeAssistant, person_entity: str) -> bool:
    """Check if a person entity is home (fail-safe: unknown/unavailable → home)."""
    state = hass.states.get(person_entity)
    if state is None or state.state in ("unavailable", "unknown"):
        return True  # fail-safe: treat as home
    return bool(state.state == "home")


async def async_send_notification(
    hass: HomeAssistant,
    tag: str,
    targets: list[dict],
    message: str,
    title: str,
    kind: str = "mold",
) -> None:
    """Send a notification to configured targets.

    For each target dict ``{entity_id, person_entity, notify_when}``:
    - If ``notify_when == "home_only"`` and *person_entity* is set and the
      person is not home, the target is skipped.
    - Otherwise the notification is sent via ``notify.send_message``.

    If *targets* is empty, or every target was skipped or failed, a persistent
    HA notification with id *tag* is created as fallback.

    Args:
        hass: Home Assistant instance.
        tag: Deduplication tag, also the persistent notification id.
        targets: List of notification target dicts.
        message: Notification body text.
        title: Notification title.
        kind: What the notification is about, for log messages.
    """
    if not targets:
        # Fallback: persistent notification in HA sidebar
        async_create(hass, message, title=title, notification_id=tag)
        return

    sent_any = False
    for target in targets:
        entity_id = target.get("entity_id", "")
        if not entity_id:
            continue

        # Home/away filtering
        person_entity = target.get("person_entity", "")
        notify_when = target.get("notify_when", "always")
        if notify_when == "home_only" and person_entity:
            if not _is_person_home(hass, person_entity):
                _LOGGER.debug(
                    "Skipping %s notification to %s — person %s is away",
                    kind,
                    entity_id,
                    person_entity,
                )
                continue

        try:
            await hass.services.async_call(
                "notify",
                "send_message",
                {
                    "entity_id": entity_id,
                    "message": message,
                    "title": title,
                    "data": {
                        "tag": tag,
                        "group": "roommind",
                    },
                },
            )
            sent_any = True
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Failed to send %s notification to %s", kind, entity_id)

    if not sent_any:
        # All targets were skipped or failed → persistent fallback
        async_create(hass, message, title=title, notification_id=tag)


async def async_send_mold_notification(
    hass: HomeAssistant,
    area_id: str,
    area_name: str,
    targets: list[dict],
    message: str,
    title: str,
    tag_suffix: str = "risk",
) -> None:
    """Send mold notification to configured targets (see async_send_notification).

    Args:
        hass: Home Assistant instance.
        area_id: Room/area identifier (used for deduplication tags).
        area_name: Human-readable area name (for display in notifications).
        targets: List of notification target dicts.
        message: Notification body text.
        title: Notification title.
        tag_suffix: Suffix for deduplication tag (e.g. "risk", "prevention").
    """
    await async_send_notification(hass, f"{NOTIF_PREFIX}{area_id}_{tag_suffix}", targets, message, title, kind="mold")


def drying_notification_targets(settings: dict) -> list[dict]:
    """Targets for drying notifications: their own, else the mold targets (empty → persistent notification)."""
    targets: list[dict] = settings.get("drying_notification_targets") or settings.get("mold_notification_targets") or []
    return targets


async def async_send_drying_notification(
    hass: HomeAssistant,
    area_id: str,
    settings: dict,
    message: str,
    title: str,
) -> None:
    """Send a drying-mode notification for a room (tag ``roommind_drying_<area_id>``).

    One tag per room, so a newer drying message replaces the previous one on the phone.
    """
    await async_send_notification(
        hass, f"{DRYING_NOTIF_PREFIX}{area_id}", drying_notification_targets(settings), message, title, kind="drying"
    )


def dismiss_mold_notification(
    hass: HomeAssistant,
    area_id: str,
    tag_suffix: str = "risk",
) -> None:
    """Dismiss the persistent mold notification for a room."""
    notification_id = f"{NOTIF_PREFIX}{area_id}_{tag_suffix}"
    async_dismiss(hass, notification_id)
