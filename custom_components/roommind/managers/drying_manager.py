"""Drying mode (laundry dried indoors) manager for RoomMind.

A drying run is an override of type ``drying``: the room heats to a high
target while this manager watches the absolute humidity of the room air.

- Moisture builds up → suggest a short airing (the open window pauses heating
  through the existing window handling).
- The air stops taking up water (absolute humidity flat and back near where the
  run started) → the run ends and the user is told to air the room out.
- The override's end time is the backstop; the run then ends by time limit.

Thermostats that cap their setpoint below the drying target and regulate on
their own sensor (Tado) get a temporary negative temperature offset for the
run, restored when it ends. Only one call per device at the start and one at
the end: Tado is a cloud API with tight rate limits.

Times are wall-clock (``time.time()``): run state survives a restart, so the
saved offsets can be restored after a crash.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from ..const import (
    DOMAIN,
    DRYING_AH_EMA_ALPHA,
    DRYING_AH_HISTORY_S,
    DRYING_DONE_AH_MARGIN,
    DRYING_DONE_FLAT_WINDOW_S,
    DRYING_DONE_MIN_RUNTIME_S,
    DRYING_DONE_RISE_TOLERANCE,
    DRYING_OFFSET_MARGIN,
    DRYING_OFFSET_MAX,
    DRYING_OFFSET_PLATFORMS,
    DRYING_OFFSET_STALE_S,
    DRYING_OFFSET_STEP,
    DRYING_OUTDOOR_RH_FALLBACK,
    DRYING_RESTORE_MAX_ATTEMPTS,
    DRYING_RESTORE_RETRY_S,
    DRYING_VENT_CLOSE_AFTER_S,
    DRYING_VENT_CLOSE_EXCHANGE,
    DRYING_VENT_CLOSE_MIN_OPEN_S,
    DRYING_VENT_MAX_OPEN_S,
    DRYING_VENT_MIN_INTERVAL_S,
    DRYING_VENT_OPEN_DELTA_AH,
    DRYING_VENT_OPEN_RH,
    DRYING_VENT_SUPPRESS_RH_DROP,
    DRYING_VENT_TEMP_DROP_K,
    OVERRIDE_DRYING,
    is_override_active,
    make_roommind_context,
)
from ..utils.device_utils import get_trv_eids
from ..utils.mold_utils import absolute_humidity
from ..utils.notification_utils import async_send_drying_notification
from ..utils.temp_utils import ha_temp_to_celsius

_LOGGER = logging.getLogger(__name__)

OFFSET_SERVICE_DOMAIN = "tado"
OFFSET_SERVICE = "set_climate_temperature_offset"
NOTIFICATION_TITLE = "RoomMind: Drying"

END_DONE = "done"
END_EXPIRED = "expired"
END_CLEARED = "cleared"


def is_drying_active(room: dict) -> bool:
    """Return True when the room holds an active drying override."""
    return room.get("override_type") == OVERRIDE_DRYING and is_override_active(room)


@dataclass
class DryingRun:
    """Runtime state of one room's drying run. Persisted except the sample history."""

    started_at: float
    until: float | None
    target: float | None
    baseline_ah: float | None = None
    ah_ema: float | None = None
    humidity: float | None = None
    last_vent_prompt_at: float | None = None
    vent_pending: bool = False  # open prompt sent, window not opened since
    window_closed_since: float | None = None
    window_open_since: float | None = None
    ah_at_open: float | None = None  # raw AH when the window opened
    temp_at_open: float | None = None
    close_prompted: bool = False  # close prompt sent for the current opening
    close_reminded: bool = False
    peak_rh: float | None = None
    saved_offsets: dict[str, float] = field(default_factory=dict)
    applied_offsets: dict[str, float] = field(default_factory=dict)
    samples: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PendingRestore:
    """An offset restore that failed and is retried."""

    offset: float
    attempts: int
    next_at: float


@dataclass
class OffsetRecord:
    """The last boost of a device: what it replaced, what was set, when the original went back.

    The integration's offset attribute lags behind (debounced cloud refresh), so a run
    started soon after a restore, or reading back its own boost, takes the original from here.
    """

    original: float
    applied: float
    restored_at: float | None = None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _offset_dict(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for eid, raw in value.items():
        num = _float_or_none(raw)
        if isinstance(eid, str) and num is not None:
            out[eid] = num
    return out


def boost_offset(target: float, max_temp: float) -> float:
    """The (negative) offset that lets a device capped at *max_temp* heat a room to *target*."""
    boost = min(DRYING_OFFSET_MAX, target - max_temp + DRYING_OFFSET_MARGIN)
    return -round(boost / DRYING_OFFSET_STEP) * DRYING_OFFSET_STEP


class DryingManager:
    """Runs drying mode per room: humidity tracking, airing prompts, end detection, offset boost."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._runs: dict[str, DryingRun] = {}
        self._pending_restores: dict[str, PendingRestore] = {}
        self._offset_records: dict[str, OffsetRecord] = {}
        self.state_dirty = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_state(self, data: dict | None) -> None:
        """Restore persisted runs and pending restores. Tolerates corrupt entries."""
        self._runs = {}
        self._pending_restores = {}
        self._offset_records = {}
        data = data if isinstance(data, dict) else {}
        runs = data.get("runs")
        for area_id, raw in (runs if isinstance(runs, dict) else {}).items():
            started_at = _float_or_none(raw.get("started_at")) if isinstance(raw, dict) else None
            if started_at is None:
                _LOGGER.debug("Drying: ignoring malformed persisted run for '%s'", area_id)
                continue
            self._runs[area_id] = DryingRun(
                started_at=started_at,
                until=_float_or_none(raw.get("until")),
                target=_float_or_none(raw.get("target")),
                baseline_ah=_float_or_none(raw.get("baseline_ah")),
                last_vent_prompt_at=_float_or_none(raw.get("last_vent_prompt_at")),
                vent_pending=raw.get("vent_pending") is True,
                window_closed_since=_float_or_none(raw.get("window_closed_since")),
                window_open_since=_float_or_none(raw.get("window_open_since")),
                ah_at_open=_float_or_none(raw.get("ah_at_open")),
                temp_at_open=_float_or_none(raw.get("temp_at_open")),
                close_prompted=raw.get("close_prompted") is True,
                close_reminded=raw.get("close_reminded") is True,
                peak_rh=_float_or_none(raw.get("peak_rh")),
                saved_offsets=_offset_dict(raw.get("saved_offsets")),
                applied_offsets=_offset_dict(raw.get("applied_offsets")),
            )
        pending = data.get("pending_restores")
        for eid, raw in (pending if isinstance(pending, dict) else {}).items():
            offset = _float_or_none(raw.get("offset")) if isinstance(raw, dict) else None
            if offset is None:
                continue
            self._pending_restores[eid] = PendingRestore(
                offset=offset,
                attempts=int(_float_or_none(raw.get("attempts")) or 0),
                next_at=_float_or_none(raw.get("next_at")) or 0.0,
            )
        records = data.get("offset_records")
        for eid, raw in (records if isinstance(records, dict) else {}).items():
            original = _float_or_none(raw.get("original")) if isinstance(raw, dict) else None
            applied = _float_or_none(raw.get("applied")) if isinstance(raw, dict) else None
            if original is None or applied is None:
                continue
            self._offset_records[eid] = OffsetRecord(original, applied, _float_or_none(raw.get("restored_at")))

    def get_state(self) -> dict:
        """Serialise state for the store."""
        return {
            "runs": {
                area_id: {
                    "started_at": run.started_at,
                    "until": run.until,
                    "target": run.target,
                    "baseline_ah": run.baseline_ah,
                    "last_vent_prompt_at": run.last_vent_prompt_at,
                    "vent_pending": run.vent_pending,
                    "window_closed_since": run.window_closed_since,
                    "window_open_since": run.window_open_since,
                    "ah_at_open": run.ah_at_open,
                    "temp_at_open": run.temp_at_open,
                    "close_prompted": run.close_prompted,
                    "close_reminded": run.close_reminded,
                    "peak_rh": run.peak_rh,
                    "saved_offsets": dict(run.saved_offsets),
                    "applied_offsets": dict(run.applied_offsets),
                }
                for area_id, run in self._runs.items()
            },
            "pending_restores": {
                eid: {"offset": p.offset, "attempts": p.attempts, "next_at": p.next_at}
                for eid, p in self._pending_restores.items()
            },
            "offset_records": {
                eid: {"original": r.original, "applied": r.applied, "restored_at": r.restored_at}
                for eid, r in self._offset_records.items()
            },
        }

    async def _async_persist(self) -> None:
        """Save the state right away (before a cloud call whose effect must survive a crash)."""
        try:
            await self.hass.data[DOMAIN]["store"].async_save_settings({"drying_state": self.get_state()})
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Drying: saving the drying state failed", exc_info=True)
            return
        self.state_dirty = False

    def run_for(self, area_id: str) -> DryingRun | None:
        """Return the run of a room, or None. For tests and diagnostics."""
        return self._runs.get(area_id)

    # ------------------------------------------------------------------
    # Per-cycle evaluation
    # ------------------------------------------------------------------

    async def async_evaluate(
        self,
        area_id: str,
        area_name: str,
        room: dict,
        current_temp: float | None,
        current_humidity: float | None,
        window_open: bool | None,
        settings: dict,
        now: float | None = None,
        *,
        offset_boost_allowed: bool = True,
        control_active: bool = True,
        outdoor_temp: float | None = None,
        outdoor_humidity: float | None = None,
    ) -> dict | None:
        """Advance the room's drying run by one cycle.

        *window_open* None means the window state is not known yet: no airing prompts then.

        *offset_boost_allowed* is False for rooms that read their temperature from the
        thermostat itself: an offset would falsify that reading. A room whose climate
        control is off ends its run (override cleared, offsets restored).

        Returns the live ``drying`` dict, or None when no run is active.
        """
        now = time.time() if now is None else now
        run = self._runs.get(area_id)

        if is_drying_active(room) and not control_active:
            _LOGGER.info("Room '%s': climate control is off, ending drying", area_id)
            await self._async_clear_override(area_id)
            await self._async_finish(area_id, area_name, settings, END_CLEARED)
            return None

        if not is_drying_active(room):
            if run is not None:
                expired = room.get("override_type") == OVERRIDE_DRYING and room.get("override_until") is not None
                await self._async_finish(area_id, area_name, settings, END_EXPIRED if expired else END_CLEARED)
            return None

        if run is None:
            run = DryingRun(started_at=now, until=room.get("override_until"), target=room.get("override_heat"))
            self._runs[area_id] = run
            self.state_dirty = True
            if offset_boost_allowed:
                await self._async_apply_offsets(area_id, room, run, now)
                if self._runs.get(area_id) is not run:
                    return None  # ended (room removed or paused) while a cloud call was running
        elif run.until != room.get("override_until") or run.target != room.get("override_heat"):
            # Restarted with other settings: same run, same offsets (no extra cloud calls)
            run.until = room.get("override_until")
            run.target = room.get("override_heat")
            self.state_dirty = True

        ah_raw = (
            absolute_humidity(current_temp, current_humidity)
            if current_temp is not None and current_humidity is not None
            else None
        )
        ah_out = (
            absolute_humidity(
                outdoor_temp, outdoor_humidity if outdoor_humidity is not None else DRYING_OUTDOOR_RH_FALLBACK
            )
            if outdoor_temp is not None
            else None
        )

        if window_open is not None:
            self._track_window(run, window_open, now, current_temp, ah_raw)
            if window_open:
                await self._async_close_prompt(area_id, area_name, settings, run, now, current_temp, ah_raw, ah_out)

        run.humidity = current_humidity
        if ah_raw is None or current_humidity is None:
            return self._live(run)

        run.ah_ema = ah_raw if run.ah_ema is None else run.ah_ema + DRYING_AH_EMA_ALPHA * (ah_raw - run.ah_ema)
        run.samples.append((now, run.ah_ema))
        run.samples = [s for s in run.samples if s[0] >= now - DRYING_AH_HISTORY_S]
        run.peak_rh = current_humidity if run.peak_rh is None else max(run.peak_rh, current_humidity)
        if run.baseline_ah is None:
            run.baseline_ah = ah_raw
            self.state_dirty = True

        if self._is_done(run, now):
            await self._async_clear_override(area_id)
            await self._async_finish(area_id, area_name, settings, END_DONE)
            return None

        if window_open is False and self._open_pays_off(run, now, current_humidity, ah_out):
            run.last_vent_prompt_at = now
            run.vent_pending = True
            self.state_dirty = True
            await async_send_drying_notification(
                self.hass,
                area_id,
                settings,
                message=(
                    f"Open the {area_name} window for a few minutes to let the moisture out "
                    f"({current_humidity:.0f}% humidity)."
                ),
                title=NOTIFICATION_TITLE,
            )

        return self._live(run)

    def _track_window(
        self, run: DryingRun, window_open: bool, now: float, temp: float | None, ah_raw: float | None
    ) -> None:
        """Note openings and closings of the window."""
        if window_open:
            run.window_closed_since = None
            if run.window_open_since is None:
                run.window_open_since = now
                run.ah_at_open = ah_raw
                run.temp_at_open = temp
                run.close_prompted = False
                run.close_reminded = False
                run.vent_pending = False
                self.state_dirty = True
            return
        if run.window_open_since is not None:
            run.window_open_since = None
            run.ah_at_open = None
            run.temp_at_open = None
            run.close_prompted = False
            run.close_reminded = False
            self.state_dirty = True
        if run.window_closed_since is None:
            run.window_closed_since = now
            self.state_dirty = True

    async def _async_close_prompt(
        self,
        area_id: str,
        area_name: str,
        settings: dict,
        run: DryingRun,
        now: float,
        temp: float | None,
        ah_raw: float | None,
        ah_out: float | None,
    ) -> None:
        """Ask to close an open window once the air exchange is good enough; remind once if it stays open."""
        open_for = now - (run.window_open_since or now)
        if not run.close_prompted:
            exchanged = (
                open_for >= DRYING_VENT_CLOSE_MIN_OPEN_S
                and ah_out is not None
                and ah_raw is not None
                and run.ah_at_open is not None
                and ah_raw <= ah_out + DRYING_VENT_CLOSE_EXCHANGE * (run.ah_at_open - ah_out)
            )
            cooled = (
                temp is not None and run.temp_at_open is not None and run.temp_at_open - temp >= DRYING_VENT_TEMP_DROP_K
            )
            if not (open_for >= DRYING_VENT_CLOSE_AFTER_S or exchanged or cooled):
                return
            run.close_prompted = True
            message = f"Close the {area_name} window — heating resumes."
        elif not run.close_reminded and open_for >= DRYING_VENT_MAX_OPEN_S:
            run.close_reminded = True
            message = f"The {area_name} window is still open — close it so heating resumes."
        else:
            return
        self.state_dirty = True
        await async_send_drying_notification(self.hass, area_id, settings, message=message, title=NOTIFICATION_TITLE)

    @staticmethod
    def _ah_flat(run: DryingRun, now: float) -> bool:
        """AH has not risen over the last flat window, with the window closed all that time."""
        if run.ah_ema is None:
            return False
        if run.window_closed_since is None or now - run.window_closed_since < DRYING_DONE_FLAT_WINDOW_S:
            return False
        older = [ema for ts, ema in run.samples if ts <= now - DRYING_DONE_FLAT_WINDOW_S]
        return bool(older) and run.ah_ema - older[-1] <= DRYING_DONE_RISE_TOLERANCE

    def _open_pays_off(self, run: DryingRun, now: float, rh: float, ah_out: float | None) -> bool:
        """Airing is worth it: closed long enough, humid, more water inside than outside, laundry still wet."""
        if run.ah_ema is None or run.baseline_ah is None or run.window_closed_since is None:
            return False
        if now - run.window_closed_since < DRYING_VENT_MIN_INTERVAL_S:
            return False
        if run.last_vent_prompt_at is not None and now - run.last_vent_prompt_at < DRYING_VENT_MIN_INTERVAL_S:
            return False  # an ignored prompt is repeated at this pace at most
        if rh < DRYING_VENT_OPEN_RH:
            return False
        reference = ah_out if ah_out is not None else run.baseline_ah
        if run.ah_ema - reference < DRYING_VENT_OPEN_DELTA_AH:
            return False
        # Laundry nearly dry: AH no longer rising and RH clearly past its peak
        peak = run.peak_rh if run.peak_rh is not None else rh
        return not (self._ah_flat(run, now) and peak - rh >= DRYING_VENT_SUPPRESS_RH_DROP)

    @classmethod
    def _is_done(cls, run: DryingRun, now: float) -> bool:
        """The laundry stopped releasing water: AH flat for a while with the window closed, back near the start."""
        if run.ah_ema is None or run.baseline_ah is None:
            return False
        if now - run.started_at < DRYING_DONE_MIN_RUNTIME_S:
            return False
        # An airing lowers the humidity for a while whether the laundry is dry or not (_ah_flat needs it closed)
        return cls._ah_flat(run, now) and run.ah_ema <= run.baseline_ah + DRYING_DONE_AH_MARGIN

    @staticmethod
    def _live(run: DryingRun) -> dict:
        return {
            "started_at": run.started_at,
            "until": run.until,
            "target": run.target,
            "humidity": run.humidity,
            "abs_humidity": round(run.ah_ema, 2) if run.ah_ema is not None else None,
            "baseline_abs_humidity": round(run.baseline_ah, 2) if run.baseline_ah is not None else None,
            "vent_suggested": run.vent_pending and run.window_open_since is None,
            "close_suggested": run.close_prompted and run.window_open_since is not None,
            "window_open_since": run.window_open_since,
            "last_vent_prompt_at": run.last_vent_prompt_at,
            "offset_boost": dict(run.applied_offsets),
        }

    # ------------------------------------------------------------------
    # Ending a run
    # ------------------------------------------------------------------

    async def _async_clear_override(self, area_id: str) -> None:
        try:
            await self.hass.data[DOMAIN]["store"].async_update_room(
                area_id,
                {"override_heat": None, "override_cool": None, "override_until": None, "override_type": None},
            )
        except KeyError:
            pass  # room deleted meanwhile

    async def _async_finish(self, area_id: str, area_name: str, settings: dict, reason: str) -> None:
        run = self._runs.pop(area_id, None)
        if run is None:
            return
        self.state_dirty = True
        await self._async_restore_offsets(run.saved_offsets)
        _LOGGER.info("Room '%s': drying ended (%s)", area_id, reason)
        if reason == END_DONE:
            message = f"Laundry in {area_name} looks dry — air the room out."
        elif reason == END_EXPIRED:
            message = f"Drying in {area_name} ended at its time limit — air the room out."
        else:
            return
        await async_send_drying_notification(self.hass, area_id, settings, message=message, title=NOTIFICATION_TITLE)

    async def async_end(self, area_id: str) -> None:
        """End a room's run without a notification (room removed or paused): restore the offsets."""
        await self._async_finish(area_id, area_id, {}, END_CLEARED)

    async def async_prune(self, area_ids: set[str]) -> None:
        """End runs of rooms no longer evaluated (deleted, resting or turned outdoor)."""
        for area_id in [a for a in self._runs if a not in area_ids]:
            await self.async_end(area_id)

    # ------------------------------------------------------------------
    # Offset boost
    # ------------------------------------------------------------------

    def _original_offset(self, eid: str, attr: object, now: float) -> float | None:
        """The offset to put back after the run; None when it cannot be known (no boost then)."""
        pending = self._pending_restores.pop(eid, None)
        if pending is not None:
            # Its earlier restore never went through: that value is still the original
            self.state_dirty = True
            return pending.offset
        value = _float_or_none(attr)
        record = self._offset_records.get(eid)
        if record is not None and (
            record.restored_at is None
            or now - record.restored_at < DRYING_OFFSET_STALE_S
            or (value is not None and value == record.applied)
        ):
            # The attribute may still show the boost: the last known original is the better guess
            return record.original
        return value

    async def _async_apply_offsets(self, area_id: str, room: dict, run: DryingRun, now: float) -> None:
        """Give capped offset-capable thermostats a negative offset so they heat past their cap.

        The original is recorded and saved before the call: restoring it is harmless if the
        call failed, while a boost that went through unrecorded would stay for good.
        """
        target = run.target
        if target is None:
            return
        registry = er.async_get(self.hass)
        devices = room.get("devices", [])
        for eid in get_trv_eids(devices):
            device: dict = next((d for d in devices if d.get("entity_id") == eid), {})
            if device.get("drying_offset_boost", True) is False:
                continue
            entry = registry.async_get(eid)
            if entry is None or entry.platform not in DRYING_OFFSET_PLATFORMS:
                continue
            state = self.hass.states.get(eid)
            max_temp = _float_or_none(state.attributes.get("max_temp")) if state else None
            if max_temp is None:
                continue
            max_c = ha_temp_to_celsius(self.hass, max_temp)
            if max_c >= target:
                continue
            offset = boost_offset(target, max_c)
            original = self._original_offset(eid, state.attributes.get("offset_celsius") if state else None, now)
            if original is None:
                _LOGGER.warning("Drying: '%s' reports no temperature offset, not boosting it", eid)
                continue
            run.saved_offsets[eid] = original
            self._offset_records[eid] = OffsetRecord(original, offset)
            await self._async_persist()
            ok = await self._async_set_offset(eid, offset)
            if self._runs.get(area_id) is not run:
                # The run ended during the call, maybe restoring before the boost landed
                await self._async_restore_offsets({eid: original})
                return
            if ok:
                run.applied_offsets[eid] = offset
                _LOGGER.info(
                    "Room '%s': drying offset %.1f on '%s' (cap %.1f, was %.1f)", area_id, offset, eid, max_c, original
                )

    async def _async_set_offset(self, entity_id: str, offset: float) -> bool:
        try:
            await self.hass.services.async_call(
                OFFSET_SERVICE_DOMAIN,
                OFFSET_SERVICE,
                {"entity_id": entity_id, "offset": offset},
                blocking=True,
                context=make_roommind_context(),
            )
        except Exception:  # noqa: BLE001
            _LOGGER.warning("Drying: setting the temperature offset of '%s' to %.1f failed", entity_id, offset)
            return False
        return True

    async def _async_restore_offsets(self, saved: dict[str, float]) -> None:
        for eid, original in saved.items():
            if await self._async_set_offset(eid, original):
                self._note_restored(eid)
            else:
                self._pending_restores[eid] = PendingRestore(
                    offset=original, attempts=1, next_at=time.time() + DRYING_RESTORE_RETRY_S
                )
                self.state_dirty = True

    def _note_restored(self, eid: str) -> None:
        record = self._offset_records.get(eid)
        if record is not None:
            record.restored_at = time.time()
            self.state_dirty = True

    async def async_retry_restores(self, now: float | None = None) -> None:
        """Retry offset restores that failed, a few times before giving up."""
        now = time.time() if now is None else now
        for eid, pending in list(self._pending_restores.items()):
            if now < pending.next_at:
                continue
            self.state_dirty = True
            if await self._async_set_offset(eid, pending.offset):
                del self._pending_restores[eid]
                self._note_restored(eid)
                continue
            pending.attempts += 1
            pending.next_at = now + DRYING_RESTORE_RETRY_S
            if pending.attempts >= DRYING_RESTORE_MAX_ATTEMPTS:
                del self._pending_restores[eid]
                _LOGGER.error(
                    "Drying: could not restore the temperature offset of '%s' to %.1f; set it back by hand",
                    eid,
                    pending.offset,
                )
