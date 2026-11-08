"""Tests for drying_manager.py — drying mode runs, airing prompts, end detection and offset boost."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.roommind.const import (
    DOMAIN,
    DRYING_DONE_MIN_RUNTIME_S,
    DRYING_RESTORE_MAX_ATTEMPTS,
    DRYING_RESTORE_RETRY_S,
    DRYING_VENT_PROMPT_COOLDOWN_S,
)
from custom_components.roommind.managers.drying_manager import DryingManager, boost_offset, is_drying_active
from custom_components.roommind.utils.mold_utils import absolute_humidity

T0 = 1_800_000_000.0
TADO = "climate.tado_bedroom"
OTHER = "climate.zigbee_bedroom"
NOTIFY = "custom_components.roommind.managers.drying_manager.async_send_drying_notification"


def _room(**overrides):
    room = {
        "area_id": "bedroom",
        "climate_mode": "heat_only",
        "devices": [{"entity_id": TADO, "type": "trv"}],
        "override_heat": 30.0,
        "override_cool": None,
        "override_until": T0 + 8 * 3600,
        "override_type": "drying",
    }
    room.update(overrides)
    return room


def _state(max_temp=25.0, offset=None):
    st = MagicMock()
    st.attributes = {"max_temp": max_temp}
    if offset is not None:
        st.attributes["offset_celsius"] = offset
    return st


@pytest.fixture
def env(hass):
    states = {TADO: _state(offset=-0.5), OTHER: _state()}
    hass.states.get = states.get
    hass.services.async_call = AsyncMock()
    hass.config.units.temperature_unit = "°C"
    store = MagicMock()
    store.async_update_room = AsyncMock()
    store.async_save_settings = AsyncMock()
    hass.data[DOMAIN] = {"store": store}
    platforms = {TADO: "tado", OTHER: "zha"}
    registry = MagicMock()
    registry.async_get = lambda eid: MagicMock(platform=platforms[eid]) if eid in platforms else None
    with (
        patch("custom_components.roommind.managers.drying_manager.er.async_get", return_value=registry),
        patch("custom_components.roommind.utils.temp_utils._is_fahrenheit", return_value=False),
    ):
        yield hass, states, store


def _offset_calls(hass):
    return [
        (c.args[2]["entity_id"], c.args[2]["offset"])
        for c in hass.services.async_call.call_args_list
        if c.args[:2] == ("tado", "set_climate_temperature_offset")
    ]


async def _eval(dm, room, temp=24.0, rh=55.0, window_open=False, now=T0, settings=None):
    return await dm.async_evaluate(
        "bedroom", "Bedroom", room, temp, rh, window_open, settings if settings is not None else {}, now=now
    )


def test_boost_offset_formula():
    assert boost_offset(30.0, 25.0) == -7.0
    assert boost_offset(35.0, 20.0) == -10.0  # capped
    assert boost_offset(27.3, 25.0) == -4.5  # rounded to 0.5


def test_is_drying_active():
    assert is_drying_active(_room(override_until=None))
    assert not is_drying_active(_room(override_type="custom"))
    assert not is_drying_active(_room(override_heat=None))


@pytest.mark.asyncio
async def test_inactive_room_returns_none(env):
    hass, *_ = env
    dm = DryingManager(hass)
    assert await _eval(dm, _room(override_type=None, override_heat=None)) is None
    hass.services.async_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_applies_tado_offset_once_and_reports_live(env):
    hass, *_ = env
    dm = DryingManager(hass)
    live = await _eval(dm, _room())
    assert _offset_calls(hass) == [(TADO, -7.0)]
    assert dm.state_dirty
    assert live == {
        "started_at": T0,
        "until": T0 + 8 * 3600,
        "target": 30.0,
        "humidity": 55.0,
        "abs_humidity": round(absolute_humidity(24.0, 55.0), 2),
        "baseline_abs_humidity": round(absolute_humidity(24.0, 55.0), 2),
        "vent_suggested": False,
        "last_vent_prompt_at": None,
        "offset_boost": {TADO: -7.0},
    }
    assert dm.run_for("bedroom").saved_offsets == {TADO: -0.5}
    # Later cycles send nothing more
    await _eval(dm, _room(), now=T0 + 30)
    assert len(_offset_calls(hass)) == 1


@pytest.mark.asyncio
async def test_offset_skipped_for_other_platforms_opt_out_and_uncapped(env):
    hass, states, _ = env
    dm = DryingManager(hass)
    devices = [
        {"entity_id": TADO, "type": "trv", "drying_offset_boost": False},
        {"entity_id": OTHER, "type": "trv"},
        {"entity_id": "climate.unknown", "type": "trv"},
    ]
    await _eval(dm, _room(devices=devices))
    assert _offset_calls(hass) == []

    dm2 = DryingManager(hass)
    states[TADO] = _state(max_temp=35.0)
    await _eval(dm2, _room())
    assert _offset_calls(hass) == []

    dm3 = DryingManager(hass)
    states[TADO] = _state(max_temp=None)
    await _eval(dm3, _room())
    assert _offset_calls(hass) == []


@pytest.mark.asyncio
async def test_missing_offset_attribute_skips_boost(env):
    """Without a readable original the boost could not be undone correctly: no boost."""
    hass, states, _ = env
    dm = DryingManager(hass)
    for attr in (None, "unknown"):
        states[TADO] = _state(offset=attr)
        dm = DryingManager(hass)
        live = await _eval(dm, _room())
        assert live["offset_boost"] == {}
        await _eval(dm, _room(override_type=None, override_heat=None, override_until=None), now=T0 + 60)
    assert _offset_calls(hass) == []


@pytest.mark.asyncio
async def test_set_failure_keeps_drying_without_boost(env):
    hass, *_ = env
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("cloud down"))
    dm = DryingManager(hass)
    live = await _eval(dm, _room())
    assert live is not None
    assert live["offset_boost"] == {}
    # The call may have reached the device: the original is restored anyway
    assert dm.run_for("bedroom").saved_offsets == {TADO: -0.5}


@pytest.mark.asyncio
async def test_clear_restores_without_notification(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        live = await _eval(dm, _room(override_type=None, override_heat=None, override_until=None), now=T0 + 60)
    assert live is None
    assert _offset_calls(hass)[-1] == (TADO, -0.5)
    notify.assert_not_awaited()
    assert dm.run_for("bedroom") is None


@pytest.mark.asyncio
async def test_expiry_restores_and_notifies(env):
    hass, *_ = env
    dm = DryingManager(hass)
    room = _room(override_until=T0 + 3600)
    await _eval(dm, room)
    expired = dict(room)
    with (
        patch("custom_components.roommind.const.time.time", return_value=T0 + 3601),
        patch(NOTIFY, new_callable=AsyncMock) as notify,
    ):
        live = await _eval(dm, expired, now=T0 + 3601)
    assert live is None
    assert _offset_calls(hass)[-1] == (TADO, -0.5)
    notify.assert_awaited_once()
    assert "time limit" in notify.call_args.kwargs["message"]


@pytest.mark.asyncio
async def test_vent_prompt_on_moisture_build_up_with_cooldown(env):
    hass, *_ = env
    dm = DryingManager(hass)
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        await _eval(dm, _room(), temp=26.0, rh=50.0)  # baseline ~12.2 g/m³
        # Humidity builds up: ~16 g/m³ at 26 °C / 65 %
        for i in range(1, 15):
            live = await _eval(dm, _room(), temp=26.0, rh=65.0, now=T0 + 30 * i)
        assert notify.await_count == 1
        assert "window" in notify.call_args.kwargs["message"]
        assert live["vent_suggested"] is True
        prompt_at = live["last_vent_prompt_at"]

        # Window opens: prompt answered
        live = await _eval(dm, _room(), temp=26.0, rh=65.0, window_open=True, now=prompt_at + 60)
        assert live["vent_suggested"] is False
        # Closed again, still humid, within cooldown: nothing new
        live = await _eval(dm, _room(), temp=26.0, rh=65.0, now=prompt_at + 600)
        assert notify.await_count == 1
        assert live["vent_suggested"] is False
        # After the cooldown: prompt again
        await _eval(dm, _room(), temp=26.0, rh=65.0, now=prompt_at + DRYING_VENT_PROMPT_COOLDOWN_S)
        assert notify.await_count == 2


@pytest.mark.asyncio
async def test_no_vent_prompt_below_rh_floor(env):
    hass, *_ = env
    dm = DryingManager(hass)
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        await _eval(dm, _room(), temp=20.0, rh=40.0)
        # Much more water but warm, so RH stays below 60 %
        for i in range(1, 20):
            await _eval(dm, _room(), temp=30.0, rh=55.0, now=T0 + 30 * i)
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_done_after_flat_humidity_clears_override_and_notifies(env):
    hass, _, store = env
    dm = DryingManager(hass)
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        await _eval(dm, _room(), temp=24.0, rh=55.0)
        now = T0
        live = None
        # Flat humidity, but before the minimum runtime nothing ends
        while now + 300 < T0 + DRYING_DONE_MIN_RUNTIME_S:
            now += 300
            live = await _eval(dm, _room(), temp=24.0, rh=55.0, now=now)
            assert live is not None
        now = T0 + DRYING_DONE_MIN_RUNTIME_S + 1
        live = await _eval(dm, _room(), temp=24.0, rh=55.0, now=now)
    assert live is None
    store.async_update_room.assert_awaited_once_with(
        "bedroom", {"override_heat": None, "override_cool": None, "override_until": None, "override_type": None}
    )
    assert _offset_calls(hass)[-1] == (TADO, -0.5)
    notify.assert_awaited_once()
    assert "looks dry" in notify.call_args.kwargs["message"]


@pytest.mark.asyncio
async def test_not_done_while_humidity_rises_or_after_airing(env):
    hass, _, store = env
    dm = DryingManager(hass)
    with patch(NOTIFY, new_callable=AsyncMock):
        await _eval(dm, _room(), temp=24.0, rh=50.0)
        now = T0
        rh = 50.0
        while now < T0 + DRYING_DONE_MIN_RUNTIME_S + 3600:
            now += 300
            rh = min(rh + 0.5, 80.0)
            assert await _eval(dm, _room(), temp=24.0, rh=rh, now=now) is not None
        # Window just closed after airing: humidity dropped back, but too soon to tell
        dm2 = DryingManager(hass)
        await _eval(dm2, _room(), temp=24.0, rh=50.0)
        await _eval(dm2, _room(), temp=24.0, rh=50.0, now=T0 + DRYING_DONE_MIN_RUNTIME_S, window_open=True)
        assert await _eval(dm2, _room(), temp=24.0, rh=50.0, now=T0 + DRYING_DONE_MIN_RUNTIME_S + 600) is not None
    store.async_update_room.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_humidity_sensor_runs_time_based(env):
    hass, _, store = env
    dm = DryingManager(hass)
    live = await _eval(dm, _room(), temp=24.0, rh=None)
    assert live["abs_humidity"] is None
    assert live["baseline_abs_humidity"] is None
    live = await _eval(dm, _room(), temp=24.0, rh=None, now=T0 + 6 * 3600)
    assert live is not None
    store.async_update_room.assert_not_awaited()


@pytest.mark.asyncio
async def test_baseline_taken_once_sensor_reports(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room(), temp=None, rh=None)
    live = await _eval(dm, _room(), temp=24.0, rh=55.0, now=T0 + 30)
    assert live["baseline_abs_humidity"] == round(absolute_humidity(24.0, 55.0), 2)


@pytest.mark.asyncio
async def test_changed_target_updates_run_without_new_offset_calls(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    live = await _eval(dm, _room(override_heat=32.0, override_until=T0 + 4 * 3600), now=T0 + 30)
    assert (live["target"], live["until"]) == (32.0, T0 + 4 * 3600)
    assert len(_offset_calls(hass)) == 1


@pytest.mark.asyncio
async def test_state_roundtrip_and_restore_after_restart(env):
    """Saved offsets survive a restart; a run whose override is gone restores them on the first cycle."""
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    state = dm.get_state()

    restarted = DryingManager(hass)
    restarted.load_state(state)
    assert restarted.run_for("bedroom").saved_offsets == {TADO: -0.5}
    assert restarted.run_for("bedroom").applied_offsets == {TADO: -7.0}
    await _eval(restarted, _room(override_type=None, override_heat=None, override_until=None), now=T0 + 600)
    assert _offset_calls(hass) == [(TADO, -7.0), (TADO, -0.5)]


def test_load_state_tolerates_garbage(hass):
    dm = DryingManager(hass)
    dm.load_state(
        {
            "runs": {"a": "x", "b": {"started_at": None}, "c": {"started_at": 1.0, "saved_offsets": {"x": "bad"}}},
            "pending_restores": {"e": None, "f": {"offset": "x"}, "g": {"offset": 1.0}},
        }
    )
    assert set(dm._runs) == {"c"}
    assert dm.run_for("c").saved_offsets == {}
    assert set(dm._pending_restores) == {"g"}
    dm.load_state(None)
    assert dm.get_state() == {"runs": {}, "pending_restores": {}, "offset_records": {}}


@pytest.mark.asyncio
async def test_prune_and_end_restore_offsets(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    await dm.async_prune({"bedroom"})
    assert dm.run_for("bedroom") is not None
    await dm.async_prune(set())
    assert dm.run_for("bedroom") is None
    assert _offset_calls(hass)[-1] == (TADO, -0.5)
    await dm.async_end("bedroom")  # nothing left: no call
    assert len(_offset_calls(hass)) == 2


@pytest.mark.asyncio
async def test_failed_restore_is_retried_then_given_up(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("cloud down"))
    with patch("custom_components.roommind.managers.drying_manager.time.time", return_value=T0):
        await dm.async_end("bedroom")
    assert set(dm.get_state()["pending_restores"]) == {TADO}

    await dm.async_retry_restores(now=T0 + 10)  # too early
    assert hass.services.async_call.await_count == 1
    now = T0
    for _ in range(DRYING_RESTORE_MAX_ATTEMPTS - 1):
        now += DRYING_RESTORE_RETRY_S
        await dm.async_retry_restores(now=now)
    assert dm.get_state()["pending_restores"] == {}
    assert hass.services.async_call.await_count == DRYING_RESTORE_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_succeeds(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("cloud down"))
    with patch("custom_components.roommind.managers.drying_manager.time.time", return_value=T0):
        await dm.async_end("bedroom")
    hass.services.async_call = AsyncMock()
    await dm.async_retry_restores(now=T0 + DRYING_RESTORE_RETRY_S)
    assert _offset_calls(hass) == [(TADO, -0.5)]
    assert dm.get_state()["pending_restores"] == {}


@pytest.mark.asyncio
async def test_new_run_reuses_original_of_pending_restore(env):
    """A restore that never went through: the next run must save the true original, not its own boost."""
    hass, states, _ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("cloud down"))
    await dm.async_end("bedroom")
    states[TADO] = _state(offset=-7.0)  # still boosted on the device
    hass.services.async_call = AsyncMock()
    await _eval(dm, _room(), now=T0 + 3600)
    assert dm.run_for("bedroom").saved_offsets == {TADO: -0.5}
    assert dm.get_state()["pending_restores"] == {}


@pytest.mark.asyncio
async def test_new_run_failing_keeps_pending_restore(env):
    hass, *_ = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    hass.services.async_call = AsyncMock(side_effect=RuntimeError("cloud down"))
    await dm.async_end("bedroom")
    await _eval(dm, _room(), now=T0 + 3600)
    # The original moved into the new run, which restores it at its end
    assert dm.get_state()["pending_restores"] == {}
    assert dm.run_for("bedroom").saved_offsets == {TADO: -0.5}


@pytest.mark.asyncio
async def test_clear_override_tolerates_deleted_room(env):
    hass, _, store = env
    store.async_update_room = AsyncMock(side_effect=KeyError("bedroom"))
    dm = DryingManager(hass)
    await dm._async_clear_override("bedroom")


@pytest.mark.asyncio
async def test_original_saved_before_the_cloud_call(env):
    """A crash after the boost landed must still find the original in the store."""
    hass, _, store = env
    dm = DryingManager(hass)

    async def _call(*args, **kwargs):
        saved = store.async_save_settings.call_args.args[0]["drying_state"]
        assert saved["runs"]["bedroom"]["saved_offsets"] == {TADO: -0.5}

    hass.services.async_call = AsyncMock(side_effect=_call)
    await _eval(dm, _room())
    store.async_save_settings.assert_awaited()


@pytest.mark.asyncio
async def test_persist_failure_is_logged_and_run_continues(env):
    hass, _, store = env
    store.async_save_settings = AsyncMock(side_effect=RuntimeError("disk full"))
    dm = DryingManager(hass)
    live = await _eval(dm, _room())
    assert live["offset_boost"] == {TADO: -7.0}
    assert dm.state_dirty


@pytest.mark.asyncio
async def test_run_ended_during_the_call_restores_after_it(env):
    """Room removed while the boost call runs: the boost must not outlive the run."""
    hass, *_ = env
    dm = DryingManager(hass)

    async def _call(domain, service, data, **kwargs):
        if data["offset"] == -7.0:
            await dm.async_end("bedroom")

    hass.services.async_call = AsyncMock(side_effect=_call)
    live = await _eval(dm, _room())
    assert live is None
    assert dm.run_for("bedroom") is None
    # boost sent, the end restored while it ran, restored again once the boost returned
    assert _offset_calls(hass) == [(TADO, -7.0), (TADO, -0.5), (TADO, -0.5)]


@pytest.mark.asyncio
async def test_stale_attribute_after_quick_restart_uses_recorded_original(env):
    """The integration still reports the boost right after a restore: the recorded original wins."""
    hass, states, _ = env
    dm = DryingManager(hass)
    with patch("custom_components.roommind.managers.drying_manager.time.time", return_value=T0 + 60):
        await _eval(dm, _room())
        await _eval(dm, _room(override_type=None, override_heat=None, override_until=None), now=T0 + 60)
    states[TADO] = _state(offset=-7.0)  # not refreshed yet
    await _eval(dm, _room(), now=T0 + 120)
    assert dm.run_for("bedroom").saved_offsets == {TADO: -0.5}


@pytest.mark.asyncio
async def test_attribute_equal_to_last_boost_is_stale(env):
    hass, states, _ = env
    dm = DryingManager(hass)
    with patch("custom_components.roommind.managers.drying_manager.time.time", return_value=T0):
        await _eval(dm, _room())
        await _eval(dm, _room(override_type=None, override_heat=None, override_until=None), now=T0)
    states[TADO] = _state(offset=-7.0)
    await _eval(dm, _room(), now=T0 + 3 * 3600)
    assert dm.run_for("bedroom").saved_offsets == {TADO: -0.5}


@pytest.mark.asyncio
async def test_offset_changed_by_user_long_after_restore_is_the_new_original(env):
    hass, states, _ = env
    dm = DryingManager(hass)
    with patch("custom_components.roommind.managers.drying_manager.time.time", return_value=T0):
        await _eval(dm, _room())
        await _eval(dm, _room(override_type=None, override_heat=None, override_until=None), now=T0)
    states[TADO] = _state(offset=1.0)
    await _eval(dm, _room(), now=T0 + 3 * 3600)
    assert dm.run_for("bedroom").saved_offsets == {TADO: 1.0}


@pytest.mark.asyncio
async def test_offset_records_survive_restart(env):
    hass, states, _ = env
    dm = DryingManager(hass)
    with patch("custom_components.roommind.managers.drying_manager.time.time", return_value=T0):
        await _eval(dm, _room())
        await _eval(dm, _room(override_type=None, override_heat=None, override_until=None), now=T0)
    restarted = DryingManager(hass)
    restarted.load_state(dm.get_state())
    states[TADO] = _state(offset=-7.0)
    await _eval(restarted, _room(), now=T0 + 300)
    assert restarted.run_for("bedroom").saved_offsets == {TADO: -0.5}
    restarted.load_state({"offset_records": {TADO: {"original": "x"}, "y": None}})
    assert restarted.get_state()["offset_records"] == {}


@pytest.mark.asyncio
async def test_no_boost_when_not_allowed(env):
    """Rooms reading the thermostat's own temperature get no offset (it would falsify the reading)."""
    hass, *_ = env
    dm = DryingManager(hass)
    live = await dm.async_evaluate(
        "bedroom", "Bedroom", _room(), 24.0, 55.0, False, {}, now=T0, offset_boost_allowed=False
    )
    assert live is not None
    assert live["offset_boost"] == {}
    assert _offset_calls(hass) == []


@pytest.mark.asyncio
async def test_control_off_ends_run_and_clears_override(env):
    hass, _, store = env
    dm = DryingManager(hass)
    await _eval(dm, _room())
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        live = await dm.async_evaluate(
            "bedroom", "Bedroom", _room(), 24.0, 55.0, False, {}, now=T0 + 60, control_active=False
        )
    assert live is None
    assert dm.run_for("bedroom") is None
    assert _offset_calls(hass)[-1] == (TADO, -0.5)
    store.async_update_room.assert_awaited_once_with(
        "bedroom", {"override_heat": None, "override_cool": None, "override_until": None, "override_type": None}
    )
    notify.assert_not_awaited()
