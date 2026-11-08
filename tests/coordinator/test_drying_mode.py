"""Drying mode in the coordinator cycle: targets, live data, mold stacking, offset restore."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import SAMPLE_ROOM, _create_coordinator, _make_store_mock, make_mock_states_get

AREA = "living_room_abc12345"
TRV = "climate.living_room"
NOTIFY = "custom_components.roommind.managers.drying_manager.async_send_drying_notification"


def _drying_room(**overrides):
    room = {
        **SAMPLE_ROOM,
        "override_heat": 30.0,
        "override_cool": None,
        "override_until": time.time() + 8 * 3600,
        "override_type": "drying",
    }
    room.update(overrides)
    return room


def _trv_state(offset=-0.5):
    return ("heat", {"max_temp": 25.0, "min_temp": 5.0, "temperature": 21.0, "offset_celsius": offset})


@pytest.fixture
def tado_registry():
    registry = MagicMock()
    registry.async_get = lambda eid: MagicMock(platform="tado") if eid == TRV else None
    with patch("custom_components.roommind.managers.drying_manager.er.async_get", return_value=registry):
        yield registry


def _offset_calls(hass):
    return [
        (c.args[2]["entity_id"], c.args[2]["offset"])
        for c in hass.services.async_call.call_args_list
        if c.args[:2] == ("tado", "set_climate_temperature_offset")
    ]


def _setup(hass, mock_config_entry, room, settings=None, **states):
    store = _make_store_mock({AREA: room}, settings)
    store.async_update_room = AsyncMock()
    hass.data = {"roommind": {"store": store}}
    hass.states.get = MagicMock(side_effect=make_mock_states_get(extra={TRV: _trv_state()}, **states))
    hass.services.async_call = AsyncMock()
    return store, _create_coordinator(hass, mock_config_entry)


@pytest.mark.asyncio
async def test_drying_heats_to_target_past_outdoor_gate(hass, mock_config_entry, tado_registry):
    """Drying is an override: 30 °C heat-only target, heating even on a warm day."""
    store, coordinator = _setup(hass, mock_config_entry, _drying_room(), temp="22.0", outdoor_temp="25.0")
    data = await coordinator._async_update_data()

    room = data["rooms"][AREA]
    assert room["heat_target"] == 30.0
    assert room["cool_target"] is None
    assert room["override_type"] == "drying"
    assert room["mode"] == "heating"
    drying = room["drying"]
    assert drying["target"] == 30.0
    assert drying["humidity"] == 55.0
    assert drying["abs_humidity"] is not None
    assert drying["offset_boost"] == {TRV: -7.0}
    assert drying["vent_suggested"] is False
    assert _offset_calls(hass) == [(TRV, -7.0)]
    saved = [c.args[0] for c in store.async_save_settings.call_args_list if "drying_state" in c.args[0]]
    assert saved[-1]["drying_state"]["runs"][AREA]["saved_offsets"] == {TRV: -0.5}


@pytest.mark.asyncio
async def test_no_drying_live_is_none(hass, mock_config_entry, tado_registry):
    _, coordinator = _setup(hass, mock_config_entry, dict(SAMPLE_ROOM))
    data = await coordinator._async_update_data()
    assert data["rooms"][AREA]["drying"] is None
    assert _offset_calls(hass) == []


@pytest.mark.asyncio
async def test_mold_prevention_does_not_stack_on_drying(hass, mock_config_entry, tado_registry):
    settings = {"mold_prevention_enabled": True, "mold_prevention_intensity": "strong"}
    _, coordinator = _setup(
        hass, mock_config_entry, _drying_room(), settings, temp="18.0", humidity="80.0", outdoor_temp="0.0"
    )
    data = await coordinator._async_update_data()
    room = data["rooms"][AREA]
    assert room["mold_prevention_active"] is True
    assert room["heat_target"] == 30.0


@pytest.mark.asyncio
async def test_startup_restores_saved_offset_without_active_drying(hass, mock_config_entry, tado_registry):
    """Crash safety: a stored run whose override is gone restores the device offset on the first cycle."""
    state = {
        "runs": {
            AREA: {
                "started_at": time.time() - 3600,
                "until": None,
                "target": 30.0,
                "saved_offsets": {TRV: -0.5},
                "applied_offsets": {TRV: -7.0},
            }
        }
    }
    store, coordinator = _setup(hass, mock_config_entry, dict(SAMPLE_ROOM), {"drying_state": state})
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        data = await coordinator._async_update_data()
    assert data["rooms"][AREA]["drying"] is None
    assert _offset_calls(hass) == [(TRV, -0.5)]
    notify.assert_not_awaited()
    saved = [c.args[0] for c in store.async_save_settings.call_args_list if "drying_state" in c.args[0]]
    assert saved[-1]["drying_state"]["runs"] == {}


@pytest.mark.asyncio
async def test_expired_drying_restores_and_notifies(hass, mock_config_entry, tado_registry):
    store, coordinator = _setup(hass, mock_config_entry, _drying_room())
    hass.async_create_task = MagicMock(side_effect=lambda coro: coro.close())  # the expiry auto-clear
    with patch(NOTIFY, new_callable=AsyncMock) as notify:
        await coordinator._async_update_data()
        expired = _drying_room(override_until=time.time() - 1)
        store.get_rooms.return_value = {AREA: expired}
        data = await coordinator._async_update_data()
    assert data["rooms"][AREA]["drying"] is None
    assert _offset_calls(hass) == [(TRV, -7.0), (TRV, -0.5)]
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_removed_room_restores_offset(hass, mock_config_entry, tado_registry):
    store, coordinator = _setup(hass, mock_config_entry, _drying_room())
    await coordinator._async_update_data()
    coordinator.async_request_refresh = AsyncMock()
    with patch("homeassistant.helpers.entity_registry.async_get"):
        await coordinator.async_room_removed(AREA)
    assert _offset_calls(hass) == [(TRV, -7.0), (TRV, -0.5)]


@pytest.mark.asyncio
async def test_paused_room_restores_offset(hass, mock_config_entry, tado_registry):
    store, coordinator = _setup(hass, mock_config_entry, _drying_room())
    await coordinator._async_update_data()
    with patch("homeassistant.helpers.entity_registry.async_get"):
        await coordinator.async_room_paused(AREA)
    assert _offset_calls(hass) == [(TRV, -7.0), (TRV, -0.5)]


@pytest.mark.asyncio
async def test_room_gone_from_config_restores_offset(hass, mock_config_entry, tado_registry):
    store, coordinator = _setup(hass, mock_config_entry, _drying_room())
    await coordinator._async_update_data()
    store.get_rooms.return_value = {}
    await coordinator._async_update_data()
    assert _offset_calls(hass) == [(TRV, -7.0), (TRV, -0.5)]


@pytest.mark.asyncio
async def test_no_offset_boost_without_external_sensor(hass, mock_config_entry, tado_registry):
    """Managed rooms read the TRV's temperature, which the offset would falsify."""
    room = _drying_room(temperature_sensor="")
    _, coordinator = _setup(hass, mock_config_entry, room)
    data = await coordinator._async_update_data()
    assert data["rooms"][AREA]["drying"]["offset_boost"] == {}
    assert _offset_calls(hass) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("room_overrides", "settings"),
    [({"climate_control_enabled": False}, None), ({}, {"climate_control_active": False})],
)
async def test_climate_control_off_ends_drying(hass, mock_config_entry, tado_registry, room_overrides, settings):
    store, coordinator = _setup(hass, mock_config_entry, _drying_room())
    await coordinator._async_update_data()
    store.get_rooms.return_value = {AREA: _drying_room(**room_overrides)}
    if settings:
        store.get_settings.return_value = {**store.get_settings.return_value, **settings}
    data = await coordinator._async_update_data()
    assert data["rooms"][AREA]["drying"] is None
    assert _offset_calls(hass) == [(TRV, -7.0), (TRV, -0.5)]
    store.async_update_room.assert_any_await(
        AREA, {"override_heat": None, "override_cool": None, "override_until": None, "override_type": None}
    )
