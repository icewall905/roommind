/**
 * Override logic shared by the panel's override section and the dashboard card.
 * Kept free of custom elements: the card is a separate bundle and must not define
 * the panel's tags a second time.
 */
import type { ClimateMode, OverrideType, RoomLiveData } from "../types";
import { localize } from "./localize";

export const OVERRIDE_DURATIONS_H = [1, 2, 4];
export const DRYING_DURATIONS_H = [4, 6, 8, 12];
export const DRYING_DEFAULT_DURATION_H = 8;
export const DRYING_DEFAULT_TARGET = 30;

export interface RoomTemps {
  comfortHeat: number;
  comfortCool: number;
  ecoHeat: number;
  ecoCool: number;
}

/** Heat/cool targets an override of this type sets, masked to the room's climate mode. */
export function overrideBand(
  type: OverrideType,
  climateMode: ClimateMode,
  temps: RoomTemps,
  custom: { heat: number; cool: number },
  dryingTarget = DRYING_DEFAULT_TARGET,
): { heat: number | null; cool: number | null } {
  let heat: number | null;
  let cool: number | null;
  if (type === "drying") {
    return { heat: dryingTarget, cool: null };
  }
  if (type === "boost") {
    heat = temps.comfortHeat;
    cool = temps.comfortCool;
  } else if (type === "eco") {
    heat = temps.ecoHeat;
    cool = temps.ecoCool;
  } else {
    heat = custom.heat;
    cool = custom.cool;
  }
  if (climateMode === "heat_only") cool = null;
  if (climateMode === "cool_only") heat = null;
  return { heat, cool };
}

export function isInvalidBand(heat: number | null, cool: number | null): boolean {
  return heat != null && cool != null && cool < heat;
}

/** WS message for `roommind/override/set`; presets take their targets from the room. */
export function overrideSetMessage(
  areaId: string,
  type: OverrideType,
  hours: number,
  climateMode: ClimateMode,
  custom: { heat: number; cool: number },
  dryingTarget = DRYING_DEFAULT_TARGET,
): Record<string, unknown> {
  const msg: Record<string, unknown> = {
    type: "roommind/override/set",
    area_id: areaId,
    override_type: type,
    duration: hours,
  };
  if (type === "drying") {
    msg.heat = dryingTarget;
  }
  if (type === "custom") {
    if (climateMode !== "cool_only") msg.heat = custom.heat;
    if (climateMode !== "heat_only") msg.cool = custom.cool;
  }
  return msg;
}

/** Remaining time as "1h 20m" / "20m", empty once the time has passed. */
export function formatRemaining(until: number, now = Date.now() / 1000): string {
  const remaining = until - now;
  if (remaining <= 0) return "";
  const h = Math.floor(remaining / 3600);
  const m = Math.floor((remaining % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

/** Local clock time "HH:MM" of an epoch-seconds timestamp. */
export function formatClock(ts: number, language: string): string {
  return new Date(ts * 1000).toLocaleTimeString(language, { hour: "2-digit", minute: "2-digit" });
}

/** Text lines for an active drying run: status first, then hints. */
export function dryingLines(live: RoomLiveData | undefined, language: string): string[] {
  const d = live?.drying;
  if (!d) return [];
  const lines: string[] = [];
  const parts = [localize("drying.label", language)];
  if (d.humidity != null) parts.push(`${Math.round(d.humidity)}%`);
  parts.push(localize("drying.ends_by", language, { time: formatClock(d.until, language) }));
  lines.push(parts.join(" · "));
  if (d.humidity == null) lines.push(localize("drying.no_humidity", language));
  if (d.vent_suggested) lines.push(localize("drying.vent_hint", language));
  if (d.close_suggested) lines.push(localize("drying.close_hint", language));
  if (Object.keys(d.offset_boost ?? {}).length > 0)
    lines.push(localize("drying.offset_boost_active", language));
  return lines;
}
