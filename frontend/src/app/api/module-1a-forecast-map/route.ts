import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { module1AResults } from "@/lib/db/schema";
import { geocode } from "@/lib/geocode";

interface ForecastRow {
  country: string;
  forecastStartWeek: string;
  weeklyCasesPer100k: number[];
  recommendedDiseaseFocus: string[];
  lat: number;
  lng: number;
}

function normalizeFocus(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return value
    .map((item) => String(item ?? "").trim())
    .filter((item) => item.length > 0);
}

function normalizeWeeklyCases(value: unknown): number[] {
  let weeklyRaw: unknown[] = [];

  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    const fromSnake = record.weekly_cases_per_100k;
    const fromCamel = record.weeklyCasesPer100k;
    const candidate = Array.isArray(fromSnake)
      ? fromSnake
      : Array.isArray(fromCamel)
        ? fromCamel
        : [];

    weeklyRaw = candidate;
  }

  const weekly = weeklyRaw
    .map((entry) => Number(entry))
    .filter((entry) => Number.isFinite(entry));

  const result = weekly.slice(0, 12);
  const filler = result.length > 0 ? result[result.length - 1] : 0;
  while (result.length < 12) {
    result.push(filler);
  }

  return result;
}

function normalizeForecastStartWeek(value: unknown): string {
  if (!value || typeof value !== "object") {
    return "Unknown";
  }

  const record = value as Record<string, unknown>;
  const fromSnake = String(record.forecast_start_week ?? "").trim();
  const fromCamel = String(record.forecastStartWeek ?? "").trim();

  return fromSnake || fromCamel || "Unknown";
}

export async function GET() {
  try {
    const rawRows = await db
      .select({
        country: module1AResults.country,
        recommendedDiseaseFocus: module1AResults.recommendedDiseaseFocus,
        twelveWeekForecast: module1AResults.twelveWeekForecast,
      })
      .from(module1AResults);

    const rows: ForecastRow[] = [];

    await Promise.all(
      rawRows.map(async (row) => {
        const country = String(row.country ?? "").trim();
        if (!country) return;

        const coords = await geocode(country);
        if (!coords) return;

        rows.push({
          country,
          lat: coords.lat,
          lng: coords.lng,
          recommendedDiseaseFocus: normalizeFocus(row.recommendedDiseaseFocus),
          weeklyCasesPer100k: normalizeWeeklyCases(row.twelveWeekForecast),
          forecastStartWeek: normalizeForecastStartWeek(row.twelveWeekForecast),
        });
      }),
    );

    return NextResponse.json({ rows });
  } catch {
    return NextResponse.json({ rows: [] }, { status: 200 });
  }
}
