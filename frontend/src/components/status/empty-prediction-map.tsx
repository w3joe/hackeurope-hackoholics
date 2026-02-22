"use client";

import { useEffect, useMemo, useState } from "react";
import { Circle, MapContainer, Popup, TileLayer } from "react-leaflet";

interface ForecastPoint {
  country: string;
  forecastStartWeek: string;
  weeklyCasesPer100k: number[];
  recommendedDiseaseFocus: string[];
  lat: number;
  lng: number;
}

function getColor(value: number): string {
  if (value < 30) return "#22c55e";
  if (value <= 60) return "#f97316";
  return "#ef4444";
}

function getRadiusMeters(value: number): number {
  const radius = 22000 + value * 1500;
  return Math.max(22000, Math.min(210000, radius));
}

function parseIsoWeek(isoWeek: string): Date | null {
  const match = isoWeek.match(/^(\d{4})-W(\d{1,2})$/);
  if (!match) return null;

  const year = Number(match[1]);
  const week = Number(match[2]);
  if (!Number.isFinite(year) || !Number.isFinite(week) || week < 1 || week > 53) {
    return null;
  }

  const jan4 = new Date(Date.UTC(year, 0, 4));
  const jan4Day = jan4.getUTCDay() || 7;
  const week1Monday = new Date(jan4);
  week1Monday.setUTCDate(jan4.getUTCDate() - jan4Day + 1);

  const selected = new Date(week1Monday);
  selected.setUTCDate(week1Monday.getUTCDate() + (week - 1) * 7);
  return selected;
}

function formatIsoWeek(date: Date): string {
  const day = date.getUTCDay() || 7;
  const thursday = new Date(date);
  thursday.setUTCDate(date.getUTCDate() + (4 - day));

  const year = thursday.getUTCFullYear();
  const jan1 = new Date(Date.UTC(year, 0, 1));
  const week = Math.ceil(
    ((thursday.getTime() - jan1.getTime()) / 86400000 + 1) / 7,
  );

  return `${year}-W${String(week).padStart(2, "0")}`;
}

function addWeeksToIsoWeek(isoWeek: string, offset: number): string {
  const parsed = parseIsoWeek(isoWeek);
  if (!parsed) return "Unknown";

  const shifted = new Date(parsed);
  shifted.setUTCDate(parsed.getUTCDate() + offset * 7);
  return formatIsoWeek(shifted);
}

export default function EmptyPredictionMap() {
  const [points, setPoints] = useState<ForecastPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [weekIndex, setWeekIndex] = useState(0);

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetch("/api/module-1a-forecast-map", {
          cache: "no-store",
        });
        const payload = (await response.json()) as { rows?: ForecastPoint[] };
        setPoints(Array.isArray(payload.rows) ? payload.rows : []);
      } catch {
        setPoints([]);
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const summaryLabel = useMemo(() => {
    if (points.length === 0) return `Week ${weekIndex + 1} of 12`;

    const start = points[0].forecastStartWeek;
    const current = addWeeksToIsoWeek(start, weekIndex);
    if (current === "Unknown") return `Week ${weekIndex + 1} of 12`;

    return `Week ${weekIndex + 1} of 12 (${current})`;
  }, [points, weekIndex]);

  return (
    <div className="flex h-full w-full flex-col">
      <div className="border-b p-3">
        <label className="flex flex-col gap-2 text-sm">
          <span className="font-medium">Forecast Horizon: {summaryLabel}</span>
          <input
            type="range"
            min={0}
            max={11}
            step={1}
            value={weekIndex}
            onChange={(event) => {
              setWeekIndex(Number(event.target.value));
            }}
          />
        </label>
        <p className="mt-2 text-xs text-muted-foreground">
          Circle color: green &lt; 30, orange 30-60, red &gt; 60 cases per 100k.
        </p>
      </div>

      <div className="relative flex-1">
        {isLoading ? (
          <p className="absolute top-3 left-3 z-[500] rounded bg-background/90 px-2 py-1 text-xs text-muted-foreground">
            Loading forecast map...
          </p>
        ) : null}
        <MapContainer
          center={[54, 15]}
          zoom={4}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {points.map((point) => {
            const value = point.weeklyCasesPer100k[weekIndex] ?? 0;
            const color = getColor(value);
            const radius = getRadiusMeters(value);
            const selectedWeek = addWeeksToIsoWeek(point.forecastStartWeek, weekIndex);
            const focusText = point.recommendedDiseaseFocus.length
              ? point.recommendedDiseaseFocus.join(", ")
              : "No specific disease focus";

            return (
              <Circle
                key={point.country}
                center={[point.lat, point.lng]}
                radius={radius}
                pathOptions={{
                  color,
                  fillColor: color,
                  fillOpacity: 0.28,
                  weight: 2,
                }}
              >
                <Popup>
                  <div className="min-w-[230px] space-y-1 text-xs">
                    <p className="text-sm font-semibold">{point.country}</p>
                    <p>
                      Cases per 100k: <strong>{value.toFixed(1)}</strong>
                    </p>
                    <p>
                      Forecast week: <strong>{selectedWeek}</strong>
                    </p>
                    <p>
                      Forecast starts: <strong>{point.forecastStartWeek}</strong>
                    </p>
                    <p>
                      Recommended disease focus: <strong>{focusText}</strong>
                    </p>
                  </div>
                </Popup>
              </Circle>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}
