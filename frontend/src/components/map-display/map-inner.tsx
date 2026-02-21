"use client";

import L from "leaflet";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import { SeverityBadge } from "../ui/severity-badge";
import { type MapPoint } from "./index";

const DEFAULT_COLOR = "#3b82f6";

export default function MapInner({ points }: { points: MapPoint[] }) {
  if (points.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
        No pharmacy locations available.
      </div>
    );
  }

  const bounds = L.latLngBounds(points.map((p) => [p.lat, p.lng]));

  return (
    <MapContainer
      bounds={bounds}
      boundsOptions={{ padding: [40, 40] }}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {points.map((point, i) => (
        <CircleMarker
          key={`${point.label}-${i}`}
          center={[point.lat, point.lng]}
          radius={8}
          pathOptions={{
            color: point.color ?? DEFAULT_COLOR,
            fillColor: point.color ?? DEFAULT_COLOR,
            fillOpacity: 0.8,
            weight: 2,
          }}
        >
          <Popup>
            {point.popup ? (
              <div className="min-w-[220px]">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <strong className="text-sm">{point.popup.name}</strong>
                  <SeverityBadge severity={point.popup.severity} />
                </div>
                <p className="text-xs text-muted-foreground my-1">
                  {point.popup.address}
                </p>
                <hr className="my-2 border-border" />
                <p className="text-xs">
                  {point.popup.drugCount === 0 ? (
                    "No drugs require restocking"
                  ) : (
                    <>
                      <strong>{point.popup.drugCount}</strong>{" "}
                      {point.popup.drugCount === 1
                        ? "drug requires restocking"
                        : "drugs require restocking"}
                    </>
                  )}
                </p>
              </div>
            ) : (
              point.label
            )}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
