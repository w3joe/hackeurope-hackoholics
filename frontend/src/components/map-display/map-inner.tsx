"use client";

import { useState } from "react";
import L from "leaflet";
import {
  CircleMarker,
  MapContainer,
  Marker,
  Polyline,
  Popup,
  TileLayer,
  useMapEvents,
} from "react-leaflet";
import { SeverityBadge } from "../ui/severity-badge";
import { type MapPoint } from "./index";

const DEFAULT_COLOR = "#3b82f6";

function scrollToTakeAction(targetId?: string) {
  if (!targetId) return;
  const element = document.getElementById(targetId);
  if (!element) return;

  element.scrollIntoView({ behavior: "smooth", block: "start" });
}

function createPinIcon(color: string) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="40" viewBox="0 0 28 40" fill="none">
      <path d="M14 1C7.37 1 2 6.37 2 13c0 8.77 9.8 19.81 11.4 21.54a.8.8 0 0 0 1.2 0C16.2 32.81 26 21.77 26 13 26 6.37 20.63 1 14 1Z" fill="${color}" stroke="white" stroke-width="2"/>
      <circle cx="14" cy="13" r="4.5" fill="white"/>
    </svg>
  `;

  return L.divIcon({
    className: "",
    html: svg,
    iconSize: [28, 40],
    iconAnchor: [14, 39],
    popupAnchor: [0, -34],
  });
}

function ResetDistributorOverlay({
  onReset,
}: {
  onReset: () => void;
}) {
  useMapEvents({
    click: () => {
      onReset();
    },
  });

  return null;
}

export default function MapInner({ points }: { points: MapPoint[] }) {
  const [selectedPharmacyId, setSelectedPharmacyId] = useState<string | null>(null);
  const selectedPoint = points.find((point) => point.id === selectedPharmacyId) ?? null;
  const distributorPoints = selectedPoint?.closestDistributors ?? [];

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
      <ResetDistributorOverlay
        onReset={() => {
          setSelectedPharmacyId(null);
        }}
      />
      {points.map((point, i) => (
        <Marker
          key={`${point.label}-${i}`}
          position={[point.lat, point.lng]}
          icon={createPinIcon(point.color ?? DEFAULT_COLOR)}
          eventHandlers={{
            click: () => {
              setSelectedPharmacyId(point.id);
            },
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
                {point.popup.drugCount > 0 ? (
                  <div className="space-y-1">
                    <p className="text-xs font-medium">Drugs to restock:</p>
                    {point.popup.drugNames?.map((drugName) => (
                      <button
                        key={`${point.label}-${drugName}`}
                        type="button"
                        className="block text-left text-xs text-blue-600 underline"
                        onClick={() => {
                          scrollToTakeAction(point.popup?.takeActionTargetId);
                        }}
                      >
                        {drugName}
                      </button>
                    ))}
                  </div>
                ) : null}
                <p className="mt-2 text-xs">
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
                {point.id === selectedPharmacyId &&
                (point.closestDistributors?.length ?? 0) > 0 ? (
                  <div className="mt-2 space-y-1 border-t pt-2 text-xs">
                    <p className="font-medium">Closest distributors:</p>
                    {point.closestDistributors?.map((distributor) => (
                      <p key={`${point.id}-${distributor.id}`}>
                        {distributor.name}
                        {typeof distributor.distanceKm === "number"
                          ? ` (${distributor.distanceKm.toFixed(2)} km)`
                          : ""}
                      </p>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : (
              point.label
            )}
          </Popup>
        </Marker>
      ))}
      {selectedPoint
        ? distributorPoints.map((distributor) => (
            <Polyline
              key={`line-${selectedPoint.id}-${distributor.id}`}
              positions={[
                [selectedPoint.lat, selectedPoint.lng],
                [distributor.lat, distributor.lng],
              ]}
              pathOptions={{
                color: "#1d4ed8",
                weight: 2,
                opacity: 0.7,
              }}
            />
          ))
        : null}
      {selectedPoint
        ? distributorPoints.map((distributor) => (
            <CircleMarker
              key={`distributor-${selectedPoint.id}-${distributor.id}`}
              center={[distributor.lat, distributor.lng]}
              radius={6}
              pathOptions={{
                color: "#1d4ed8",
                fillColor: "#60a5fa",
                fillOpacity: 0.95,
                weight: 2,
              }}
            >
              <Popup>
                <div className="min-w-[180px] text-xs">
                  <p className="text-sm font-semibold">{distributor.name}</p>
                  {typeof distributor.distanceKm === "number" ? (
                    <p className="text-muted-foreground">
                      Distance: {distributor.distanceKm.toFixed(2)} km
                    </p>
                  ) : null}
                </div>
              </Popup>
            </CircleMarker>
          ))
        : null}
    </MapContainer>
  );
}
