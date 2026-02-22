"use client";

import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";

export interface MapPointPopup {
  name: string;
  address: string;
  severity: import("@/lib/types").Severity;
  drugCount: number;
  drugNames?: string[];
  takeActionTargetId?: string;
}

export interface ClosestDistributorPoint {
  id: string;
  name: string;
  lat: number;
  lng: number;
  distanceKm?: number;
}

export interface MapPoint {
  id: string;
  lat: number;
  lng: number;
  label: string;
  color?: string;
  popup?: MapPointPopup;
  closestDistributors?: ClosestDistributorPoint[];
}

const MapInner = dynamic(() => import("./map-inner"), { ssr: false });

export function MapDisplay({ points }: { points: MapPoint[] }) {
  return (
    <Card className="max-w-5xl mx-4 shadow-sm w-full">
      <CardHeader>
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Store Locations
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[400px] w-full rounded-lg overflow-hidden">
          <MapInner points={points} />
        </div>
      </CardContent>
    </Card>
  );
}
