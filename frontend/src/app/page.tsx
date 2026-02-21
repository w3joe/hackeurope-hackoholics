import { AlertsDisplay } from "@/components/alert-display";
import { Alert } from "@/components/alert-display/types";
import { MapDisplay } from "@/components/map-display";
import { TakeAction } from "@/components/take-action";
import { Branch } from "@/components/take-action/types";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { db } from "@/lib/db";
import {
  alerts as alertsTable,
  pharmacies,
  vaccineStock,
} from "@/lib/db/schema";
import { geocodeBranches } from "@/lib/geocode";
import { Severity, severityConfig } from "@/lib/types";
import { desc, isNotNull, sql } from "drizzle-orm";

const severityRank: Record<Severity, number> = {
  low: 0,
  watch: 1,
  urgent: 2,
};

function toSeverity(value: string): Severity {
  if (value === "low" || value === "watch" || value === "urgent") {
    return value;
  }

  return "low";
}

function toEpochMillis(timestamp: number | string): number {
  const parsed = Number(timestamp);
  if (!Number.isFinite(parsed)) return 0;

  return parsed < 1_000_000_000_000 ? parsed * 1000 : parsed;
}

function formatAddress(
  address: string,
  postalCode: string,
  city: string,
  country: string,
): string {
  return [address, `${postalCode} ${city}`, country].join(", ");
}

export default async function Page() {
  const [alertRows, pharmacyRows, shortageRows] = await Promise.all([
    db
      .select({
        id: alertsTable.id,
        affectedStoreIds: alertsTable.affectedStoreIds,
        timestamp: alertsTable.timestamp,
        description: alertsTable.description,
        severity: alertsTable.severity,
      })
      .from(alertsTable)
      .orderBy(desc(alertsTable.timestamp)),
    db
      .select({
        storeId: pharmacies.storeId,
        name: pharmacies.name,
        address: pharmacies.address,
        postalCode: pharmacies.postalCode,
        city: pharmacies.city,
        country: pharmacies.country,
      })
      .from(pharmacies),
    db
      .select({
        storeId: vaccineStock.storeId,
        shortageCount: sql<number>`sum(case when ${vaccineStock.stockQuantity} < ${vaccineStock.minStockLevel} then 1 else 0 end)::int`,
      })
      .from(vaccineStock)
      .where(isNotNull(vaccineStock.storeId))
      .groupBy(vaccineStock.storeId),
  ]);

  const alerts: Alert[] = alertRows.map((row) => ({
    id: row.id,
    affectedStoreIds: row.affectedStoreIds,
    timestamp: toEpochMillis(row.timestamp),
    description: row.description,
    severity: toSeverity(row.severity),
  }));

  const highestSeverityByStore = new Map<string, Severity>();
  for (const alert of alerts) {
    for (const storeId of alert.affectedStoreIds) {
      const current = highestSeverityByStore.get(storeId);
      if (!current || severityRank[alert.severity] > severityRank[current]) {
        highestSeverityByStore.set(storeId, alert.severity);
      }
    }
  }

  const shortageCountByStore = new Map<string, number>();
  for (const row of shortageRows) {
    if (!row.storeId) continue;
    shortageCountByStore.set(row.storeId, Number(row.shortageCount));
  }

  const branchesForMap = pharmacyRows.map((store) => ({
    id: store.storeId,
    name: store.name ?? store.storeId,
    address: formatAddress(
      store.address,
      store.postalCode,
      store.city,
      store.country,
    ),
    severity: highestSeverityByStore.get(store.storeId) ?? "low",
    drugCount: shortageCountByStore.get(store.storeId) ?? 0,
  }));

  const branchData: Branch[] = [
    {
      id: "b1",
      name: "Apotheke am Brandenburger Tor",
      address: "Unter den Linden 40, 10117 Berlin",
      severity: "urgent",
      drugs: [
        {
          id: "d1",
          name: "Oseltamivir",
          currentStock: 12,
          suggestedQuantity: 200,
        },
        {
          id: "d2",
          name: "Amoxicillin",
          currentStock: 45,
          suggestedQuantity: 150,
        },
        {
          id: "d3",
          name: "Ibuprofen",
          currentStock: 80,
          suggestedQuantity: 300,
        },
      ],
    },
    {
      id: "b2",
      name: "Marienapotheke München",
      address: "Marienplatz 2, 80331 München",
      severity: "watch",
      drugs: [
        {
          id: "d4",
          name: "Paracetamol",
          currentStock: 100,
          suggestedQuantity: 250,
        },
        {
          id: "d5",
          name: "Doxycycline",
          currentStock: 30,
          suggestedQuantity: 120,
        },
      ],
    },
    {
      id: "b3",
      name: "Ratsapotheke Hamburg",
      address: "Rathausmarkt 1, 20095 Hamburg",
      severity: "low",
      drugs: [
        {
          id: "d6",
          name: "Cetirizine",
          currentStock: 200,
          suggestedQuantity: 50,
        },
      ],
    },
  ];

  const [mapBranches, branches] = await Promise.all([
    geocodeBranches(branchesForMap),
    geocodeBranches(branchData),
  ]);

  const mapPoints = mapBranches.map((b) => ({
    lat: b.lat,
    lng: b.lng,
    label: b.name,
    color: severityConfig[b.severity].markerColor,
    popup: {
      name: b.name,
      address: b.address,
      severity: b.severity,
      drugCount: b.drugCount,
    },
  }));

  return (
    <main className="flex flex-col items-center w-full min-h-screen gap-4">
      <SidebarTrigger size="icon-lg" className={"m-2 self-start"} />
      <AlertsDisplay alerts={alerts} />
      <MapDisplay points={mapPoints} />
      <TakeAction branches={branches} />
    </main>
  );
}
