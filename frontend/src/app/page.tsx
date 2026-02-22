import { AlertsDisplay } from "@/components/alert-display";
import { Alert } from "@/components/alert-display/types";
import { MapDisplay } from "@/components/map-display";
import type { ClosestDistributorPoint } from "@/components/map-display";
import { TakeAction } from "@/components/take-action";
import { Branch } from "@/components/take-action/types";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { db } from "@/lib/db";
import {
  alerts as alertsTable,
  confirmOrders,
  orchestrationResults,
  pharmacies,
} from "@/lib/db/schema";
import {
  toDirectiveDrugId,
  toReplenishmentDirectives,
  toUiSeverity,
} from "@/lib/orchestration/types";
import { toConfirmOrderLineItems } from "@/lib/orders/types";
import { Severity, severityConfig } from "@/lib/types";
import { desc } from "drizzle-orm";

export const dynamic = "force-dynamic";

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

function parseClosestDistributors(value: unknown): ClosestDistributorPoint[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const distributors: ClosestDistributorPoint[] = [];

  value.forEach((entry, index) => {
    if (!entry || typeof entry !== "object") {
      return;
    }

    const candidate = entry as Record<string, unknown>;
    const lat = Number(candidate.lat);
    const lng = Number(candidate.lng ?? candidate.lon);
    const name = String(candidate.name ?? "").trim();

    if (!Number.isFinite(lat) || !Number.isFinite(lng) || !name) {
      return;
    }

    const distanceKmRaw = Number(candidate.distance_km);

    distributors.push({
      id: String(candidate.dist_id ?? `${name}-${index}`).trim(),
      name,
      lat,
      lng,
      distanceKm: Number.isFinite(distanceKmRaw) ? distanceKmRaw : undefined,
    });
  });

  return distributors;
}

export default async function Page() {
  const [alertRows, pharmacyRows] =
    await Promise.all([
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
          latitude: pharmacies.latitude,
          longitude: pharmacies.longitude,
          closestDistributors: pharmacies.closestDistributors,
        })
        .from(pharmacies),
    ]);

  let confirmedOrderRows: { lineItems: unknown }[] = [];
  try {
    confirmedOrderRows = await db
      .select({
        lineItems: confirmOrders.lineItems,
      })
      .from(confirmOrders);
  } catch {
    confirmedOrderRows = [];
  }

  let latestOrchestrationResult: { replenishmentDirectives: unknown } | null = null;
  try {
    const rows = await db
      .select({
        replenishmentDirectives: orchestrationResults.replenishmentDirectives,
      })
      .from(orchestrationResults)
      .orderBy(desc(orchestrationResults.createdAt))
      .limit(1);

    latestOrchestrationResult = rows[0] ?? null;
  } catch {
    latestOrchestrationResult = null;
  }

  const alerts: Alert[] = alertRows.map((row) => ({
    id: row.id,
    affectedStoreIds: row.affectedStoreIds,
    timestamp: toEpochMillis(row.timestamp),
    description: row.description,
    severity: toSeverity(row.severity),
  }));
  const visibleAlerts = alerts.slice(0, 5);

  const confirmedDrugIds = new Set(
    confirmedOrderRows.flatMap((order) =>
      toConfirmOrderLineItems(order.lineItems).map((lineItem) => lineItem.drugId),
    ),
  );

  const highestSeverityByStore = new Map<string, Severity>();
  for (const alert of alerts) {
    for (const storeId of alert.affectedStoreIds) {
      const current = highestSeverityByStore.get(storeId);
      if (!current || severityRank[alert.severity] > severityRank[current]) {
        highestSeverityByStore.set(storeId, alert.severity);
      }
    }
  }

  const pharmacyByStoreId = new Map(
    pharmacyRows.map((store) => [
      store.storeId,
      {
        name: store.name ?? store.storeId,
        address: formatAddress(
          store.address,
          store.postalCode,
          store.city,
          store.country,
        ),
      },
    ]),
  );

  const directives = toReplenishmentDirectives(
    latestOrchestrationResult?.replenishmentDirectives,
  ).sort((left, right) => left.priorityRank - right.priorityRank);

  const branchByStoreId = new Map<string, Branch>();
  for (const directive of directives) {
    const pharmacyMeta = pharmacyByStoreId.get(directive.pharmacyId);

    const branch: Branch =
      branchByStoreId.get(directive.pharmacyId) ?? {
        id: directive.pharmacyId,
        name: directive.pharmacyName || pharmacyMeta?.name || directive.pharmacyId,
        address: directive.location || pharmacyMeta?.address || "Unknown location",
        severity: toUiSeverity(directive.severity),
        drugs: [],
      };

    branch.severity =
      severityRank[toUiSeverity(directive.severity)] > severityRank[branch.severity]
        ? toUiSeverity(directive.severity)
        : branch.severity;

    directive.drugsToDeliver.forEach((drug, index) => {
      const drugId = toDirectiveDrugId(directive, drug, index);
      if (confirmedDrugIds.has(drugId)) {
        return;
      }

      branch.drugs.push({
        id: drugId,
        name: drug.drugName,
        manufacturer: directive.assignedDistributorName,
        currentStock: drug.currentStockQuantity,
        suggestedQuantity: drug.quantity,
        unitPriceUsd: drug.unitCostUsd,
      });
    });

    if (branch.drugs.length > 0) {
      branchByStoreId.set(directive.pharmacyId, branch);
    }
  }
  const branchData = Array.from(branchByStoreId.values());
  const takeActionByStore = new Map(
    branchData.map((branch) => [
      branch.id,
      {
        drugCount: branch.drugs.length,
        drugNames: Array.from(new Set(branch.drugs.map((drug) => drug.name))),
      },
    ]),
  );

  const mapPoints = pharmacyRows
    .filter(
      (store) =>
        typeof store.latitude === "number" && typeof store.longitude === "number",
    )
    .map((store) => {
      const severity = highestSeverityByStore.get(store.storeId) ?? "low";
      const takeAction = takeActionByStore.get(store.storeId);

      return {
        id: store.storeId,
        lat: store.latitude as number,
        lng: store.longitude as number,
        label: store.name ?? store.storeId,
        color: severityConfig[severity].markerColor,
        closestDistributors: parseClosestDistributors(store.closestDistributors),
        popup: {
          name: store.name ?? store.storeId,
          address: formatAddress(
            store.address,
            store.postalCode,
            store.city,
            store.country,
          ),
          severity,
          drugCount: takeAction?.drugCount ?? 0,
          drugNames: takeAction?.drugNames ?? [],
          takeActionTargetId:
            takeAction && takeAction.drugCount > 0
              ? `take-action-${store.storeId}`
              : undefined,
        },
      };
    });

  return (
    <main className="flex flex-col items-center w-full min-h-screen gap-4">
      <SidebarTrigger size="icon-lg" className={"m-2 self-start"} />
      <AlertsDisplay alerts={visibleAlerts} />
      <MapDisplay points={mapPoints} />
      <TakeAction branches={branchData} />
    </main>
  );
}
