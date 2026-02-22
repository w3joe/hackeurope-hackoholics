import type { Severity } from "@/lib/types";

export interface DirectiveDrug {
  drugName: string;
  quantity: number;
  currentStockQuantity: number;
  unitCostUsd: number;
  totalCostUsd: number;
}

export interface ReplenishmentDirectiveRecord {
  pharmacyId: string;
  pharmacyName: string;
  location: string;
  severity: string;
  priorityRank: number;
  assignedDistributorName: string;
  drugsToDeliver: DirectiveDrug[];
}

function toStringValue(value: unknown, fallback = ""): string {
  const stringValue = String(value ?? "").trim();
  return stringValue || fallback;
}

function toNumberValue(value: unknown, fallback = 0): number {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : fallback;
}

export function toUiSeverity(value: string): Severity {
  const normalized = value.toUpperCase();
  if (normalized === "LOW") return "low";
  if (normalized === "MEDIUM") return "watch";
  return "urgent";
}

export function toDirectiveDrugId(
  directive: ReplenishmentDirectiveRecord,
  drug: DirectiveDrug,
  index: number,
): string {
  const drugToken = drug.drugName.toLowerCase().replace(/\s+/g, "-").trim();
  const distributorToken = directive.assignedDistributorName
    .toLowerCase()
    .replace(/\s+/g, "-")
    .trim();

  return [
    directive.pharmacyId,
    String(directive.priorityRank),
    distributorToken,
    drugToken,
    String(index),
  ].join("::");
}

export function toReplenishmentDirectives(
  value: unknown,
): ReplenishmentDirectiveRecord[] {
  const rawList = Array.isArray(value)
    ? value
    : value &&
        typeof value === "object" &&
        Array.isArray((value as Record<string, unknown>).replenishment_directives)
      ? ((value as Record<string, unknown>).replenishment_directives as unknown[])
      : [];

  return rawList
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const directive = entry as Record<string, unknown>;
      const drugsRaw = Array.isArray(directive.drugs_to_deliver)
        ? directive.drugs_to_deliver
        : [];
      const drugsToDeliver = drugsRaw
        .map((drugEntry) => {
          if (!drugEntry || typeof drugEntry !== "object") {
            return null;
          }

          const drug = drugEntry as Record<string, unknown>;
          const quantity = Math.max(0, Math.trunc(toNumberValue(drug.quantity, 0)));
          const drugName = toStringValue(drug.drug_name);
          const currentStockQuantity = Math.max(
            0,
            Math.trunc(toNumberValue(drug.current_stock_quantity, 0)),
          );
          const unitCostUsd = Math.max(0, toNumberValue(drug.unit_cost_usd, 0));
          const totalCostUsd = Math.max(0, toNumberValue(drug.total_cost_usd, 0));

          if (!drugName || quantity <= 0) {
            return null;
          }

          return {
            drugName,
            quantity,
            currentStockQuantity,
            unitCostUsd,
            totalCostUsd,
          };
        })
        .filter((drug): drug is DirectiveDrug => drug !== null);

      const pharmacyId = toStringValue(directive.pharmacy_id);
      if (!pharmacyId) {
        return null;
      }

      return {
        pharmacyId,
        pharmacyName: toStringValue(directive.pharmacy_name, pharmacyId),
        location: toStringValue(directive.location, "Unknown location"),
        severity: toStringValue(directive.severity, "LOW"),
        priorityRank: Math.trunc(toNumberValue(directive.priority_rank, index + 1)),
        assignedDistributorName: toStringValue(
          directive.assigned_distributor_name,
          "Unknown Distributor",
        ),
        drugsToDeliver,
      };
    })
    .filter(
      (directive): directive is ReplenishmentDirectiveRecord =>
        directive !== null && directive.drugsToDeliver.length > 0,
    );
}
