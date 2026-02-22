export interface ConfirmOrderLineItem {
  drugId: string;
  drugName: string;
  manufacturer: string;
  requestedQuantity: number;
  unitPriceUsd: number;
  lineTotalUsd: number;
}

export interface ConfirmOrder {
  id: string;
  storeId: string;
  storeName: string;
  storeAddress: string;
  lineItems: ConfirmOrderLineItem[];
  createdAt: string;
}

export function toConfirmOrderLineItems(value: unknown): ConfirmOrderLineItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const candidate = entry as Record<string, unknown>;
      const drugId = String(candidate.drugId ?? "").trim();
      const drugName = String(candidate.drugName ?? "").trim();
      const manufacturer = String(candidate.manufacturer ?? "").trim();
      const requestedQuantity = Number(candidate.requestedQuantity);
      const unitPriceUsd = Number(candidate.unitPriceUsd ?? candidate.unitCostUsd ?? 0);
      const lineTotalUsd = Number(
        candidate.lineTotalUsd ??
          candidate.totalCostUsd ??
          requestedQuantity * unitPriceUsd,
      );

      if (
        !drugId ||
        !drugName ||
        !manufacturer ||
        !Number.isFinite(requestedQuantity) ||
        !Number.isFinite(unitPriceUsd) ||
        !Number.isFinite(lineTotalUsd)
      ) {
        return null;
      }

      return {
        drugId,
        drugName,
        manufacturer,
        requestedQuantity,
        unitPriceUsd,
        lineTotalUsd,
      };
    })
    .filter((lineItem): lineItem is ConfirmOrderLineItem => lineItem !== null);
}
