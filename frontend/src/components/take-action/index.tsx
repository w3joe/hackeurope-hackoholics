"use client";

import { useCallback, useMemo, useState } from "react";
import {
  ColumnDef,
  OnChangeFn,
  RowSelectionState,
} from "@tanstack/react-table";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { SeverityBadge } from "../ui/severity-badge";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Checkbox } from "../ui/checkbox";
import { DataTable } from "../ui/data-table";
import { Branch, Drug } from "./types";
import { useRouter } from "next/navigation";

function initializeBranchSelection(
  branches: Branch[],
): Record<string, RowSelectionState> {
  return Object.fromEntries(
    branches.map((branch) => [
      branch.id,
      Object.fromEntries(branch.drugs.map((drug) => [drug.id, true])),
    ]),
  );
}

function BranchTable({
  branch,
  editedQuantities,
  rowSelection,
  onRowSelectionChange,
  onUpdateQuantity,
}: {
  branch: Branch;
  editedQuantities: Record<string, number>;
  rowSelection: RowSelectionState;
  onRowSelectionChange: OnChangeFn<RowSelectionState>;
  onUpdateQuantity: (drugId: string, qty: number) => void;
}) {
  const columns = useMemo<ColumnDef<Drug>[]>(
    () => [
      {
        id: "select",
        header: ({ table }) => (
          <Checkbox
            checked={table.getIsAllRowsSelected()}
            onCheckedChange={(value) => table.toggleAllRowsSelected(!!value)}
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            checked={row.getIsSelected()}
            onCheckedChange={(value) => row.toggleSelected(!!value)}
          />
        ),
      },
      {
        accessorKey: "name",
        header: "Drug Name",
        cell: ({ row }) => (
          <span className="font-medium">{row.getValue("name")}</span>
        ),
      },
      {
        accessorKey: "currentStock",
        header: () => <span className="block text-right">Current Stock</span>,
        cell: ({ row }) => (
          <span className="block text-right">
            {row.getValue("currentStock")}
          </span>
        ),
      },
      {
        id: "suggestedQuantity",
        header: () => <span className="block text-right">Suggested Qty</span>,
        cell: ({ row }) => {
          const drug = row.original;
          const qty = editedQuantities[drug.id] ?? drug.suggestedQuantity;
          return (
            <div className="flex justify-end">
              <Input
                type="number"
                min={0}
                value={qty}
                onChange={(e) =>
                  onUpdateQuantity(drug.id, Number(e.target.value))
                }
                className="w-20 text-right"
              />
            </div>
          );
        },
      },
    ],
    [editedQuantities, onUpdateQuantity],
  );

  return (
    <DataTable
      columns={columns}
      data={branch.drugs}
      rowSelection={rowSelection}
      onRowSelectionChange={onRowSelectionChange}
      getRowId={(row) => row.id}
    />
  );
}

export function TakeAction({ branches }: { branches: Branch[] }) {
  const router = useRouter();
  const [localBranches, setLocalBranches] = useState<Branch[]>(branches);
  const [editedQuantities, setEditedQuantities] = useState<
    Record<string, Record<string, number>>
  >({});
  const [rowSelections, setRowSelections] = useState<
    Record<string, RowSelectionState>
  >(() => initializeBranchSelection(branches));
  const [pendingBranchId, setPendingBranchId] = useState<string | null>(null);

  const updateQuantity = useCallback(
    (branchId: string, drugId: string, qty: number) => {
      setEditedQuantities((prev) => ({
        ...prev,
        [branchId]: { ...prev[branchId], [drugId]: qty },
      }));
    },
    [],
  );

  const updateBranchSelection = useCallback(
    (
      branchId: string,
      updater: Parameters<OnChangeFn<RowSelectionState>>[0],
    ) => {
      setRowSelections((prev) => {
        const current = prev[branchId] ?? {};
        const next = typeof updater === "function" ? updater(current) : updater;

        return {
          ...prev,
          [branchId]: next,
        };
      });
    },
    [],
  );

  const placeOrder = useCallback(
    async (branch: Branch) => {
      const selectedIds = new Set(
        Object.entries(rowSelections[branch.id] ?? {})
          .filter(([, isSelected]) => Boolean(isSelected))
          .map(([drugId]) => drugId),
      );

      const selectedDrugs = branch.drugs.filter((drug) =>
        selectedIds.has(drug.id),
      );
      if (selectedDrugs.length === 0) {
        return;
      }

      const lineItems = selectedDrugs
        .map((drug) => ({
          drugId: drug.id,
          drugName: drug.name,
          manufacturer: drug.manufacturer,
          requestedQuantity:
            editedQuantities[branch.id]?.[drug.id] ?? drug.suggestedQuantity,
          unitPriceUsd: Number.isFinite(drug.unitPriceUsd)
            ? drug.unitPriceUsd
            : 0,
        }))
        .filter(
          (lineItem) =>
            Number.isFinite(lineItem.requestedQuantity) &&
            lineItem.requestedQuantity > 0,
        )
        .map((lineItem) => ({
          drugId: lineItem.drugId,
          drugName: lineItem.drugName,
          manufacturer: lineItem.manufacturer,
          requestedQuantity: lineItem.requestedQuantity,
          unitPriceUsd: lineItem.unitPriceUsd,
          lineTotalUsd: Number(
            (lineItem.requestedQuantity * lineItem.unitPriceUsd).toFixed(2),
          ),
        }));

      const orderedDrugIds = new Set(
        lineItems.map((lineItem) => lineItem.drugId),
      );

      if (lineItems.length === 0) {
        return;
      }

      setPendingBranchId(branch.id);

      try {
        const response = await fetch("/api/confirm-orders", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            storeId: branch.id,
            storeName: branch.name,
            storeAddress: branch.address,
            lineItems,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to place order");
        }

        setLocalBranches((prev) =>
          prev
            .map((candidateBranch) => {
              if (candidateBranch.id !== branch.id) {
                return candidateBranch;
              }

              return {
                ...candidateBranch,
                drugs: candidateBranch.drugs.filter(
                  (drug) => !orderedDrugIds.has(drug.id),
                ),
              };
            })
            .filter((candidateBranch) => candidateBranch.drugs.length > 0),
        );

        setEditedQuantities((prev) => {
          const branchEdited = prev[branch.id] ?? {};
          const nextBranchEdited = Object.fromEntries(
            Object.entries(branchEdited).filter(
              ([drugId]) => !orderedDrugIds.has(drugId),
            ),
          );

          return {
            ...prev,
            [branch.id]: nextBranchEdited,
          };
        });

        setRowSelections((prev) => {
          const remainingDrugs = branch.drugs.filter(
            (drug) => !orderedDrugIds.has(drug.id),
          );
          return {
            ...prev,
            [branch.id]: Object.fromEntries(
              remainingDrugs.map((drug) => [drug.id, true]),
            ),
          };
        });

        router.push("/orders");
      } finally {
        setPendingBranchId(null);
      }
    },
    [editedQuantities, router, rowSelections],
  );

  return (
    <Card className="max-w-5xl mx-4 shadow-sm w-full">
      <CardHeader>
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Take Action
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {localBranches.length === 0 ? (
          <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">
            No pending actions.
          </div>
        ) : null}
        {localBranches.map((branch) => {
          const selectedCount = Object.values(
            rowSelections[branch.id] ?? {},
          ).filter(Boolean).length;

          return (
            <Card
              key={branch.id}
              id={`take-action-${branch.id}`}
              className="scroll-mt-24"
            >
              <CardHeader>
                <div className="flex items-center gap-2">
                  <CardTitle className="text-sm font-semibold">
                    {branch.name}
                  </CardTitle>
                  <SeverityBadge severity={branch.severity} />
                </div>
                <p className="text-xs text-muted-foreground">
                  {branch.address}
                </p>
              </CardHeader>
              <CardContent>
                <BranchTable
                  branch={branch}
                  editedQuantities={editedQuantities[branch.id] ?? {}}
                  rowSelection={rowSelections[branch.id] ?? {}}
                  onRowSelectionChange={(updater) =>
                    updateBranchSelection(branch.id, updater)
                  }
                  onUpdateQuantity={(drugId, qty) =>
                    updateQuantity(branch.id, drugId, qty)
                  }
                />
              </CardContent>
              <CardFooter className="justify-end">
                <Button
                  disabled={
                    selectedCount === 0 || pendingBranchId === branch.id
                  }
                  onClick={() => {
                    void placeOrder(branch);
                  }}
                >
                  {pendingBranchId === branch.id
                    ? "Placing Order..."
                    : "Place Order"}
                </Button>
              </CardFooter>
            </Card>
          );
        })}
      </CardContent>
    </Card>
  );
}
