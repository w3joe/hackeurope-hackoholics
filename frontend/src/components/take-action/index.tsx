"use client";

import { useCallback, useMemo, useState } from "react";
import { ColumnDef, RowSelectionState } from "@tanstack/react-table";
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

function BranchTable({
  branch,
  editedQuantities,
  onUpdateQuantity,
}: {
  branch: Branch;
  editedQuantities: Record<string, number>;
  onUpdateQuantity: (drugId: string, qty: number) => void;
}) {
  const [rowSelection, setRowSelection] = useState<RowSelectionState>(() =>
    Object.fromEntries(branch.drugs.map((d) => [d.id, true])),
  );

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
      onRowSelectionChange={setRowSelection}
      getRowId={(row) => row.id}
    />
  );
}

export function TakeAction({ branches }: { branches: Branch[] }) {
  const [editedQuantities, setEditedQuantities] = useState<
    Record<string, Record<string, number>>
  >({});

  const updateQuantity = useCallback(
    (branchId: string, drugId: string, qty: number) => {
      setEditedQuantities((prev) => ({
        ...prev,
        [branchId]: { ...prev[branchId], [drugId]: qty },
      }));
    },
    [],
  );

  return (
    <Card className="max-w-5xl mx-4 shadow-sm w-full">
      <CardHeader>
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Take Action
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {branches.map((branch) => (
            <Card key={branch.id}>
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
                  onUpdateQuantity={(drugId, qty) =>
                    updateQuantity(branch.id, drugId, qty)
                  }
                />
              </CardContent>
              <CardFooter className="justify-end">
                <Button>Place Order</Button>
              </CardFooter>
            </Card>
          ))}
      </CardContent>
    </Card>
  );
}
