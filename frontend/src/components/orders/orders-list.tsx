"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { ConfirmOrder, ConfirmOrderLineItem } from "@/lib/orders/types";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

function getManufacturerGroups(lineItems: ConfirmOrderLineItem[]) {
  const grouped = new Map<string, ConfirmOrderLineItem[]>();

  for (const lineItem of lineItems) {
    const current = grouped.get(lineItem.manufacturer) ?? [];
    current.push(lineItem);
    grouped.set(lineItem.manufacturer, current);
  }

  return Array.from(grouped.entries());
}

function toFileSafeToken(value: string): string {
  return value
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .slice(0, 64);
}

function downloadPurchaseOrderPdf(
  order: ConfirmOrder,
  manufacturer: string,
  lineItems: ConfirmOrderLineItem[],
) {
  const doc = new jsPDF();
  const createdAt = new Date(order.createdAt);
  const manufacturerTotal = Number(
    lineItems.reduce((sum, lineItem) => sum + lineItem.lineTotalUsd, 0).toFixed(2),
  );

  doc.setFontSize(16);
  doc.text("Purchase Order", 14, 16);

  doc.setFontSize(11);
  doc.text(`Order ID: ${order.id}`, 14, 25);
  doc.text(`Created At: ${createdAt.toLocaleString()}`, 14, 31);
  doc.text(`Store: ${order.storeName} (${order.storeId})`, 14, 37);
  doc.text(`Address: ${order.storeAddress}`, 14, 43);
  doc.text(`Manufacturer: ${manufacturer}`, 14, 49);

  autoTable(doc, {
    startY: 56,
    head: [["Drug", "Requested Qty", "Unit Price (USD)", "Line Total (USD)"]],
    body: lineItems.map((lineItem) => [
      lineItem.drugName,
      String(lineItem.requestedQuantity),
      lineItem.unitPriceUsd.toFixed(2),
      lineItem.lineTotalUsd.toFixed(2),
    ]),
    styles: {
      fontSize: 10,
    },
  });

  const finalY = (doc as jsPDF & { lastAutoTable?: { finalY?: number } }).lastAutoTable
    ?.finalY;
  doc.setFontSize(11);
  doc.text(
    `Manufacturer Total (USD): ${manufacturerTotal.toFixed(2)}`,
    14,
    (finalY ?? 56) + 10,
  );

  const manufacturerToken = toFileSafeToken(manufacturer) || "manufacturer";
  doc.save(`purchase-order-${order.id}-${manufacturerToken}.pdf`);
}

export function OrdersList({ orders }: { orders: ConfirmOrder[] }) {
  if (orders.length === 0) {
    return (
      <Card className="mx-4 w-full max-w-5xl shadow-sm">
        <CardContent className="p-6 text-sm text-muted-foreground">
          No orders yet.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="mx-4 flex w-full max-w-5xl flex-col gap-4">
      {orders.map((order) => {
        const manufacturerGroups = getManufacturerGroups(order.lineItems);
        const orderTotal = Number(
          order.lineItems
            .reduce((sum, lineItem) => sum + lineItem.lineTotalUsd, 0)
            .toFixed(2),
        );

        return (
          <Card key={order.id} className="shadow-sm">
            <CardHeader>
              <CardTitle className="text-sm font-semibold">{order.storeName}</CardTitle>
              <p className="text-xs text-muted-foreground">{order.storeAddress}</p>
              <p className="text-xs text-muted-foreground">
                Placed: {new Date(order.createdAt).toLocaleString()}
              </p>
              <p className="text-xs font-medium">Order Total (USD): {orderTotal.toFixed(2)}</p>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {manufacturerGroups.map(([manufacturer, items]) => (
                <div key={`${order.id}-${manufacturer}`} className="rounded-md border p-3">
                  <p className="mb-2 text-xs text-muted-foreground">
                    Manufacturer Total (USD):
                    {" "}
                    {items
                      .reduce((sum, lineItem) => sum + lineItem.lineTotalUsd, 0)
                      .toFixed(2)}
                  </p>
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <p className="text-sm font-medium">{manufacturer}</p>
                    <Button
                      size="sm"
                      onClick={() => downloadPurchaseOrderPdf(order, manufacturer, items)}
                    >
                      Download Purchase Order
                    </Button>
                  </div>
                  <div className="space-y-1 text-sm">
                    {items.map((item) => (
                      <p key={`${manufacturer}-${item.drugId}`}>
                        {item.drugName} - {item.requestedQuantity} x ${item.unitPriceUsd.toFixed(2)} = ${item.lineTotalUsd.toFixed(2)}
                      </p>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
