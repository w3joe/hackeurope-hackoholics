import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { confirmOrders } from "@/lib/db/schema";
import {
  type ConfirmOrderLineItem,
  toConfirmOrderLineItems,
} from "@/lib/orders/types";

interface ConfirmOrderPayload {
  storeId: string;
  storeName: string;
  storeAddress: string;
  lineItems: ConfirmOrderLineItem[];
}

function parsePayload(value: unknown): ConfirmOrderPayload | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const candidate = value as Record<string, unknown>;
  const storeId = String(candidate.storeId ?? "").trim();
  const storeName = String(candidate.storeName ?? "").trim();
  const storeAddress = String(candidate.storeAddress ?? "").trim();
  const lineItems = toConfirmOrderLineItems(candidate.lineItems);
  const validLineItems = lineItems.filter(
    (lineItem) => lineItem.requestedQuantity > 0,
  );

  if (!storeId || !storeName || !storeAddress || validLineItems.length === 0) {
    return null;
  }

  return {
    storeId,
    storeName,
    storeAddress,
    lineItems: validLineItems,
  };
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as unknown;
    const payload = parsePayload(body);

    if (!payload) {
      return NextResponse.json(
        { error: "Invalid payload" },
        { status: 400 },
      );
    }

    const [order] = await db
      .insert(confirmOrders)
      .values({
        storeId: payload.storeId,
        storeName: payload.storeName,
        storeAddress: payload.storeAddress,
        lineItems: payload.lineItems,
      })
      .returning({ id: confirmOrders.id });

    return NextResponse.json({ ok: true, id: order.id });
  } catch {
    return NextResponse.json(
      { error: "Failed to create order" },
      { status: 500 },
    );
  }
}
