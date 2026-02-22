import { OrdersList } from "@/components/orders/orders-list";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { db } from "@/lib/db";
import { confirmOrders } from "@/lib/db/schema";
import { type ConfirmOrder, toConfirmOrderLineItems } from "@/lib/orders/types";
import { desc } from "drizzle-orm";

export const dynamic = "force-dynamic";

export default async function OrdersPage() {
  let orderRows: {
    id: string;
    storeId: string;
    storeName: string;
    storeAddress: string;
    lineItems: unknown;
    createdAt: string | null;
  }[] = [];

  try {
    orderRows = await db
      .select({
        id: confirmOrders.id,
        storeId: confirmOrders.storeId,
        storeName: confirmOrders.storeName,
        storeAddress: confirmOrders.storeAddress,
        lineItems: confirmOrders.lineItems,
        createdAt: confirmOrders.createdAt,
      })
      .from(confirmOrders)
      .orderBy(desc(confirmOrders.createdAt));
  } catch {
    orderRows = [];
  }

  const orders: ConfirmOrder[] = orderRows.map((row) => ({
    id: row.id,
    storeId: row.storeId,
    storeName: row.storeName,
    storeAddress: row.storeAddress,
    lineItems: toConfirmOrderLineItems(row.lineItems),
    createdAt: row.createdAt ?? new Date().toISOString(),
  }));

  return (
    <main className="flex min-h-screen w-full flex-col gap-4 items-center">
      <SidebarTrigger size="icon-lg" className="m-2 self-start" />
      <OrdersList orders={orders} />
    </main>
  );
}
