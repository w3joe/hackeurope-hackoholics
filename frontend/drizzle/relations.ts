import { relations } from "drizzle-orm/relations";
import { pharmacies, vaccineStock, confirmOrders } from "./schema";

export const vaccineStockRelations = relations(vaccineStock, ({one}) => ({
	pharmacy: one(pharmacies, {
		fields: [vaccineStock.storeId],
		references: [pharmacies.storeId]
	}),
}));

export const pharmaciesRelations = relations(pharmacies, ({many}) => ({
	vaccineStocks: many(vaccineStock),
	confirmOrders: many(confirmOrders),
}));

export const confirmOrdersRelations = relations(confirmOrders, ({one}) => ({
	pharmacy: one(pharmacies, {
		fields: [confirmOrders.storeId],
		references: [pharmacies.storeId]
	}),
}));