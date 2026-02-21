import { relations } from "drizzle-orm/relations";
import { pharmacies, vaccineStock } from "./schema";

export const vaccineStockRelations = relations(vaccineStock, ({one}) => ({
	pharmacy: one(pharmacies, {
		fields: [vaccineStock.storeId],
		references: [pharmacies.storeId]
	}),
}));

export const pharmaciesRelations = relations(pharmacies, ({many}) => ({
	vaccineStocks: many(vaccineStock),
}));