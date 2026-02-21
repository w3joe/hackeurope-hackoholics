import {
  pgTable,
  index,
  text,
  timestamp,
  foreignKey,
  bigserial,
  date,
  integer,
  check,
  uuid,
  bigint,
  jsonb,
  numeric,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const pharmacies = pgTable(
  "pharmacies",
  {
    storeId: text("store_id").primaryKey().notNull(),
    name: text().default("dm-drogerie markt"),
    country: text().notNull(),
    city: text().notNull(),
    address: text().notNull(),
    postalCode: text("postal_code").notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).defaultNow(),
  },
  (table) => [
    index("idx_pharmacies_city").using(
      "btree",
      table.city.asc().nullsLast().op("text_ops"),
    ),
    index("idx_pharmacies_country").using(
      "btree",
      table.country.asc().nullsLast().op("text_ops"),
    ),
  ],
);

export const vaccineStock = pgTable(
  "vaccine_stock",
  {
    id: bigserial({ mode: "bigint" }).primaryKey().notNull(),
    storeId: text("store_id"),
    snapshotDate: date("snapshot_date").notNull(),
    targetDisease: text("target_disease").notNull(),
    vaccineBrand: text("vaccine_brand").notNull(),
    manufacturer: text().notNull(),
    stockQuantity: integer("stock_quantity").notNull(),
    minStockLevel: integer("min_stock_level").notNull(),
    expiryDate: date("expiry_date").notNull(),
    storageType: text("storage_type").notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).defaultNow(),
  },
  (table) => [
    index("idx_stock_expiry").using(
      "btree",
      table.expiryDate.asc().nullsLast().op("date_ops"),
    ),
    index("idx_stock_store_id").using(
      "btree",
      table.storeId.asc().nullsLast().op("text_ops"),
    ),
    foreignKey({
      columns: [table.storeId],
      foreignColumns: [pharmacies.storeId],
      name: "vaccine_stock_store_id_fkey",
    }).onDelete("cascade"),
  ],
);

export const alerts = pgTable(
  "alerts",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    affectedStoreIds: text().array().notNull(),
    // You can use { mode: "bigint" } if numbers are exceeding js number limitations
    timestamp: bigint({ mode: "number" }).notNull(),
    description: text().notNull(),
    severity: text().notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).defaultNow(),
  },
  (table) => [
    check(
      "alerts_severity_check",
      sql`severity = ANY (ARRAY['low'::text, 'watch'::text, 'urgent'::text])`,
    ),
  ],
);

export const orchestrationResults = pgTable("orchestration_results", {
  id: uuid().defaultRandom().primaryKey().notNull(),
  replenishmentDirectives: jsonb("replenishment_directives")
    .default([])
    .notNull(),
  grandTotalCostUsd: numeric("grand_total_cost_usd", {
    precision: 12,
    scale: 2,
  })
    .default("0")
    .notNull(),
  overallSystemSummary: text("overall_system_summary").default("").notNull(),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  }).defaultNow(),
});
