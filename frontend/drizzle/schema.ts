import {
  pgTable,
  check,
  uuid,
  text,
  numeric,
  jsonb,
  timestamp,
  bigint,
  doublePrecision,
  foreignKey,
  bigserial,
  date,
  integer,
  index,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const module1AResults = pgTable(
  "module_1a_results",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    country: text().notNull(),
    riskLevel: text("risk_level").notNull(),
    spreadLikelihood: numeric("spread_likelihood", { precision: 5, scale: 2 })
      .default("0")
      .notNull(),
    reasoning: text().default("").notNull(),
    recommendedDiseaseFocus: jsonb("recommended_disease_focus")
      .default([])
      .notNull(),
    twelveWeekForecast: jsonb("twelve_week_forecast").default({}).notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).defaultNow(),
  },
  (table) => [
    check(
      "module_1a_results_risk_level_check",
      sql`risk_level = ANY (ARRAY['LOW'::text, 'MEDIUM'::text, 'HIGH'::text, 'CRITICAL'::text])`,
    ),
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

export const pharmacies = pgTable("pharmacies", {
  storeId: text("store_id").primaryKey().notNull(),
  name: text().default("dm-drogerie markt"),
  country: text().notNull(),
  city: text().notNull(),
  address: text().notNull(),
  postalCode: text("postal_code").notNull(),
  latitude: doublePrecision(),
  longitude: doublePrecision(),
  closestDistributors: jsonb("closest_distributors"),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  }).defaultNow(),
});

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
  },
  (table) => [
    foreignKey({
      columns: [table.storeId],
      foreignColumns: [pharmacies.storeId],
      name: "vaccine_stock_store_id_fkey",
    }).onDelete("cascade"),
  ],
);

export const confirmOrders = pgTable(
  "confirm_orders",
  {
    id: uuid().defaultRandom().primaryKey().notNull(),
    storeId: text("store_id").notNull(),
    storeName: text("store_name").notNull(),
    storeAddress: text("store_address").notNull(),
    lineItems: jsonb("line_items").notNull(),
    createdAt: timestamp("created_at", {
      withTimezone: true,
      mode: "string",
    }).defaultNow(),
  },
  (table) => [
    index("idx_confirm_orders_created_at").using(
      "btree",
      table.createdAt.asc().nullsLast().op("timestamptz_ops"),
    ),
    index("idx_confirm_orders_store_id").using(
      "btree",
      table.storeId.asc().nullsLast().op("text_ops"),
    ),
    foreignKey({
      columns: [table.storeId],
      foreignColumns: [pharmacies.storeId],
      name: "confirm_orders_store_id_pharmacies_store_id_fk",
    }).onDelete("cascade"),
  ],
);
