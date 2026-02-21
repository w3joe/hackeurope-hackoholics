-- Current sql file was generated after introspecting the database
-- If you want to run this migration please uncomment this code before executing migrations
/*
CREATE TABLE "pharmacies" (
	"store_id" text PRIMARY KEY NOT NULL,
	"name" text DEFAULT 'dm-drogerie markt',
	"country" text NOT NULL,
	"city" text NOT NULL,
	"address" text NOT NULL,
	"postal_code" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "vaccine_stock" (
	"id" bigserial PRIMARY KEY NOT NULL,
	"store_id" text,
	"snapshot_date" date NOT NULL,
	"target_disease" text NOT NULL,
	"vaccine_brand" text NOT NULL,
	"manufacturer" text NOT NULL,
	"stock_quantity" integer NOT NULL,
	"min_stock_level" integer NOT NULL,
	"expiry_date" date NOT NULL,
	"storage_type" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
CREATE TABLE "alerts" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"affectedStoreIds" text[] NOT NULL,
	"timestamp" bigint NOT NULL,
	"description" text NOT NULL,
	"severity" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now(),
	CONSTRAINT "alerts_severity_check" CHECK (severity = ANY (ARRAY['low'::text, 'watch'::text, 'urgent'::text]))
);
--> statement-breakpoint
ALTER TABLE "vaccine_stock" ADD CONSTRAINT "vaccine_stock_store_id_fkey" FOREIGN KEY ("store_id") REFERENCES "public"."pharmacies"("store_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "idx_pharmacies_city" ON "pharmacies" USING btree ("city" text_ops);--> statement-breakpoint
CREATE INDEX "idx_pharmacies_country" ON "pharmacies" USING btree ("country" text_ops);--> statement-breakpoint
CREATE INDEX "idx_stock_expiry" ON "vaccine_stock" USING btree ("expiry_date" date_ops);--> statement-breakpoint
CREATE INDEX "idx_stock_store_id" ON "vaccine_stock" USING btree ("store_id" text_ops);
*/