CREATE TABLE "confirm_orders" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"store_id" text NOT NULL,
	"store_name" text NOT NULL,
	"store_address" text NOT NULL,
	"line_items" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now()
);
--> statement-breakpoint
ALTER TABLE "confirm_orders" ADD CONSTRAINT "confirm_orders_store_id_pharmacies_store_id_fk" FOREIGN KEY ("store_id") REFERENCES "public"."pharmacies"("store_id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "idx_confirm_orders_store_id" ON "confirm_orders" USING btree ("store_id" text_ops);--> statement-breakpoint
CREATE INDEX "idx_confirm_orders_created_at" ON "confirm_orders" USING btree ("created_at" timestamptz_ops);