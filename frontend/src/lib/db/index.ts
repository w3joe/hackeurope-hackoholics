import { config } from "dotenv";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";

config({ path: ".env.local" });

const globalForDb = globalThis as unknown as {
  client: ReturnType<typeof postgres> | undefined;
  db: ReturnType<typeof drizzle> | undefined;
};

const client =
  globalForDb.client ??
  postgres(process.env.DATABASE_URL!, {
    max: 1,
  });

const db = globalForDb.db ?? drizzle({ client });

if (process.env.NODE_ENV !== "production") {
  globalForDb.client = client;
  globalForDb.db = db;
}

export { client, db };
