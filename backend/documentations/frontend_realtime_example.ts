/**
 * Frontend: Subscribe to Alerts via Supabase Realtime
 *
 * Install: npm install @supabase/supabase-js
 *
 * DB columns match Alert interface exactly (camelCase).
 */

import { createClient, RealtimeChannel } from "@supabase/supabase-js";

const SUPABASE_URL = "https://your-project.supabase.co";
const SUPABASE_ANON_KEY = "your-anon-key";

export type Severity = "low" | "watch" | "urgent";

export interface Alert {
  id: string;
  affectedStoreIds: string[];
  timestamp: number;
  description: string;
  severity: Severity;
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const channel: RealtimeChannel = supabase
  .channel("alerts")
  .on(
    "postgres_changes",
    { event: "INSERT", schema: "public", table: "alerts" },
    (payload) => {
      const alert = payload.new as Alert;
      console.log("New alert:", alert);
      // showNotification(alert);
    }
  )
  .subscribe();

// Unsubscribe when done:
// supabase.removeChannel(channel);
