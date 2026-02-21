export type { Severity } from "@/lib/types";

export interface Alert {
  id: string;
  affectedStoreIds: string[];
  timestamp: number;
  description: string;
  severity: import("@/lib/types").Severity;
}
