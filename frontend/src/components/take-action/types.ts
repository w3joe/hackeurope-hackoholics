import { Severity } from "@/lib/types";

export interface Drug {
  id: string;
  name: string;
  currentStock: number;
  suggestedQuantity: number;
}

export interface Branch {
  id: string;
  name: string;
  address: string;
  lat?: number;
  lng?: number;
  severity: Severity;
  drugs: Drug[];
}

export type ResolvedBranch = Branch & { lat: number; lng: number };
