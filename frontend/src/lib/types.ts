import {
  AlertTriangleIcon,
  EyeIcon,
  LucideIcon,
  TrendingDownIcon,
} from "lucide-react";

export type Severity = "low" | "watch" | "urgent";

export interface SeverityConfig {
  icon: LucideIcon;
  label: string;
  bg: string;
  text: string;
  badge: "default" | "secondary" | "destructive";
  markerColor: string;
}

export const severityConfig: Record<Severity, SeverityConfig> = {
  low: {
    icon: TrendingDownIcon,
    label: "Low",
    bg: "bg-green-100",
    text: "text-green-500",
    badge: "secondary",
    markerColor: "#22c55e",
  },
  watch: {
    icon: EyeIcon,
    label: "Watch",
    bg: "bg-orange-100",
    text: "text-orange-500",
    badge: "default",
    markerColor: "#f97316",
  },
  urgent: {
    icon: AlertTriangleIcon,
    label: "Urgent",
    bg: "bg-red-100",
    text: "text-red-500",
    badge: "destructive",
    markerColor: "#ef4444",
  },
};
