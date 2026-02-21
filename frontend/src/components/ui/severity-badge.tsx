import { Severity, severityConfig } from "@/lib/types";
import { cn } from "@/lib/utils";

export function SeverityBadge({
  severity,
  className,
}: {
  severity: Severity;
  className?: string;
}) {
  const config = severityConfig[severity];

  return (
    <span
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap shrink-0 rounded-4xl px-2 py-0.5 text-xs font-medium",
        config.bg,
        config.text,
        className,
      )}
    >
      {config.label}
    </span>
  );
}
