"use client";

import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { ScrollArea, ScrollBar } from "../ui/scroll-area";
import { SeverityBadge } from "../ui/severity-badge";
import { Alert } from "./types";
import { severityConfig } from "@/lib/types";

export function AlertsDisplay({ alerts }: { alerts: Alert[] }) {
  return (
    <Card className="max-w-5xl mx-4 shadow-sm">
      <CardHeader>
        <CardTitle className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Active Notices
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="w-full">
          <div className="flex gap-3 pb-2">
            {alerts.map((alert) => {
              const config = severityConfig[alert.severity];
              const Icon = config.icon;
              const time = new Date(alert.timestamp).toLocaleDateString(
                "en-US",
                {
                  month: "short",
                  day: "numeric",
                  hour: "2-digit",
                  minute: "2-digit",
                },
              );

              return (
                <div
                  key={alert.id}
                  className={`flex min-w-[280px] max-w-[340px] shrink-0 items-start gap-3 rounded-lg ${config.bg} p-3`}
                >
                  <div
                    className={`mt-0.5 rounded-full p-1.5 ${config.text}/20`}
                  >
                    <Icon className={`h-3.5 w-3.5 ${config.text}`} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium leading-snug text-card-foreground">
                        {alert.description}
                      </p>
                      <SeverityBadge severity={alert.severity} />
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">{time}</p>
                  </div>
                </div>
              );
            })}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
