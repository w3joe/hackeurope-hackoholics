"use client";

import { SidebarTrigger } from "@/components/ui/sidebar";
import dynamic from "next/dynamic";
import Image from "next/image";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import { Suspense, useMemo, useState } from "react";

const EU_COUNTRIES = [
  "Austria",
  "Belgium",
  "Bulgaria",
  "Croatia",
  "Cyprus",
  "Czechia",
  "Denmark",
  "Estonia",
  "Finland",
  "France",
  "Germany",
  "Greece",
  "Hungary",
  "Ireland",
  "Italy",
  "Latvia",
  "Lithuania",
  "Luxembourg",
  "Malta",
  "Netherlands",
  "Poland",
  "Portugal",
  "Romania",
  "Slovakia",
  "Slovenia",
  "Spain",
  "Sweden",
] as const;

const PATHOGENS = ["Influenza", "SRV"] as const;

const parseCountry = parseAsStringLiteral([...EU_COUNTRIES]).withDefault(
  "Germany",
);
const parsePathogen = parseAsStringLiteral([...PATHOGENS]).withDefault(
  "Influenza",
);
const parseStatusTab = parseAsStringLiteral([
  "prediction",
  "validation",
] as const).withDefault("prediction");

const EmptyPredictionMap = dynamic(
  () => import("@/components/status/empty-prediction-map"),
  { ssr: false },
);

function StatusPageContent() {
  const [activeTab, setActiveTab] = useQueryState("tab", parseStatusTab);
  const [country, setCountry] = useQueryState("country", parseCountry);
  const [pathogen, setPathogen] = useQueryState("pathogen", parsePathogen);
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const [errorUrl, setErrorUrl] = useState<string | null>(null);
  const [notFoundUrl, setNotFoundUrl] = useState<string | null>(null);

  const apiBaseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

  const chartUrl = useMemo(() => {
    const countryEncoded = encodeURIComponent(country);
    const pathogenEncoded = encodeURIComponent(pathogen);
    return `${apiBaseUrl}/chart/validation/${countryEncoded}/${pathogenEncoded}`;
  }, [apiBaseUrl, country, pathogen]);

  const hasError = errorUrl === chartUrl;
  const hasNoData = notFoundUrl === chartUrl;
  const isLoading = loadedUrl !== chartUrl && !hasError && !hasNoData;

  return (
    <main className="flex min-h-screen w-full flex-col gap-4 p-4 items-center">
      <SidebarTrigger size="icon-lg" className="self-start" />

      <section className="flex w-full gap-2 rounded-lg border p-2 max-w-5xl">
        <button
          type="button"
          onClick={() => {
            void setActiveTab("prediction");
          }}
          className={`rounded-md px-4 py-2 text-sm font-medium ${
            activeTab === "prediction"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground"
          }`}
        >
          12 week prediction
        </button>
        <button
          type="button"
          onClick={() => {
            void setActiveTab("validation");
          }}
          className={`rounded-md px-4 py-2 text-sm font-medium ${
            activeTab === "validation"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground"
          }`}
        >
          Validation Graph
        </button>
      </section>

      {activeTab === "prediction" ? (
        <section className="h-[560px] w-full overflow-hidden rounded-lg border max-w-5xl ">
          <EmptyPredictionMap />
        </section>
      ) : (
        <>
          <section className="flex w-full flex-wrap items-end gap-4 rounded-lg border p-4 max-w-5xl">
            <label className="flex min-w-56 flex-col gap-2 text-sm">
              <span className="font-medium">Country</span>
              <select
                value={country}
                onChange={(event) => {
                  void setCountry(
                    event.target.value as (typeof EU_COUNTRIES)[number],
                  );
                }}
                className="rounded-md border bg-background px-3 py-2"
              >
                {EU_COUNTRIES.map((countryOption) => (
                  <option key={countryOption} value={countryOption}>
                    {countryOption}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex min-w-56 flex-col gap-2 text-sm">
              <span className="font-medium">Pathogen</span>
              <select
                value={pathogen}
                onChange={(event) => {
                  void setPathogen(
                    event.target.value as (typeof PATHOGENS)[number],
                  );
                }}
                className="rounded-md border bg-background px-3 py-2"
              >
                {PATHOGENS.map((pathogenOption) => (
                  <option key={pathogenOption} value={pathogenOption}>
                    {pathogenOption}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section className="relative flex w-full justify-center rounded-lg border p-4 max-w-5xl">
            {isLoading ? (
              <p className="absolute top-4 text-sm text-muted-foreground">
                Loading chart...
              </p>
            ) : null}

            {hasNoData ? (
              <p className="absolute top-4 text-sm text-muted-foreground">
                No data exists for the selected combination.
              </p>
            ) : null}

            {hasError ? (
              <p className="absolute top-4 text-sm text-destructive">
                Unable to load chart for this country/pathogen combination.
              </p>
            ) : null}

            <Image
              src={chartUrl}
              alt={`Validation chart for ${country} (${pathogen})`}
              width={1600}
              height={900}
              unoptimized
              className={`h-auto w-full max-w-6xl rounded-md ${hasError || hasNoData ? "hidden" : ""}`}
              onLoad={() => {
                setLoadedUrl(chartUrl);
                setErrorUrl(null);
                setNotFoundUrl(null);
              }}
              onError={() => {
                void (async () => {
                  try {
                    const response = await fetch(chartUrl, {
                      cache: "no-store",
                    });
                    if (response.status === 404) {
                      setNotFoundUrl(chartUrl);
                      setErrorUrl(null);
                      return;
                    }
                  } catch {
                    // no-op: handled below
                  }

                  setNotFoundUrl(null);
                  setErrorUrl(chartUrl);
                })();
              }}
            />
          </section>
        </>
      )}
    </main>
  );
}

export default function StatusPage() {
  return (
    <Suspense
      fallback={
        <main className="flex min-h-screen w-full flex-col gap-4 p-4">
          <SidebarTrigger size="icon-lg" className="self-start" />
          <p className="text-sm text-muted-foreground">Loading chart...</p>
        </main>
      }
    >
      <StatusPageContent />
    </Suspense>
  );
}
