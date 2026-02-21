interface NominatimResult {
  lat: string;
  lon: string;
}

const cache = new Map<string, { lat: number; lng: number }>();
const failedCache = new Set<string>();

function buildAddressCandidates(address: string): string[] {
  const trimmed = address.trim();
  const parts = trimmed
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);

  const candidates: string[] = [];
  const addCandidate = (value: string) => {
    const normalized = value.trim();
    if (!normalized || candidates.includes(normalized)) return;
    candidates.push(normalized);
  };

  addCandidate(trimmed);

  const hasLikelyVenuePrefix =
    parts.length >= 2 && !/\d/.test(parts[0]) && /\d/.test(parts[1]);

  if (hasLikelyVenuePrefix) {
    addCandidate(parts.slice(1).join(", "));
  }

  if (parts.length >= 3) {
    addCandidate(parts.slice(-3).join(", "));
  }

  if (parts.length >= 2) {
    addCandidate(parts.slice(-2).join(", "));
  }

  return candidates;
}

export async function geocode(
  address: string,
): Promise<{ lat: number; lng: number } | null> {
  const cached = cache.get(address);
  if (cached) return cached;
  if (failedCache.has(address)) return null;

  const candidates = buildAddressCandidates(address);

  for (const candidate of candidates) {
    const candidateCached = cache.get(candidate);
    if (candidateCached) {
      cache.set(address, candidateCached);
      return candidateCached;
    }

    if (failedCache.has(candidate)) {
      continue;
    }

    const url = new URL("https://nominatim.openstreetmap.org/search");
    url.searchParams.set("q", candidate);
    url.searchParams.set("format", "json");
    url.searchParams.set("limit", "1");

    const res = await fetch(url, {
      headers: { "User-Agent": "HackEurope2026-Frontend/1.0" },
      next: { revalidate: 86400 },
    });

    if (!res.ok) {
      failedCache.add(candidate);
      continue;
    }

    const data: NominatimResult[] = await res.json();
    if (data.length === 0) {
      failedCache.add(candidate);
      continue;
    }

    const result = { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) };
    cache.set(candidate, result);
    cache.set(address, result);
    return result;
  }

  failedCache.add(address);
  return null;
}

export async function geocodeBranches<
  T extends { address: string; lat?: number; lng?: number },
>(branches: T[]): Promise<(T & { lat: number; lng: number })[]> {
  const results: (T & { lat: number; lng: number })[] = [];

  for (const branch of branches) {
    if (branch.lat != null && branch.lng != null) {
      results.push(branch as T & { lat: number; lng: number });
      continue;
    }

    const coords = await geocode(branch.address);
    if (coords) {
      results.push({ ...branch, lat: coords.lat, lng: coords.lng });
    } else {
      // Fallback to 0,0 if geocoding fails — should not happen with valid addresses
      results.push({ ...branch, lat: 0, lng: 0 });
    }
  }

  return results;
}
