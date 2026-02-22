-- Distributors table — vaccine/pharma distributors for logistics routing
-- Run this in the Supabase SQL Editor
-- Module 3 retrieves closest distributors per pharmacy from this table

CREATE TABLE IF NOT EXISTS distributors (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  country TEXT NOT NULL,
  city TEXT NOT NULL,
  latitude NUMERIC(10, 6),
  longitude NUMERIC(10, 6),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_distributors_country ON distributors(country);
CREATE INDEX IF NOT EXISTS idx_distributors_city ON distributors(city);

-- Seed with major vaccine manufacturers (EU hubs for logistics)
INSERT INTO distributors (id, name, country, city, latitude, longitude) VALUES
('DIST-SANOFI', 'Sanofi Vaccines', 'France', 'Lyon', 45.7640, 4.8357),
('DIST-PFIZER', 'Pfizer/BioNTech', 'Germany', 'Berlin', 52.5200, 13.4050),
('DIST-GSK', 'GSK Vaccines', 'Belgium', 'Wavre', 50.7167, 4.6167),
('DIST-MODERNA', 'Moderna', 'Switzerland', 'Basel', 47.5596, 7.5886),
('DIST-MSD', 'MSD', 'Netherlands', 'Haarlem', 52.3874, 4.6462),
('DIST-VIATRIS', 'Viatris', 'Germany', 'Munich', 48.1351, 11.5820),
('DIST-NOVAVAX', 'Novavax', 'Sweden', 'Stockholm', 59.3293, 18.0686),
('DIST-ASTRAZENECA', 'AstraZeneca', 'United Kingdom', 'Cambridge', 52.2053, 0.1218)
ON CONFLICT (id) DO NOTHING;
