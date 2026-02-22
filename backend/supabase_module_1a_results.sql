-- Module 1A Risk Assessment results — stores TDA/Holt-Winters risk assessments per country
-- Run this in the Supabase SQL Editor
-- Table is truncated on each Module 1A run, then repopulated with new results

DROP TABLE IF EXISTS module_1a_results CASCADE;

CREATE TABLE module_1a_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  country TEXT NOT NULL,
  risk_level TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
  spread_likelihood NUMERIC(5, 2) NOT NULL DEFAULT 0,
  reasoning TEXT NOT NULL DEFAULT '',
  recommended_disease_focus JSONB NOT NULL DEFAULT '[]',
  twelve_week_forecast JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RPC for truncating table (called at start of each Module 1A run)
CREATE OR REPLACE FUNCTION truncate_module_1a_results()
RETURNS void AS $$
BEGIN
  TRUNCATE module_1a_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Enable Realtime for frontend subscription (optional)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime' AND tablename = 'module_1a_results'
  ) THEN
    ALTER PUBLICATION supabase_realtime ADD TABLE module_1a_results;
  END IF;
END $$;
