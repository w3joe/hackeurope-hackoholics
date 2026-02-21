-- Orchestration results table — stores replenishment directives after pipeline completion
-- Run this in the Supabase SQL Editor

DROP TABLE IF EXISTS orchestration_results CASCADE;

CREATE TABLE orchestration_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  replenishment_directives JSONB NOT NULL DEFAULT '[]',
  grand_total_cost_usd NUMERIC(12, 2) NOT NULL DEFAULT 0,
  overall_system_summary TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Realtime for frontend subscription (optional, same pattern as alerts)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime' AND tablename = 'orchestration_results'
  ) THEN
    ALTER PUBLICATION supabase_realtime ADD TABLE orchestration_results;
  END IF;
END $$;
