-- Alerts table for real-time push notifications to frontend
-- Run this in the Supabase SQL Editor after vaccine_stock setup
-- Columns use camelCase to match Alert interface exactly

DROP TABLE IF EXISTS alerts CASCADE;

CREATE TABLE alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  "affectedStoreIds" TEXT[] NOT NULL,
  "timestamp" BIGINT NOT NULL,
  description TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('low', 'watch', 'urgent')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Realtime for this table so frontend can subscribe to INSERTs (idempotent)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime' AND tablename = 'alerts'
  ) THEN
    ALTER PUBLICATION supabase_realtime ADD TABLE alerts;
  END IF;
END $$;

-- Optional: RLS for secure access (adjust as needed for your auth model)
-- ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Allow anon read" ON alerts FOR SELECT USING (true);
-- CREATE POLICY "Allow service role insert" ON alerts FOR INSERT WITH CHECK (true);
