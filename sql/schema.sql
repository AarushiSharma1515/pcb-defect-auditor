CREATE TABLE inspections (
    id              SERIAL PRIMARY KEY,
    board_id        VARCHAR(100) NOT NULL,
    image_path      TEXT NOT NULL,
    defect_type     VARCHAR(50) NOT NULL,
    confidence      NUMERIC(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    inspected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_ms   INTEGER
);

CREATE INDEX idx_inspections_defect_type ON inspections (defect_type);
CREATE INDEX idx_inspections_inspected_at ON inspections (inspected_at);