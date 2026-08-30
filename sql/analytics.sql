-- Defect count by type
SELECT defect_type, COUNT(*) AS total
FROM inspections
GROUP BY defect_type
ORDER BY total DESC;

-- Average confidence by defect type
SELECT defect_type, ROUND(AVG(confidence), 4) AS avg_confidence
FROM inspections
GROUP BY defect_type;

-- Daily inspection volume, last 30 days
SELECT DATE(inspected_at) AS day, COUNT(*) AS inspections
FROM inspections
WHERE inspected_at >= NOW() - INTERVAL '30 days'
GROUP BY day
ORDER BY day;

-- Rolling 7-inspection average confidence per board
SELECT 
    board_id, 
    inspected_at, 
    confidence, 
    AVG(confidence) OVER (
        PARTITION BY board_id 
        ORDER BY inspected_at 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_confidence
FROM inspections
ORDER BY board_id, inspected_at;