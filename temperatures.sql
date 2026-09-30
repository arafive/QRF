SELECT      kss.omirl_code,
            a.label,
            b.publication_date,
            t.day,
            t.temperature_max,
            t.temperature_min  
FROM        ligurian_bulletin.temperatures AS t
JOIN        ligurian_bulletin.areas AS a ON t.area_id = a.id
JOIN        ligurian_bulletin.bulletins AS b ON t.bulletin_id = b.id
LEFT JOIN   ligurian_bulletin.kalman_standard_stations AS kss ON a.kalman_code_id = kss.id
WHERE       t.day >= '2026-01-01' AND
            b.publication_date IS NOT NULL
ORDER BY    t.bulletin_id, t.day; 