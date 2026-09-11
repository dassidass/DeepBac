-- Tiny example corpus, enough to run the pipeline end to end without the
-- production database. Load it after schema.sql, then:
--
--   node src/scripts/sync_chunks.js
--   node src/scripts/backfill_qdrant.js
--
-- The two lessons below are abbreviated stand-ins for real course content; they
-- exist to prove the wiring, not to teach anything.

SET NAMES utf8mb4;

INSERT INTO subjects (id, name, description, icon, color) VALUES
  (1, 'التاريخ والجغرافيا', 'History and Geography, Algerian BAC syllabus', '📚', '#6f42c1')
ON DUPLICATE KEY UPDATE name = VALUES(name);

INSERT INTO branches (id, name_ar, name_en) VALUES
  (1, 'آداب وفلسفة', 'Literature and Philosophy')
ON DUPLICATE KEY UPDATE name_ar = VALUES(name_ar);

INSERT INTO units (id, subject_id, branch_id, name) VALUES
  (1, 1, 1, 'الوحدة الأولى: العالم بعد الحرب العالمية الثانية')
ON DUPLICATE KEY UPDATE name = VALUES(name);

INSERT INTO courses (id, subject_id, unit_id, title, description, category, level) VALUES
  (1, 1, 1, 'الدرس الأول: تطور العالم في ظل الثنائية القطبية',
   'Bipolarity and the origins of the Cold War', 'history', 'basic'),
  (2, 1, 1, 'الدرس الثاني: الأزمات الدولية',
   'International crises of the Cold War period', 'history', 'basic')
ON DUPLICATE KEY UPDATE title = VALUES(title);

INSERT INTO course_parts (id, course_id, title, content, position) VALUES
  (1, 1, 'مفهوم الثنائية القطبية',
   'الثنائية القطبية هي نظام دولي ساد بعد الحرب العالمية الثانية، تقاسمت فيه الولايات المتحدة الأمريكية والاتحاد السوفياتي النفوذ في العالم، وقام كل قطب على تكتل عسكري واقتصادي وإيديولوجي خاص به.',
   0),
  (2, 1, 'مظاهر الصراع بين المعسكرين',
   'تجلى الصراع في سباق التسلح، والحرب الباردة الإعلامية، والأحلاف العسكرية مثل الحلف الأطلسي وحلف وارسو، إضافة إلى الصراع على مناطق النفوذ في العالم الثالث.',
   1),
  (3, 2, 'أزمة برلين',
   'اندلعت أزمة برلين سنة 1948 عندما فرض الاتحاد السوفياتي حصارا على القطاع الغربي من المدينة، فردت الدول الغربية بالجسر الجوي، وانتهت الأزمة سنة 1949 بتقسيم ألمانيا إلى دولتين.',
   0)
ON DUPLICATE KEY UPDATE content = VALUES(content);
