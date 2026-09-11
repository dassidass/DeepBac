-- DeepBac RAG backend — minimal MySQL schema.
--
-- This is the subset of the platform schema the retrieval service actually
-- touches. The full platform adds users, subscriptions, exercises, progress and
-- payments; none of those are needed to run or reproduce retrieval.
--
-- utf8mb4 throughout. Ordinary Arabic — letters, diacritics, presentation forms
-- — fits in three bytes and would survive the legacy `utf8` alias. What would
-- not: emoji in a typed question, and the Arabic Mathematical Alphabetic
-- Symbols block (U+1EE00-U+1EEFF) that appears in transcribed formulas. On a
-- three-byte column MySQL rejects or truncates those, and a truncation inside
-- curriculum content is irreversible.

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- Curriculum: subjects → units → courses → course_parts
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS subjects (
  id          INT PRIMARY KEY AUTO_INCREMENT,
  name        VARCHAR(100) NOT NULL,
  description TEXT,
  icon        VARCHAR(10),
  color       VARCHAR(7) DEFAULT '#007bff',
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  -- Retrieval filters by subject name coming from the client, so the name has
  -- to be unique or a query could silently address the wrong corpus.
  UNIQUE KEY uq_subjects_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Academic streams (الشعب). A subject is taught to several streams with
-- different unit lists, which is why units carry an optional branch.
CREATE TABLE IF NOT EXISTS branches (
  id         INT PRIMARY KEY AUTO_INCREMENT,
  name_ar    VARCHAR(100) NOT NULL,
  name_en    VARCHAR(100),
  icon       VARCHAR(10),
  color      VARCHAR(7) DEFAULT '#007bff',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS units (
  id          INT PRIMARY KEY AUTO_INCREMENT,
  subject_id  INT NOT NULL,
  branch_id   INT NULL,
  name        VARCHAR(200) NOT NULL,
  description TEXT,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
  FOREIGN KEY (branch_id)  REFERENCES branches(id) ON DELETE SET NULL,
  UNIQUE KEY uq_units_subject_branch_name (subject_id, branch_id, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS courses (
  id          INT PRIMARY KEY AUTO_INCREMENT,
  subject_id  INT NOT NULL,
  unit_id     INT NULL,
  title       VARCHAR(200) NOT NULL,
  description TEXT,
  -- Legacy single-blob lesson body. New content lives in course_parts; this
  -- column is still read as a fallback so pre-migration lessons stay indexed.
  content     LONGTEXT,
  category    VARCHAR(50) DEFAULT NULL,
  level       ENUM('basic','intermediate','advanced') DEFAULT 'basic',
  duration    INT DEFAULT 0,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
  FOREIGN KEY (unit_id)    REFERENCES units(id) ON DELETE SET NULL,
  KEY idx_courses_subject (subject_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- One authored section of a lesson, and the unit of retrieval. Teachers write
-- parts at the granularity students read them, which is why chunking follows
-- part boundaries instead of a character window.
CREATE TABLE IF NOT EXISTS course_parts (
  id         INT PRIMARY KEY AUTO_INCREMENT,
  course_id  INT NOT NULL,
  title      VARCHAR(200) NOT NULL,
  content    LONGTEXT,
  position   INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
  KEY idx_course_parts_course (course_id, position)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Retrieval corpus
-- ---------------------------------------------------------------------------

-- Derived, never authored by hand: rebuilt from course_parts by
-- services/ragChunkSync.js. Deleting the whole table is safe — a sync restores
-- it — which is the property that makes re-chunking a cheap experiment.
--
-- The vector for each row lives in Qdrant, keyed back to `id` through the
-- point payload. Text stays here so an answer's citations are always read from
-- the system of record rather than from a possibly stale index.
CREATE TABLE IF NOT EXISTS rag_chunks (
  id            INT PRIMARY KEY AUTO_INCREMENT,
  course_id     INT NOT NULL,
  -- The authored section this chunk came from, and the knowledge tracer's
  -- micro-skill key: `dkt/` predicts mastery per course_parts.id, so a weakness
  -- prediction joins straight to retrievable text. Null only on the legacy
  -- path, where a whole courses.content blob became one chunk and there is no
  -- section to point at.
  part_id       INT NULL,
  chunk_index   INT NOT NULL,
  section_title VARCHAR(255),
  content       LONGTEXT NOT NULL,
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
  FOREIGN KEY (part_id)   REFERENCES course_parts(id) ON DELETE CASCADE,
  -- Second line of defence for part-backed chunks, and it keeps the Qdrant
  -- point id (derived from course_id and chunk_index) collision-free. It does
  -- NOT constrain legacy rows: InnoDB treats each NULL as distinct, so a unique
  -- index containing one never rejects a duplicate. Idempotency on both paths
  -- comes from the DELETE that precedes every insert in ragChunkSync.js.
  UNIQUE KEY unique_chunk (course_id, part_id, chunk_index),
  KEY idx_rag_chunks_course (course_id, chunk_index),
  -- Resolving "which chunks belong to section 149" is the DKT → RAG direction,
  -- and it runs on every recommendation the client acts on.
  KEY idx_rag_chunks_part (part_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Optional: conversation history, when the service is deployed with the chat UI
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ai_chat_sessions (
  id         INT PRIMARY KEY AUTO_INCREMENT,
  user_id    INT NOT NULL,
  subject    VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_ai_chat_sessions_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ai_chat_messages (
  id         INT PRIMARY KEY AUTO_INCREMENT,
  session_id INT NOT NULL,
  role       ENUM('user','assistant') NOT NULL,
  content    MEDIUMTEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES ai_chat_sessions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
