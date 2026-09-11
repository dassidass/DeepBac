/**
 * MySQL connection pool.
 *
 * MySQL is the system of record for this service. It holds the curriculum
 * (subjects → units → courses → course_parts) and the derived retrieval corpus
 * (`rag_chunks`). Qdrant only stores vectors plus a small payload that points
 * back at a `rag_chunks.id`, so MySQL always owns the authoritative text.
 *
 * A unix socket is preferred when one exists, because shared hosts frequently
 * resolve `localhost` to ::1 while the MySQL grant only allows 127.0.0.1.
 */

const mysql = require('mysql2/promise');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

dotenv.config();
dotenv.config({ path: path.join(__dirname, '..', '..', '.env'), override: false });

/** First non-empty value among the given environment variable names. */
const getEnv = (...keys) => {
  for (const key of keys) {
    const value = process.env[key];
    if (value != null && String(value).trim() !== '') return String(value);
  }
  return undefined;
};

const dbUser = getEnv('DB_USER', 'MYSQL_USER');
// An empty password is legitimate on local XAMPP/WAMP installs, so only an
// entirely absent variable counts as missing.
let dbPassword = getEnv('DB_PASSWORD', 'MYSQL_PASSWORD');
if (dbPassword === undefined && process.env.DB_PASSWORD === '') dbPassword = '';

const dbName = getEnv('DB_NAME', 'MYSQL_DATABASE');
const rawHost = getEnv('DB_HOST', 'MYSQL_HOST') || 'localhost';
const host = rawHost === 'localhost' ? '127.0.0.1' : rawHost;

const missingEnv = [
  ...(dbUser ? [] : ['DB_USER']),
  ...(dbPassword !== undefined ? [] : ['DB_PASSWORD']),
  ...(dbName ? [] : ['DB_NAME'])
];

const socketPath = [
  process.env.DB_SOCKET,
  '/var/run/mysqld/mysqld.sock',
  '/var/lib/mysql/mysql.sock',
  '/tmp/mysql.sock'
]
  .filter(Boolean)
  .find((candidate) => {
    try {
      return fs.existsSync(candidate);
    } catch {
      return false;
    }
  });

const dbConfig = {
  ...(socketPath ? { socketPath } : { host, port: Number(process.env.DB_PORT || 3306) }),
  user: dbUser,
  password: dbPassword,
  database: dbName,
  waitForConnections: true,
  connectionLimit: Number(process.env.DB_POOL_SIZE || 10),
  queueLimit: 0,
  connectTimeout: 60000,
  charset: 'utf8mb4_general_ci'
};

/** @type {import('mysql2/promise').Pool | null} */
let pool = null;

if (missingEnv.length > 0) {
  console.error('Database environment variables missing:', missingEnv.join(', '));
  console.error('Copy .env.example to .env and fill in DB_HOST / DB_USER / DB_PASSWORD / DB_NAME.');
} else {
  pool = mysql.createPool(dbConfig);
}

/** Cheap liveness probe used by the health endpoint and by scripts before long jobs. */
async function pingDatabase() {
  if (!pool) return { ok: false, reason: 'no_pool' };
  try {
    const conn = await pool.getConnection();
    await conn.ping();
    conn.release();
    return { ok: true, database: dbName, via: socketPath ? 'socket' : 'tcp' };
  } catch (err) {
    return { ok: false, reason: err.message };
  }
}

module.exports = { pool, pingDatabase, dbConfig };
