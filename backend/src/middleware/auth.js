/**
 * JWT bearer authentication.
 *
 * Retrieval endpoints are not public: answers are generated from paid course
 * content, and every call costs an embedding request plus a generation request.
 * The token is issued by the main platform; this service only verifies it,
 * which keeps the RAG backend deployable as a separate process that shares
 * nothing with the platform except `JWT_SECRET` and the database.
 */

const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET || '';

if (!JWT_SECRET) {
  console.error('JWT_SECRET is not set — every authenticated request will be rejected.');
}

/**
 * Accepts `Authorization: Bearer <token>` and attaches `req.user`.
 * Historical tokens on this platform carry the user id under one of three
 * names, so all three are accepted and normalised to `userId`.
 */
function verifyToken(req, res, next) {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({ error: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    const userId = decoded.userId ?? decoded.id ?? decoded.user_id;
    req.user = { ...decoded, userId };
    return next();
  } catch {
    return res.status(401).json({ error: 'Invalid token.' });
  }
}

/**
 * Development escape hatch: with `AUTH_DISABLED=1` the middleware passes every
 * request through. Intended for reproducing the evaluation runs locally without
 * standing up the platform's auth service. Never enable it in production.
 */
function authMiddleware(req, res, next) {
  if (/^(1|true|yes)$/i.test(String(process.env.AUTH_DISABLED || '').trim())) {
    req.user = { userId: 0, role: 'anonymous' };
    return next();
  }
  return verifyToken(req, res, next);
}

module.exports = { verifyToken, authMiddleware };
