const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
const PUBLIC_DIR = path.join(__dirname, 'public');

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body)
  });
  res.end(body);
}

function serveStatic(req, res) {
  const cleanPath = req.url === '/' ? '/index.html' : req.url;
  const filePath = path.join(PUBLIC_DIR, path.normalize(cleanPath).replace(/^\/+/, ''));

  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    const ext = path.extname(filePath);
    const typeMap = {
      '.html': 'text/html; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
      '.js': 'application/javascript; charset=utf-8'
    };

    res.writeHead(200, { 'Content-Type': typeMap[ext] || 'application/octet-stream' });
    res.end(data);
  });
}

async function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => {
      try {
        resolve(JSON.parse(body || '{}'));
      } catch {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

async function handleChat(req, res) {
  if (!API_KEY) {
    sendJson(res, 500, { error: 'Missing OPENAI_API_KEY environment variable.' });
    return;
  }

  let payload;
  try {
    payload = await parseBody(req);
  } catch (err) {
    sendJson(res, 400, { error: err.message });
    return;
  }

  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  if (messages.length === 0) {
    sendJson(res, 400, { error: 'messages is required.' });
    return;
  }

  const input = messages.map(msg => ({
    role: msg.role,
    content: [{ type: 'input_text', text: String(msg.content || '') }]
  }));

  try {
    const response = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: MODEL,
        input,
        temperature: 0.7,
        max_output_tokens: 600
      })
    });

    const data = await response.json();
    if (!response.ok) {
      sendJson(res, response.status, { error: data.error?.message || 'OpenAI request failed', raw: data });
      return;
    }

    sendJson(res, 200, {
      text: data.output_text || 'No text returned.',
      usage: data.usage || null,
      model: data.model || MODEL
    });
  } catch (err) {
    sendJson(res, 500, { error: `Server error: ${err.message}` });
  }
}

const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/api/chat') {
    handleChat(req, res);
    return;
  }

  if (req.method === 'GET') {
    serveStatic(req, res);
    return;
  }

  res.writeHead(405);
  res.end('Method Not Allowed');
});

server.listen(PORT, () => {
  console.log(`Chat app running at http://localhost:${PORT}`);
});
