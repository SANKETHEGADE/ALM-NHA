const http = require('http');
const express = require('express');
const cors = require('cors');

const sceneRoutes = require('./modules/scene/routes');
const alertRoutes = require('./modules/alerts/routes');
const { initWebSocketServer } = require('./websocket/server');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json());

app.use('/api/v1/sessions', sceneRoutes);
app.use('/api/v1/alerts', alertRoutes);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'ALM-NHCE Scene & Alerts API' });
});

initWebSocketServer(server);

const PORT = process.env.PORT || 4000;

if (process.env.NODE_ENV !== 'test') {
  server.listen(PORT, () => {
    console.log(`[Backend] Server listening on http://localhost:${PORT}`);
    console.log(`[Backend] WebSocket available on ws://localhost:${PORT}/ws`);
  });
}

module.exports = { app, server };
