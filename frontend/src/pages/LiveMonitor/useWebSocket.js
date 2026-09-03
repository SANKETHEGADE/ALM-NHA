import { useState, useEffect, useRef, useCallback } from 'react';

export function useWebSocket(sessionId, options = {}) {
  const {
    url = (typeof window !== 'undefined'
      ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:4000/ws/sessions/${sessionId || 'default'}`
      : `ws://localhost:4000/ws/sessions/${sessionId || 'default'}`),
    autoReconnect = true,
    reconnectInterval = 3000,
    onResultReady = null,
    onAlertRaised = null
  } = options;

  const [status, setStatus] = useState('disconnected');
  const [latestResult, setLatestResult] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [messages, setMessages] = useState([]);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const shouldReconnectRef = useRef(autoReconnect);

  const connect = useCallback(() => {
    if (!sessionId) return;
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setStatus('connecting');

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setMessages((prev) => [data, ...prev.slice(0, 49)]);

          if (data.event === 'result_ready') {
            setLatestResult(data.payload);
            if (onResultReady) onResultReady(data.payload);
          } else if (data.event === 'alert_raised') {
            const newAlert = data.payload;
            setAlerts((prev) => {
              if (newAlert.id && prev.some((a) => a.id === newAlert.id)) {
                return prev;
              }
              return [newAlert, ...prev];
            });
            if (onAlertRaised) onAlertRaised(data.payload);
          }
        } catch (err) {
          // ignore parse error
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        wsRef.current = null;
        if (shouldReconnectRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = () => {
        setStatus('error');
        ws.close();
      };
    } catch (err) {
      setStatus('error');
    }
  }, [sessionId, url, reconnectInterval, onResultReady, onAlertRaised]);

  useEffect(() => {
    shouldReconnectRef.current = autoReconnect;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect, autoReconnect]);

  const dismissAlert = useCallback((alertId) => {
    setAlerts((prev) => prev.filter((a) => (a.id || a.alert_id) !== alertId));
  }, []);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  const injectMockResult = useCallback((mockResult) => {
    setLatestResult(mockResult);
    if (mockResult.alerts && Array.isArray(mockResult.alerts)) {
      setAlerts(mockResult.alerts);
    }
  }, []);

  return {
    status,
    isConnected: status === 'connected',
    latestResult,
    alerts,
    messages,
    dismissAlert,
    clearAlerts,
    injectMockResult,
    reconnect: connect
  };
}

export default useWebSocket;

