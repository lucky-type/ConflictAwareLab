// useExperimentWebSocket.ts - Custom hook for WebSocket connection
import { useEffect, useRef, useCallback } from 'react';
import { ExperimentWebSocket, ExperimentProgress, SimulationFrame } from '../services/api';

interface UseExperimentWebSocketOptions {
  experimentId: number | null;
  onProgress?: (data: ExperimentProgress) => void;
  onSimulationFrame?: (data: SimulationFrame) => void;
  onError?: (error: Event) => void;
}

export function useExperimentWebSocket({
  experimentId,
  onProgress,
  onSimulationFrame,
  onError
}: UseExperimentWebSocketOptions) {
  const wsRef = useRef<ExperimentWebSocket | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    // Create WebSocket connection
    const ws = new ExperimentWebSocket();
    wsRef.current = ws;

    // Set up event handlers
    if (onProgress) {
      ws.on('progress', onProgress);
    }
    if (onSimulationFrame) {
      ws.on('simulation_frame', onSimulationFrame);
    }

    // Connect
    ws.connect(experimentId);

    // Cleanup
    return () => {
      if (onProgress) ws.off('progress', onProgress);
      if (onSimulationFrame) ws.off('simulation_frame', onSimulationFrame);
      ws.disconnect();
      wsRef.current = null;
    };
  }, [experimentId, onProgress, onSimulationFrame, onError]);

  const disconnect = useCallback(() => {
    wsRef.current?.disconnect();
  }, []);

  return { disconnect };
}
