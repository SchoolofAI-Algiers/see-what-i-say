"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import HelmetVisorFrame from "@/components/HelmetVisorFrame";
import ArcReactorHUD from "@/components/ArcReactorHUD";
import AudioSpectrumHUD from "@/components/AudioSpectrumHUD";
import ArmorDiagnosticsHUD from "@/components/ArmorDiagnosticsHUD";
import CommandConsoleHUD from "@/components/CommandConsoleHUD";
import VisionCanvas from "@/components/VisionCanvas";
import BootSequence from "@/components/BootSequence";
import { DetectionBox } from "@/components/TargetReticleHUD";
import { EffectMode } from "@/components/CommandDock";

interface SpeechRecognitionResultItem {
  transcript: string;
}

interface SpeechRecognitionResultList {
  [index: number]: {
    [index: number]: SpeechRecognitionResultItem;
  };
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionInstance {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((err: unknown) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

export default function StarkVisionCockpit() {
  const [bootComplete, setBootComplete] = useState<boolean>(true); // Immediate cockpit entry for seamless demo
  const [query, setQuery] = useState<string>("sofa");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [effectMode, setEffectMode] = useState<EffectMode>("highlight");
  const [boxThreshold, setBoxThreshold] = useState<number>(0.35);
  const [textThreshold, setTextThreshold] = useState<number>(0.25);
  const [cameraActive, setCameraActive] = useState<boolean>(false); // Start with clean penthouse background, click to enable webcam

  // Telemetry & WebSocket state
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [inferenceMs, setInferenceMs] = useState<number>(28);
  const [vramUsedMb, setVramUsedMb] = useState<number>(3190);
  const [fps, setFps] = useState<number>(60.0);

  // Default locked detection matching the sofa in the clean room
  const [detections, setDetections] = useState<DetectionBox[]>([
    {
      id: 1,
      label: "SOFA",
      confidence: 0.968,
      box: [0.30, 0.54, 0.70, 0.82],
      distance: 2.15,
      isAmbiguous: false,
    },
  ]);

  const wsRef = useRef<WebSocket | null>(null);
  const fpsFramesRef = useRef<number[]>([]);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  // Submit Query handler
  const handleSubmitQuery = useCallback(
    (newQuery: string) => {
      if (!newQuery.trim()) return;
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "update_query",
            query: newQuery.trim(),
            box_threshold: boxThreshold,
            text_threshold: textThreshold,
          })
        );
      }
    },
    [boxThreshold, textThreshold]
  );

  // Locked 60.0 FPS Viewport Render Measurement
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();

    const measureFps = (now: number) => {
      animId = requestAnimationFrame(measureFps);
      const delta = now - lastTime;
      lastTime = now;
      if (delta > 0) {
        const currentFps = 1000 / delta;
        fpsFramesRef.current.push(currentFps);
        if (fpsFramesRef.current.length > 30) {
          fpsFramesRef.current.shift();
          const avgFps =
            fpsFramesRef.current.reduce((a, b) => a + b, 0) /
            fpsFramesRef.current.length;
          setFps(Math.min(60.0, avgFps));
        }
      }
    };
    animId = requestAnimationFrame(measureFps);

    return () => cancelAnimationFrame(animId);
  }, []);

  // WebSocket Connection to FastAPI Grounding Engine
  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connectWebSocket = () => {
      try {
        const host = window.location.hostname || "localhost";
        const wsUrl = `ws://${host}:8000/api/ws/ground`;
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
          setWsConnected(true);
        };

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.detections && data.detections.length > 0) {
              setDetections(data.detections);
            }
            if (data.inference_ms) {
              setInferenceMs(data.inference_ms);
            }
            if (data.vram_used_mb) {
              setVramUsedMb(data.vram_used_mb);
            }
          } catch (e) {
            console.error("Error parsing WebSocket packet:", e);
          }
        };

        socket.onclose = () => {
          setWsConnected(false);
          reconnectTimeout = setTimeout(connectWebSocket, 3000);
        };

        socket.onerror = () => {
          socket?.close();
        };

        wsRef.current = socket;
      } catch {
        reconnectTimeout = setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      clearTimeout(reconnectTimeout);
      socket?.close();
    };
  }, []);

  // Web Speech API Voice Recognition (Hands-free STT)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const windowWithSpeech = window as unknown as {
        SpeechRecognition?: new () => SpeechRecognitionInstance;
        webkitSpeechRecognition?: new () => SpeechRecognitionInstance;
      };

      const SpeechRecognitionClass =
        windowWithSpeech.SpeechRecognition ||
        windowWithSpeech.webkitSpeechRecognition;

      if (SpeechRecognitionClass) {
        const recognition = new SpeechRecognitionClass();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onresult = (event: SpeechRecognitionEvent) => {
          let transcript = "";
          for (let i = event.resultIndex; i < 100; i++) {
            if (!event.results[i] || !event.results[i][0]) break;
            transcript += event.results[i][0].transcript;
          }
          if (transcript.trim()) {
            const cleanText = transcript.trim().toLowerCase();
            setQuery(cleanText);
            handleSubmitQuery(cleanText);
          }
        };

        recognition.onerror = () => {
          setIsRecording(false);
        };

        recognition.onend = () => {
          if (isRecording) {
            try {
              recognition.start();
            } catch {
              setIsRecording(false);
            }
          }
        };

        recognitionRef.current = recognition;
      }
    }
  }, [isRecording, handleSubmitQuery]);

  // Toggle Recording handler
  const handleToggleRecording = useCallback(() => {
    setIsRecording((prev) => {
      const nextState = !prev;
      if (recognitionRef.current) {
        if (nextState) {
          try {
            recognitionRef.current.start();
          } catch {
            console.warn("Speech recognition failed to start");
          }
        } else {
          try {
            recognitionRef.current.stop();
          } catch {
            console.warn("Speech recognition failed to stop");
          }
        }
      }
      return nextState;
    });
  }, []);

  // Send Frame + Query to Grounding DINO via WebSocket
  const handleFrameCapture = useCallback(
    (base64Jpeg: string) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "ground_frame",
            image_data: base64Jpeg,
            query: query,
            box_threshold: boxThreshold,
            text_threshold: textThreshold,
          })
        );
      }
    },
    [query, boxThreshold, textThreshold]
  );

  return (
    <main className="relative w-screen h-screen overflow-hidden bg-[#05080e] select-none">
      {/* Optional Cinematic JARVIS Boot Sequence */}
      {!bootComplete && (
        <BootSequence onComplete={() => setBootComplete(true)} />
      )}

      {/* 1. Underlying Video / Photorealistic Penthouse Layer */}
      <VisionCanvas
        detections={detections}
        effectMode={effectMode}
        cameraActive={cameraActive}
        onFrameCapture={handleFrameCapture}
        className="absolute inset-0 z-0"
      />

      {/* 2. Authentic Mark 85 Helmet Visor Canopy (Faceted Armor Silhouette) */}
      <HelmetVisorFrame />

      {/* 3. Tactical Heads-Up Display Overlays */}
      <div className="absolute inset-0 pointer-events-none z-30 flex flex-col justify-between p-4 md:p-6">
        {/* Top Header Rail */}
        <div className="w-full flex items-start justify-between gap-4 pointer-events-auto">
          {/* Top-Left: Arc Reactor HUD */}
          <ArcReactorHUD className="pl-6 pt-3" />

          {/* Top-Center: Audio Spectrum Equalizer */}
          <AudioSpectrumHUD isRecording={isRecording} className="pt-2.5" />

          {/* Top-Right: Armor & GPU Telemetry Diagnostics */}
          <ArmorDiagnosticsHUD
            fps={fps}
            inferenceMs={inferenceMs}
            vramUsedMb={vramUsedMb}
            wsConnected={wsConnected}
            className="pr-6 pt-3"
          />
        </div>

        {/* Center Crosshair & Compass Strip */}
        <div className="w-full flex items-center justify-center pointer-events-none">
          <div className="font-mono text-[9px] tracking-widest text-slate-500 uppercase flex items-center gap-3">
            <span>HDG 042°</span>
            <span className="text-stark-cyan text-glow-cyan">───┼───</span>
            <span>ALT 1.2M</span>
          </div>
        </div>

        {/* Bottom Cockpit Chin Console (1:1 with Reference Mockup) */}
        <div className="w-full flex justify-center pointer-events-auto pb-2">
          <CommandConsoleHUD
            query={query}
            onQueryChange={setQuery}
            onSubmitQuery={handleSubmitQuery}
            isRecording={isRecording}
            onToggleRecording={handleToggleRecording}
            effectMode={effectMode}
            onEffectModeChange={setEffectMode}
            boxThreshold={boxThreshold}
            onBoxThresholdChange={setBoxThreshold}
            textThreshold={textThreshold}
            onTextThresholdChange={setTextThreshold}
            cameraActive={cameraActive}
            onToggleCamera={() => setCameraActive(!cameraActive)}
          />
        </div>
      </div>
    </main>
  );
}
