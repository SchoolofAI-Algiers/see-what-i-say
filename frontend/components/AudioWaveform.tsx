"use client";

import React, { useEffect, useRef, useState } from "react";

interface AudioWaveformProps {
  isRecording?: boolean;
  onTranscription?: (text: string) => void;
  className?: string;
}

export default function AudioWaveform({
  isRecording = false,
  className = "",
}: AudioWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [hasMicAccess, setHasMicAccess] = useState<boolean>(false);
  const [rmsLevel, setRmsLevel] = useState<number>(0);

  // Initialize Web Audio API on record trigger or user interaction
  useEffect(() => {
    let animId: number;

    const startAudio = async () => {
      try {
        if (!audioContextRef.current) {
          const AudioContextClass =
            window.AudioContext ||
            (window as unknown as { webkitAudioContext: typeof AudioContext })
              .webkitAudioContext;
          audioContextRef.current = new AudioContextClass();
        }

        if (audioContextRef.current.state === "suspended") {
          await audioContextRef.current.resume();
        }

        if (!streamRef.current) {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: true,
          });
          streamRef.current = stream;
          setHasMicAccess(true);

          const analyser = audioContextRef.current.createAnalyser();
          analyser.fftSize = 64;
          analyser.smoothingTimeConstant = 0.8;
          analyserRef.current = analyser;

          const source =
            audioContextRef.current.createMediaStreamSource(stream);
          source.connect(analyser);
          sourceRef.current = source;
        }
      } catch (err) {
        console.warn("Microphone access not granted or unavailable, using synthetic visualizer:", err);
      }
    };

    if (isRecording && !streamRef.current) {
      startAudio();
    }

    // Render loop
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let tick = 0;

    const render = () => {
      animId = requestAnimationFrame(render);
      tick += 0.05;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const barCount = 32;
      const barWidth = 3;
      const gap = (width - barCount * barWidth) / (barCount - 1);

      if (analyserRef.current && hasMicAccess) {
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Compute RMS level
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const avg = sum / bufferLength;
        setRmsLevel(Math.min(100, Math.round((avg / 255) * 100)));

        for (let i = 0; i < barCount; i++) {
          const index = Math.floor((i / barCount) * bufferLength);
          const value = dataArray[index] || 10;
          const barHeight = Math.max(4, (value / 255) * height * 0.9);

          const x = i * (barWidth + gap);
          const y = height / 2 - barHeight / 2;

          // Dual-tone Gradient
          const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
          gradient.addColorStop(0, "#FFBF00"); // Stark Gold top
          gradient.addColorStop(0.5, "#00F5FF"); // Cyan center
          gradient.addColorStop(1, "#FFBF00"); // Stark Gold bottom

          ctx.fillStyle = gradient;
          ctx.shadowColor = isRecording ? "#FFBF00" : "#00F5FF";
          ctx.shadowBlur = 6;
          ctx.fillRect(x, y, barWidth, barHeight);
        }
      } else {
        // Ambient idle breathing pattern
        setRmsLevel(isRecording ? 45 : 12);
        for (let i = 0; i < barCount; i++) {
          const centerDist = Math.abs(i - barCount / 2) / (barCount / 2);
          const wave =
            Math.sin(tick + i * 0.3) * 0.5 + Math.cos(tick * 1.5 - i * 0.2) * 0.5;
          const baseMultiplier = isRecording ? 0.75 : 0.28;
          const barHeight = Math.max(
            3,
            (1 - centerDist * 0.6) * (wave * 0.5 + 0.5) * height * baseMultiplier
          );

          const x = i * (barWidth + gap);
          const y = height / 2 - barHeight / 2;

          const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
          gradient.addColorStop(0, isRecording ? "#FFBF00" : "#00F5FF");
          gradient.addColorStop(1, isRecording ? "#FF2E2E" : "#1E90FF");

          ctx.fillStyle = gradient;
          ctx.shadowColor = isRecording ? "#FFBF00" : "#00F5FF";
          ctx.shadowBlur = 4;
          ctx.fillRect(x, y, barWidth, barHeight);
        }
      }
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [isRecording, hasMicAccess]);

  return (
    <div
      className={`hud-glass px-4 py-2 flex flex-col items-center gap-1.5 border-b-2 ${
        isRecording ? "border-b-stark-gold" : "border-b-stark-cyan"
      } ${className}`}
    >
      <div className="flex items-center justify-between w-full gap-4">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isRecording
                ? "bg-stark-gold animate-ping"
                : "bg-stark-cyan animate-pulse"
            }`}
          />
          <span className="font-orbitron text-[10px] font-bold tracking-widest text-stark-cyan uppercase">
            {isRecording ? "TRANSMITTING VOICE COMMAND" : "AUDIO SPECTRUM // RECEPTION"}
          </span>
        </div>
        <div className="font-mono text-[9px] text-slate-400 flex items-center gap-2">
          <span>GAIN: <strong className="text-stark-gold">{rmsLevel}%</strong></span>
          <span className="text-slate-600">|</span>
          <span className={isRecording ? "text-stark-gold font-bold animate-pulse" : "text-slate-400"}>
            {isRecording ? "REC ACTIVE" : "VOX STANDBY"}
          </span>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={240}
        height={32}
        className="w-[240px] h-[32px]"
      />

      <div className="w-full flex justify-between items-center text-[8px] font-mono text-slate-400 tracking-wider">
        <span>20 Hz</span>
        <span className="text-stark-gold">MARK 85 NEURAL ACOUSTIC</span>
        <span>20 kHz</span>
      </div>
    </div>
  );
}
