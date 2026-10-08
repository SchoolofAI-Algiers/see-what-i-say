"use client";

import React, { useEffect, useRef } from "react";

interface AudioSpectrumHUDProps {
  isRecording?: boolean;
  className?: string;
}

export default function AudioSpectrumHUD({
  isRecording = false,
  className = "",
}: AudioSpectrumHUDProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let animId: number;

    const initAudio = async () => {
      try {
        if (!audioCtxRef.current) {
          const AudioContextClass =
            window.AudioContext ||
            (window as unknown as { webkitAudioContext: typeof AudioContext })
              .webkitAudioContext;
          audioCtxRef.current = new AudioContextClass();
        }

        if (audioCtxRef.current.state === "suspended") {
          await audioCtxRef.current.resume();
        }

        if (!streamRef.current) {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: true,
          });
          streamRef.current = stream;

          const analyser = audioCtxRef.current.createAnalyser();
          analyser.fftSize = 64;
          analyser.smoothingTimeConstant = 0.75;
          analyserRef.current = analyser;

          const source = audioCtxRef.current.createMediaStreamSource(stream);
          source.connect(analyser);
        }
      } catch (err) {
        console.warn("Using simulated Jarvis acoustic waveform:", err);
      }
    };

    if (isRecording && !streamRef.current) {
      initAudio();
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let tick = 0;

    const render = () => {
      animId = requestAnimationFrame(render);
      tick += 0.04;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const barCount = 28;
      const barWidth = 4;
      const gap = 3;
      const totalWidth = barCount * barWidth + (barCount - 1) * gap;
      const startX = (width - totalWidth) / 2;

      let freqData: Uint8Array | null = null;
      if (analyserRef.current) {
        const buffer = new ArrayBuffer(analyserRef.current.frequencyBinCount);
        const dataArr = new Uint8Array(buffer);
        analyserRef.current.getByteFrequencyData(dataArr);
        freqData = dataArr;
      }

      for (let i = 0; i < barCount; i++) {
        let barHeight = 6;

        if (freqData) {
          const idx = Math.floor((i / barCount) * freqData.length);
          const val = freqData[idx] || 10;
          barHeight = Math.max(6, (val / 255) * height * 0.95);
        } else {
          // idle breathing wave when mic is quiet
          const centerDist = Math.abs(i - barCount / 2) / (barCount / 2);
          const wave =
            Math.sin(tick * 2 + i * 0.35) * 0.45 +
            Math.cos(tick * 3 - i * 0.2) * 0.35 +
            0.5;
          const mult = isRecording ? 0.9 : 0.55;
          barHeight = Math.max(
            5,
            (1 - centerDist * 0.45) * wave * height * mult
          );
        }

        const x = startX + i * (barWidth + gap);

        // draw segmented bars
        const segmentCount = Math.floor(barHeight / 5);
        for (let s = 0; s < segmentCount; s++) {
          const segY = height - (s + 1) * 5;
          const isTopHalf = s > segmentCount * 0.5;

          ctx.fillStyle = isTopHalf ? "#FFBF00" : "#00F5FF";
          ctx.shadowColor = isTopHalf ? "#FFBF00" : "#00F5FF";
          ctx.shadowBlur = 4;
          ctx.fillRect(x, segY, barWidth, 3.5);
        }
      }
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [isRecording]);

  return (
    <div
      className={`flex flex-col items-center gap-1 select-none pointer-events-auto ${className}`}
    >
      <div className="flex items-center gap-2">
        <div className="w-10 h-[1.5px] bg-gradient-to-r from-transparent to-stark-cyan opacity-80" />

        <div className="font-orbitron text-stark-gold text-sm font-bold text-glow-gold">
          &#123;
        </div>

        {/* waveform canvas */}
        <canvas
          ref={canvasRef}
          width={240}
          height={38}
          className="w-[220px] h-[36px]"
        />

        <div className="font-orbitron text-stark-gold text-sm font-bold text-glow-gold">
          &#125;
        </div>

        <div className="w-10 h-[1.5px] bg-gradient-to-l from-transparent to-stark-cyan opacity-80" />
      </div>

      {/* Center Caption */}
      <div className="flex items-center gap-2">
        <span className="w-1 h-1 rounded-full bg-stark-cyan animate-ping" />
        <span className="font-orbitron text-[9px] md:text-[10px] font-bold tracking-[0.25em] text-stark-cyan text-glow-cyan uppercase">
          AUDIO SPECTRUM // RECEPTION
        </span>
        <span className="w-1 h-1 rounded-full bg-stark-gold animate-ping" />
      </div>
    </div>
  );
}
