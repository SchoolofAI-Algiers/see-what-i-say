"use client";

import React, { useRef, useEffect, useState } from "react";
import TargetReticleHUD, { DetectionBox } from "./TargetReticleHUD";
import { EffectMode } from "./CommandDock";

interface VisionCanvasProps {
  detections: DetectionBox[];
  effectMode: EffectMode;
  cameraActive: boolean;
  onFrameCapture?: (base64Jpeg: string) => void;
  className?: string;
}

export default function VisionCanvas({
  detections,
  effectMode,
  cameraActive,
  onFrameCapture,
  className = "",
}: VisionCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const [streamReady, setStreamReady] = useState(false);

  // Initialize Webcam when cameraActive is true
  useEffect(() => {
    let currentStream: MediaStream | null = null;

    if (cameraActive) {
      navigator.mediaDevices
        ?.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: "user",
          },
          audio: false,
        })
        .then((stream) => {
          currentStream = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadedmetadata = () => {
              videoRef.current?.play();
              setStreamReady(true);
            };
          }
        })
        .catch((err) => {
          console.warn("Camera access not available, displaying Stark Penthouse Lab:", err);
          setStreamReady(false);
        });
    } else {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      setStreamReady(false);
    }

    return () => {
      if (currentStream) {
        currentStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraActive]);

  // Frame Capture Interval for Grounding DINO
  useEffect(() => {
    if (!onFrameCapture) return;

    const interval = setInterval(() => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = 640;
      canvas.height = 360;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const base64Jpeg = canvas.toDataURL("image/jpeg", 0.7);
      onFrameCapture(base64Jpeg);
    }, 250);

    return () => clearInterval(interval);
  }, [onFrameCapture]);

  return (
    <div
      className={`relative w-full h-full overflow-hidden bg-stark-obsidian ${className}`}
    >
      {/* Hidden frame capture canvas */}
      <canvas ref={captureCanvasRef} className="hidden" />

      {/* 1. Default Photorealistic Stark Penthouse Living Room Backdrop */}
      <div
        className="absolute inset-0 bg-cover bg-center transition-opacity duration-500 pointer-events-none"
        style={{
          backgroundImage: "url('/stark_clean_room.jpg')",
          opacity: streamReady && cameraActive ? 0 : 1,
        }}
      />

      {/* 2. Live Fullscreen Edge-to-Edge Webcam Stream */}
      <video
        ref={videoRef}
        playsInline
        autoPlay
        muted
        className={`w-full h-full object-cover transition-all duration-500 ${
          streamReady && cameraActive ? "opacity-100" : "opacity-0 pointer-events-none"
        } ${effectMode === "blur" ? "filter blur-sm" : ""} ${
          effectMode === "isolate" && detections.length > 0 ? "filter brightness-50" : ""
        }`}
      />

      {/* 3. Subtle CRT Scanlines & Holographic Texture */}
      <div className="absolute inset-0 scanlines opacity-30 pointer-events-none" />

      {/* 4. Visor Edge Vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          boxShadow:
            "inset 0 0 120px rgba(0, 0, 0, 0.9), inset 0 0 50px rgba(0, 245, 255, 0.15)",
        }}
      />

      {/* 5. Render Dynamic Target Reticles */}
      <div className="absolute inset-0 pointer-events-none">
        {detections.map((detection) => (
          <TargetReticleHUD
            key={detection.id}
            detection={detection}
            effectMode={effectMode}
            recolorTint="rgba(0, 245, 255, 0.28)"
          />
        ))}
      </div>
    </div>
  );
}
