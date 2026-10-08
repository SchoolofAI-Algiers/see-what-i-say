"use client";
/* eslint-disable @next/next/no-img-element */

import React, { useState } from "react";
import TargetReticleHUD, { DetectionBox } from "./TargetReticleHUD";
import { EffectMode } from "./CommandDock";

interface VisionCanvasProps {
  detections: DetectionBox[];
  effectMode: EffectMode;
  cameraActive: boolean;
  className?: string;
}

export default function VisionCanvas({
  detections,
  effectMode,
  cameraActive,
  className = "",
}: VisionCanvasProps) {
  const [streamError, setStreamError] = useState(false);

  // Direct backend MJPEG video feed URL (proxied via Next.js rewrites)
  const videoFeedUrl = "/api/video_feed";

  return (
    <div
      className={`relative w-full h-full overflow-hidden bg-stark-obsidian ${className}`}
    >
      {/* 1. Photorealistic Stark Penthouse Lab Backdrop (active when camera is off or stream error) */}
      <div
        className="absolute inset-0 bg-cover bg-center transition-opacity duration-500 pointer-events-none"
        style={{
          backgroundImage: "url('/stark_clean_room.jpg')",
          opacity: cameraActive && !streamError ? 0 : 1,
        }}
      />

      {/* 2. Real-Time Python MJPEG Video Stream (30 FPS, hardware captured by OpenCV) */}
      {cameraActive && !streamError && (
        <img
          src={videoFeedUrl}
          alt="STARK VISION Live Feed"
          onError={() => setStreamError(true)}
          onLoad={() => setStreamError(false)}
          className={`w-full h-full object-cover transition-all duration-300 ${
            effectMode === "blur" ? "filter blur-sm" : ""
          } ${
            effectMode === "isolate" && detections.length > 0
              ? "filter brightness-50"
              : ""
          }`}
        />
      )}

      {/* 3. Subtle Tactical CRT Scanlines & Holographic Grid */}
      <div className="absolute inset-0 scanlines opacity-30 pointer-events-none" />

      {/* 4. Visor Edge Chamfer Vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          boxShadow:
            "inset 0 0 120px rgba(0, 0, 0, 0.9), inset 0 0 50px rgba(0, 245, 255, 0.15)",
        }}
      />

      {/* 5. Mark 85 Tactical HUD Target Reticles */}
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
