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
  const [videoFeedUrl, setVideoFeedUrl] = useState("http://localhost:8000/api/video_feed");
  const [retryKey, setRetryKey] = useState(0);

  // hit fastapi directly on port 8000
  React.useEffect(() => {
    if (typeof window !== "undefined") {
      const host = window.location.hostname || "localhost";
      setVideoFeedUrl(`http://${host}:8000/api/video_feed`);
    }
  }, []);

  // reconnectng retry if stream drops
  React.useEffect(() => {
    if (streamError) {
      const timer = setTimeout(() => {
        setRetryKey((k) => k + 1);
        setStreamError(false);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [streamError]);

  return (
    <div
      className={`relative w-full h-full overflow-hidden bg-stark-obsidian ${className}`}
    >
      {/* fallback backgroud when cam is offline */}
      <div
        className="absolute inset-0 bg-cover bg-center transition-opacity duration-500 pointer-events-none"
        style={{
          backgroundImage: "url('/stark_clean_room.jpg')",
          opacity: cameraActive && !streamError ? 0 : 1,
        }}
      />

      {/* mjpeg stream from backend */}
      {cameraActive && !streamError && (
        <img
          key={retryKey}
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

      {/* scanline overlay */}
      <div className="absolute inset-0 scanlines opacity-30 pointer-events-none" />

      {/* corner vignette */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          boxShadow:
            "inset 0 0 120px rgba(0, 0, 0, 0.9), inset 0 0 50px rgba(0, 245, 255, 0.15)",
        }}
      />

      {/* tracking reticle layer */}
      <div className="absolute inset-0 pointer-events-none z-10">
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
