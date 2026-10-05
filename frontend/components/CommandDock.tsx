"use client";

import React, { useState, useEffect } from "react";
import {
  Mic,
  MicOff,
  Search,
  Sliders,
  Sparkles,
  Camera,
  Layers,
  Eye,
  Crosshair,
} from "lucide-react";

export type EffectMode = "highlight" | "recolor" | "blur" | "isolate";

interface CommandDockProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSubmitQuery: (query: string) => void;
  isRecording: boolean;
  onToggleRecording: () => void;
  effectMode: EffectMode;
  onEffectModeChange: (mode: EffectMode) => void;
  boxThreshold: number;
  onBoxThresholdChange: (val: number) => void;
  textThreshold: number;
  onTextThresholdChange: (val: number) => void;
  cameraActive: boolean;
  onToggleCamera: () => void;
  className?: string;
}

const PRESET_QUERIES = [
  "person",
  "laptop",
  "cell phone",
  "bottle",
  "chair",
  "backpack",
];

export default function CommandDock({
  query,
  onQueryChange,
  onSubmitQuery,
  isRecording,
  onToggleRecording,
  effectMode,
  onEffectModeChange,
  boxThreshold,
  onBoxThresholdChange,
  textThreshold,
  onTextThresholdChange,
  cameraActive,
  onToggleCamera,
  className = "",
}: CommandDockProps) {
  const [showConfig, setShowConfig] = useState(false);

  // Global Keyboard Shortcuts (Space for mic, 1-4 for effect modes)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        if (e.key === "Enter") {
          onSubmitQuery(query);
        }
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        onToggleRecording();
      } else if (e.key === "1") {
        onEffectModeChange("highlight");
      } else if (e.key === "2") {
        onEffectModeChange("recolor");
      } else if (e.key === "3") {
        onEffectModeChange("blur");
      } else if (e.key === "4") {
        onEffectModeChange("isolate");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [query, onToggleRecording, onSubmitQuery, onEffectModeChange]);

  return (
    <div
      className={`fixed bottom-5 left-1/2 -translate-x-1/2 w-[92%] max-w-4xl flex flex-col items-center gap-2 z-40 ${className}`}
    >
      {/* Expandable Configuration Drawer */}
      {showConfig && (
        <div className="hud-glass clip-chamfer-all w-full p-4 mb-2 flex flex-wrap items-center justify-between gap-4 border border-stark-gold/40 animate-in fade-in slide-in-from-bottom-2 duration-200">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-stark-gold" />
            <span className="font-orbitron text-xs text-stark-gold font-bold uppercase tracking-wider">
              Neural Grounding Calibrations
            </span>
          </div>

          <div className="flex items-center gap-6 flex-wrap">
            {/* Box Threshold Slider */}
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-slate-300">
                Box Confidence:
              </span>
              <input
                type="range"
                min="0.1"
                max="0.85"
                step="0.05"
                value={boxThreshold}
                onChange={(e) => onBoxThresholdChange(parseFloat(e.target.value))}
                className="w-24 accent-stark-cyan cursor-pointer"
              />
              <span className="font-mono text-xs text-stark-cyan font-bold w-10">
                {boxThreshold.toFixed(2)}
              </span>
            </div>

            {/* Text Threshold Slider */}
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-slate-300">
                Text Alignment:
              </span>
              <input
                type="range"
                min="0.1"
                max="0.85"
                step="0.05"
                value={textThreshold}
                onChange={(e) =>
                  onTextThresholdChange(parseFloat(e.target.value))
                }
                className="w-24 accent-stark-gold cursor-pointer"
              />
              <span className="font-mono text-xs text-stark-gold font-bold w-10">
                {textThreshold.toFixed(2)}
              </span>
            </div>

            {/* Toggle Camera button */}
            <button
              onClick={onToggleCamera}
              className={`px-3 py-1 text-[11px] font-orbitron uppercase border flex items-center gap-1.5 transition-all ${
                cameraActive
                  ? "border-stark-cyan text-stark-cyan hover:bg-stark-cyan/10"
                  : "border-stark-crimson text-stark-crimson hover:bg-stark-crimson/10"
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              {cameraActive ? "Feed Active" : "Camera Muted"}
            </button>
          </div>
        </div>
      )}

      {/* Main Floating Capsule Dock */}
      <div className="hud-glass-dock clip-chamfer-dock w-full p-2.5 px-4 flex items-center gap-3">
        {/* Push-to-Talk Mic Button */}
        <button
          onClick={onToggleRecording}
          title="Push to Talk (Shortcut: SPACE)"
          className={`relative p-3.5 rounded-lg flex items-center justify-center transition-all duration-300 ${
            isRecording
              ? "bg-gradient-to-br from-stark-gold to-stark-crimson text-black shadow-[0_0_25px_#FFBF00] scale-105"
              : "hud-glass text-stark-cyan hover:text-white hover:border-stark-gold hover:shadow-[0_0_15px_#00F5FF]"
          }`}
        >
          {isRecording ? (
            <Mic className="w-5 h-5 animate-pulse" />
          ) : (
            <MicOff className="w-5 h-5" />
          )}
          <span className="absolute -bottom-1 -right-1 text-[8px] font-mono px-1 bg-black/80 rounded border border-stark-gold text-stark-gold">
            SPACE
          </span>
        </button>

        {/* Search Input Bar */}
        <div className="flex-1 relative flex items-center">
          <Search className="w-4 h-4 text-stark-cyan absolute left-3 pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                onSubmitQuery(query);
              }
            }}
            placeholder="Speak or enter prompt (e.g. 'person in black shirt', 'water bottle')..."
            className="w-full bg-slate-900/70 border border-stark-cyan/40 focus:border-stark-gold focus:shadow-[0_0_15px_rgba(255,191,0,0.3)] text-slate-100 placeholder:text-slate-500 pl-9 pr-20 py-2.5 rounded text-xs font-mono tracking-wide outline-none transition-all"
          />
          <button
            onClick={() => onSubmitQuery(query)}
            className="absolute right-1.5 px-3 py-1 bg-stark-cyan/20 hover:bg-stark-cyan/40 border border-stark-cyan text-stark-cyan text-[10px] font-orbitron font-bold tracking-wider uppercase rounded transition-all"
          >
            GROUND
          </button>
        </div>

        {/* Effect Mode Tabs */}
        <div className="hidden md:flex items-center gap-1 bg-slate-950/80 p-1 rounded border border-slate-800">
          <button
            onClick={() => onEffectModeChange("highlight")}
            title="Hotkey: 1"
            className={`px-2.5 py-1.5 rounded text-[10px] font-orbitron font-semibold uppercase flex items-center gap-1 transition-all ${
              effectMode === "highlight"
                ? "bg-stark-cyan text-black font-bold shadow-[0_0_10px_#00F5FF]"
                : "text-slate-400 hover:text-stark-cyan"
            }`}
          >
            <Crosshair className="w-3 h-3" />
            <span>[1] Reticle</span>
          </button>

          <button
            onClick={() => onEffectModeChange("recolor")}
            title="Hotkey: 2"
            className={`px-2.5 py-1.5 rounded text-[10px] font-orbitron font-semibold uppercase flex items-center gap-1 transition-all ${
              effectMode === "recolor"
                ? "bg-stark-gold text-black font-bold shadow-[0_0_10px_#FFBF00]"
                : "text-slate-400 hover:text-stark-gold"
            }`}
          >
            <Sparkles className="w-3 h-3" />
            <span>[2] Recolor</span>
          </button>

          <button
            onClick={() => onEffectModeChange("blur")}
            title="Hotkey: 3"
            className={`px-2.5 py-1.5 rounded text-[10px] font-orbitron font-semibold uppercase flex items-center gap-1 transition-all ${
              effectMode === "blur"
                ? "bg-stark-plasma text-white font-bold shadow-[0_0_10px_#1E90FF]"
                : "text-slate-400 hover:text-stark-plasma"
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>[3] Blur</span>
          </button>

          <button
            onClick={() => onEffectModeChange("isolate")}
            title="Hotkey: 4"
            className={`px-2.5 py-1.5 rounded text-[10px] font-orbitron font-semibold uppercase flex items-center gap-1 transition-all ${
              effectMode === "isolate"
                ? "bg-stark-crimson text-white font-bold shadow-[0_0_10px_#FF2E2E]"
                : "text-slate-400 hover:text-stark-crimson"
            }`}
          >
            <Eye className="w-3 h-3" />
            <span>[4] Isolate</span>
          </button>
        </div>

        {/* Calibrations toggle */}
        <button
          onClick={() => setShowConfig(!showConfig)}
          title="Adjust detection thresholds"
          className={`p-2.5 rounded border transition-all ${
            showConfig
              ? "hud-glass-gold text-stark-gold border-stark-gold"
              : "hud-glass text-slate-400 hover:text-stark-cyan border-slate-700"
          }`}
        >
          <Sliders className="w-4 h-4" />
        </button>
      </div>

      {/* Preset Quick Query Chips */}
      <div className="flex items-center gap-2 overflow-x-auto max-w-full px-2 py-0.5">
        <span className="text-[9px] font-mono text-slate-400 uppercase tracking-widest hidden sm:inline">
          PROMPTS:
        </span>
        {PRESET_QUERIES.map((preset) => (
          <button
            key={preset}
            onClick={() => {
              onQueryChange(preset);
              onSubmitQuery(preset);
            }}
            className="hud-glass px-2.5 py-0.5 rounded-full text-[10px] font-mono text-slate-300 hover:text-stark-cyan hover:border-stark-cyan transition-all flex items-center gap-1"
          >
            <span className="text-stark-gold">#</span>
            {preset}
          </button>
        ))}
      </div>
    </div>
  );
}
