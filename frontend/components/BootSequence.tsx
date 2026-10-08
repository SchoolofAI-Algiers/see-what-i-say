"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Zap } from "lucide-react";

interface BootSequenceProps {
  onComplete: () => void;
}

const BOOT_LOGS = [
  "INITIALIZING STARK VISION // MARK LXXXV NEURAL VISOR...",
  "CONNECTING TO GPU HARDWARE: NVIDIA GEFORCE RTX 4060 (8GB VRAM)...",
  "LOADING MULTIMODAL GROUNDING BACKBONE: IDEA-Research/grounding-dino-base...",
  "SYNCHRONIZING SWIN-T VISUAL FEATURE EXTRACTOR & CROSS-ATTENTION MAPS...",
  "ALLOCATING SPATIAL TENSORS IN CUDA FP16 MEMORY...",
  "ACOUSTIC SENSORS LINKED: 16 kHz MONO SPEECH-TO-TEXT STREAM...",
  "JARVIS PROTOCOLS ACTIVATED. ALL SYSTEMS NOMINAL.",
];

export default function BootSequence({ onComplete }: BootSequenceProps) {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    // plays greeting sound over browser speech synth if supported
    try {
      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(
          "JARVIS online. Mark 85 neural interface calibrated. Welcome back, Mr. Stark."
        );
        utterance.rate = 1.05;
        utterance.pitch = 0.95;
        // try finding english male voice
        const voices = window.speechSynthesis.getVoices();
        const jarvisVoice = voices.find(
          (v) =>
            v.lang.includes("en-GB") ||
            v.name.includes("Daniel") ||
            v.name.includes("George") ||
            v.name.includes("Male")
        );
        if (jarvisVoice) utterance.voice = jarvisVoice;
        window.speechSynthesis.speak(utterance);
      }
    } catch (e) {
      console.warn("Speech synthesis unavailable:", e);
    }

    // cycle through log lines
    const stepInterval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < BOOT_LOGS.length - 1) {
          return prev + 1;
        } else {
          clearInterval(stepInterval);
          setTimeout(onComplete, 600);
          return prev;
        }
      });
    }, 320);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === " ") {
        onComplete();
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      clearInterval(stepInterval);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onComplete]);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
      onClick={onComplete}
      className="fixed inset-0 z-50 bg-[#070B12]/95 backdrop-blur-2xl flex flex-col items-center justify-center cursor-pointer select-none"
    >
      <div className="relative w-full max-w-xl p-8 flex flex-col items-center gap-6">
        {/* center spinner graphic */}
        <div className="relative w-36 h-36 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border-2 border-dashed border-stark-cyan/40 animate-radar-spin" />
          <div className="absolute inset-2 rounded-full border border-stark-gold/50 animate-radar-reverse" />
          <div className="w-20 h-20 rounded-full bg-stark-cyan/10 border-2 border-stark-cyan flex items-center justify-center shadow-[0_0_30px_#00F5FF]">
            <Zap className="w-10 h-10 text-stark-cyan animate-pulse" />
          </div>
        </div>

        {/* heading */}
        <div className="flex flex-col items-center gap-1">
          <h1 className="font-orbitron text-2xl font-bold tracking-widest text-stark-cyan text-glow-cyan">
            STARK VISION // MK-85
          </h1>
          <p className="font-mono text-xs text-stark-gold tracking-widest uppercase">
            Multimodal Referring Segmentation Engine
          </p>
        </div>

        {/* progress indicator */}
        <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-stark-cyan/30">
          <motion.div
            className="h-full bg-gradient-to-r from-stark-cyan via-stark-gold to-stark-cyan"
            initial={{ width: "0%" }}
            animate={{
              width: `${((currentStep + 1) / BOOT_LOGS.length) * 100}%`,
            }}
            transition={{ duration: 0.2 }}
          />
        </div>

        {/* boot logs */}
        <div className="w-full bg-black/60 p-4 rounded border border-stark-cyan/30 font-mono text-[11px] h-32 overflow-hidden flex flex-col justify-end gap-1">
          {BOOT_LOGS.slice(0, currentStep + 1).map((log, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2 text-slate-300"
            >
              <span className="text-stark-gold font-bold">{">"}</span>
              <span
                className={
                  idx === currentStep
                    ? "text-stark-cyan font-semibold text-glow-cyan"
                    : "text-slate-400"
                }
              >
                {log}
              </span>
            </motion.div>
          ))}
        </div>

        {/* skip hint */}
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
          <span className="px-1.5 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-300">
            ESC
          </span>
          <span>or click anywhere to enter cockpit immediately</span>
        </div>
      </div>
    </motion.div>
  );
}
