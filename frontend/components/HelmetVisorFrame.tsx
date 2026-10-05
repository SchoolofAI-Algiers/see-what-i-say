"use client";

import React from "react";

export default function HelmetVisorFrame() {
  return (
    <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
      <svg
        viewBox="0 0 1920 1080"
        preserveAspectRatio="none"
        className="w-full h-full"
      >
        <defs>
          {/* Armor plate gradients */}
          <linearGradient id="armorGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#0a0f19" />
            <stop offset="50%" stopColor="#060910" />
            <stop offset="100%" stopColor="#030509" />
          </linearGradient>

          {/* Gold edge glow filter */}
          <filter id="goldGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Cyan edge glow filter */}
          <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* 1. TOP BROW VISOR ARMOR PLATE */}
        <polygon
          points="
            0,0 1920,0 1920,45 
            1660,45 1520,30 
            1260,50 1120,70 
            960,82 
            800,70 660,50 
            400,30 260,45 
            0,45
          "
          fill="url(#armorGradient)"
        />
        {/* Brow Bevel Edge (Cyan) */}
        <polyline
          points="
            0,45 260,45 400,30 660,50 800,70 960,82 1120,70 1260,50 1520,30 1660,45 1920,45
          "
          fill="none"
          stroke="#00F5FF"
          strokeWidth="2"
          opacity="0.95"
          filter="url(#cyanGlow)"
        />
        {/* Brow Secondary Gold Accent Line */}
        <polyline
          points="
            40,36 260,36 400,22 660,42 800,62 960,74 1120,62 1260,42 1520,22 1660,36 1880,36
          "
          fill="none"
          stroke="#FFBF00"
          strokeWidth="1.5"
          opacity="0.85"
          filter="url(#goldGlow)"
        />

        {/* 2. LEFT CHEEK VISOR ARMOR CANOPY */}
        <polygon
          points="
            0,0 260,45 
            190,160 270,380 
            180,550 
            270,750 180,920 
            320,1010 0,1080
          "
          fill="url(#armorGradient)"
        />
        {/* Left Cheek Inner Contour Edge */}
        <polyline
          points="
            260,45 190,160 270,380 180,550 270,750 180,920 320,1010
          "
          fill="none"
          stroke="#00F5FF"
          strokeWidth="2.5"
          opacity="0.95"
          filter="url(#cyanGlow)"
        />
        <polyline
          points="
            240,45 170,160 250,380 160,550 250,750 160,920 300,1010
          "
          fill="none"
          stroke="#FFBF00"
          strokeWidth="1.5"
          opacity="0.8"
        />

        {/* 3. RIGHT CHEEK VISOR ARMOR CANOPY */}
        <polygon
          points="
            1920,0 1660,45 
            1730,160 1650,380 
            1740,550 
            1650,750 1740,920 
            1600,1010 1920,1080
          "
          fill="url(#armorGradient)"
        />
        {/* Right Cheek Inner Contour Edge */}
        <polyline
          points="
            1660,45 1730,160 1650,380 1740,550 1650,750 1740,920 1600,1010
          "
          fill="none"
          stroke="#00F5FF"
          strokeWidth="2.5"
          opacity="0.95"
          filter="url(#cyanGlow)"
        />
        <polyline
          points="
            1680,45 1750,160 1670,380 1760,550 1670,750 1760,920 1620,1010
          "
          fill="none"
          stroke="#FFBF00"
          strokeWidth="1.5"
          opacity="0.8"
        />

        {/* 4. BOTTOM CHIN VISOR CONSOLE TRAY */}
        <polygon
          points="
            0,1080 320,1010 
            500,1010 560,950 
            1360,950 1420,1010 
            1600,1010 1920,1080
          "
          fill="url(#armorGradient)"
        />
        {/* Bottom Chin Bevel Contour */}
        <polyline
          points="
            320,1010 500,1010 560,950 1360,950 1420,1010 1600,1010
          "
          fill="none"
          stroke="#FFBF00"
          strokeWidth="2.5"
          opacity="0.95"
          filter="url(#goldGlow)"
        />
        <polyline
          points="
            340,1020 505,1020 565,962 1355,962 1415,1020 1580,1020
          "
          fill="none"
          stroke="#00F5FF"
          strokeWidth="1.2"
          opacity="0.85"
        />

        {/* 5. TACTICAL CORNER BOLTS & SENSOR NODES */}
        <circle cx="270" cy="55" r="3.5" fill="#00F5FF" opacity="0.9" />
        <circle cx="270" cy="55" r="7" fill="none" stroke="#FFBF00" strokeWidth="1" />

        <circle cx="1650" cy="55" r="3.5" fill="#00F5FF" opacity="0.9" />
        <circle cx="1650" cy="55" r="7" fill="none" stroke="#FFBF00" strokeWidth="1" />

        <circle cx="530" cy="975" r="3.5" fill="#FFBF00" opacity="0.9" />
        <circle cx="530" cy="975" r="7" fill="none" stroke="#00F5FF" strokeWidth="1" />

        <circle cx="1390" cy="975" r="3.5" fill="#FFBF00" opacity="0.9" />
        <circle cx="1390" cy="975" r="7" fill="none" stroke="#00F5FF" strokeWidth="1" />
      </svg>
    </div>
  );
}
