"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";

interface ArcReactorHUDProps {
  className?: string;
}

export default function ArcReactorHUD({ className = "" }: ArcReactorHUDProps) {
  const threeRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });

  useEffect(() => {
    const container = threeRef.current;
    if (!container) return;

    const size = 100;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = 4.8;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(size, size);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    // Glowing core sphere
    const coreGeo = new THREE.SphereGeometry(0.4, 24, 24);
    const coreMat = new THREE.MeshBasicMaterial({ color: 0x00f5ff });
    const core = new THREE.Mesh(coreGeo, coreMat);
    group.add(core);

    // Core point light
    const light = new THREE.PointLight(0x00f5ff, 3, 10);
    group.add(light);

    // 10 Palladium coils
    const coilGroup = new THREE.Group();
    const coilGeo = new THREE.BoxGeometry(0.12, 0.32, 0.16);
    const coilMat = new THREE.MeshStandardMaterial({
      color: 0xffbf00,
      metalness: 0.9,
      roughness: 0.2,
      emissive: 0x664a00,
    });
    for (let i = 0; i < 10; i++) {
      const angle = (i / 10) * Math.PI * 2;
      const mesh = new THREE.Mesh(coilGeo, coilMat);
      mesh.position.x = Math.cos(angle) * 0.95;
      mesh.position.y = Math.sin(angle) * 0.95;
      mesh.rotation.z = angle + Math.PI / 2;
      coilGroup.add(mesh);
    }
    group.add(coilGroup);

    // Inner wireframe ring
    const innerRingGeo = new THREE.TorusGeometry(0.72, 0.04, 8, 32);
    const innerRingMat = new THREE.MeshBasicMaterial({
      color: 0x00f5ff,
      wireframe: true,
    });
    const innerRing = new THREE.Mesh(innerRingGeo, innerRingMat);
    group.add(innerRing);

    // Outer gold ring
    const outerRingGeo = new THREE.TorusGeometry(1.3, 0.07, 12, 48);
    const outerRingMat = new THREE.MeshStandardMaterial({
      color: 0xffbf00,
      metalness: 0.95,
      roughness: 0.2,
    });
    const outerRing = new THREE.Mesh(outerRingGeo, outerRingMat);
    group.add(outerRing);

    // Lighting
    const amb = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(amb);
    const dir = new THREE.DirectionalLight(0xffbf00, 1.5);
    dir.position.set(2, 3, 2);
    scene.add(dir);

    // Mouse parallax
    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current.targetX = (e.clientX / window.innerWidth - 0.5) * 0.5;
      mouseRef.current.targetY = -(e.clientY / window.innerHeight - 0.5) * 0.5;
    };
    window.addEventListener("mousemove", onMouseMove);

    let animId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const t = clock.getElapsedTime();

      innerRing.rotation.z = t * 0.9;
      coilGroup.rotation.z = -t * 0.45;
      outerRing.rotation.z = t * 0.2;

      const pulse = 1 + Math.sin(t * 5) * 0.08;
      core.scale.set(pulse, pulse, pulse);

      mouseRef.current.x += (mouseRef.current.targetX - mouseRef.current.x) * 0.08;
      mouseRef.current.y += (mouseRef.current.targetY - mouseRef.current.y) * 0.08;
      group.rotation.y = mouseRef.current.x;
      group.rotation.x = -mouseRef.current.y;

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      cancelAnimationFrame(animId);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div className={`flex flex-col gap-1.5 select-none ${className}`}>
      {/* Top Header Tag */}
      <div className="flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-stark-cyan animate-ping" />
        <span className="font-orbitron text-[10px] md:text-[11px] font-bold tracking-wider text-stark-gold text-glow-gold uppercase">
          ARC REACTOR MK85 // CORE STATUS{" "}
          <span className="text-stark-cyan text-glow-cyan">[98% ONLINE]</span>
        </span>
      </div>

      {/* Main Dial Area */}
      <div className="relative w-44 h-32 flex items-center justify-center">
        {/* SVG Surrounding Telemetry Arc & Ticks */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 180 130"
        >
          {/* Outer Cyan Arc */}
          <path
            d="M 28 65 A 50 50 0 0 1 152 65"
            fill="none"
            stroke="#00F5FF"
            strokeWidth="1.5"
            strokeDasharray="4 2"
            opacity="0.8"
          />
          {/* Inner Gold Arc */}
          <path
            d="M 38 65 A 40 40 0 0 1 142 65"
            fill="none"
            stroke="#FFBF00"
            strokeWidth="1.5"
            opacity="0.9"
          />

          {/* Left Arc Radial Ticks */}
          <path
            d="M 20 40 L 14 36 M 16 55 L 10 55 M 18 70 L 12 72 M 22 85 L 16 90"
            stroke="#00F5FF"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.85"
          />

          {/* Left Vertical Arc Text */}
          <text
            x="8"
            y="78"
            fill="#00F5FF"
            fontSize="7"
            fontFamily="Orbitron"
            letterSpacing="2"
            transform="rotate(-90 8 78)"
            opacity="0.9"
          >
            ROTATING TELEMETRY
          </text>

          {/* Right Arc Radial Ticks */}
          <path
            d="M 160 40 L 166 36 M 164 55 L 170 55 M 162 70 L 168 72 M 158 85 L 164 90"
            stroke="#FFBF00"
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.85"
          />
        </svg>

        {/* 3D WebGL Canvas */}
        <div
          ref={threeRef}
          className="w-[100px] h-[100px] relative z-10 filter drop-shadow-[0_0_15px_#00F5FF]"
        />
      </div>

      {/* Bottom Footer Telemetry */}
      <div className="font-mono text-[9px] text-slate-300 tracking-wider flex items-center gap-1.5">
        <span className="text-stark-cyan font-bold">ARC REACTOR MK85</span>
        <span className="text-slate-600">{"//"}</span>
        <span className="text-stark-gold">CORE STATUS [98% ONLINE]</span>
      </div>
    </div>
  );
}
