"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";

interface ArcReactor3DProps {
  className?: string;
}

export default function ArcReactor3D({ className = "" }: ArcReactor3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const width = 120;
    const height = 120;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.z = 5.2;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Group for mouse parallax
    const reactorGroup = new THREE.Group();
    scene.add(reactorGroup);

    // center core sphere
    const coreGeo = new THREE.SphereGeometry(0.35, 32, 32);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0x00f5ff,
      transparent: true,
      opacity: 0.95,
    });
    const coreSphere = new THREE.Mesh(coreGeo, coreMat);
    reactorGroup.add(coreSphere);

    // core point light
    const pointLight = new THREE.PointLight(0x00f5ff, 2.5, 10);
    reactorGroup.add(pointLight);

    // perimeter coils
    const coilsGroup = new THREE.Group();
    const coilGeo = new THREE.BoxGeometry(0.12, 0.35, 0.15);
    const coilMat = new THREE.MeshStandardMaterial({
      color: 0xffbf00,
      metalness: 0.85,
      roughness: 0.2,
      emissive: 0x664a00,
    });

    const coilCount = 10;
    for (let i = 0; i < coilCount; i++) {
      const angle = (i / coilCount) * Math.PI * 2;
      const coil = new THREE.Mesh(coilGeo, coilMat);
      coil.position.x = Math.cos(angle) * 0.95;
      coil.position.y = Math.sin(angle) * 0.95;
      coil.rotation.z = angle + Math.PI / 2;
      coilsGroup.add(coil);
    }
    reactorGroup.add(coilsGroup);

    // inner wireframe ring
    const innerRingGeo = new THREE.TorusGeometry(0.7, 0.04, 16, 64);
    const innerRingMat = new THREE.MeshBasicMaterial({
      color: 0x00f5ff,
      wireframe: true,
    });
    const innerRing = new THREE.Mesh(innerRingGeo, innerRingMat);
    reactorGroup.add(innerRing);

    // outer housing ring
    const outerRingGeo = new THREE.TorusGeometry(1.35, 0.08, 16, 64);
    const outerRingMat = new THREE.MeshStandardMaterial({
      color: 0xffbf00,
      metalness: 0.9,
      roughness: 0.3,
    });
    const outerRing = new THREE.Mesh(outerRingGeo, outerRingMat);
    reactorGroup.add(outerRing);

    // corner brackets
    const bracketGeo = new THREE.TorusGeometry(1.5, 0.02, 8, 3, Math.PI / 3);
    const bracketMat = new THREE.MeshBasicMaterial({
      color: 0x00f5ff,
      transparent: true,
      opacity: 0.7,
    });
    for (let i = 0; i < 3; i++) {
      const bracket = new THREE.Mesh(bracketGeo, bracketMat);
      bracket.rotation.z = (i * Math.PI * 2) / 3;
      reactorGroup.add(bracket);
    }

    // Ambient & Directional Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffbf00, 1.2);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);

    // Mouse movement listener for parallax
    const handleMouseMove = (e: MouseEvent) => {
      const { innerWidth, innerHeight } = window;
      mouseRef.current.targetX = (e.clientX / innerWidth - 0.5) * 0.6;
      mouseRef.current.targetY = -(e.clientY / innerHeight - 0.5) * 0.6;
    };
    window.addEventListener("mousemove", handleMouseMove);

    // Animation Loop
    let animId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Continuous rotation
      innerRing.rotation.z = elapsedTime * 0.8;
      coilsGroup.rotation.z = -elapsedTime * 0.4;
      outerRing.rotation.z = elapsedTime * 0.15;

      // Subtle pulse on core
      const pulse = 1 + Math.sin(elapsedTime * 4) * 0.08;
      coreSphere.scale.set(pulse, pulse, pulse);

      // Lerp mouse parallax
      mouseRef.current.x +=
        (mouseRef.current.targetX - mouseRef.current.x) * 0.08;
      mouseRef.current.y +=
        (mouseRef.current.targetY - mouseRef.current.y) * 0.08;

      reactorGroup.rotation.y = mouseRef.current.x;
      reactorGroup.rotation.x = -mouseRef.current.y;

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animId);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      className={`hud-glass clip-chamfer-tl p-3.5 flex items-center gap-3.5 border-l-2 border-l-stark-gold ${className}`}
    >
      <div
        ref={containerRef}
        className="w-[120px] h-[120px] relative flex-shrink-0 flex items-center justify-center filter drop-shadow-[0_0_12px_rgba(0,245,255,0.6)]"
      />
      <div className="flex flex-col gap-1 pr-2">
        <div className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-stark-cyan animate-ping" />
          <span className="font-orbitron text-xs font-bold tracking-wider text-stark-cyan text-glow-cyan uppercase">
            Arc Reactor MK-85
          </span>
        </div>
        <div className="font-mono text-[11px] text-slate-300 tracking-tight">
          CORE STATUS: <span className="text-stark-gold font-bold">ONLINE</span>
        </div>
        <div className="font-mono text-[10px] text-slate-400">
          POWER OUTPUT: <span className="text-stark-cyan">98.4% GW</span>
        </div>
        <div className="w-28 bg-slate-800/80 h-1.5 rounded-full overflow-hidden border border-stark-cyan/30 mt-1">
          <div
            className="h-full bg-gradient-to-r from-stark-cyan via-stark-gold to-stark-cyan animate-pulse"
            style={{ width: "98.4%" }}
          />
        </div>
        <span className="font-mono text-[9px] text-slate-400 tracking-widest uppercase">
          TELEMETRY // SYNC LOCKED
        </span>
      </div>
    </div>
  );
}
