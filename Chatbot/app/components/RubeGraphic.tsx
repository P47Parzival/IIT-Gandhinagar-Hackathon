'use client';

import { useEffect, useRef } from 'react';

export function RubeGraphic({
  className = 'h-5 w-5',          // 16 × 16 px by default
}: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;

    // ---- size comes from the Tailwind class you pass ----
    const container = canvas.parentElement!;
    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      const size = Math.min(rect.width, rect.height);
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      ctx.scale(dpr, dpr);
      // store for particles
      canvas.dataset.size = size.toString();
    };
    updateSize();

    // particles are recreated when size changes
    let particles: Particle[] = [];
    const makeParticles = () => {
      const s = Number(canvas.dataset.size);
      particles = [];
      for (let i = 0; i < 18; i++) particles.push(new Particle(s));
    };
    makeParticles();

    let time = 0;
    const animate = () => {
      const size = Number(canvas.dataset.size);
      ctx.clearRect(0, 0, size, size);
      time += 0.016;

      particles.forEach(p => {
        p.update(time);
        p.draw(ctx, size);
      });

      requestAnimationFrame(animate);
    };
    animate();

    // react to Tailwind size changes (e.g. responsive navbar)
    const ro = new ResizeObserver(() => {
      updateSize();
      makeParticles();
    });
    ro.observe(container);

    return () => ro.disconnect();
  }, []);

  return (
    <div className={`inline-block ${className}`}>
      <canvas ref={canvasRef} className="block w-full h-full" />
    </div>
  );
}

/* ---------- particle logic (scaled for tiny canvas) ---------- */
class Particle {
  private cx: number;
  private cy: number;
  private radius: number;
  private angle: number;
  private speed: number;
  private hue: number;
  private orbit: number;

  constructor(size: number) {
    this.cx = size / 2;
    this.cy = size / 2;
    // tiny orbit & radius for 16-px canvas
    this.orbit = Math.random() * 3.2 + 2.2;   // 2.2 → 5.4 px
    this.radius = Math.random() * 1.6 + 0.6; // 0.4 → 1.3 px
    this.angle = Math.random() * Math.PI * 2;
    this.speed = Math.random() * 2.2 + 1.6;
    this.hue = Math.random() * 60 + 200;
  }

  update(time: number) {
    this.angle += this.speed * 0.04;
    const pulse = 0.7 + 0.3 * Math.sin(time * 4 + this.hue);
    this.orbit = 2.2 + 3.2 * pulse;
  }

  draw(ctx: CanvasRenderingContext2D, size: number) {
    const x = this.cx + Math.cos(this.angle) * this.orbit;
    const y = this.cy + Math.sin(this.angle) * this.orbit;

    const grad = ctx.createRadialGradient(x, y, 0, x, y, this.radius);
    grad.addColorStop(0, `hsla(${this.hue},100%,70%,1)`);
    grad.addColorStop(1, `hsla(${this.hue},100%,50%,0)`);

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, this.radius, 0, Math.PI * 2);
    ctx.fill();
  }
}