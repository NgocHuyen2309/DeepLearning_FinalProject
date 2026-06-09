'use client';

import React, { useState, useRef, MouseEvent, TouchEvent } from 'react';
import styles from './HeatmapSlider.module.css';

interface HeatmapSliderProps {
  originalImage: string; // URL or Base64
  heatmapImage: string; // URL or Base64
}

export default function HeatmapSlider({ originalImage, heatmapImage }: HeatmapSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50); // percentage 0-100
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = (clientX: number) => {
    if (!isDragging || !containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    
    setSliderPosition(percent);
  };

  const handleMouseMove = (e: MouseEvent) => handleMove(e.clientX);
  const handleTouchMove = (e: TouchEvent) => handleMove(e.touches[0].clientX);

  const startDrag = () => setIsDragging(true);
  const stopDrag = () => setIsDragging(false);

  return (
    <div 
      className={styles.sliderContainer} 
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseUp={stopDrag}
      onMouseLeave={stopDrag}
      onTouchMove={handleTouchMove}
      onTouchEnd={stopDrag}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={originalImage} alt="Original" className={styles.imageLayer} />
      
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img 
        src={heatmapImage} 
        alt="Heatmap" 
        className={styles.overlayLayer} 
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        draggable={false}
      />
      
      <div 
        className={styles.sliderHandle} 
        style={{ left: `${sliderPosition}%` }}
        onMouseDown={startDrag}
        onTouchStart={startDrag}
      ></div>

      <div className={styles.labels}>
        <span className={styles.label}>Heatmap</span>
        <span className={styles.label}>Original</span>
      </div>
    </div>
  );
}
