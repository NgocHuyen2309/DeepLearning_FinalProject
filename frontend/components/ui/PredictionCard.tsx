import React from 'react';
import styles from './PredictionCard.module.css';

interface PredictionCardProps {
  modelName: string;
  isFake: boolean;
  confidence: number; // 0 to 100
}

export default function PredictionCard({ modelName, isFake, confidence }: PredictionCardProps) {
  const statusClass = isFake ? styles.fake : styles.real;
  
  return (
    <div className={`${styles.card} ${statusClass}`}>
      <div className={styles.header}>
        <span className={styles.modelName}>{modelName}</span>
        <span className={styles.confidence}>
          {confidence.toFixed(1)}% {isFake ? 'FAKE' : 'REAL'}
        </span>
      </div>
      <div className={styles.progressBarContainer}>
        <div 
          className={styles.progressBar} 
          style={{ width: `${confidence}%` }}
        ></div>
      </div>
    </div>
  );
}
