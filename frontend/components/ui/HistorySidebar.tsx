'use client';

import React from 'react';
import styles from './HistorySidebar.module.css';
import { usePredictionStore } from '@/store/usePredictionStore';

export default function HistorySidebar() {
  const { history, setCurrentScan } = usePredictionStore();

  return (
    <aside className={styles.sidebar}>
      <h2 className={styles.title}>Recent Scans</h2>
      {history.length === 0 ? (
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>No history yet.</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {history.map((scan) => {
            const isFake = scan.results?.resnet?.isFake;
            return (
              <div 
                key={scan.id} 
                className={styles.historyItem}
                onClick={() => setCurrentScan(scan)}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={scan.originalImageUrl} alt="History thumbnail" className={styles.historyImage} />
                <div className={styles.historyInfo}>
                  {new Date(scan.timestamp).toLocaleTimeString()} - {' '}
                  <span className={isFake ? styles.fakeLabel : styles.realLabel}>
                    {isFake ? 'FAKE' : 'REAL'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </aside>
  );
}
