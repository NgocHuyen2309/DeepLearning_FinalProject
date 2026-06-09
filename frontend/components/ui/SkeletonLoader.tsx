import React from 'react';
import styles from './SkeletonLoader.module.css';

export default function SkeletonLoader() {
  return (
    <div className={styles.skeletonContainer}>
      <div className={`${styles.skeletonBlock} ${styles.title}`}></div>
      <div className={`${styles.skeletonBlock} ${styles.imageBox}`}></div>
      <div className={`${styles.skeletonBlock} ${styles.text}`}></div>
      <div className={`${styles.skeletonBlock} ${styles.textShort}`}></div>
      <div className={`${styles.skeletonBlock} ${styles.text}`}></div>
    </div>
  );
}
