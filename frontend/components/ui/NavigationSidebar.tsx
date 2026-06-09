'use client';

import React from 'react';
import styles from './NavigationSidebar.module.css';
import { Activity, History } from 'lucide-react';

interface NavigationSidebarProps {
  activeTab: 'analysis' | 'history';
  setActiveTab: (tab: 'analysis' | 'history') => void;
}

export default function NavigationSidebar({ activeTab, setActiveTab }: NavigationSidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <h2 className={styles.title}>Menu</h2>
      <nav className={styles.nav}>
        <button 
          className={`${styles.navItem} ${activeTab === 'analysis' ? styles.active : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          <Activity size={20} />
          Forensics Analysis
        </button>
        <button 
          className={`${styles.navItem} ${activeTab === 'history' ? styles.active : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <History size={20} />
          History & Stats
        </button>
      </nav>
    </aside>
  );
}
