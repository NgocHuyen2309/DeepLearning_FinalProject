'use client';

import React from 'react';
import styles from './page.module.css';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const dailyData = [
  { name: 'Mon', uploads: 400, fake: 240, real: 160 },
  { name: 'Tue', uploads: 300, fake: 139, real: 161 },
  { name: 'Wed', uploads: 450, fake: 200, real: 250 },
  { name: 'Thu', uploads: 278, fake: 178, real: 100 },
  { name: 'Fri', uploads: 189, fake: 89, real: 100 },
  { name: 'Sat', uploads: 239, fake: 139, real: 100 },
  { name: 'Sun', uploads: 349, fake: 149, real: 200 },
];

const performanceData = [
  { name: '10:00', latency: 400 },
  { name: '10:05', latency: 300 },
  { name: '10:10', latency: 200 },
  { name: '10:15', latency: 278 },
  { name: '10:20', latency: 189 },
];

export default function Dashboard() {
  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Admin Dashboard</h1>
      
      <div className={styles.grid}>
        <div className={styles.card}>
          <h2>Weekly Uploads (Fake vs Real)</h2>
          <div style={{ width: '100%', height: 300, marginTop: '1rem' }}>
            <ResponsiveContainer>
              <BarChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="fake" fill="#ef4444" />
                <Bar dataKey="real" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={styles.card}>
          <h2>System Latency (ms)</h2>
          <div style={{ width: '100%', height: 300, marginTop: '1rem' }}>
            <ResponsiveContainer>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="latency" stroke="#3b82f6" activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
