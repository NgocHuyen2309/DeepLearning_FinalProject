'use client';

import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';

interface RadarAnalysisProps {
  data: {
    subject: string;
    A: number; // Fake Probability
    fullMark: number;
  }[];
}

export default function RadarAnalysis({ data }: RadarAnalysisProps) {
  return (
    <div style={{ width: '100%', height: 320, backgroundColor: 'var(--card-bg)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', padding: '1rem', marginTop: '1rem' }}>
      <h3 style={{ textAlign: 'center', marginBottom: '1rem', color: 'var(--text-primary)', fontSize: '1.2rem' }}>XAI Region Analysis</h3>
      <div style={{ width: '100%', height: 250 }}>
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} />
          <Radar name="Fake Probability %" dataKey="A" stroke="#ef4444" fill="#ef4444" fillOpacity={0.5} />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
