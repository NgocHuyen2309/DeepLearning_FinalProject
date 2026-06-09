'use client';

import React from 'react';
import { usePredictionStore } from '@/store/usePredictionStore';
import { BarChart, Bar, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import styles from './HistoryDashboard.module.css';

interface HistoryDashboardProps {
  onScanSelect: () => void;
}

export default function HistoryDashboard({ onScanSelect }: HistoryDashboardProps) {
  const { history, setCurrentScan } = usePredictionStore();

  if (history.length === 0) {
    return (
      <div className={styles.emptyState}>
        <p>No scanning history available yet. Upload an image to start.</p>
      </div>
    );
  }

  // Chuẩn bị dữ liệu cho 3 biểu đồ
  const chartData = history.map((scan, index) => {
    const resnetConf = scan.results?.resnet?.confidence || 0;
    const vitConf = scan.results?.vit?.confidence || 0;
    
    // Tính Độ bất định (Uncertainty) = 100 - |Conf - 50| * 2
    const resnetUncertainty = 100 - Math.abs(resnetConf - 50) * 2;
    const vitUncertainty = 100 - Math.abs(vitConf - 50) * 2;
    
    // Tính Chỉ số bất thường khu vực (Anomaly Index)
    const rScores = scan.results?.resnet?.regionalScores;
    const resnetAnomaly = rScores ? (rScores.eyes + rScores.mouth + rScores.skinTexture + rScores.lighting) / 4 : 0;
    const vScores = scan.results?.vit?.regionalScores;
    const vitAnomaly = vScores ? (vScores.eyes + vScores.mouth + vScores.skinTexture + vScores.lighting) / 4 : 0;
    
    return {
      name: `Lần quét ${history.length - index}`,
      ResNet_Latency: scan.results?.resnet?.processingTimeMs || 0,
      ViT_Latency: scan.results?.vit?.processingTimeMs || 0,
      ResNet_Uncertainty: parseFloat(resnetUncertainty.toFixed(1)),
      ViT_Uncertainty: parseFloat(vitUncertainty.toFixed(1)),
      ResNet_Anomaly: parseFloat(resnetAnomaly.toFixed(1)),
      ViT_Anomaly: parseFloat(vitAnomaly.toFixed(1)),
    };
  }).reverse();

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Phân tích Chuyên sâu Thời gian thực (Real-time Metrics)</h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.5rem' }}>
        
        {/* Biểu đồ 1: Tốc độ phản hồi (Latency) */}
        <div className={styles.chartSection}>
          <h3 className={styles.sectionTitle}>Độ trễ xử lý AI (Processing Latency)</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Thời gian tính bằng mili-giây (ms) mô hình cần để đưa ra kết quả.</p>
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
                <YAxis tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
                <Tooltip contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
                <Legend />
                <Line type="monotone" name="ResNet Latency (ms)" dataKey="ResNet_Latency" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                <Line type="monotone" name="ViT Latency (ms)" dataKey="ViT_Latency" stroke="#10b981" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Biểu đồ 2: Độ bất định (Uncertainty Score) */}
        <div className={styles.chartSection}>
          <h3 className={styles.sectionTitle}>Độ bất định (Uncertainty Score)</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Mức độ "phân vân" của AI. 100% = Hên xui, 0% = Cực kỳ tự tin.</p>
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
                <Tooltip contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
                <Legend />
                <Line type="stepAfter" name="ResNet Uncertainty (%)" dataKey="ResNet_Uncertainty" stroke="#f59e0b" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                <Line type="stepAfter" name="ViT Uncertainty (%)" dataKey="ViT_Uncertainty" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Biểu đồ 3: Chỉ số bất thường khu vực (Regional Anomaly Index) */}
        <div className={styles.chartSection} style={{ gridColumn: '1 / -1' }}>
          <h3 className={styles.sectionTitle}>Chỉ số Bất thường Khu vực (Regional Anomaly Index)</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Trung bình điểm XAI của Mắt, Miệng, Da và Ánh sáng (Phản ánh độ biến dạng của khuôn mặt).</p>
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorResNet" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorViT" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ec4899" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
                <Tooltip contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
                <Legend />
                <Area type="monotone" name="ResNet Anomaly Index" dataKey="ResNet_Anomaly" stroke="#ef4444" fillOpacity={1} fill="url(#colorResNet)" />
                <Area type="monotone" name="ViT Anomaly Index" dataKey="ViT_Anomaly" stroke="#ec4899" fillOpacity={1} fill="url(#colorViT)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      <div className={styles.listSection}>
        <h3 className={styles.sectionTitle}>Lịch sử Quét (Recent Scans)</h3>
        <div className={styles.grid}>
          {history.map((scan) => {
            const resnetFake = scan.results?.resnet?.isFake;
            return (
              <div key={scan.id} className={styles.historyCard} onClick={() => {
                setCurrentScan(scan);
                onScanSelect();
              }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={scan.originalImageUrl} alt="Scan" className={styles.image} />
                <div className={styles.info}>
                  <p className={styles.time}>{new Date(scan.timestamp).toLocaleString()}</p>
                  <p>ResNet: <span style={{ color: resnetFake ? '#ef4444' : '#10b981', fontWeight: 'bold' }}>{resnetFake ? 'FAKE' : 'REAL'}</span></p>
                  <p>ViT: <span style={{ color: scan.results?.vit?.isFake ? '#ef4444' : '#10b981', fontWeight: 'bold' }}>{scan.results?.vit?.isFake ? 'FAKE' : 'REAL'}</span></p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
