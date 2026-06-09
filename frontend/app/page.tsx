'use client';

import React, { useState, useRef, useEffect } from 'react';
import styles from './page.module.css';
import ImageUploader from '@/components/ui/ImageUploader';
import PredictionCard from '@/components/ui/PredictionCard';
import HeatmapSlider from '@/components/ui/HeatmapSlider';
import SkeletonLoader from '@/components/ui/SkeletonLoader';
import NavigationSidebar from '@/components/ui/NavigationSidebar';
import RadarAnalysis from '@/components/ui/RadarAnalysis';
import HistoryDashboard from '@/components/ui/HistoryDashboard';
import { usePredictionStore } from '@/store/usePredictionStore';
import { Download, Moon, Sun } from 'lucide-react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export default function Home() {
  const { isProcessing, currentScan, addToHistory, setProcessing, setCurrentScan, reset } = usePredictionStore();
  const [activeModel, setActiveModel] = useState<'resnet' | 'vit'>('resnet');
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [activeTab, setActiveTab] = useState<'analysis' | 'history'>('analysis');
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Load lịch sử từ Backend mỗi khi refresh trang
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/scans`);
        if (res.ok) {
          const historyData = await res.json();
          const mappedHistory = historyData.map((data: any) => ({
            id: data.id,
            timestamp: data.timestamp,
            originalImageUrl: data.originalImageUrl ? `${API_BASE_URL}${data.originalImageUrl}` : '',
            results: {
              resnet: {
                isFake: data.resnetIsFake,
                confidence: data.resnetConfidence,
                heatmapUrl: data.resnetHeatmapUrl ? `${API_BASE_URL}${data.resnetHeatmapUrl}` : '',
                regionalScores: {
                  eyes: data.resnetEyes || 0,
                  mouth: data.resnetMouth || 0,
                  skinTexture: data.resnetSkinTexture || 0,
                  lighting: data.resnetLighting || 0,
                  background: data.resnetBackground || 0
                },
                processingTimeMs: data.resnetProcessingTimeMs || 0
              },
              vit: {
                isFake: data.vitIsFake,
                confidence: data.vitConfidence,
                heatmapUrl: data.vitHeatmapUrl ? `${API_BASE_URL}${data.vitHeatmapUrl}` : '',
                regionalScores: {
                  eyes: data.vitEyes || 0,
                  mouth: data.vitMouth || 0,
                  skinTexture: data.vitSkinTexture || 0,
                  lighting: data.vitLighting || 0,
                  background: data.vitBackground || 0
                },
                processingTimeMs: data.vitProcessingTimeMs || 0
              }
            }
          }));
          usePredictionStore.setState({ history: mappedHistory });
        }
      } catch (error) {
        console.error("Lỗi khi lấy lịch sử:", error);
      }
    };
    fetchHistory();
  }, []);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const handleImageUpload = async (file: File, url: string) => {
    reset();
    setProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Gọi API xuống Java Core Backend
      const response = await fetch(`${API_BASE_URL}/api/v1/scans`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Lỗi khi gửi ảnh tới Backend');
      }

      const data = await response.json();
      
      // Jackson bên Java tự serialize boolean isFake thành 'fake' nên ta map lại cho an toàn
      const newScan = {
        id: data.id || Math.random().toString(36).substr(2, 9),
        timestamp: data.timestamp || Date.now(),
        originalImageUrl: data.originalImageUrl ? `${API_BASE_URL}${data.originalImageUrl}` : url,
        results: {
          resnet: { 
            isFake: data.results?.resnet?.isFake ?? data.results?.resnet?.fake ?? false, 
            confidence: data.results?.resnet?.confidence || 0, 
            heatmapUrl: data.results?.resnet?.heatmapUrl 
              ? `${API_BASE_URL}${data.results.resnet.heatmapUrl}` 
              : url,
            regionalScores: data.results?.resnet?.regionalScores || null,
            processingTimeMs: data.results?.resnet?.processingTimeMs || 0
          },
          vit: { 
            isFake: data.results?.vit?.isFake ?? data.results?.vit?.fake ?? false, 
            confidence: data.results?.vit?.confidence || 0, 
            heatmapUrl: data.results?.vit?.heatmapUrl 
              ? `${API_BASE_URL}${data.results.vit.heatmapUrl}` 
              : url,
            regionalScores: data.results?.vit?.regionalScores || null,
            processingTimeMs: data.results?.vit?.processingTimeMs || 0
          }
        }
      };
      
      setCurrentScan(newScan);
      addToHistory(newScan);
    } catch (error) {
      console.error("Lỗi:", error);
      alert("Đã xảy ra lỗi khi kết nối với máy chủ!");
    } finally {
      setProcessing(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!reportRef.current) return;
    
    try {
      const canvas = await html2canvas(reportRef.current, { scale: 2 });
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`Forensics_Report_${Date.now()}.pdf`);
    } catch (error) {
      console.error("Failed to generate PDF", error);
      alert("Failed to generate PDF");
    }
  };

  // Dữ liệu mặc định khi chưa có ảnh
  const mockRadarData = [
    { subject: 'Eyes', A: 0, fullMark: 100 },
    { subject: 'Mouth', A: 0, fullMark: 100 },
    { subject: 'Skin Texture', A: 0, fullMark: 100 },
    { subject: 'Lighting', A: 0, fullMark: 100 },
    { subject: 'Background', A: 0, fullMark: 100 },
  ];

  const generateRadarData = () => {
    if (!currentScan) return mockRadarData;
    const modelResult = currentScan.results?.[activeModel];
    if (!modelResult || !modelResult.regionalScores) return mockRadarData;

    const scores = modelResult.regionalScores;
    return [
      { subject: 'Eyes', A: scores.eyes || 5, fullMark: 100 },
      { subject: 'Mouth', A: scores.mouth || 5, fullMark: 100 },
      { subject: 'Skin Texture', A: scores.skinTexture || 5, fullMark: 100 },
      { subject: 'Lighting', A: scores.lighting || 5, fullMark: 100 },
      { subject: 'Background', A: scores.background || 5, fullMark: 100 },
    ];
  };

  const currentRadarData = generateRadarData();

  return (
    <div className={styles.layout}>
      <NavigationSidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <div className={styles.content}>
        <div className={styles.container}>
          <header className={styles.header}>
            <div className={styles.titleBox}>
              <h1 className={styles.title}>Deepfake Forensics</h1>
              <p className={styles.subtitle}>Advanced Dual-Model Analysis & Reporting</p>
            </div>
            
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <button 
                onClick={toggleTheme}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  padding: '0.75rem', backgroundColor: 'var(--card-bg)',
                  color: 'var(--text-primary)', borderRadius: '50%',
                  boxShadow: 'var(--shadow-md)', border: '1px solid var(--border-color)'
                }}
              >
                {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
              </button>

              {activeTab === 'analysis' && currentScan && (
                <button 
                  onClick={handleDownloadPDF}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '0.5rem',
                    padding: '0.75rem 1.5rem', backgroundColor: 'var(--primary-color)',
                    color: 'white', borderRadius: 'var(--radius-md)', fontWeight: 'bold',
                    boxShadow: 'var(--shadow-md)'
                  }}
                >
                  <Download size={20} />
                  Export PDF
                </button>
              )}
            </div>
          </header>

          <main className={styles.main}>
            {activeTab === 'analysis' ? (
              <>
                <section className={styles.uploadSection}>
                  <h2 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Upload Target</h2>
                  <ImageUploader onImageUpload={handleImageUpload} isProcessing={isProcessing} />
                </section>

                <section className={styles.resultSection} ref={reportRef}>
                  <h2 style={{ marginBottom: '1rem', color: 'var(--text-primary)' }}>Analysis Report</h2>
                  
                  {isProcessing && <SkeletonLoader />}
                  
                  {!isProcessing && !currentScan && (
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                      Upload an image to start forensics
                    </div>
                  )}

                  {!isProcessing && currentScan && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', flex: 1 }}>
                      <div style={{ display: 'flex', gap: '1rem' }}>
                        <button 
                          onClick={() => setActiveModel('resnet')}
                          style={{ 
                            flex: 1, padding: '0.5rem', borderRadius: 'var(--radius-md)', 
                            backgroundColor: activeModel === 'resnet' ? 'var(--primary-color)' : 'transparent',
                            color: activeModel === 'resnet' ? 'white' : 'var(--primary-color)',
                            border: '1px solid var(--primary-color)', fontWeight: 'bold'
                          }}
                        >
                          ResNet-50 Analysis
                        </button>
                        <button 
                          onClick={() => setActiveModel('vit')}
                          style={{ 
                            flex: 1, padding: '0.5rem', borderRadius: 'var(--radius-md)', 
                            backgroundColor: activeModel === 'vit' ? 'var(--primary-color)' : 'transparent',
                            color: activeModel === 'vit' ? 'white' : 'var(--primary-color)',
                            border: '1px solid var(--primary-color)', fontWeight: 'bold'
                          }}
                        >
                          ViT Analysis
                        </button>
                      </div>

                      {activeModel === 'resnet' && currentScan.results?.resnet && (
                        <>
                          <PredictionCard modelName="ResNet-50 (Local Artifacts)" isFake={currentScan.results.resnet.isFake} confidence={currentScan.results.resnet.confidence} />
                          <HeatmapSlider originalImage={currentScan.originalImageUrl} heatmapImage={currentScan.results.resnet.heatmapUrl} />
                        </>
                      )}

                      {activeModel === 'vit' && currentScan.results?.vit && (
                        <>
                          <PredictionCard modelName="ViT (Global Context)" isFake={currentScan.results.vit.isFake} confidence={currentScan.results.vit.confidence} />
                          <HeatmapSlider originalImage={currentScan.originalImageUrl} heatmapImage={currentScan.results.vit.heatmapUrl} />
                        </>
                      )}
                      
                      <RadarAnalysis data={currentRadarData} />
                    </div>
                  )}
                </section>
              </>
            ) : (
              <section style={{ gridColumn: '1 / -1' }}>
                <HistoryDashboard onScanSelect={() => setActiveTab('analysis')} />
              </section>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
