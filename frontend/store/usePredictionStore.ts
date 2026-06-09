import { create } from 'zustand';

export interface ScanResult {
  id: string;
  timestamp: number;
  originalImageUrl: string;
  results: {
    resnet: { 
      isFake: boolean; confidence: number; heatmapUrl: string; 
      regionalScores?: { eyes: number; mouth: number; skinTexture: number; lighting: number; background: number; };
      processingTimeMs?: number;
    } | null;
    vit: { 
      isFake: boolean; confidence: number; heatmapUrl: string; 
      regionalScores?: { eyes: number; mouth: number; skinTexture: number; lighting: number; background: number; };
      processingTimeMs?: number;
    } | null;
  } | null;
}

interface PredictionState {
  isProcessing: boolean;
  currentScan: ScanResult | null;
  history: ScanResult[];
  
  setProcessing: (status: boolean) => void;
  setCurrentScan: (scan: ScanResult | null) => void;
  addToHistory: (scan: ScanResult) => void;
  clearHistory: () => void;
  reset: () => void;
}

export const usePredictionStore = create<PredictionState>((set) => ({
  isProcessing: false,
  currentScan: null,
  history: [],
  
  setProcessing: (status) => set({ isProcessing: status }),
  setCurrentScan: (scan) => set({ currentScan: scan }),
  addToHistory: (scan) => set((state) => {
    // Keep only last 5 scans in memory
    const newHistory = [scan, ...state.history.filter(h => h.id !== scan.id)].slice(0, 5);
    return { history: newHistory };
  }),
  clearHistory: () => set({ history: [] }),
  reset: () => set({ isProcessing: false, currentScan: null }),
}));
