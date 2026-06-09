'use client';

import React, { useState, useRef, ChangeEvent, DragEvent } from 'react';
import styles from './ImageUploader.module.css';
import { UploadCloud } from 'lucide-react';

interface ImageUploaderProps {
  onImageUpload: (file: File, url: string) => void;
  isProcessing?: boolean;
}

export default function ImageUploader({ onImageUpload, isProcessing = false }: ImageUploaderProps) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files.length > 0) {
      processFiles(e.target.files);
    }
  };

  const processFiles = (fileList: FileList) => {
    // Demo processing first file. In a full batch system, this would queue them.
    const file = fileList[0];
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      onImageUpload(file, url);
    } else {
      alert("Please upload an image file.");
    }
  };

  return (
    <div 
      className={`${styles.uploaderContainer} ${dragActive ? styles.dragActive : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => !isProcessing && inputRef.current?.click()}
      style={{ opacity: isProcessing ? 0.5 : 1, cursor: isProcessing ? 'not-allowed' : 'pointer' }}
    >
      <input 
        ref={inputRef}
        type="file" 
        accept="image/*"
        multiple
        onChange={handleChange} 
        className={styles.hiddenInput}
        disabled={isProcessing}
      />
      
      <UploadCloud className={styles.icon} />
      <p className={styles.text}>{isProcessing ? 'Processing in Queue...' : 'Drag & Drop multiple images here'}</p>
      <p className={styles.subText}>{isProcessing ? 'Please wait' : 'or click to browse from your computer'}</p>
    </div>
  );
}
