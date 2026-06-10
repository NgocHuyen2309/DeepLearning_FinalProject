package com.deepfake.core_backend.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Entity
@Table(name = "scan_history")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ScanHistory {

    @Id
    private String id;
    
    private long timestamp;
    @Column(columnDefinition = "TEXT")
    private String originalImageUrl;
    
    private boolean resnetIsFake;
    private double resnetConfidence;
    
    @Column(columnDefinition = "TEXT")
    private String resnetHeatmapUrl;
    
    private boolean vitIsFake;
    private double vitConfidence;
    
    @Column(columnDefinition = "TEXT")
    private String vitHeatmapUrl;
    
    // ResNet Regional Scores
    private Integer resnetEyes;
    private Integer resnetMouth;
    private Integer resnetSkinTexture;
    private Integer resnetLighting;
    private Integer resnetBackground;
    
    // ViT Regional Scores
    private Integer vitEyes;
    private Integer vitMouth;
    private Integer vitSkinTexture;
    private Integer vitLighting;
    private Integer vitBackground;
    
    // Tốc độ phản hồi (Latency)
    private Long resnetProcessingTimeMs;
    private Long vitProcessingTimeMs;
}
