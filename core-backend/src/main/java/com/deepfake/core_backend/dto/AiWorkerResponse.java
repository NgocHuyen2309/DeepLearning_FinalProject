package com.deepfake.core_backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class AiWorkerResponse {
    private String model;
    private String prediction; // "Real" or "Fake"
    private double confidence;
    @JsonProperty("heatmap_base64")
    private String heatmapBase64;
    
    @JsonProperty("regional_scores")
    private RegionalScoresDto regionalScores;
    
    @JsonProperty("processing_time_ms")
    private Long processingTimeMs;
}
