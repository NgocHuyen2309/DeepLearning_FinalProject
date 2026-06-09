package com.deepfake.core_backend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ModelResultDto {
    private boolean isFake;
    private double confidence;
    private String heatmapUrl;
    private RegionalScoresDto regionalScores;
    private Long processingTimeMs;
}
