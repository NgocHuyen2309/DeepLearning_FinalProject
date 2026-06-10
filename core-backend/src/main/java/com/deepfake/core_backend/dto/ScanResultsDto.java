package com.deepfake.core_backend.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class ScanResultsDto {
    private ModelResultDto resnet;
    private ModelResultDto vit;
}
