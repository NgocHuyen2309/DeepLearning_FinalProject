package com.deepfake.core_backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class RegionalScoresDto {
    private Integer eyes;
    private Integer mouth;
    @JsonProperty("skin_texture")
    private Integer skinTexture;
    private Integer lighting;
    private Integer background;
}
