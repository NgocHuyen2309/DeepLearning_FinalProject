package com.deepfake.core_backend.service;

import com.deepfake.core_backend.dto.AiWorkerResponse;
import com.deepfake.core_backend.dto.ModelResultDto;
import com.deepfake.core_backend.dto.ScanResponse;
import com.deepfake.core_backend.dto.ScanResultsDto;
import com.deepfake.core_backend.entity.ScanHistory;
import com.deepfake.core_backend.repository.ScanHistoryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Service
public class ScanOrchestratorService {

    private final StorageService storageService;
    private final AiWorkerClient aiWorkerClient;
    private final ScanHistoryRepository scanHistoryRepository;

    @Autowired
    public ScanOrchestratorService(StorageService storageService, 
                                   AiWorkerClient aiWorkerClient, 
                                   ScanHistoryRepository scanHistoryRepository) {
        this.storageService = storageService;
        this.aiWorkerClient = aiWorkerClient;
        this.scanHistoryRepository = scanHistoryRepository;
    }

    public ScanResponse processScan(MultipartFile file) throws Exception {
        String scanId = UUID.randomUUID().toString();
        long timestamp = System.currentTimeMillis();

        // 1. Lưu file gốc
        String originalImageUrl = storageService.saveOriginalImage(file);

        // 2. Gọi song song 2 mô hình AI
        CompletableFuture<AiWorkerResponse> resnetFuture = aiWorkerClient.analyzeWithResnet(file);
        CompletableFuture<AiWorkerResponse> vitFuture = aiWorkerClient.analyzeWithVit(file);

        // Chờ cả 2 hoàn thành
        CompletableFuture.allOf(resnetFuture, vitFuture).join();

        AiWorkerResponse resnetResponse = resnetFuture.get();
        AiWorkerResponse vitResponse = vitFuture.get();

        // 3. Giải mã và lưu Heatmap Base64 thành file ảnh
        String resnetHeatmapUrl = storageService.saveBase64Heatmap(resnetResponse.getHeatmapBase64(), "resnet", scanId);
        String vitHeatmapUrl = storageService.saveBase64Heatmap(vitResponse.getHeatmapBase64(), "vit", scanId);

        // 4. Lưu kết quả vào Database
        ScanHistory history = new ScanHistory();
        history.setId(scanId);
        history.setTimestamp(timestamp);
        history.setOriginalImageUrl(originalImageUrl);
        
        history.setResnetIsFake("Fake".equalsIgnoreCase(resnetResponse.getPrediction()));
        history.setResnetConfidence(resnetResponse.getConfidence());
        history.setResnetHeatmapUrl(resnetHeatmapUrl);
        if (resnetResponse.getRegionalScores() != null) {
            history.setResnetEyes(resnetResponse.getRegionalScores().getEyes());
            history.setResnetMouth(resnetResponse.getRegionalScores().getMouth());
            history.setResnetSkinTexture(resnetResponse.getRegionalScores().getSkinTexture());
            history.setResnetLighting(resnetResponse.getRegionalScores().getLighting());
            history.setResnetBackground(resnetResponse.getRegionalScores().getBackground());
        }
        history.setResnetProcessingTimeMs(resnetResponse.getProcessingTimeMs());
        
        history.setVitIsFake("Fake".equalsIgnoreCase(vitResponse.getPrediction()));
        history.setVitConfidence(vitResponse.getConfidence());
        history.setVitHeatmapUrl(vitHeatmapUrl);
        if (vitResponse.getRegionalScores() != null) {
            history.setVitEyes(vitResponse.getRegionalScores().getEyes());
            history.setVitMouth(vitResponse.getRegionalScores().getMouth());
            history.setVitSkinTexture(vitResponse.getRegionalScores().getSkinTexture());
            history.setVitLighting(vitResponse.getRegionalScores().getLighting());
            history.setVitBackground(vitResponse.getRegionalScores().getBackground());
        }
        history.setVitProcessingTimeMs(vitResponse.getProcessingTimeMs());

        scanHistoryRepository.save(history);

        // 5. Build DTO trả về cho Frontend
        ModelResultDto resnetDto = new ModelResultDto(
            history.isResnetIsFake(), history.getResnetConfidence(), resnetHeatmapUrl, resnetResponse.getRegionalScores(), resnetResponse.getProcessingTimeMs()
        );
        ModelResultDto vitDto = new ModelResultDto(
            history.isVitIsFake(), history.getVitConfidence(), vitHeatmapUrl, vitResponse.getRegionalScores(), vitResponse.getProcessingTimeMs()
        );
        
        ScanResultsDto resultsDto = new ScanResultsDto(resnetDto, vitDto);

        return new ScanResponse(scanId, timestamp, originalImageUrl, resultsDto);
    }
}
