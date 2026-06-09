package com.deepfake.core_backend.controller;

import com.deepfake.core_backend.dto.ScanResponse;
import com.deepfake.core_backend.entity.ScanHistory;
import com.deepfake.core_backend.repository.ScanHistoryRepository;
import com.deepfake.core_backend.service.ScanOrchestratorService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api/v1/scans")
public class ScanController {

    private final ScanOrchestratorService orchestratorService;
    private final ScanHistoryRepository repository;

    @Autowired
    public ScanController(ScanOrchestratorService orchestratorService, ScanHistoryRepository repository) {
        this.orchestratorService = orchestratorService;
        this.repository = repository;
    }

    @PostMapping
    public ResponseEntity<ScanResponse> uploadAndScan(@RequestParam("file") MultipartFile file) {
        try {
            ScanResponse response = orchestratorService.processScan(file);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().build();
        }
    }

    @GetMapping
    public ResponseEntity<List<ScanHistory>> getHistory() {
        return ResponseEntity.ok(repository.findAllByOrderByTimestampDesc());
    }
}
