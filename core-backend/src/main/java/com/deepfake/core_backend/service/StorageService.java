package com.deepfake.core_backend.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import jakarta.annotation.PostConstruct;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.UUID;

@Service
public class StorageService {

    @Value("${storage.upload-dir}")
    private String uploadDir;

    @PostConstruct
    public void init() {
        try {
            Files.createDirectories(Paths.get(uploadDir));
        } catch (IOException e) {
            throw new RuntimeException("Could not create upload directory!");
        }
    }

    public String saveOriginalImage(MultipartFile file) {
        try {
            String filename = UUID.randomUUID().toString() + "_" + file.getOriginalFilename();
            Path root = Paths.get(uploadDir);
            Files.copy(file.getInputStream(), root.resolve(filename));
            return "/uploads/" + filename;
        } catch (Exception e) {
            throw new RuntimeException("Could not store the file. Error: " + e.getMessage());
        }
    }

    public String saveBase64Heatmap(String base64String, String modelName, String scanId) {
        if (base64String == null || base64String.isEmpty()) return null;
        try {
            String cleanBase64 = base64String;
            if (base64String.contains(",")) {
                cleanBase64 = base64String.split(",")[1];
            }
            
            byte[] decodedBytes = Base64.getDecoder().decode(cleanBase64);
            String filename = scanId + "_" + modelName + "_heatmap.png";
            File outputFile = new File(uploadDir, filename);
            
            try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                fos.write(decodedBytes);
            }
            return "/uploads/" + filename;
        } catch (Exception e) {
            throw new RuntimeException("Error decoding and saving Base64: " + e.getMessage());
        }
    }

    @org.springframework.scheduling.annotation.Scheduled(cron = "0 0 0 * * ?") // Chạy vào lúc 00:00 hàng ngày
    public void cleanupOldFiles() {
        try {
            Path dirPath = Paths.get(uploadDir);
            long threshold = System.currentTimeMillis() - (48L * 60 * 60 * 1000); // 48 hours ago
            Files.list(dirPath).forEach(file -> {
                try {
                    File f = file.toFile();
                    if (f.lastModified() < threshold) {
                        f.delete();
                    }
                } catch (Exception e) {
                    // Ignore delete error
                }
            });
        } catch (IOException e) {
            System.err.println("Failed to cleanup storage: " + e.getMessage());
        }
    }
}
