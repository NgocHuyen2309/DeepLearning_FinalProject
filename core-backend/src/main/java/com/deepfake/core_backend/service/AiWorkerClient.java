package com.deepfake.core_backend.service;

import com.deepfake.core_backend.dto.AiWorkerResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;

@Service
public class AiWorkerClient {

    private final WebClient webClient;

    @Autowired
    public AiWorkerClient(WebClient aiWorkerWebClient) {
        this.webClient = aiWorkerWebClient;
    }

    public CompletableFuture<AiWorkerResponse> analyzeWithResnet(MultipartFile file) throws IOException {
        return callWorkerApi("/predict/resnet", file);
    }

    public CompletableFuture<AiWorkerResponse> analyzeWithVit(MultipartFile file) throws IOException {
        return callWorkerApi("/predict/vit", file);
    }

    private CompletableFuture<AiWorkerResponse> callWorkerApi(String endpoint, MultipartFile file) throws IOException {
        byte[] fileBytes = file.getBytes();
        ByteArrayResource resource = new ByteArrayResource(fileBytes) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);

        return webClient.post()
                .uri(endpoint)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(body))
                .retrieve()
                .bodyToMono(AiWorkerResponse.class)
                .toFuture();
    }
}
