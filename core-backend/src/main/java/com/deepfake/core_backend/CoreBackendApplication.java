package com.deepfake.core_backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class CoreBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(CoreBackendApplication.class, args);
	}

}
