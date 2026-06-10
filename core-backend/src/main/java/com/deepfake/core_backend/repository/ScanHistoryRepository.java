package com.deepfake.core_backend.repository;

import com.deepfake.core_backend.entity.ScanHistory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ScanHistoryRepository extends JpaRepository<ScanHistory, String> {
    List<ScanHistory> findAllByOrderByTimestampDesc();
}
