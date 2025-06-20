
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import AIOutfitAnalyzer from '@/components/AIOutfitAnalyzer';
import AIEventRecommender from '@/components/AIEventRecommender';
import AITrendForecasting from '@/components/AITrendForecasting';
import AIStyleInsights from '@/components/AIStyleInsights';

// mockWardrobeItems constant removed as AIEventRecommender now fetches its own data.

const AIStudioPage = () => {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/ai-studio/analyzer" replace />} />
      <Route path="/analyzer" element={<AIOutfitAnalyzer />} />
      <Route path="/event-recommender" element={<AIEventRecommender />} />
      <Route path="/insights" element={<AIStyleInsights />} />
      <Route path="/trends" element={<AITrendForecasting />} />
    </Routes>
  );
};

export default AIStudioPage;
