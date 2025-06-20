
import React from 'react';
import WardrobeManager from '@/components/WardrobeManager';
import ErrorBoundary from '@/components/ErrorBoundary';

const WardrobePage = () => {
  return (
    <ErrorBoundary>
      <WardrobeManager />
    </ErrorBoundary>
  );
};

export default WardrobePage;
