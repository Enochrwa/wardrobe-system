
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { AuthProvider } from '@/contexts/AuthContext'; // Added AuthProvider import
import IndexPage from './pages/IndexPage';
import DashboardPage from './pages/DashboardPage';
import WardrobePage from './pages/WardrobePage';
import CommunityPage from './pages/CommunityPage';
import ProfilePage from './pages/ProfilePage';
import SettingsPage from './pages/SettingsPage';
import AdminPage from './pages/AdminPage';
import NotFoundPage from './pages/NotFoundPage';
import Header from './components/Header';
import AIStudioPage from './pages/AIStudioPage';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider> {/* Added AuthProvider wrapper */}
        <Router>
          <div className="min-h-screen bg-background">
            <Header />
          <main className="pt-16">
            <Routes>
              <Route path="/" element={<IndexPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/wardrobe" element={<WardrobePage />} />
              <Route path="/ai-studio/*" element={<AIStudioPage />} />
              <Route path="/community" element={<CommunityPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/admin" element={<AdminPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </main>
        </div>
      </Router>
      </AuthProvider> {/* Closed AuthProvider wrapper */}
    </ThemeProvider>
  );
}

export default App;
