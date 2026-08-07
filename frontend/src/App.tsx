import { Route, Routes } from 'react-router-dom';
import NavBar from './components/NavBar';
import Sidebar from './components/Sidebar';
import ChatPage from './pages/ChatPage';
import ConflictsPage from './pages/ConflictsPage';
import ConnectorsPage from './pages/ConnectorsPage';
import EntityGraphPage from './pages/EntityGraphPage';
import MemoriesPage from './pages/MemoriesPage';
import PatrolPage from './pages/PatrolPage';

export default function App() {
  return (
    <div className="flex h-screen bg-white text-gray-900">
      <NavBar />
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/memories" element={<MemoriesPage />} />
          <Route path="/graph" element={<EntityGraphPage />} />
          <Route path="/connectors" element={<ConnectorsPage />} />
          <Route path="/patrol" element={<PatrolPage />} />
          <Route path="/conflicts" element={<ConflictsPage />} />
        </Routes>
      </main>
    </div>
  );
}
