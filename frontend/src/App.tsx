import { Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatPage from './pages/ChatPage';
import MemoriesPage from './pages/MemoriesPage';

export default function App() {
  return (
    <div className="flex h-screen bg-white text-gray-900">
      <Sidebar />
      <main className="flex-1 min-w-0 overflow-hidden">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/memories" element={<MemoriesPage />} />
        </Routes>
      </main>
    </div>
  );
}
