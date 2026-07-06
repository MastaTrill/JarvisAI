import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { DashboardOverview } from './components/dashboard/overview'
import { Sidebar } from './components/layout/sidebar'
import { Header } from './components/layout/header'
import { Toaster } from '@/components/ui/use-toast'

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-gray-50">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Header />
          <main className="flex-1 p-6 overflow-y-auto">
            <Routes>
              <Route path="/" element={<DashboardOverview />} />
              <Route path="/models" element={<div>Models page coming soon</div>} />
              <Route path="/agents" element={<div>Agents page coming soon</div>} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
        <Toaster />
      </div>
    </BrowserRouter>
  )
}

export default App