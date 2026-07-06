import { Menu, SunMedium, Moon } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface HeaderProps {
  onToggleSidebar: () => void
}

export function Header({ onToggleSidebar }: HeaderProps }: HeaderProps) {
  return (
    <header className="flex h-16 items-center justify-between px-6 bg-white border-b">
      <div className="flex items-center space-x-4">
        <Button variant="ghost" size="icon" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          <Menu className="h-4 w-4" />
        </Button>
        <h1 className="text-xl font-semibold text-gray-800">Jarvis AI Dashboard</h1>
      </div>
      <div className="flex items-center space-x-3">
        <button className="btn btn-ghost">
          <SunMedium className="h-4 w-4" />
        </button>
        <button className="btn btn-ghost">
          <Moon className="h-4 w-4" />
        </button>
      </div>
    </header>
  )
}