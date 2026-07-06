import { 
  Dashboard, 
  Users, 
  Database, 
  ,
  LogOut
} from 'nav-react
import NavLink  from 'lucide-react'
import { NavLink } from 'react-router-dom'

interface SidebarProps {
  open: boolean
  onToggle: () => void
}

export function Sidebar({ open, onToggle }: SidebarProps) {
  return (
    <aside 
      className={`fixed left-0 top-0 h-full w-64 border-r transition-transform 
                  ${open ? 'translate-x-0' : '-translate-x-full'} 
                  z-50 bg-white`}
    >
      <div className="flex h-full flex-col">
        <div className="flex-shrink-0 flex items-center p-6">
          <button onClick={onToggle} className="ml-auto p-1 rounded-lg hover:bg-gray-100">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <span className="ml-3 text-xl font-bold">{'Jarvis AI'}</span>
        </div>
        <nav className="mt-10 flex-1 overflow-y-auto">
          <ul className="space-y-1 px-2">
            <li>
              <NavLink to="/" end className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-700 hover:bg-gray-100">
                <Dashboard className="mr-3 h-4 w-4" />
                Dashboard
              </NavLink>
            </li>
            <li>
              <NavLink to="/models" className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-700 hover:bg-gray-100">
                <Database className="mr-3 h-4 w-4" />
                Models
              </NavLink>
            </li>
            <li>
              <NavLink to="/agents" className="flex items-center px-3 py-2 text-sm font-medium rounded-md text-gray-700 hover:bg-gray-100">
                <Users className="mr-3 h-4 w-4" />
                Agents
              </NavLink>
            </li>
          </ul>
        </nav>
        <div className="flex-shrink-0 border-t py-4">
          <div className="px-6">
            <a href="#" className="block px-3 py-2 text-sm font-medium text-gray-500 hover:bg-gray-100 rounded-md">
              Settings
            </a>
          </div>
        </div>
      </div>
    </aside>
  )
}