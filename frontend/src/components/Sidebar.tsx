import { NavLink } from 'react-router-dom';
import { clsx } from 'clsx';
import {
  LayoutDashboard,
  Upload,
  Tags,
  FileCode2,
  FileText,
  ShieldAlert,
  Clock,
  Cpu,
  Activity,
  X,
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/upload', label: 'Upload', icon: Upload },
  { to: '/entities', label: 'Entities', icon: Tags },
  { to: '/icd-codes', label: 'ICD Codes', icon: FileCode2 },
  { to: '/summary', label: 'Summary', icon: FileText },
  { to: '/risk', label: 'Risk', icon: ShieldAlert },
  { to: '/timeline', label: 'Timeline', icon: Clock },
  { to: '/models', label: 'Models', icon: Cpu },
];

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed top-0 left-0 z-50 h-full bg-surface border-r border-border',
          'flex flex-col transition-transform duration-200 ease-in-out',
          'w-[var(--sidebar-width)]',
          'lg:translate-x-0 lg:z-30',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Logo area */}
        <div className="flex items-center justify-between h-16 px-5 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary-500">
              <Activity className="w-4.5 h-4.5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-text-primary tracking-tight">
                ClinIQ
              </h1>
              <p className="text-[10px] font-medium text-text-muted uppercase tracking-wider -mt-0.5">
                Clinical NLP
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-3">
          <div className="space-y-1">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                onClick={onClose}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary-500 text-white shadow-sm'
                      : 'text-text-secondary hover:text-text-primary hover:bg-gray-100 dark:hover:bg-gray-800'
                  )
                }
                end={to === '/'}
              >
                <Icon className="w-[18px] h-[18px] flex-shrink-0" />
                {label}
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-3 px-2">
            <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center">
              <span className="text-xs font-semibold text-primary-600 dark:text-primary-300">
                CQ
              </span>
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium text-text-primary truncate">
                ClinIQ Platform
              </p>
              <p className="text-xs text-text-muted">v0.1.0</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
