import { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Menu, Sun, Moon, Bell } from 'lucide-react';
import { clsx } from 'clsx';
import { Sidebar } from './Sidebar';

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('cliniq_theme') === 'dark' ||
        (!localStorage.getItem('cliniq_theme') &&
          window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    return false;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
      localStorage.setItem('cliniq_theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('cliniq_theme', 'light');
    }
  }, [darkMode]);

  return (
    <div className="min-h-screen bg-surface-dim">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main content */}
      <div className="lg:pl-[var(--sidebar-width)]">
        {/* Top bar */}
        <header className="sticky top-0 z-20 h-16 bg-surface/80 backdrop-blur-md border-b border-border">
          <div className="flex items-center justify-between h-full px-4 lg:px-6">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              >
                <Menu className="w-5 h-5" />
              </button>
              <div className="hidden sm:block">
                <h2 className="text-sm font-medium text-text-muted">
                  Clinical NLP Platform
                </h2>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Notifications */}
              <button
                className={clsx(
                  'relative p-2 rounded-lg transition-colors',
                  'text-text-muted hover:text-text-primary',
                  'hover:bg-gray-100 dark:hover:bg-gray-800'
                )}
              >
                <Bell className="w-5 h-5" />
                <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-red-500" />
              </button>

              {/* Dark mode toggle */}
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={clsx(
                  'p-2 rounded-lg transition-colors',
                  'text-text-muted hover:text-text-primary',
                  'hover:bg-gray-100 dark:hover:bg-gray-800'
                )}
                title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {darkMode ? (
                  <Sun className="w-5 h-5" />
                ) : (
                  <Moon className="w-5 h-5" />
                )}
              </button>

              {/* User avatar */}
              <div className="ml-2 w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center cursor-pointer">
                <span className="text-xs font-semibold text-white">AD</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
