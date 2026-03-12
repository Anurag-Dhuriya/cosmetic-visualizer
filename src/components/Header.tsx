import { Moon, Sun } from 'lucide-react';
import { useTheme } from './../contexts/ThemeContext';

export function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex items-center justify-between px-6 py-5 border-b border-slate-200 dark:border-slate-700 bg-gradient-to-r from-slate-50 to-white dark:from-slate-800 dark:to-slate-800 transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
          <span className="text-white font-semibold text-lg">CV</span>
        </div>
        <div>
          <h1 className="text-slate-900 dark:text-slate-100 transition-colors">Cosmetic Visualizer</h1>
          <p className="text-slate-500 dark:text-slate-400 text-sm transition-colors">Sidebar preview • Demo UI</p>
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        <button
          onClick={toggleTheme}
          className="p-2.5 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-all border border-slate-200 dark:border-slate-600"
          aria-label="Toggle theme"
        >
          {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
        </button>
        <button className="px-5 py-2.5 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40">
          Save
        </button>
        <button className="px-5 py-2.5 bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-600 transition-all border border-slate-200 dark:border-slate-600">
          History
        </button>
      </div>
    </div>
  );
}