interface SidebarProps {
  selectedCategory: string;
  onSelectCategory: (category: string) => void;
}

const categories = ['Face', 'Lips', 'Nose', 'Eyebrows'];

export function Sidebar({ selectedCategory, onSelectCategory }: SidebarProps) {
  return (
    <div className="w-48 flex-shrink-0 space-y-6">
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-3 transition-colors">Categories</h2>
        <div className="space-y-1">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => onSelectCategory(category)}
              className={`w-full text-left px-4 py-2.5 rounded-lg transition-all ${
                selectedCategory === category
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-md shadow-blue-500/30'
                  : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>
      
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-3 transition-colors">Quick Compare</h2>
        <div className="grid grid-cols-2 gap-2">
          <div className="aspect-square bg-gradient-to-br from-slate-700 to-slate-800 dark:from-slate-600 dark:to-slate-700 rounded-lg shadow-md transition-colors"></div>
          <div className="aspect-square bg-gradient-to-br from-slate-700 to-slate-800 dark:from-slate-600 dark:to-slate-700 rounded-lg shadow-md transition-colors"></div>
        </div>
      </div>
      
      <div>
        <h2 className="text-slate-900 dark:text-slate-100 mb-3 transition-colors">History</h2>
        <div className="bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-700 dark:to-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-600 transition-colors">
          <div className="space-y-1.5 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-slate-700 dark:text-slate-300 transition-colors">Patient 1</span>
            </div>
            <div className="flex items-center justify-between text-slate-600 dark:text-slate-400 transition-colors">
              <span>#A</span>
              <span className="text-slate-500 dark:text-slate-400 transition-colors">View</span>
              <span className="text-slate-500 dark:text-slate-400 transition-colors">Re-transform</span>
            </div>
            <div className="space-y-0.5 text-slate-600 dark:text-slate-400 transition-colors">
              <div>Lips</div>
              <div>60% •</div>
              <div>Nose</div>
              <div>30%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}